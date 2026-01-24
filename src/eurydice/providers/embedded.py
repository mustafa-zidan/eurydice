"""Embedded inference provider for running Orpheus locally."""

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from eurydice.config import GenerationParams
from eurydice.exceptions import ConnectionError, DependencyError, ProviderError
from eurydice.providers.base import Provider
from eurydice.types import Voice

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer
else:
    AutoModelForCausalLM = None
    AutoTokenizer = None


def _check_dependencies() -> None:
    """Check if embedded dependencies are available."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        raise DependencyError(
            "transformers",
            "pip install eurydice-tts[embedded]",
        ) from e


class EmbeddedProvider(Provider):
    """
    Embedded inference provider for running Orpheus model locally.

    This provider loads the Orpheus TTS model directly using transformers,
    allowing text-to-speech generation without requiring an external server.

    Example:
        provider = EmbeddedProvider(model="canopylabs/orpheus-3b-0.1-ft")
        async with provider:
            async for token in provider.generate_tokens("Hello!", Voice.LEO, params):
                print(token)
    """

    def __init__(
        self,
        model: str = "canopylabs/orpheus-3b-0.1-ft",
        device: str | None = None,
        torch_dtype: str = "auto",
        server_url: str = "",  # Unused, for API compatibility
        timeout: float = 120.0,  # Unused, for API compatibility
    ):
        """
        Initialize the embedded provider.

        Args:
            model: HuggingFace model ID or local path
            device: Device to use ("cuda", "mps", "cpu", or None for auto)
            torch_dtype: Torch dtype for model loading ("auto", "float16", "bfloat16", "float32")
            server_url: Unused, present for API compatibility with other providers
            timeout: Unused, present for API compatibility with other providers
        """
        self.model_id = model
        self._device = device
        self._torch_dtype = torch_dtype
        self._model: AutoModelForCausalLM | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "embedded"

    def _detect_device(self) -> str:
        """Auto-detect the best available device."""
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _get_torch_dtype(self):
        """Get the torch dtype from string specification."""
        import torch

        if self._torch_dtype == "auto":
            # Use bfloat16 if available, else float16 for GPU, float32 for CPU
            device = self._device or self._detect_device()
            if device == "cpu":
                return torch.float32
            if hasattr(torch, "bfloat16"):
                return torch.bfloat16
            return torch.float16

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self._torch_dtype, torch.float32)

    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        _check_dependencies()

        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self._device or self._detect_device()
        dtype = self._get_torch_dtype()

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None,
            low_cpu_mem_usage=True,
        )

        # Move to device if not using device_map
        if device == "cpu":
            self._model = self._model.to(device)

        self._model.eval()
        self._loaded = True

    async def connect(self) -> bool:
        """
        Load the model and verify it's ready for inference.

        Returns:
            True if model loaded successfully
        """
        try:
            if not self._loaded:
                # Run model loading in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_model)
            return True
        except Exception as e:
            raise ConnectionError(f"Failed to load model {self.model_id}: {e}") from e

    async def generate_tokens(
        self,
        text: str,
        voice: Voice,
        params: GenerationParams,
    ) -> AsyncIterator[str]:
        """
        Stream tokens from the locally loaded model.

        Args:
            text: Text to convert to speech
            voice: Voice to use
            params: Generation parameters

        Yields:
            Individual token strings as they're generated (e.g., "<custom_token_12345>")
        """
        if not self._loaded or self._model is None or self._tokenizer is None:
            raise ProviderError("Model not loaded. Call connect() first.")

        prompt = self._format_prompt(text, voice)

        # Tokenize input
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # Generate tokens using streaming
        loop = asyncio.get_event_loop()

        # Use a queue for async streaming
        token_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def generate_in_thread():
            """Run generation in thread and put tokens in queue."""

            def _generate():
                import torch

                with torch.no_grad():
                    generated_ids = inputs["input_ids"].clone()

                    for _ in range(params.max_tokens):
                        outputs = self._model(
                            input_ids=generated_ids,
                            use_cache=True,
                        )
                        next_token_logits = outputs.logits[:, -1, :]

                        # Apply temperature
                        if params.temperature > 0:
                            next_token_logits = next_token_logits / params.temperature

                        # Apply top_p (nucleus) sampling
                        if params.top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(
                                next_token_logits, descending=True
                            )
                            cumulative_probs = torch.cumsum(
                                torch.softmax(sorted_logits, dim=-1), dim=-1
                            )
                            sorted_indices_to_remove = cumulative_probs > params.top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                ..., :-1
                            ].clone()
                            sorted_indices_to_remove[..., 0] = 0

                            indices_to_remove = sorted_indices_to_remove.scatter(
                                1, sorted_indices, sorted_indices_to_remove
                            )
                            next_token_logits[indices_to_remove] = float("-inf")

                        # Sample next token
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)

                        # Check for end of sequence
                        if next_token.item() == self._tokenizer.eos_token_id:
                            break

                        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                        # Decode just the new token
                        token_str = self._tokenizer.decode(
                            next_token[0], skip_special_tokens=False
                        )

                        # Put token in queue (blocking call from sync context)
                        asyncio.run_coroutine_threadsafe(
                            token_queue.put(token_str), loop
                        ).result()

                # Signal completion
                asyncio.run_coroutine_threadsafe(token_queue.put(None), loop).result()

            await loop.run_in_executor(None, _generate)

        # Start generation task
        gen_task = asyncio.create_task(generate_in_thread())

        # Yield tokens as they arrive
        try:
            while True:
                token = await token_queue.get()
                if token is None:
                    break
                yield token
        finally:
            await gen_task

    def _format_prompt(self, text: str, voice: Voice) -> str:
        """Format prompt with voice and special tokens."""
        return f"<|audio|>{voice.value}: {text}<|eot_id|>"

    async def close(self) -> None:
        """Clean up model resources."""
        if self._model is not None:
            import torch

            # Free GPU memory
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._tokenizer = None
        self._loaded = False
