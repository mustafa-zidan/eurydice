"""vLLM inference provider using orpheus-speech for CUDA GPUs."""

import asyncio
from collections.abc import AsyncIterator

from eurydice.config import GenerationParams
from eurydice.exceptions import ConnectionError, DependencyError, ProviderError
from eurydice.providers.base import Provider
from eurydice.types import Voice


def _check_dependencies() -> None:
    """Check if vLLM/orpheus-speech dependencies are available."""
    try:
        import torch

        if not torch.cuda.is_available():
            raise DependencyError(
                "CUDA",
                "vLLM requires CUDA. Use 'orpheus-cpp' provider for CPU/Metal.",
            )
        import orpheus_tts  # noqa: F401
    except ImportError as e:
        raise DependencyError(
            "orpheus-speech",
            "pip install eurydice-tts[vllm]",
        ) from e


class VLLMProvider(Provider):
    """
    vLLM inference provider using orpheus-speech for CUDA GPUs.

    This provider uses the orpheus-speech library which leverages vLLM
    for high-performance GPU inference. It's the fastest option for
    CUDA-enabled systems.

    Note: This provider yields audio chunks directly instead of tokens,
    bypassing the standard token processing pipeline.

    Example:
        provider = VLLMProvider(model="canopylabs/orpheus-tts-0.1-finetune-prod")
        async with provider:
            audio = await tts.generate("Hello!")
    """

    # Marker to indicate this provider yields audio, not tokens
    yields_audio: bool = True

    def __init__(
        self,
        model: str = "canopylabs/orpheus-tts-0.1-finetune-prod",
        max_model_len: int = 2048,
        dtype: str = "bfloat16",
        server_url: str = "",  # Unused, for API compatibility
        timeout: float = 120.0,  # Unused, for API compatibility
        **engine_kwargs,
    ):
        """
        Initialize the vLLM provider.

        Args:
            model: HuggingFace model ID or local path
            max_model_len: Maximum sequence length for vLLM
            dtype: Torch dtype ("bfloat16", "float16", "float32")
            server_url: Unused, present for API compatibility
            timeout: Unused, present for API compatibility
            **engine_kwargs: Additional arguments passed to vLLM engine
        """
        self.model_id = model
        self._max_model_len = max_model_len
        self._dtype = dtype
        self._engine_kwargs = engine_kwargs
        self._orpheus_model = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "vllm"

    def _get_torch_dtype(self):
        """Get the torch dtype from string specification."""
        import torch

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self._dtype, torch.bfloat16)

    def _load_model(self) -> None:
        """Load the orpheus-speech model with vLLM backend."""
        _check_dependencies()

        from orpheus_tts import OrpheusModel

        self._orpheus_model = OrpheusModel(
            model_name=self.model_id,
            dtype=self._get_torch_dtype(),
            max_model_len=self._max_model_len,
            **self._engine_kwargs,
        )
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
            raise ConnectionError(f"Failed to load vLLM model {self.model_id}: {e}") from e

    async def generate_tokens(
        self,
        text: str,
        voice: Voice,
        params: GenerationParams,
    ) -> AsyncIterator[str]:
        """
        This method is not used for VLLMProvider.

        VLLMProvider yields audio directly via generate_audio().
        This method exists only for API compatibility.

        Raises:
            ProviderError: Always raises, use generate_audio() instead
        """
        raise ProviderError(
            "VLLMProvider yields audio directly. "
            "Use generate_audio() or let Eurydice client handle this automatically."
        )
        # Make this an async generator to satisfy the type signature
        yield  # pragma: no cover

    async def generate_audio(
        self,
        text: str,
        voice: Voice,
        params: GenerationParams,
    ) -> AsyncIterator[bytes]:
        """
        Stream audio chunks from the vLLM-powered orpheus-speech model.

        Args:
            text: Text to convert to speech
            voice: Voice to use
            params: Generation parameters

        Yields:
            Audio chunks as int16 PCM bytes at 24kHz
        """
        if not self._loaded or self._orpheus_model is None:
            raise ProviderError("Model not loaded. Call connect() first.")

        loop = asyncio.get_event_loop()

        # Queue for async streaming
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        def _generate():
            """Run generation in thread and put audio chunks in queue."""
            try:
                # generate_speech yields audio chunks directly
                for audio_chunk in self._orpheus_model.generate_speech(
                    prompt=text,
                    voice=voice.value,
                    temperature=params.temperature,
                    top_p=params.top_p,
                    repetition_penalty=params.repetition_penalty,
                    max_tokens=params.max_tokens,
                ):
                    if audio_chunk is not None:
                        # audio_chunk is already bytes
                        if isinstance(audio_chunk, bytes):
                            audio_bytes = audio_chunk
                        else:
                            # Convert numpy array if needed
                            import numpy as np

                            audio_data = np.asarray(audio_chunk).flatten()
                            if audio_data.dtype != np.int16:
                                if audio_data.dtype in (np.float32, np.float64):
                                    audio_data = (audio_data * 32767).astype(np.int16)
                                else:
                                    audio_data = audio_data.astype(np.int16)
                            audio_bytes = audio_data.tobytes()

                        asyncio.run_coroutine_threadsafe(
                            audio_queue.put(audio_bytes), loop
                        ).result()
            finally:
                asyncio.run_coroutine_threadsafe(audio_queue.put(None), loop).result()

        # Start generation in thread pool
        gen_future = loop.run_in_executor(None, _generate)

        # Stream audio chunks as they arrive
        try:
            while True:
                audio_chunk = await audio_queue.get()
                if audio_chunk is None:
                    break
                yield audio_chunk
        finally:
            await gen_future

    async def close(self) -> None:
        """Clean up model resources."""
        # vLLM handles its own cleanup
        self._orpheus_model = None
        self._loaded = False
