"""Orpheus-cpp inference provider using llama.cpp backend."""

import asyncio
from collections.abc import AsyncIterator

from eurydice.config import GenerationParams
from eurydice.exceptions import ConnectionError, DependencyError, ProviderError
from eurydice.providers.base import Provider
from eurydice.types import Voice


def _check_dependencies() -> None:
    """Check if orpheus-cpp dependencies are available."""
    try:
        import orpheus_cpp  # noqa: F401
    except ImportError as e:
        raise DependencyError(
            "orpheus-cpp",
            "uv sync --extra cpp",
        ) from e


class OrpheusCppProvider(Provider):
    """
    Orpheus-cpp inference provider using llama.cpp backend.

    This provider uses the orpheus-cpp library which provides fast inference
    via llama.cpp. It's optimized for CPU and Apple Silicon (Metal) and
    doesn't require a GPU or external server.

    Note: This provider yields audio chunks directly instead of tokens,
    bypassing the standard token processing pipeline.

    Example:
        provider = OrpheusCppProvider()
        async with provider:
            async for token in provider.generate_tokens("Hello!", Voice.LEO, params):
                print(token)
    """

    # Marker to indicate this provider yields audio, not tokens
    yields_audio: bool = True

    def __init__(
        self,
        model_path: str | None = None,
        verbose: bool = False,
        lang: str = "en",
        server_url: str = "",  # Unused, for API compatibility
        timeout: float = 120.0,  # Unused, for API compatibility
        model: str = "",  # Unused, for API compatibility
    ):
        """
        Initialize the orpheus-cpp provider.

        Args:
            model_path: Path to GGUF model file (auto-downloads if None)
            verbose: Enable verbose output from orpheus-cpp
            lang: Language code (default: "en")
            server_url: Unused, present for API compatibility with other providers
            timeout: Unused, present for API compatibility with other providers
            model: Unused, present for API compatibility with other providers
        """
        self._model_path = model_path
        self._verbose = verbose
        self._lang = lang
        self._orpheus = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "orpheus-cpp"

    def _load_model(self) -> None:
        """Load the orpheus-cpp model."""
        _check_dependencies()

        from orpheus_cpp import OrpheusCpp

        kwargs = {
            "verbose": self._verbose,
            "lang": self._lang,
        }
        if self._model_path:
            kwargs["model_path"] = self._model_path

        self._orpheus = OrpheusCpp(**kwargs)
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
            raise ConnectionError(f"Failed to load orpheus-cpp model: {e}") from e

    async def generate_tokens(
        self,
        text: str,
        voice: Voice,
        params: GenerationParams,
    ) -> AsyncIterator[str]:
        """
        This method is not used for OrpheusCppProvider.

        OrpheusCppProvider yields audio directly via generate_audio().
        This method exists only for API compatibility.

        Raises:
            ProviderError: Always raises, use generate_audio() instead
        """
        raise ProviderError(
            "OrpheusCppProvider yields audio directly. "
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
        Stream audio chunks from the orpheus-cpp model.

        Args:
            text: Text to convert to speech
            voice: Voice to use
            params: Generation parameters (temperature, top_p used)

        Yields:
            Audio chunks as int16 PCM bytes at 24kHz
        """
        if not self._loaded or self._orpheus is None:
            raise ProviderError("Model not loaded. Call connect() first.")

        import numpy as np

        loop = asyncio.get_event_loop()

        # Create options for generation
        options = {
            "voice_id": voice.value,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "repetition_penalty": params.repetition_penalty,
        }

        # Queue for async streaming
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        def _generate():
            """Run generation in thread and put audio chunks in queue."""
            try:
                for _sample_rate, chunk in self._orpheus.stream_tts_sync(text, options=options):
                    # Convert numpy array to int16 PCM bytes
                    if isinstance(chunk, np.ndarray):
                        # Ensure proper shape and dtype
                        audio_data = chunk.flatten()
                        if audio_data.dtype != np.int16:
                            # Normalize float to int16
                            if audio_data.dtype in (np.float32, np.float64):
                                audio_data = (audio_data * 32767).astype(np.int16)
                            else:
                                audio_data = audio_data.astype(np.int16)
                        audio_bytes = audio_data.tobytes()
                    else:
                        audio_bytes = bytes(chunk)

                    asyncio.run_coroutine_threadsafe(audio_queue.put(audio_bytes), loop).result()
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
        self._orpheus = None
        self._loaded = False
