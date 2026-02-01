"""Main Eurydice TTS client."""

import asyncio

from eurydice.audio.decoder import SNACDecoder, is_audio_available
from eurydice.audio.formats import calculate_duration, create_wav
from eurydice.audio.tokens import TokenProcessor
from eurydice.cache.base import Cache
from eurydice.cache.key import generate_cache_key
from eurydice.cache.memory import MemoryCache
from eurydice.config import GenerationParams, TTSConfig
from eurydice.exceptions import AudioDecodingError, EurydiceError
from eurydice.providers.auto import detect_best_provider, get_provider_info
from eurydice.providers.base import Provider
from eurydice.providers.embedded import EmbeddedProvider
from eurydice.providers.lmstudio import LMStudioProvider
from eurydice.providers.orpheus_cpp import OrpheusCppProvider
from eurydice.providers.vllm import VLLMProvider
from eurydice.types import AudioFormat, AudioResult, Voice


class Eurydice:
    """
    Main client for Eurydice TTS.

    Named after Orpheus's wife in Greek mythology, this library brings text to life
    through speech, serving as a bridge to the Orpheus TTS model.

    Example usage:
        # Simple async usage
        async with Eurydice() as tts:
            audio = await tts.generate("Hello, world!")
            audio.save("hello.wav")

        # With custom configuration
        config = TTSConfig(provider="lmstudio", cache_enabled=True)
        tts = Eurydice(config)
        await tts.connect()
        audio = await tts.generate("Hello!", voice=Voice.TARA)
        await tts.close()

        # Sync usage
        tts = Eurydice()
        audio = tts.generate_sync("Hello, world!")
    """

    # Provider registry
    PROVIDERS: dict[str, type[Provider]] = {
        "lmstudio": LMStudioProvider,
        "embedded": EmbeddedProvider,
        "orpheus-cpp": OrpheusCppProvider,
        "vllm": VLLMProvider,
    }

    def __init__(
        self,
        config: TTSConfig | None = None,
        provider: Provider | None = None,
        cache: Cache | None = None,
    ):
        """
        Initialize the Eurydice client.

        Args:
            config: TTS configuration (uses defaults if None)
            provider: Custom provider instance (created from config if None)
            cache: Custom cache instance (created from config if None)
        """
        self.config = config or TTSConfig()

        # Initialize provider
        if provider:
            self._provider = provider
        else:
            self._provider = self._create_provider()

        # Initialize cache
        if cache is not None:
            self._cache = cache
        elif self.config.cache_enabled:
            self._cache = MemoryCache()
        else:
            self._cache = None

        # Audio processing (lazy loaded)
        self._decoder: SNACDecoder | None = None
        self._token_processor = TokenProcessor()

    def _create_provider(self) -> Provider:
        """Create provider instance from config."""
        provider_name = self.config.provider

        # Handle auto-detection
        if provider_name == "auto":
            provider_name = detect_best_provider()

        provider_cls = self.PROVIDERS.get(provider_name)
        if not provider_cls:
            raise EurydiceError(f"Unknown provider: {provider_name}")

        return provider_cls(
            server_url=self.config.get_server_url(),
            model=self.config.model,
            timeout=self.config.timeout,
        )

    async def _generate_from_token_provider(
        self,
        text: str,
        voice: Voice,
        params: GenerationParams,
    ) -> list[bytes]:
        """Generate audio from a provider that yields tokens."""
        # Initialize decoder if needed
        if self._decoder is None:
            if not is_audio_available():
                raise EurydiceError(
                    "Audio dependencies not installed. Install with: pip install eurydice[audio]"
                )
            self._decoder = SNACDecoder()

        # Reset token processor
        self._token_processor.reset()

        audio_segments = []

        async for token in self._provider.generate_tokens(text, voice, params):
            token_id = self._token_processor.process_token(token)

            if token_id is not None and self._token_processor.has_complete_frame():
                frame_tokens = self._token_processor.get_frame_tokens()
                audio_bytes = self._decoder.decode_frame(frame_tokens)

                if audio_bytes:
                    audio_segments.append(audio_bytes)

        return audio_segments

    async def _generate_from_audio_provider(
        self,
        text: str,
        voice: Voice,
        params: GenerationParams,
    ) -> list[bytes]:
        """Generate audio from a provider that yields audio chunks directly."""
        audio_segments = []

        # Provider must have generate_audio method
        if not hasattr(self._provider, "generate_audio"):
            raise EurydiceError(
                f"Provider {self._provider.name} marked as yields_audio but missing generate_audio()"
            )

        async for audio_chunk in self._provider.generate_audio(text, voice, params):
            if audio_chunk:
                audio_segments.append(audio_chunk)

        return audio_segments

    async def connect(self) -> bool:
        """
        Test connection and initialize resources.

        Returns:
            True if connection successful
        """
        connected = await self._provider.connect()
        if connected and is_audio_available():
            self._decoder = SNACDecoder()
        return connected

    async def generate(
        self,
        text: str,
        voice: Voice | None = None,
        params: GenerationParams | None = None,
        format: AudioFormat = AudioFormat.WAV,
        use_cache: bool = True,
    ) -> AudioResult:
        """
        Generate speech from text.

        Args:
            text: The text to convert to speech
            voice: Voice to use (default from config)
            params: Generation parameters (default from config)
            format: Output audio format
            use_cache: Whether to use cache (if available)

        Returns:
            AudioResult containing the generated audio

        Raises:
            EurydiceError: If generation fails
            AudioDecodingError: If audio decoding fails
        """
        voice = voice or self.config.default_voice
        params = params or self.config.generation

        # Check cache first
        if use_cache and self._cache:
            cache_key = generate_cache_key(text, voice, params, self.config.model)
            cached = await self._cache.get(cache_key)
            if cached:
                return cached

        # Check if the provider yields audio directly (e.g., OrpheusCppProvider)
        if getattr(self._provider, "yields_audio", False):
            audio_segments = await self._generate_from_audio_provider(text, voice, params)
        else:
            audio_segments = await self._generate_from_token_provider(text, voice, params)

        if not audio_segments:
            raise AudioDecodingError("No audio generated. Check if Orpheus model is loaded.")

        # Combine and format audio
        combined = b"".join(audio_segments)
        duration = calculate_duration(combined, self.config.sample_rate)

        if format == AudioFormat.WAV:
            audio_data = create_wav(combined, self.config.sample_rate)
        else:
            audio_data = combined

        result = AudioResult(
            audio_data=audio_data,
            duration=duration,
            format=format,
            sample_rate=self.config.sample_rate,
            voice=voice,
            cached=False,
        )

        # Store in cache
        if use_cache and self._cache:
            cache_key = generate_cache_key(text, voice, params, self.config.model)
            await self._cache.set(cache_key, result, self.config.cache_ttl_seconds)

        return result

    def generate_sync(
        self,
        text: str,
        voice: Voice | None = None,
        **kwargs,
    ) -> AudioResult:
        """
        Synchronous wrapper for generate().

        Args:
            text: The text to convert to speech
            voice: Voice to use (default from config)
            **kwargs: Additional arguments passed to generate()

        Returns:
            AudioResult containing the generated audio
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.generate(text, voice, **kwargs))

    async def generate_to_file(
        self,
        text: str,
        path: str,
        voice: Voice | None = None,
        **kwargs,
    ) -> AudioResult:
        """
        Generate speech and save to file.

        Args:
            text: The text to convert to speech
            path: Path to save the audio file
            voice: Voice to use (default from config)
            **kwargs: Additional arguments passed to generate()

        Returns:
            AudioResult (audio is also saved to file)
        """
        result = await self.generate(text, voice, **kwargs)
        result.save(path)
        return result

    async def close(self) -> None:
        """Clean up resources."""
        await self._provider.close()
        if self._cache:
            await self._cache.close()

    async def __aenter__(self) -> "Eurydice":
        await self.connect()
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    # Utility methods

    @staticmethod
    def available_voices() -> list[Voice]:
        """List all available voices."""
        return list(Voice)

    async def is_available(self) -> bool:
        """Check if TTS service is available."""
        try:
            return await self._provider.connect()
        except Exception:
            return False

    @staticmethod
    def is_audio_available() -> bool:
        """Check if audio dependencies are installed."""
        return is_audio_available()

    @staticmethod
    def available_providers() -> dict[str, dict]:
        """
        Get information about all available providers.

        Returns:
            Dict mapping provider names to their availability info
        """
        return get_provider_info()

    @staticmethod
    def detect_best_provider() -> str:
        """
        Detect the best available provider based on installed dependencies.

        Returns:
            Provider name string
        """
        return detect_best_provider()


# Backwards compatibility alias
OrpheusTTS = Eurydice
