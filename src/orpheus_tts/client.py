"""Main OrpheusTTS client."""

import asyncio
from typing import Optional, Type

from orpheus_tts.audio.decoder import SNACDecoder, is_audio_available
from orpheus_tts.audio.formats import create_wav, calculate_duration
from orpheus_tts.audio.tokens import TokenProcessor
from orpheus_tts.cache.base import Cache
from orpheus_tts.cache.key import generate_cache_key
from orpheus_tts.cache.memory import MemoryCache
from orpheus_tts.config import TTSConfig, GenerationParams
from orpheus_tts.exceptions import AudioDecodingError, OrpheusError
from orpheus_tts.providers.base import Provider
from orpheus_tts.providers.lmstudio import LMStudioProvider
from orpheus_tts.types import AudioFormat, AudioResult, Voice


class OrpheusTTS:
    """
    Main client for Orpheus TTS.

    Example usage:
        # Simple async usage
        async with OrpheusTTS() as tts:
            audio = await tts.generate("Hello, world!")
            audio.save("hello.wav")

        # With custom configuration
        config = TTSConfig(provider="lmstudio", cache_enabled=True)
        tts = OrpheusTTS(config)
        await tts.connect()
        audio = await tts.generate("Hello!", voice=Voice.TARA)
        await tts.close()

        # Sync usage
        tts = OrpheusTTS()
        audio = tts.generate_sync("Hello, world!")
    """

    # Provider registry
    PROVIDERS: dict[str, Type[Provider]] = {
        "lmstudio": LMStudioProvider,
    }

    def __init__(
        self,
        config: Optional[TTSConfig] = None,
        provider: Optional[Provider] = None,
        cache: Optional[Cache] = None,
    ):
        """
        Initialize the OrpheusTTS client.

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
        self._decoder: Optional[SNACDecoder] = None
        self._token_processor = TokenProcessor()

    def _create_provider(self) -> Provider:
        """Create provider instance from config."""
        provider_cls = self.PROVIDERS.get(self.config.provider)
        if not provider_cls:
            raise OrpheusError(f"Unknown provider: {self.config.provider}")

        return provider_cls(
            server_url=self.config.get_server_url(),
            model=self.config.model,
            timeout=self.config.timeout,
        )

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
        voice: Optional[Voice] = None,
        params: Optional[GenerationParams] = None,
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
            OrpheusError: If generation fails
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

        # Initialize decoder if needed
        if self._decoder is None:
            if not is_audio_available():
                raise OrpheusError(
                    "Audio dependencies not installed. "
                    "Install with: pip install orpheus-tts[audio]"
                )
            self._decoder = SNACDecoder()

        # Reset token processor
        self._token_processor.reset()

        audio_segments = []

        async for token in self._provider.generate_tokens(text, voice, params):
            token_id = self._token_processor.process_token(token)
            if token_id is not None:
                # Convert to audio when we have enough tokens
                if self._token_processor.has_complete_frame():
                    frame_tokens = self._token_processor.get_frame_tokens()
                    audio_bytes = self._decoder.decode_frame(frame_tokens)
                    if audio_bytes:
                        audio_segments.append(audio_bytes)

        if not audio_segments:
            raise AudioDecodingError(
                "No audio generated. Check if Orpheus model is loaded."
            )

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
        voice: Optional[Voice] = None,
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
        voice: Optional[Voice] = None,
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

    async def __aenter__(self) -> "OrpheusTTS":
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
