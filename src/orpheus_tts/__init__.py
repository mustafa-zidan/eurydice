"""
Orpheus TTS - Text-to-speech library using Orpheus TTS model.

Example usage:
    from orpheus_tts import OrpheusTTS, Voice

    # Async usage
    async with OrpheusTTS() as tts:
        audio = await tts.generate("Hello, world!", voice=Voice.LEO)
        audio.save("hello.wav")

    # Sync usage
    tts = OrpheusTTS()
    audio = tts.generate_sync("Hello, world!")
    audio.save("hello.wav")

With caching:
    from orpheus_tts import OrpheusTTS, TTSConfig, FilesystemCache

    config = TTSConfig(cache_enabled=True)
    cache = FilesystemCache("~/.orpheus-tts/cache")

    async with OrpheusTTS(config, cache=cache) as tts:
        # First call generates audio
        audio1 = await tts.generate("Hello!")

        # Second call returns cached audio
        audio2 = await tts.generate("Hello!")
        assert audio2.cached == True
"""

from orpheus_tts.client import OrpheusTTS
from orpheus_tts.config import TTSConfig, GenerationParams, SAMPLE_RATE
from orpheus_tts.types import Voice, AudioFormat, AudioResult
from orpheus_tts.exceptions import (
    OrpheusError,
    ConfigurationError,
    ProviderError,
    ConnectionError,
    ModelNotFoundError,
    AudioDecodingError,
    CacheError,
    DependencyError,
)

# Providers
from orpheus_tts.providers import Provider, LMStudioProvider

# Cache
from orpheus_tts.cache import Cache, MemoryCache, FilesystemCache, generate_cache_key

# Audio utilities
from orpheus_tts.audio import is_audio_available

__version__ = "0.1.0"

__all__ = [
    # Main client
    "OrpheusTTS",
    # Configuration
    "TTSConfig",
    "GenerationParams",
    "SAMPLE_RATE",
    # Types
    "Voice",
    "AudioFormat",
    "AudioResult",
    # Exceptions
    "OrpheusError",
    "ConfigurationError",
    "ProviderError",
    "ConnectionError",
    "ModelNotFoundError",
    "AudioDecodingError",
    "CacheError",
    "DependencyError",
    # Providers
    "Provider",
    "LMStudioProvider",
    # Cache
    "Cache",
    "MemoryCache",
    "FilesystemCache",
    "generate_cache_key",
    # Utilities
    "is_audio_available",
]
