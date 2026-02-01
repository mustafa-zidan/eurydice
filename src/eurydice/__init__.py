"""
Eurydice - Text-to-speech library using the Orpheus TTS model.

Named after Orpheus's wife in Greek mythology. Like Orpheus who tried to bring
Eurydice back from the underworld, this library brings text to life through speech.

Example usage:
    from eurydice import Eurydice, Voice

    # Async usage
    async with Eurydice() as tts:
        audio = await tts.generate("Hello, world!", voice=Voice.LEO)
        audio.save("hello.wav")

    # Sync usage
    tts = Eurydice()
    audio = tts.generate_sync("Hello, world!")
    audio.save("hello.wav")

With caching:
    from eurydice import Eurydice, TTSConfig, FilesystemCache

    config = TTSConfig(cache_enabled=True)
    cache = FilesystemCache("~/.eurydice/cache")

    async with Eurydice(config, cache=cache) as tts:
        # First call generates audio
        audio1 = await tts.generate("Hello!")

        # Second call returns cached audio
        audio2 = await tts.generate("Hello!")
        assert audio2.cached == True

With embedded provider (no external server needed):
    from eurydice import Eurydice, TTSConfig

    config = TTSConfig(provider="embedded")

    async with Eurydice(config) as tts:
        audio = await tts.generate("Hello from local model!")
        audio.save("hello.wav")
"""

# Audio utilities
from eurydice.audio import is_audio_available

# Cache
from eurydice.cache import Cache, FilesystemCache, MemoryCache, generate_cache_key
from eurydice.client import Eurydice, OrpheusTTS  # OrpheusTTS is a backwards compat alias
from eurydice.config import SAMPLE_RATE, GenerationParams, TTSConfig
from eurydice.exceptions import (
    AudioDecodingError,
    CacheError,
    ConfigurationError,
    ConnectionError,
    DependencyError,
    EurydiceError,
    ModelNotFoundError,
    OrpheusError,  # Backwards compatibility alias
    ProviderError,
)

# Providers
from eurydice.providers import (
    EmbeddedProvider,
    LMStudioProvider,
    OrpheusCppProvider,
    Provider,
    VLLMProvider,
)
from eurydice.types import AudioFormat, AudioResult, Voice

__version__ = "0.1.0"

__all__ = [
    # Main client
    "Eurydice",
    "OrpheusTTS",  # Backwards compatibility
    # Configuration
    "TTSConfig",
    "GenerationParams",
    "SAMPLE_RATE",
    # Types
    "Voice",
    "AudioFormat",
    "AudioResult",
    # Exceptions
    "EurydiceError",
    "OrpheusError",  # Backwards compatibility
    "ConfigurationError",
    "ProviderError",
    "ConnectionError",
    "ModelNotFoundError",
    "AudioDecodingError",
    "CacheError",
    "DependencyError",
    # Providers
    "Provider",
    "EmbeddedProvider",
    "LMStudioProvider",
    "OrpheusCppProvider",
    "VLLMProvider",
    # Cache
    "Cache",
    "MemoryCache",
    "FilesystemCache",
    "generate_cache_key",
    # Utilities
    "is_audio_available",
]
