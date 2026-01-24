# Orpheus TTS

Text-to-speech library using the Orpheus TTS model with audio caching and provider abstraction.

## Features

- **Multiple Providers**: Support for LM Studio (with Ollama and embedded model support planned)
- **Audio Caching**: Built-in caching to avoid regenerating the same audio
- **Multiple Voices**: 8 high-quality voices (tara, leah, jess, leo, dan, mia, zac, zoe)
- **Async-First**: Designed for async/await with sync wrappers for convenience
- **Type Hints**: Full type annotations throughout

## Installation

```bash
# Basic installation (requires external LM Studio server)
pip install orpheus-tts

# With audio decoding support (recommended)
pip install orpheus-tts[audio]

# Development installation
pip install orpheus-tts[dev]
```

## Quick Start

### Prerequisites

1. Install [LM Studio](https://lmstudio.ai/)
2. Download and load the Orpheus TTS model (`orpheus-3b-0.1-ft`)
3. Start the LM Studio server (default: `http://localhost:1234`)

### Basic Usage

```python
from orpheus_tts import OrpheusTTS, Voice

# Async usage (recommended)
async with OrpheusTTS() as tts:
    audio = await tts.generate("Hello, world!", voice=Voice.LEO)
    audio.save("hello.wav")

# Sync usage
tts = OrpheusTTS()
audio = tts.generate_sync("Hello, world!")
audio.save("hello.wav")
```

### With Caching

```python
from orpheus_tts import OrpheusTTS, TTSConfig, FilesystemCache

config = TTSConfig(cache_enabled=True)
cache = FilesystemCache("~/.orpheus-tts/cache")

async with OrpheusTTS(config, cache=cache) as tts:
    # First call generates audio
    audio1 = await tts.generate("Hello!")

    # Second call returns cached audio (much faster)
    audio2 = await tts.generate("Hello!")
    assert audio2.cached == True
```

### Custom Configuration

```python
from orpheus_tts import OrpheusTTS, TTSConfig, GenerationParams, Voice

config = TTSConfig(
    provider="lmstudio",
    server_url="http://localhost:1234/v1",
    model="orpheus-3b-0.1-ft",
    default_voice=Voice.TARA,
    cache_enabled=True,
    generation=GenerationParams(
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    ),
)

async with OrpheusTTS(config) as tts:
    audio = await tts.generate("Custom configuration example!")
    print(f"Duration: {audio.duration:.2f}s")
```

## Available Voices

| Voice | Description |
|-------|-------------|
| `tara` | Female voice |
| `leah` | Female voice |
| `jess` | Female voice |
| `leo` | Male voice (default) |
| `dan` | Male voice |
| `mia` | Female voice |
| `zac` | Male voice |
| `zoe` | Female voice |

## API Reference

### OrpheusTTS

Main client class for text-to-speech generation.

```python
class OrpheusTTS:
    def __init__(
        self,
        config: Optional[TTSConfig] = None,
        provider: Optional[Provider] = None,
        cache: Optional[Cache] = None,
    ): ...

    async def generate(
        self,
        text: str,
        voice: Optional[Voice] = None,
        params: Optional[GenerationParams] = None,
        format: AudioFormat = AudioFormat.WAV,
        use_cache: bool = True,
    ) -> AudioResult: ...

    def generate_sync(self, text: str, **kwargs) -> AudioResult: ...
    async def generate_to_file(self, text: str, path: str, **kwargs) -> AudioResult: ...
    async def is_available(self) -> bool: ...
    @staticmethod
    def available_voices() -> list[Voice]: ...
```

### AudioResult

Result object containing generated audio.

```python
@dataclass
class AudioResult:
    audio_data: bytes      # Raw audio bytes
    duration: float        # Duration in seconds
    format: AudioFormat    # WAV or RAW
    sample_rate: int       # Sample rate (24000 Hz)
    voice: Voice           # Voice used
    cached: bool           # Whether from cache

    def save(self, path: str) -> None: ...
    def to_base64(self) -> str: ...
```

## Caching

Two cache backends are available:

### MemoryCache

In-memory LRU cache (default when caching enabled):

```python
from orpheus_tts import MemoryCache

cache = MemoryCache(
    max_size=100,           # Maximum items to store
    default_ttl_seconds=3600,  # 1 hour TTL (optional)
)
```

### FilesystemCache

Persistent disk-based cache:

```python
from orpheus_tts import FilesystemCache

cache = FilesystemCache(
    cache_dir="~/.orpheus-tts/cache",
    default_ttl_seconds=86400,  # 24 hour TTL (optional)
)
```

## License

MIT License
