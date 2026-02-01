# Eurydice 🎵

> *Named after Orpheus's wife in Greek mythology. Like Orpheus who tried to bring Eurydice back from the underworld, this library brings text to life through speech.*

[![PyPI version](https://badge.fury.io/py/eurydice-tts.svg?icon=si%3Apython)](https://badge.fury.io/py/eurydice-tts)
[![Python Versions](https://img.shields.io/pypi/pyversions/eurydice-tts.svg)](https://pypi.org/project/eurydice-tts/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/mustafa-zidan/eurydice/actions/workflows/test.yml/badge.svg)](https://github.com/mustafa-zidan/eurydice/actions/workflows/test.yml)

A Python library for text-to-speech using the Orpheus TTS model, featuring audio caching and provider abstraction.

## Features

- 🎤 **8 High-Quality Voices** - tara, leah, jess, leo, dan, mia, zac, zoe
- ⚡ **Audio Caching** - Memory and filesystem caching to avoid regenerating audio
- 🔌 **Multiple Providers** - vLLM (CUDA), orpheus-cpp (CPU/Metal), embedded (transformers), LM Studio
- 🔍 **Auto-Detection** - Automatically selects the best available provider
- 🔄 **Async-First** - Built for async/await with sync wrappers for convenience
- 📦 **Type Hints** - Full type annotations throughout
- 🧪 **Well Tested** - Comprehensive test suite

## Installation

```bash
# Basic installation (requires external LM Studio server)
uv add eurydice-tts

# With audio decoding support (recommended)
uv add eurydice-tts[audio]

# With embedded model support (transformers, requires GPU)
uv add eurydice-tts[embedded]

# With orpheus-cpp support (llama.cpp, fast CPU/Metal inference)
uv add eurydice-tts[cpp]

# With vLLM support (CUDA GPU, fastest inference)
uv add eurydice-tts[vllm]

# For development
uv add eurydice-tts[dev]

# All extras
uv add eurydice-tts[all]
```

## Quick Start

### Recommended: Auto-Detection

Eurydice can automatically detect and use the best available provider:

```python
from eurydice import Eurydice, TTSConfig, Voice

# Auto-detect provider (vLLM > orpheus-cpp > embedded > lmstudio)
config = TTSConfig(provider="auto")
async with Eurydice(config) as tts:
    audio = await tts.generate("Hello, world!", voice=Voice.LEO)
    audio.save("hello.wav")
```

### Alternative: LM Studio

If using LM Studio as your provider:

1. Install [LM Studio](https://lmstudio.ai/)
2. Download and load the Orpheus TTS model (`orpheus-3b-0.1-ft`)
3. Start the LM Studio server (default: `http://localhost:1234`)

### Basic Usage

```python
from eurydice import Eurydice, Voice

# Async usage (recommended)
async with Eurydice() as tts:
    audio = await tts.generate("Hello, world!", voice=Voice.LEO)
    audio.save("hello.wav")
    print(f"Generated {audio.duration:.2f}s of audio")

# Sync usage
tts = Eurydice()
audio = tts.generate_sync("Hello, world!")
audio.save("hello.wav")
```

### With Caching

```python
from eurydice import Eurydice, TTSConfig, FilesystemCache

# Configure with filesystem cache for persistence
config = TTSConfig(cache_enabled=True)
cache = FilesystemCache("~/.eurydice/cache")

async with Eurydice(config, cache=cache) as tts:
    # First call generates audio
    audio1 = await tts.generate("Hello!")
    print(f"Cached: {audio1.cached}")  # False

    # Second call returns cached audio instantly
    audio2 = await tts.generate("Hello!")
    print(f"Cached: {audio2.cached}")  # True
```

### Custom Configuration

```python
from eurydice import Eurydice, TTSConfig, GenerationParams, Voice

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

async with Eurydice(config) as tts:
    audio = await tts.generate("Custom configuration example!")
    print(f"Duration: {audio.duration:.2f}s")
```

## Available Voices

| Voice | ID     | Description          |
|-------|--------|----------------------|
| Tara  | `tara` | Female voice         |
| Leah  | `leah` | Female voice         |
| Jess  | `jess` | Female voice         |
| Leo   | `leo`  | Male voice (default) |
| Dan   | `dan`  | Male voice           |
| Mia   | `mia`  | Female voice         |
| Zac   | `zac`  | Male voice           |
| Zoe   | `zoe`  | Female voice         |

## API Reference

### Eurydice

Main client class for text-to-speech generation.

```python
class Eurydice:
    def __init__(
        self,
        config: Optional[TTSConfig] = None,
        provider: Optional[Provider] = None,
        cache: Optional[Cache] = None,
    ) -> None: ...

    async def generate(
        self,
        text: str,
        voice: Optional[Voice] = None,
        params: Optional[GenerationParams] = None,
        format: AudioFormat = AudioFormat.WAV,
        use_cache: bool = True,
    ) -> AudioResult: ...

    def generate_sync(self, text: str, **kwargs) -> AudioResult: ...

    async def generate_to_file(
        self, text: str, path: str, **kwargs
    ) -> AudioResult: ...

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
    cached: bool           # Whether result came from cache

    def save(self, path: str) -> None: ...
    def to_base64(self) -> str: ...
```

### TTSConfig

Configuration for the TTS client.

```python
@dataclass
class TTSConfig:
    provider: str = "lmstudio"
    server_url: Optional[str] = None
    model: str = "orpheus-3b-0.1-ft"
    default_voice: Voice = Voice.LEO
    generation: GenerationParams = GenerationParams()
    cache_enabled: bool = True
    cache_ttl_seconds: Optional[int] = None
    sample_rate: int = 24000
    timeout: float = 120.0
```

## Caching

Eurydice supports two caching backends:

### MemoryCache

In-memory LRU cache (default when caching is enabled):

```python
from eurydice import MemoryCache

cache = MemoryCache(
    max_size=100,              # Maximum items to store
    default_ttl_seconds=3600,  # 1 hour TTL (optional)
)
```

### FilesystemCache

Persistent disk-based cache:

```python
from eurydice import FilesystemCache

cache = FilesystemCache(
    cache_dir="~/.eurydice/cache",
    default_ttl_seconds=86400,  # 24 hour TTL (optional)
)
```

### Cache Keys

Cache keys are content-addressed using SHA256 of:
- Input text
- Voice selection
- Generation parameters (temperature, top_p, etc.)
- Model identifier

This ensures that different configurations produce different cache entries.

## Providers

### Auto-Detection (Recommended)

Let Eurydice automatically select the best available provider:

```python
from eurydice import Eurydice, TTSConfig

# Auto-detect the best provider
config = TTSConfig(provider="auto")
async with Eurydice(config) as tts:
    audio = await tts.generate("Hello with auto-detected provider!")
    audio.save("hello.wav")

# Check what providers are available
print(Eurydice.available_providers())
# {'vllm': {'available': True, ...}, 'orpheus-cpp': {'available': False, ...}, ...}

# See which provider would be selected
print(Eurydice.detect_best_provider())
# 'vllm' (if CUDA available) or 'orpheus-cpp' (if installed) or 'embedded' or 'lmstudio'
```

**Provider priority:** vLLM (CUDA) > orpheus-cpp (CPU/Metal) > embedded (transformers) > LM Studio

### vLLM Provider (Fastest, CUDA Required)

The fastest option for NVIDIA GPUs. Uses vLLM for optimized inference:

```python
from eurydice import Eurydice, TTSConfig

# Using config
config = TTSConfig(provider="vllm")
async with Eurydice(config) as tts:
    audio = await tts.generate("Hello from vLLM!")
    audio.save("hello.wav")

# Or with custom options
from eurydice import VLLMProvider

provider = VLLMProvider(
    model="canopylabs/orpheus-tts-0.1-finetune-prod",
    max_model_len=8192,
    dtype="bfloat16",  # or "float16", "float32"
)

async with Eurydice(provider=provider) as tts:
    audio = await tts.generate("Fast GPU inference!")
```

**Requirements:**
- Install with `uv add eurydice-tts[vllm]`
- NVIDIA GPU with CUDA support
- Sufficient VRAM (8GB+ recommended)

### LM Studio (Default)

Uses the OpenAI-compatible API provided by LM Studio:

```python
from eurydice import LMStudioProvider

provider = LMStudioProvider(
    server_url="http://localhost:1234/v1",
    model="orpheus-3b-0.1-ft",
    timeout=120.0,
)
```

### Embedded Provider

Run models locally without any external server. This provider loads the Orpheus model directly using transformers:

```python
from eurydice import Eurydice, TTSConfig, EmbeddedProvider

# Using config
config = TTSConfig(provider="embedded")
async with Eurydice(config) as tts:
    audio = await tts.generate("Hello from local model!")
    audio.save("hello.wav")

# Or using provider directly
provider = EmbeddedProvider(
    model="canopylabs/orpheus-3b-0.1-ft",  # HuggingFace model ID
    device="cuda",  # or "mps" for Apple Silicon, "cpu" for CPU
    torch_dtype="auto",  # or "float16", "bfloat16", "float32"
)

async with Eurydice(provider=provider) as tts:
    audio = await tts.generate("Hello!")
```

**Requirements:** Install with `uv add eurydice-tts[embedded]` to get the required dependencies (transformers, accelerate, torch).

**Device auto-detection:** If no device is specified, the provider automatically detects the best available device (CUDA > MPS > CPU).

### Orpheus-cpp Provider (Recommended for CPU/Metal)

The fastest option for CPU and Apple Silicon. Uses llama.cpp under the hood with optimized GGUF models:

```python
from eurydice import Eurydice, TTSConfig

# Using config
config = TTSConfig(provider="orpheus-cpp")
async with Eurydice(config) as tts:
    audio = await tts.generate("Hello from llama.cpp!")
    audio.save("hello.wav")

# Or with custom options
from eurydice import OrpheusCppProvider

provider = OrpheusCppProvider(
    model_path="/path/to/model.gguf",  # Optional, auto-downloads if not specified
    verbose=False,
    lang="en",
)

async with Eurydice(provider=provider) as tts:
    audio = await tts.generate("Fast inference!")
```

**Requirements:** Install with `uv add eurydice-tts[cpp]`

**Platform support:**
- Linux/Windows: CPU inference
- macOS with Apple Silicon: Metal acceleration (very fast)

## Examples

See the [examples/](examples/) directory for more usage examples:

- `basic_usage.py` - Simple text-to-speech generation
- `with_caching.py` - Using the caching system
- `batch_generation.py` - Generating audio for multiple texts
- `sync_usage.py` - Synchronous API usage

## Development

### Setup

```bash
git clone https://github.com/mustafa-zidan/eurydice.git
cd eurydice
uv venv
source .venv/bin/activate
uv sync --all-extras
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=eurydice --cov-report=html

# Run specific test
uv run pytest tests/test_types.py -v
```

### Linting

```bash
uv run ruff check .
uv run ruff format .
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) - The underlying TTS model
- [SNAC](https://github.com/hubertsiuzdak/snac) - Neural audio codec
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM serving
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - CPU/Metal inference via orpheus-cpp
- [LM Studio](https://lmstudio.ai/) - Local LLM inference server

---

Made with ❤️ by [Mustafa Abuelfadl](https://github.com/mustafa-zidan)
