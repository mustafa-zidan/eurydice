### Orpheus TTS Development Guide

This document provides essential information for developers working on the `orpheus-tts` project.

---

### 1. Build & Configuration

The project uses `hatchling` as the build backend and `uv` or `pip` for dependency management.

#### Environment Setup
- **Base installation**: `uv sync`
- **With audio support**: `uv sync --extra audio`
- **With full development tools**: `uv sync --all-extras`

#### Key Dependencies
- `httpx`: Used for provider communication (e.g., LM Studio).
- `snac`, `torch`, `numpy`: Required for audio decoding.
- `transformers`, `accelerate`: Required for the embedded provider.

---

### 2. Testing Information

Testing is handled by `pytest` and `pytest-asyncio`.

#### Running Tests
- **All tests**: `pytest`
- **Specific file**: `pytest tests/test_providers/test_base.py`
- **With coverage**: `pytest --cov=orpheus_tts`

#### Adding New Tests
- Place tests in the `tests/` directory following the naming convention `test_*.py`.
- Use fixtures defined in `tests/conftest.py` (e.g., `mock_provider`, `memory_cache`).
- Mark asynchronous tests with `@pytest.mark.asyncio`.

#### Simple Test Example
Below is a demonstration of how to create a simple test with a mock provider.

```python
import pytest
from orpheus_tts import OrpheusTTS, Voice
from orpheus_tts.providers.base import Provider
from orpheus_tts.config import GenerationParams
from typing import AsyncIterator

class SimpleMockProvider(Provider):
    @property
    def name(self) -> str: return "mock"
    async def connect(self) -> bool: return True
    async def close(self) -> None: pass
    async def generate_tokens(self, text, voice, params) -> AsyncIterator[str]:
        for token in ["test", "tokens"]:
            yield token

@pytest.mark.asyncio
async def test_minimal_flow():
    provider = SimpleMockProvider()
    tts = OrpheusTTS(provider=provider)
    assert await tts.connect() is True
```

---

### 3. Additional Development Information

#### Code Style
- **Linter/Formatter**: The project uses `ruff`.
- **Line Length**: 100 characters.
- **Type Hinting**: Mandatory for all new functions and classes.
- **Configuration**: Managed via `pyproject.toml`.

#### Provider Implementation
To add a new inference provider, inherit from `orpheus_tts.providers.base.Provider` and implement the abstract methods. Ensure `generate_tokens` is an `AsyncIterator`.

#### Audio Decoding
Audio decoding depends on the `snac` library. If you are working on the audio pipeline, ensure you have the `[audio]` extra installed.
