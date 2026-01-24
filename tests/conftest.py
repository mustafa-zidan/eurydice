"""Pytest fixtures for orpheus-tts tests."""

import pytest
from typing import AsyncIterator, Optional

from orpheus_tts import (
    OrpheusTTS,
    TTSConfig,
    Voice,
    AudioResult,
    AudioFormat,
    GenerationParams,
)
from orpheus_tts.providers.base import Provider
from orpheus_tts.cache import MemoryCache, FilesystemCache


class MockProvider(Provider):
    """Mock provider for testing without network."""

    def __init__(self):
        self.connected = True
        self._tokens: list[str] = []

    @property
    def name(self) -> str:
        return "mock"

    async def connect(self) -> bool:
        return self.connected

    async def generate_tokens(
        self,
        text: str,
        voice: Voice,
        params: GenerationParams,
    ) -> AsyncIterator[str]:
        for token in self._tokens:
            yield token

    async def close(self) -> None:
        pass

    def set_tokens(self, tokens: list[str]) -> None:
        """Set tokens to return during generation."""
        self._tokens = tokens


@pytest.fixture
def mock_provider():
    """Create a mock provider."""
    return MockProvider()


@pytest.fixture
def memory_cache():
    """Create an in-memory cache."""
    return MemoryCache(max_size=10)


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    return str(tmp_path / "cache")


@pytest.fixture
def filesystem_cache(temp_cache_dir):
    """Create a filesystem cache in temp directory."""
    return FilesystemCache(cache_dir=temp_cache_dir)


@pytest.fixture
def tts_config():
    """Create a test TTS config."""
    return TTSConfig(cache_enabled=False)


@pytest.fixture
def sample_audio():
    """Create a sample AudioResult for testing."""
    return AudioResult(
        audio_data=b"test audio data",
        duration=1.0,
        format=AudioFormat.WAV,
        sample_rate=24000,
        voice=Voice.LEO,
        cached=False,
    )
