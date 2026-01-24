"""Tests for memory cache."""

import time

import pytest

from eurydice import AudioFormat, AudioResult, Voice
from eurydice.cache import MemoryCache


@pytest.fixture
def sample_audio():
    return AudioResult(
        audio_data=b"test audio data",
        duration=1.0,
        format=AudioFormat.WAV,
        sample_rate=24000,
        voice=Voice.LEO,
    )


class TestMemoryCache:
    """Tests for MemoryCache."""

    @pytest.mark.asyncio
    async def test_set_and_get(self, memory_cache, sample_audio):
        """Test basic set and get operations."""
        await memory_cache.set("key1", sample_audio)
        result = await memory_cache.get("key1")

        assert result is not None
        assert result.audio_data == sample_audio.audio_data
        assert result.cached is True

    @pytest.mark.asyncio
    async def test_get_missing_key(self, memory_cache):
        """Test getting a non-existent key."""
        result = await memory_cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_contains(self, memory_cache, sample_audio):
        """Test contains method."""
        await memory_cache.set("key1", sample_audio)

        assert await memory_cache.contains("key1") is True
        assert await memory_cache.contains("key2") is False

    @pytest.mark.asyncio
    async def test_delete(self, memory_cache, sample_audio):
        """Test delete operation."""
        await memory_cache.set("key1", sample_audio)

        assert await memory_cache.delete("key1") is True
        assert await memory_cache.get("key1") is None
        assert await memory_cache.delete("key1") is False

    @pytest.mark.asyncio
    async def test_clear(self, memory_cache, sample_audio):
        """Test clearing the cache."""
        await memory_cache.set("key1", sample_audio)
        await memory_cache.set("key2", sample_audio)

        count = await memory_cache.clear()
        assert count == 2
        assert await memory_cache.get("key1") is None
        assert await memory_cache.get("key2") is None

    @pytest.mark.asyncio
    async def test_lru_eviction(self, sample_audio):
        """Test LRU eviction when cache is full."""
        cache = MemoryCache(max_size=2)

        await cache.set("key1", sample_audio)
        await cache.set("key2", sample_audio)
        await cache.set("key3", sample_audio)

        # key1 should be evicted
        assert await cache.get("key1") is None
        assert await cache.get("key2") is not None
        assert await cache.get("key3") is not None

    @pytest.mark.asyncio
    async def test_ttl_expiration(self, sample_audio):
        """Test TTL-based expiration."""
        cache = MemoryCache(max_size=10, default_ttl_seconds=1)

        await cache.set("key1", sample_audio)
        assert await cache.get("key1") is not None

        # Wait for expiration
        time.sleep(1.1)

        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_size_property(self, memory_cache, sample_audio):
        """Test size property."""
        assert memory_cache.size == 0

        await memory_cache.set("key1", sample_audio)
        assert memory_cache.size == 1

        await memory_cache.set("key2", sample_audio)
        assert memory_cache.size == 2

    @pytest.mark.asyncio
    async def test_close(self, memory_cache, sample_audio):
        """Test closing the cache."""
        await memory_cache.set("key1", sample_audio)
        assert memory_cache.size == 1
        await memory_cache.close()
        assert memory_cache.size == 0
