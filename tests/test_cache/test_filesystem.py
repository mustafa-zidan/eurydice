"""Tests for filesystem cache."""

import pytest

from eurydice import AudioFormat, AudioResult, Voice
from eurydice.cache import FilesystemCache


@pytest.fixture
def sample_audio():
    return AudioResult(
        audio_data=b"test audio data for filesystem",
        duration=2.0,
        format=AudioFormat.WAV,
        sample_rate=24000,
        voice=Voice.TARA,
    )


class TestFilesystemCache:
    """Tests for FilesystemCache."""

    @pytest.mark.asyncio
    async def test_set_and_get(self, filesystem_cache, sample_audio):
        """Test basic set and get operations."""
        await filesystem_cache.set("key1", sample_audio)
        result = await filesystem_cache.get("key1")

        assert result is not None
        assert result.audio_data == sample_audio.audio_data
        assert result.duration == sample_audio.duration
        assert result.voice == sample_audio.voice
        assert result.cached is True

    @pytest.mark.asyncio
    async def test_get_missing_key(self, filesystem_cache):
        """Test getting a non-existent key."""
        result = await filesystem_cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_contains(self, filesystem_cache, sample_audio):
        """Test contains method."""
        await filesystem_cache.set("key1", sample_audio)

        assert await filesystem_cache.contains("key1") is True
        assert await filesystem_cache.contains("key2") is False

    @pytest.mark.asyncio
    async def test_delete(self, filesystem_cache, sample_audio):
        """Test delete operation."""
        await filesystem_cache.set("key1", sample_audio)

        assert await filesystem_cache.delete("key1") is True
        assert await filesystem_cache.get("key1") is None
        assert await filesystem_cache.delete("key1") is False

    @pytest.mark.asyncio
    async def test_clear(self, filesystem_cache, sample_audio):
        """Test clearing the cache."""
        await filesystem_cache.set("key1", sample_audio)
        await filesystem_cache.set("key2", sample_audio)

        count = await filesystem_cache.clear()
        assert count == 2
        assert await filesystem_cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_persistence(self, temp_cache_dir, sample_audio):
        """Test that cache persists across instances."""
        # Write with first instance
        cache1 = FilesystemCache(cache_dir=temp_cache_dir)
        await cache1.set("persistent_key", sample_audio)

        # Read with a new instance
        cache2 = FilesystemCache(cache_dir=temp_cache_dir)
        result = await cache2.get("persistent_key")

        assert result is not None
        assert result.audio_data == sample_audio.audio_data

    @pytest.mark.asyncio
    async def test_cache_size_bytes(self, filesystem_cache, sample_audio):
        """Test cache size calculation."""
        initial_size = filesystem_cache.cache_size_bytes()
        assert initial_size == 0

        await filesystem_cache.set("key1", sample_audio)
        size_after = filesystem_cache.cache_size_bytes()
        assert size_after > 0

    @pytest.mark.asyncio
    async def test_close(self, filesystem_cache):
        """Test closing the cache."""
        # For FilesystemCache, close() is a pass but should be awaitable
        await filesystem_cache.close()
