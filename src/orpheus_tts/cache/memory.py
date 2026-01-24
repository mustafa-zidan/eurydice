"""In-memory LRU cache implementation."""

import time
from collections import OrderedDict
from typing import Optional, Tuple

from orpheus_tts.cache.base import Cache
from orpheus_tts.types import AudioResult, AudioFormat, Voice


class MemoryCache(Cache):
    """In-memory LRU cache with optional TTL."""

    def __init__(
        self,
        max_size: int = 100,
        default_ttl_seconds: Optional[int] = None,
    ):
        """
        Initialize memory cache.

        Args:
            max_size: Maximum number of items to store
            default_ttl_seconds: Default TTL in seconds (None = no expiry)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl_seconds
        self._cache: OrderedDict[str, Tuple[AudioResult, Optional[float]]] = (
            OrderedDict()
        )

    async def get(self, key: str) -> Optional[AudioResult]:
        """Get item from cache."""
        if key not in self._cache:
            return None

        audio, expires_at = self._cache[key]

        # Check expiration
        if expires_at and time.time() > expires_at:
            del self._cache[key]
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)

        # Return copy with cached flag set
        return AudioResult(
            audio_data=audio.audio_data,
            duration=audio.duration,
            format=audio.format,
            sample_rate=audio.sample_rate,
            voice=audio.voice,
            cached=True,
        )

    async def set(
        self,
        key: str,
        audio: AudioResult,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Store item in cache."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expires_at = time.time() + ttl if ttl else None

        # Remove oldest if at capacity
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        self._cache[key] = (audio, expires_at)

    async def delete(self, key: str) -> bool:
        """Delete item from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    async def clear(self) -> int:
        """Clear all items from cache."""
        count = len(self._cache)
        self._cache.clear()
        return count

    async def contains(self, key: str) -> bool:
        """Check if key exists and is valid."""
        result = await self.get(key)
        return result is not None

    @property
    def size(self) -> int:
        """Current number of items in cache."""
        return len(self._cache)
