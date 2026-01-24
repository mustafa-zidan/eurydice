"""Abstract base class for cache implementations."""

from abc import ABC, abstractmethod
from typing import Optional

from orpheus_tts.types import AudioResult


class Cache(ABC):
    """Abstract base class for audio caching."""

    @abstractmethod
    async def get(self, key: str) -> Optional[AudioResult]:
        """
        Retrieve cached audio by key.

        Args:
            key: Cache key

        Returns:
            AudioResult if found and not expired, None otherwise
        """
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        audio: AudioResult,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Store audio in cache.

        Args:
            key: Cache key
            audio: Audio result to cache
            ttl_seconds: Time-to-live in seconds (None = use default or no expiry)
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete cached item.

        Args:
            key: Cache key

        Returns:
            True if item existed and was deleted
        """
        pass

    @abstractmethod
    async def clear(self) -> int:
        """
        Clear all cached items.

        Returns:
            Count of items cleared
        """
        pass

    @abstractmethod
    async def contains(self, key: str) -> bool:
        """
        Check if key exists in cache (and not expired).

        Args:
            key: Cache key

        Returns:
            True if key exists and is valid
        """
        pass

    async def close(self) -> None:
        """Clean up resources. Override if needed."""
        pass
