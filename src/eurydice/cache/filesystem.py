"""Filesystem-based cache implementation."""

import json
import time
from pathlib import Path

from eurydice.cache.base import Cache
from eurydice.types import AudioFormat, AudioResult, Voice


class FilesystemCache(Cache):
    """Disk-based cache for persistent storage."""

    def __init__(
        self,
        cache_dir: str = "~/.eurydice/cache",
        default_ttl_seconds: int | None = None,
    ):
        """
        Initialize filesystem cache.

        Args:
            cache_dir: Directory to store cache files
            default_ttl_seconds: Default TTL in seconds (None = no expiry)
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.default_ttl = default_ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_paths(self, key: str) -> tuple[Path, Path]:
        """Get paths for audio and metadata files."""
        audio_path = self.cache_dir / f"{key}.wav"
        meta_path = self.cache_dir / f"{key}.json"
        return audio_path, meta_path

    async def get(self, key: str) -> AudioResult | None:
        """Get item from the cache."""
        audio_path, meta_path = self._get_paths(key)

        if not audio_path.exists() or not meta_path.exists():
            return None

        try:
            with open(meta_path) as f:
                meta = json.load(f)

            # Check expiration
            if meta.get("expires_at") and time.time() > meta["expires_at"]:
                await self.delete(key)
                return None

            with open(audio_path, "rb") as f:
                audio_data = f.read()

            return AudioResult(
                audio_data=audio_data,
                duration=meta["duration"],
                format=AudioFormat(meta["format"]),
                sample_rate=meta["sample_rate"],
                voice=Voice(meta["voice"]),
                cached=True,
            )
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    async def set(
        self,
        key: str,
        audio: AudioResult,
        ttl_seconds: int | None = None,
    ) -> None:
        """Store item in cache."""
        audio_path, meta_path = self._get_paths(key)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl

        meta = {
            "duration": audio.duration,
            "format": audio.format.value,
            "sample_rate": audio.sample_rate,
            "voice": audio.voice.value,
            "created_at": time.time(),
            "expires_at": time.time() + ttl if ttl else None,
        }

        with open(audio_path, "wb") as f:
            f.write(audio.audio_data)

        with open(meta_path, "w") as f:
            json.dump(meta, f)

    async def delete(self, key: str) -> bool:
        """Delete item from cache."""
        audio_path, meta_path = self._get_paths(key)
        deleted = False

        for path in [audio_path, meta_path]:
            if path.exists():
                path.unlink()
                deleted = True

        return deleted

    async def clear(self) -> int:
        """Clear all items from the cache."""
        count = 0
        for f in self.cache_dir.glob("*.wav"):
            f.unlink()
            count += 1
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
        return count

    async def contains(self, key: str) -> bool:
        """Check if the key exists and is valid."""
        audio_path, _ = self._get_paths(key)
        if not audio_path.exists():
            return False
        # Also check expiration
        result = await self.get(key)
        return result is not None

    def cache_size_bytes(self) -> int:
        """Get the total size of cached files in bytes."""
        total = 0
        for f in self.cache_dir.glob("*.wav"):
            total += f.stat().st_size
        return total

    async def close(self) -> None:
        """Clean up resources."""
        pass