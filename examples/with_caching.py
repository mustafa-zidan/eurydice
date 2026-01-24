#!/usr/bin/env python3
"""Example demonstrating Eurydice's caching system."""

import asyncio
import time

from eurydice import Eurydice, FilesystemCache, MemoryCache, TTSConfig, Voice


async def demo_memory_cache():
    """Demonstrate in-memory caching."""
    print("=== Memory Cache Demo ===\n")

    # Create cache with 10 item limit
    cache = MemoryCache(max_size=10)
    config = TTSConfig(cache_enabled=True)

    async with Eurydice(config, cache=cache) as tts:
        text = "This is a test of the caching system."

        # First generation
        start = time.time()
        audio1 = await tts.generate(text, voice=Voice.LEO)
        time1 = time.time() - start
        print(f"First generation: {time1:.2f}s, cached={audio1.cached}")

        # Second generation (should be cached)
        start = time.time()
        audio2 = await tts.generate(text, voice=Voice.LEO)
        time2 = time.time() - start
        print(f"Second generation: {time2:.4f}s, cached={audio2.cached}")

        # Different voice (new generation)
        start = time.time()
        audio3 = await tts.generate(text, voice=Voice.TARA)
        time3 = time.time() - start
        print(f"Different voice: {time3:.2f}s, cached={audio3.cached}")

    print(f"\nCache speedup: {time1 / max(time2, 0.0001):.1f}x faster\n")


async def demo_filesystem_cache():
    """Demonstrate filesystem caching with persistence."""
    print("=== Filesystem Cache Demo ===\n")

    cache_dir = "/tmp/eurydice_cache_demo"
    cache = FilesystemCache(cache_dir=cache_dir)
    config = TTSConfig(cache_enabled=True)

    text = "This audio is cached to disk for persistence."

    # First run - generate and cache
    print("First run (generating)...")
    async with Eurydice(config, cache=cache) as tts:
        audio = await tts.generate(text)
        print(f"Generated: {audio.duration:.2f}s, cached={audio.cached}")
        audio.save("output_cached.wav")

    # Simulate new session by creating new client
    print("\nSecond run (new session, should be cached)...")
    cache2 = FilesystemCache(cache_dir=cache_dir)
    async with Eurydice(config, cache=cache2) as tts:
        audio = await tts.generate(text)
        print(f"Retrieved: {audio.duration:.2f}s, cached={audio.cached}")

    print(f"\nCache stored in: {cache_dir}")
    print(f"Cache size: {cache.cache_size_bytes() / 1024:.1f} KB\n")


async def demo_cache_ttl():
    """Demonstrate cache TTL (time-to-live)."""
    print("=== Cache TTL Demo ===\n")

    # Cache with 2 second TTL
    cache = MemoryCache(max_size=10, default_ttl_seconds=2)
    config = TTSConfig(cache_enabled=True)

    async with Eurydice(config, cache=cache) as tts:
        text = "This cache entry will expire in 2 seconds."

        # Generate
        audio1 = await tts.generate(text)
        print(f"Generated: cached={audio1.cached}")

        # Immediate retrieval
        audio2 = await tts.generate(text)
        print(f"Immediate: cached={audio2.cached}")

        # Wait for expiration
        print("Waiting 3 seconds for cache to expire...")
        await asyncio.sleep(3)

        # Should regenerate
        audio3 = await tts.generate(text)
        print(f"After expiry: cached={audio3.cached}")


async def main():
    """Run all demos."""
    await demo_memory_cache()
    await demo_filesystem_cache()
    await demo_cache_ttl()


if __name__ == "__main__":
    asyncio.run(main())
