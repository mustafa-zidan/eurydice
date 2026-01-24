#!/usr/bin/env python3
"""Example of batch audio generation with Eurydice."""

import asyncio
from pathlib import Path

from eurydice import Eurydice, MemoryCache, TTSConfig


async def generate_batch(texts: list[str], output_dir: str = "output"):
    """Generate audio for multiple texts."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Use caching to avoid regenerating duplicates
    cache = MemoryCache(max_size=100)
    config = TTSConfig(cache_enabled=True)

    async with Eurydice(config, cache=cache) as tts:
        for i, text in enumerate(texts, 1):
            print(f"[{i}/{len(texts)}] Generating: {text[:50]}...")

            audio = await tts.generate(text)
            filename = output_path / f"audio_{i:03d}.wav"
            audio.save(str(filename))

            status = "cached" if audio.cached else f"{audio.duration:.2f}s"
            print(f"  -> {filename.name} ({status})")

    print(f"\nGenerated {len(texts)} audio files in {output_dir}/")


async def generate_with_voices(text: str, output_dir: str = "output"):
    """Generate the same text with all available voices."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    async with Eurydice() as tts:
        for voice in Eurydice.available_voices():
            print(f"Generating with voice: {voice.value}")
            audio = await tts.generate(text, voice=voice)

            filename = output_path / f"voice_{voice.value}.wav"
            audio.save(str(filename))
            print(f"  -> {filename.name} ({audio.duration:.2f}s)")

    print(f"\nGenerated audio with all {len(Eurydice.available_voices())} voices in {output_dir}/")


async def main():
    """Run batch generation examples."""
    # Example 1: Generate multiple different texts
    texts = [
        "Welcome to the Eurydice text-to-speech demo.",
        "This is the second sentence being generated.",
        "And here's a third one for good measure.",
        "We can generate as many as we need.",
        "Each one is cached for faster retrieval later.",
    ]

    print("=== Batch Generation ===\n")
    await generate_batch(texts, "batch_output")

    # Example 2: Same text with all voices
    print("\n=== All Voices Demo ===\n")
    await generate_with_voices(
        "Hello! This is a demonstration of all available voices.",
        "voices_output",
    )


if __name__ == "__main__":
    asyncio.run(main())
