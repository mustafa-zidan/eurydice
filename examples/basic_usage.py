#!/usr/bin/env python3
"""Basic usage example for Eurydice TTS."""

import asyncio

from eurydice import Eurydice, Voice


async def main():
    """Generate speech from text."""
    # Create TTS client
    async with Eurydice() as tts:
        # Check if TTS is available
        if not await tts.is_available():
            print("TTS server not available. Make sure LM Studio is running.")
            return

        # Generate speech with default voice (Leo)
        print("Generating speech...")
        audio = await tts.generate("Hello! Welcome to Eurydice text to speech.")

        # Save to file
        audio.save("output_basic.wav")
        print(f"Saved audio to output_basic.wav ({audio.duration:.2f}s)")

        # Generate with different voice
        audio_tara = await tts.generate(
            "This is Tara speaking. How are you today?",
            voice=Voice.TARA,
        )
        audio_tara.save("output_tara.wav")
        print(f"Saved Tara's voice to output_tara.wav ({audio_tara.duration:.2f}s)")


if __name__ == "__main__":
    asyncio.run(main())
