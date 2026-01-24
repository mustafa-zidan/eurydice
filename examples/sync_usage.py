#!/usr/bin/env python3
"""Example of synchronous usage for Eurydice TTS."""

from eurydice import Eurydice, TTSConfig, Voice


def main():
    """Generate speech using synchronous API."""
    print("=== Sync Usage Example ===\n")

    # Create TTS client with config
    config = TTSConfig(
        default_voice=Voice.LEO,
        cache_enabled=True,
    )
    tts = Eurydice(config)

    # Generate using sync wrapper
    print("Generating speech...")
    audio = tts.generate_sync("Hello! This is a synchronous example.")

    # Save the audio
    audio.save("output_sync.wav")
    print(f"Saved: output_sync.wav ({audio.duration:.2f}s)")

    # Generate with different voice
    audio2 = tts.generate_sync(
        "And this is Mia speaking synchronously.",
        voice=Voice.MIA,
    )
    audio2.save("output_sync_mia.wav")
    print(f"Saved: output_sync_mia.wav ({audio2.duration:.2f}s)")

    # Show audio info
    print("\nAudio Info:")
    print(f"  Format: {audio.format.value}")
    print(f"  Sample Rate: {audio.sample_rate} Hz")
    print(f"  Voice: {audio.voice.value}")
    print(f"  Cached: {audio.cached}")
    print(f"  Size: {len(audio.audio_data) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
