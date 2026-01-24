"""Audio format utilities."""

import io
import wave

from eurydice.config import SAMPLE_RATE


def create_wav(
    audio_bytes: bytes,
    sample_rate: int = SAMPLE_RATE,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """
    Create a WAV file from raw audio bytes.

    Args:
        audio_bytes: Raw PCM audio data (int16)
        sample_rate: Audio sample rate in Hz
        channels: Number of audio channels (1 = mono, 2 = stereo)
        sample_width: Bytes per sample (2 = 16-bit)

    Returns:
        Complete WAV file as bytes
    """
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)

    return wav_buffer.getvalue()


def calculate_duration(
    audio_bytes: bytes,
    sample_rate: int = SAMPLE_RATE,
    sample_width: int = 2,
) -> float:
    """
    Calculate audio duration in seconds.

    Args:
        audio_bytes: Raw PCM audio data
        sample_rate: Audio sample rate in Hz
        sample_width: Bytes per sample

    Returns:
        Duration in seconds
    """
    return len(audio_bytes) / (sample_width * sample_rate)
