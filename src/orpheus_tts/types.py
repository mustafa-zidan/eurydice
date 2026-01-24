"""Core types for orpheus-tts."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import base64


class Voice(str, Enum):
    """Available TTS voices."""

    TARA = "tara"
    LEAH = "leah"
    JESS = "jess"
    LEO = "leo"
    DAN = "dan"
    MIA = "mia"
    ZAC = "zac"
    ZOE = "zoe"

    @classmethod
    def from_string(cls, name: str) -> "Voice":
        """Get voice from string, with fallback to default."""
        try:
            return cls(name.lower())
        except ValueError:
            return cls.LEO

    @classmethod
    def list_all(cls) -> list[str]:
        """List all available voice names."""
        return [v.value for v in cls]


class AudioFormat(str, Enum):
    """Supported audio output formats."""

    WAV = "wav"
    RAW = "raw"  # Raw PCM bytes


@dataclass
class AudioResult:
    """Result from speech generation."""

    audio_data: bytes
    duration: float
    format: AudioFormat
    sample_rate: int
    voice: Voice
    cached: bool = False

    def to_base64(self) -> str:
        """Encode audio data as base64 string."""
        return base64.b64encode(self.audio_data).decode("utf-8")

    def save(self, path: str) -> None:
        """Save audio to file."""
        with open(path, "wb") as f:
            f.write(self.audio_data)

    def as_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "audio_data": self.to_base64(),
            "duration": self.duration,
            "format": self.format.value,
            "sample_rate": self.sample_rate,
            "voice": self.voice.value,
            "cached": self.cached,
        }
