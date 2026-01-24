"""Configuration for eurydice."""

from dataclasses import dataclass, field

from eurydice.types import Voice

# Audio constants
SAMPLE_RATE = 24000  # SNAC model uses 24kHz


@dataclass
class GenerationParams:
    """Parameters for speech generation."""

    temperature: float = 0.6
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    max_tokens: int = 4096


@dataclass
class TTSConfig:
    """Main configuration for OrpheusTTS client."""

    # Provider settings
    provider: str = "lmstudio"  # "lmstudio", "ollama", "embedded"
    server_url: str | None = None  # Auto-detected if None
    model: str = "orpheus-3b-0.1-ft"

    # Default voice
    default_voice: Voice = Voice.LEO

    # Generation defaults
    generation: GenerationParams = field(default_factory=GenerationParams)

    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int | None = None  # None = no expiry

    # Audio settings
    sample_rate: int = SAMPLE_RATE

    # Connection settings
    timeout: float = 120.0

    def get_server_url(self) -> str:
        """Get server URL with provider-specific defaults."""
        if self.server_url:
            return self.server_url
        defaults = {
            "lmstudio": "http://localhost:1234/v1",
            "ollama": "http://localhost:11434",
            "embedded": "",
        }
        return defaults.get(self.provider, "http://localhost:1234/v1")
