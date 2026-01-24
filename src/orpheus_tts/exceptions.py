"""Exception hierarchy for orpheus-tts."""


class OrpheusError(Exception):
    """Base exception for all orpheus-tts errors."""

    pass


class ConfigurationError(OrpheusError):
    """Invalid configuration."""

    pass


class ProviderError(OrpheusError):
    """Error from inference provider."""

    pass


class ConnectionError(ProviderError):
    """Cannot connect to provider."""

    pass


class ModelNotFoundError(ProviderError):
    """Requested model not available."""

    pass


class AudioDecodingError(OrpheusError):
    """Error decoding audio from tokens."""

    pass


class CacheError(OrpheusError):
    """Error with cache operations."""

    pass


class DependencyError(OrpheusError):
    """Required dependency not installed."""

    def __init__(self, package: str, install_hint: str = ""):
        self.package = package
        self.install_hint = install_hint or f"pip install {package}"
        super().__init__(
            f"Required package '{package}' not installed. "
            f"Install with: {self.install_hint}"
        )
