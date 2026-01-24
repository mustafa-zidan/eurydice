"""Exception hierarchy for Eurydice."""


class EurydiceError(Exception):
    """Base exception for all Eurydice errors."""

    pass


# Backwards compatibility alias
OrpheusError = EurydiceError


class ConfigurationError(EurydiceError):
    """Invalid configuration."""

    pass


class ProviderError(EurydiceError):
    """Error from inference provider."""

    pass


class ConnectionError(ProviderError):
    """Cannot connect to provider."""

    pass


class ModelNotFoundError(ProviderError):
    """Requested model not available."""

    pass


class AudioDecodingError(EurydiceError):
    """Error decoding audio from tokens."""

    pass


class CacheError(EurydiceError):
    """Error with cache operations."""

    pass


class DependencyError(EurydiceError):
    """Required dependency not installed."""

    def __init__(self, package: str, install_hint: str = ""):
        self.package = package
        self.install_hint = install_hint or f"pip install {package}"
        super().__init__(
            f"Required package '{package}' not installed. Install with: {self.install_hint}"
        )
