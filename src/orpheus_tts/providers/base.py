"""Abstract base class for inference providers."""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from orpheus_tts.config import GenerationParams
from orpheus_tts.types import Voice


class Provider(ABC):
    """Abstract base class for inference providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier."""
        pass

    @abstractmethod
    async def connect(self) -> bool:
        """Test connection to the provider. Returns True if successful."""
        pass

    @abstractmethod
    async def generate_tokens(
        self,
        text: str,
        voice: Voice,
        params: GenerationParams,
    ) -> AsyncIterator[str]:
        """
        Stream tokens from the model.

        Args:
            text: Text to convert to speech
            voice: Voice to use
            params: Generation parameters

        Yields:
            Individual token strings as they're generated
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        pass

    async def __aenter__(self) -> "Provider":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()
