"""LM Studio inference provider."""

import json
from typing import AsyncIterator, Optional

import httpx

from orpheus_tts.config import GenerationParams
from orpheus_tts.exceptions import ConnectionError, ProviderError
from orpheus_tts.providers.base import Provider
from orpheus_tts.types import Voice


class LMStudioProvider(Provider):
    """LM Studio inference provider using OpenAI-compatible API."""

    def __init__(
        self,
        server_url: str = "http://localhost:1234/v1",
        model: str = "orpheus-3b-0.1-ft",
        timeout: float = 120.0,
    ):
        """
        Initialize the LM Studio provider.

        Args:
            server_url: LM Studio server URL (e.g., "http://localhost:1234/v1")
            model: Model name to use
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def name(self) -> str:
        return "lmstudio"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def connect(self) -> bool:
        """Test connection to LM Studio server."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.server_url}/models")
            return response.status_code == 200
        except httpx.ConnectError:
            return False
        except Exception:
            return False

    async def generate_tokens(
        self,
        text: str,
        voice: Voice,
        params: GenerationParams,
    ) -> AsyncIterator[str]:
        """Stream tokens from LM Studio."""
        prompt = self._format_prompt(text, voice)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "repeat_penalty": params.repetition_penalty,
            "stream": True,
        }

        client = await self._get_client()
        try:
            async with client.stream(
                "POST",
                f"{self.server_url}/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status_code != 200:
                    raise ProviderError(f"LM Studio API error: {response.status_code}")

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                token_text = data["choices"][0].get("text", "")
                                if token_text:
                                    yield token_text
                        except json.JSONDecodeError:
                            continue

        except httpx.ConnectError as e:
            raise ConnectionError(
                f"Cannot connect to LM Studio at {self.server_url}. Is it running?"
            ) from e

    def _format_prompt(self, text: str, voice: Voice) -> str:
        """Format prompt with voice and special tokens."""
        return f"<|audio|>{voice.value}: {text}<|eot_id|>"

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
