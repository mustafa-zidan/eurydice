"""Token parsing utilities for Orpheus TTS."""

from typing import Optional

# Token processing constants
CUSTOM_TOKEN_PREFIX = "<custom_token_"


class TokenProcessor:
    """Processes tokens from LLM output into audio codec values."""

    def __init__(self):
        self._buffer: list[int] = []
        self._count: int = 0

    def reset(self) -> None:
        """Reset the token buffer."""
        self._buffer = []
        self._count = 0

    @property
    def buffer(self) -> list[int]:
        """Get the current token buffer."""
        return self._buffer

    @property
    def count(self) -> int:
        """Get the current token count."""
        return self._count

    def parse_token(self, token_string: str, index: int) -> Optional[int]:
        """
        Convert token string to numeric ID for audio processing.

        Args:
            token_string: Raw token string from LLM (e.g., "<custom_token_4102>")
            index: Current token index in the sequence

        Returns:
            Token ID for audio codec, or None if invalid
        """
        token_string = token_string.strip()

        last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)
        if last_token_start == -1:
            return None

        last_token = token_string[last_token_start:]

        if last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">"):
            try:
                number_str = last_token[14:-1]
                token_id = int(number_str) - 10 - ((index % 7) * 4096)
                return token_id
            except ValueError:
                return None

        return None

    def process_token(self, token_string: str) -> Optional[int]:
        """
        Process a token and add it to the buffer if valid.

        Args:
            token_string: Raw token string from LLM

        Returns:
            Token ID if valid and added, None otherwise
        """
        token_id = self.parse_token(token_string, self._count)
        if token_id is not None and token_id > 0:
            self._buffer.append(token_id)
            self._count += 1
            return token_id
        return None

    def has_complete_frame(self) -> bool:
        """Check if buffer has enough tokens for audio conversion."""
        return self._count % 7 == 0 and self._count > 27

    def get_frame_tokens(self) -> list[int]:
        """Get the last 28 tokens for audio conversion."""
        return self._buffer[-28:]
