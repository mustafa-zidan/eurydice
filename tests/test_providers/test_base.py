"""Tests for provider base class."""

import pytest

from eurydice import GenerationParams, Voice


class TestMockProvider:
    """Tests for mock provider functionality."""

    @pytest.mark.asyncio
    async def test_connect_success(self, mock_provider):
        """Test successful connection."""
        assert await mock_provider.connect() is True

    @pytest.mark.asyncio
    async def test_connect_failure(self, mock_provider):
        """Test connection failure."""
        mock_provider.connected = False
        assert await mock_provider.connect() is False

    @pytest.mark.asyncio
    async def test_generate_tokens(self, mock_provider):
        """Test token generation."""
        mock_provider.set_tokens(["token1", "token2", "token3"])

        tokens = []
        async for token in mock_provider.generate_tokens("test", Voice.LEO, GenerationParams()):
            tokens.append(token)

        assert tokens == ["token1", "token2", "token3"]

    @pytest.mark.asyncio
    async def test_close(self, mock_provider):
        """Test provider close."""
        await mock_provider.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_provider):
        """Test provider as a context manager."""
        async with mock_provider as p:
            assert p.name == "mock"
