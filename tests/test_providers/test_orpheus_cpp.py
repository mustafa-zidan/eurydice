"""Tests for the orpheus-cpp inference provider."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from eurydice import GenerationParams, Voice
from eurydice.exceptions import DependencyError, ProviderError
from eurydice.providers.orpheus_cpp import OrpheusCppProvider, _check_dependencies


class TestOrpheusCppProvider:
    """Tests for OrpheusCppProvider."""

    def test_name(self):
        """Test provider name."""
        with patch.dict(sys.modules, {"orpheus_cpp": MagicMock()}):
            provider = OrpheusCppProvider()
            assert provider.name == "orpheus-cpp"

    def test_yields_audio_flag(self):
        """Test that provider is marked as yielding audio directly."""
        with patch.dict(sys.modules, {"orpheus_cpp": MagicMock()}):
            provider = OrpheusCppProvider()
            assert provider.yields_audio is True

    def test_initial_state(self):
        """Test initial provider state."""
        with patch.dict(sys.modules, {"orpheus_cpp": MagicMock()}):
            provider = OrpheusCppProvider()
            assert provider._orpheus is None
            assert provider._loaded is False

    def test_custom_params(self):
        """Test custom initialization parameters."""
        with patch.dict(sys.modules, {"orpheus_cpp": MagicMock()}):
            provider = OrpheusCppProvider(
                model_path="/path/to/model.gguf",
                verbose=True,
                lang="de",
            )
            assert provider._model_path == "/path/to/model.gguf"
            assert provider._verbose is True
            assert provider._lang == "de"

    def test_api_compatibility_params(self):
        """Test that server_url, timeout, and model params are accepted for API compatibility."""
        with patch.dict(sys.modules, {"orpheus_cpp": MagicMock()}):
            provider = OrpheusCppProvider(
                server_url="http://unused",
                timeout=60.0,
                model="unused",
            )
            # Should not raise, params are ignored but accepted
            assert provider.name == "orpheus-cpp"

    @pytest.mark.asyncio
    async def test_generate_tokens_raises_error(self):
        """Test that generate_tokens raises error (use generate_audio instead)."""
        with patch.dict(sys.modules, {"orpheus_cpp": MagicMock()}):
            provider = OrpheusCppProvider()
            provider._loaded = True
            provider._orpheus = MagicMock()
            params = GenerationParams()

            with pytest.raises(ProviderError, match="yields audio directly"):
                async for _ in provider.generate_tokens("test", Voice.LEO, params):
                    pass

    @pytest.mark.asyncio
    async def test_generate_audio_requires_connect(self):
        """Test that generate_audio raises error if not connected."""
        with patch.dict(sys.modules, {"orpheus_cpp": MagicMock()}):
            provider = OrpheusCppProvider()
            params = GenerationParams()

            with pytest.raises(ProviderError, match="Model not loaded"):
                async for _ in provider.generate_audio("test", Voice.LEO, params):
                    pass

    @pytest.mark.asyncio
    async def test_close_cleans_up(self):
        """Test that close clears model state."""
        with patch.dict(sys.modules, {"orpheus_cpp": MagicMock()}):
            provider = OrpheusCppProvider()
            provider._orpheus = MagicMock()
            provider._loaded = True

            await provider.close()

            assert provider._orpheus is None
            assert provider._loaded is False

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        """Test that close can be called multiple times."""
        with patch.dict(sys.modules, {"orpheus_cpp": MagicMock()}):
            provider = OrpheusCppProvider()
            await provider.close()
            await provider.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test context manager protocol."""
        with patch.dict(sys.modules, {"orpheus_cpp": MagicMock()}):
            provider = OrpheusCppProvider()
            provider._orpheus = MagicMock()
            provider._loaded = True

            async with provider:
                assert provider._loaded is True

            assert provider._orpheus is None
            assert provider._loaded is False


class TestDependencyCheck:
    """Tests for dependency checking."""

    def test_check_dependencies_raises_when_missing(self):
        """Test dependency check fails when orpheus-cpp is missing."""
        with (
            patch.dict(sys.modules, {"orpheus_cpp": None}),
            patch("builtins.__import__") as mock_import,
        ):

            def side_effect(name, *args, **kwargs):
                if name == "orpheus_cpp":
                    raise ImportError("No module named 'orpheus_cpp'")
                return MagicMock()

            mock_import.side_effect = side_effect
            with pytest.raises(DependencyError):
                _check_dependencies()


class TestOrpheusCppProviderIntegration:
    """Integration tests that mock the model loading."""

    @pytest.mark.asyncio
    async def test_connect_loads_model(self):
        """Test that connect loads the orpheus-cpp model."""
        mock_orpheus_cpp = MagicMock()
        mock_instance = MagicMock()
        mock_orpheus_cpp.OrpheusCpp.return_value = mock_instance

        with patch.dict(sys.modules, {"orpheus_cpp": mock_orpheus_cpp}):
            provider = OrpheusCppProvider(verbose=True, lang="en")

            result = await provider.connect()

            assert result is True
            assert provider._loaded is True
            mock_orpheus_cpp.OrpheusCpp.assert_called_once_with(
                verbose=True,
                lang="en",
            )

    @pytest.mark.asyncio
    async def test_connect_is_idempotent(self):
        """Test that connect only loads model once."""
        mock_orpheus_cpp = MagicMock()
        mock_instance = MagicMock()
        mock_orpheus_cpp.OrpheusCpp.return_value = mock_instance

        with patch.dict(sys.modules, {"orpheus_cpp": mock_orpheus_cpp}):
            provider = OrpheusCppProvider()

            await provider.connect()
            await provider.connect()  # Second call should be no-op

            # Should only be called once
            assert mock_orpheus_cpp.OrpheusCpp.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_audio_streams_chunks(self):
        """Test that generate_audio streams audio chunks."""
        import numpy as np

        mock_orpheus_cpp = MagicMock()
        mock_instance = MagicMock()

        # Simulate yielding audio chunks
        audio_chunk1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        audio_chunk2 = np.array([0.4, 0.5, 0.6], dtype=np.float32)
        mock_instance.stream_tts_sync.return_value = [
            (24000, audio_chunk1),
            (24000, audio_chunk2),
        ]
        mock_orpheus_cpp.OrpheusCpp.return_value = mock_instance

        with patch.dict(sys.modules, {"orpheus_cpp": mock_orpheus_cpp}):
            provider = OrpheusCppProvider()
            await provider.connect()

            params = GenerationParams()
            chunks = []
            async for chunk in provider.generate_audio("Hello", Voice.TARA, params):
                chunks.append(chunk)

            assert len(chunks) == 2
            # Verify chunks are bytes
            assert all(isinstance(c, bytes) for c in chunks)

            # Verify stream_tts_sync was called with correct args
            mock_instance.stream_tts_sync.assert_called_once()
            call_args = mock_instance.stream_tts_sync.call_args
            assert call_args[0][0] == "Hello"
            assert call_args[1]["options"]["voice_id"] == "tara"
