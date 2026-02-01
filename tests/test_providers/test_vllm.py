"""Tests for the vLLM inference provider."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from eurydice import GenerationParams, Voice
from eurydice.exceptions import DependencyError, ProviderError
from eurydice.providers.vllm import VLLMProvider


class TestVLLMProvider:
    """Tests for VLLMProvider."""

    def test_name(self):
        """Test provider name."""
        provider = VLLMProvider()
        assert provider.name == "vllm"

    def test_yields_audio_flag(self):
        """Test that provider is marked as yielding audio directly."""
        provider = VLLMProvider()
        assert provider.yields_audio is True

    def test_initial_state(self):
        """Test initial provider state."""
        provider = VLLMProvider()
        assert provider._orpheus_model is None
        assert provider._loaded is False

    def test_default_model(self):
        """Test default model ID."""
        provider = VLLMProvider()
        assert provider.model_id == "canopylabs/orpheus-tts-0.1-finetune-prod"

    def test_custom_params(self):
        """Test custom initialization parameters."""
        provider = VLLMProvider(
            model="custom/model",
            max_model_len=4096,
            dtype="float16",
        )
        assert provider.model_id == "custom/model"
        assert provider._max_model_len == 4096
        assert provider._dtype == "float16"

    def test_api_compatibility_params(self):
        """Test that server_url and timeout params are accepted for API compatibility."""
        provider = VLLMProvider(
            server_url="http://unused",
            timeout=60.0,
        )
        # Should not raise, params are ignored but accepted
        assert provider.name == "vllm"

    @pytest.mark.asyncio
    async def test_generate_tokens_raises_error(self):
        """Test that generate_tokens raises error (use generate_audio instead)."""
        provider = VLLMProvider()
        provider._loaded = True
        provider._orpheus_model = MagicMock()
        params = GenerationParams()

        with pytest.raises(ProviderError, match="yields audio directly"):
            async for _ in provider.generate_tokens("test", Voice.LEO, params):
                pass

    @pytest.mark.asyncio
    async def test_generate_audio_requires_connect(self):
        """Test that generate_audio raises error if not connected."""
        provider = VLLMProvider()
        params = GenerationParams()

        with pytest.raises(ProviderError, match="Model not loaded"):
            async for _ in provider.generate_audio("test", Voice.LEO, params):
                pass

    @pytest.mark.asyncio
    async def test_close_cleans_up(self):
        """Test that close clears model state."""
        provider = VLLMProvider()
        provider._orpheus_model = MagicMock()
        provider._loaded = True

        await provider.close()

        assert provider._orpheus_model is None
        assert provider._loaded is False

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        """Test that close can be called multiple times."""
        provider = VLLMProvider()
        await provider.close()
        await provider.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test context manager protocol."""
        provider = VLLMProvider()
        provider._orpheus_model = MagicMock()
        provider._loaded = True

        async with provider:
            assert provider._loaded is True

        assert provider._orpheus_model is None
        assert provider._loaded is False

    def test_get_torch_dtype(self):
        """Test torch dtype resolution."""
        import torch

        provider = VLLMProvider(dtype="float16")
        assert provider._get_torch_dtype() == torch.float16

        provider = VLLMProvider(dtype="bfloat16")
        assert provider._get_torch_dtype() == torch.bfloat16

        provider = VLLMProvider(dtype="float32")
        assert provider._get_torch_dtype() == torch.float32


class TestDependencyCheck:
    """Tests for dependency checking."""

    def test_check_dependencies_raises_when_no_cuda(self):
        """Test dependency check fails when CUDA is not available."""
        # On systems without CUDA, the check should fail
        import torch

        if not torch.cuda.is_available():
            from eurydice.providers.vllm import _check_dependencies

            with pytest.raises(DependencyError, match="CUDA"):
                _check_dependencies()
        else:
            # If CUDA is available, skip this test
            pytest.skip("CUDA is available, skipping no-CUDA test")


class TestVLLMProviderIntegration:
    """Integration tests that mock the model loading."""

    @pytest.mark.asyncio
    async def test_connect_loads_model(self):
        """Test that connect loads the vLLM model."""
        mock_orpheus_tts = MagicMock()
        mock_model_instance = MagicMock()
        mock_orpheus_tts.OrpheusModel.return_value = mock_model_instance

        provider = VLLMProvider()

        with (
            patch.dict(sys.modules, {"orpheus_tts": mock_orpheus_tts}),
            patch.object(provider, "_load_model") as mock_load,
        ):
            mock_load.side_effect = lambda: setattr(provider, "_loaded", True)
            result = await provider.connect()

            assert result is True
            mock_load.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_is_idempotent(self):
        """Test that connect only loads model once."""
        provider = VLLMProvider()
        provider._loaded = True  # Pretend already loaded

        with patch.object(provider, "_load_model") as mock_load:
            await provider.connect()
            await provider.connect()

            # Should not be called since already loaded
            mock_load.assert_not_called()
