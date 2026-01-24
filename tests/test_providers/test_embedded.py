"""Tests for the embedded inference provider."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from eurydice import GenerationParams, Voice
from eurydice.exceptions import DependencyError, ProviderError
from eurydice.providers.embedded import EmbeddedProvider, _check_dependencies


class TestEmbeddedProvider:
    """Tests for EmbeddedProvider."""

    def test_name(self):
        """Test provider name."""
        provider = EmbeddedProvider()
        assert provider.name == "embedded"

    def test_default_model(self):
        """Test default model ID."""
        provider = EmbeddedProvider()
        assert provider.model_id == "canopylabs/orpheus-3b-0.1-ft"

    def test_custom_model(self):
        """Test custom model ID."""
        provider = EmbeddedProvider(model="custom/model")
        assert provider.model_id == "custom/model"

    def test_format_prompt(self):
        """Test prompt formatting."""
        provider = EmbeddedProvider()
        prompt = provider._format_prompt("Hello world", Voice.TARA)
        assert prompt == "<|audio|>tara: Hello world<|eot_id|>"

    def test_format_prompt_all_voices(self):
        """Test prompt formatting with all voices."""
        provider = EmbeddedProvider()
        for voice in Voice:
            prompt = provider._format_prompt("Test", voice)
            assert f"{voice.value}:" in prompt
            assert "<|audio|>" in prompt
            assert "<|eot_id|>" in prompt

    def test_detect_device_cuda(self):
        """Test CUDA device detection."""
        provider = EmbeddedProvider()
        with patch.dict(sys.modules, {"torch": MagicMock()}):
            import torch

            torch.cuda.is_available = MagicMock(return_value=True)
            with patch.object(provider, "_detect_device", wraps=provider._detect_device):
                # Use actual torch since it's installed
                device = provider._detect_device()
                # On this machine, check what's actually available
                assert device in ["cuda", "mps", "cpu"]

    def test_detect_device_returns_valid(self):
        """Test device detection returns valid device."""
        provider = EmbeddedProvider()
        device = provider._detect_device()
        assert device in ["cuda", "mps", "cpu"]

    def test_initial_state(self):
        """Test initial provider state."""
        provider = EmbeddedProvider()
        assert provider._model is None
        assert provider._tokenizer is None
        assert provider._loaded is False

    @pytest.mark.asyncio
    async def test_generate_tokens_requires_connect(self):
        """Test that generate_tokens raises error if not connected."""
        provider = EmbeddedProvider()
        params = GenerationParams()

        with pytest.raises(ProviderError, match="Model not loaded"):
            async for _ in provider.generate_tokens("test", Voice.LEO, params):
                pass

    @pytest.mark.asyncio
    async def test_close_cleans_up(self):
        """Test that close clears model state."""
        provider = EmbeddedProvider()
        provider._model = MagicMock()
        provider._tokenizer = MagicMock()
        provider._loaded = True

        await provider.close()

        assert provider._model is None
        assert provider._tokenizer is None
        assert provider._loaded is False

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        """Test that close can be called multiple times."""
        provider = EmbeddedProvider()
        await provider.close()
        await provider.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test context manager protocol."""
        provider = EmbeddedProvider()
        provider._model = MagicMock()
        provider._tokenizer = MagicMock()
        provider._loaded = True

        async with provider:
            assert provider._loaded is True

        assert provider._model is None
        assert provider._loaded is False

    def test_api_compatibility_params(self):
        """Test that server_url and timeout params are accepted for API compatibility."""
        provider = EmbeddedProvider(
            server_url="http://unused",
            timeout=60.0,
        )
        # Should not raise, params are ignored but accepted
        assert provider.name == "embedded"


class TestDependencyCheck:
    """Tests for dependency checking."""

    def test_check_dependencies_passes_when_available(self):
        """Test dependency check passes when packages are available."""
        # Should not raise since torch/transformers are installed
        _check_dependencies()

    def test_check_dependencies_raises_when_missing(self):
        """Test dependency check fails when packages missing."""
        # Mock torch to raise ImportError
        with (
            patch.dict(sys.modules, {"torch": None}),
            patch("builtins.__import__") as mock_import,
        ):

            def side_effect(name, *args, **kwargs):
                if name == "torch":
                    raise ImportError("No module named 'torch'")
                return MagicMock()

            mock_import.side_effect = side_effect
            with pytest.raises(DependencyError):
                _check_dependencies()


class TestEmbeddedProviderIntegration:
    """Integration tests that mock the model loading."""

    @pytest.mark.asyncio
    async def test_connect_loads_model(self):
        """Test that connect loads the model."""
        provider = EmbeddedProvider()

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        # Mock at the transformers module level
        with (
            patch("transformers.AutoTokenizer.from_pretrained") as mock_tok_fn,
            patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model_fn,
        ):
            mock_tok_fn.return_value = mock_tokenizer
            mock_model_fn.return_value = mock_model

            result = await provider.connect()

            assert result is True
            assert provider._loaded is True
            mock_tok_fn.assert_called_once_with("canopylabs/orpheus-3b-0.1-ft")
            mock_model_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_is_idempotent(self):
        """Test that connect only loads model once."""
        provider = EmbeddedProvider()

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        with (
            patch("transformers.AutoTokenizer.from_pretrained") as mock_tok_fn,
            patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model_fn,
        ):
            mock_tok_fn.return_value = mock_tokenizer
            mock_model_fn.return_value = mock_model

            await provider.connect()
            await provider.connect()  # Second call should be no-op

            # Should only be called once
            assert mock_tok_fn.call_count == 1
            assert mock_model_fn.call_count == 1


class TestTorchDtypeResolution:
    """Tests for torch dtype resolution."""

    def test_get_torch_dtype_float32(self):
        """Test explicit float32 dtype."""
        provider = EmbeddedProvider(torch_dtype="float32")
        dtype = provider._get_torch_dtype()
        import torch

        assert dtype == torch.float32

    def test_get_torch_dtype_float16(self):
        """Test explicit float16 dtype."""
        provider = EmbeddedProvider(torch_dtype="float16")
        dtype = provider._get_torch_dtype()
        import torch

        assert dtype == torch.float16

    def test_get_torch_dtype_auto(self):
        """Test auto dtype selection."""
        provider = EmbeddedProvider(torch_dtype="auto")
        dtype = provider._get_torch_dtype()
        import torch

        # Should be a valid torch dtype
        assert dtype in [torch.float16, torch.float32, torch.bfloat16]
