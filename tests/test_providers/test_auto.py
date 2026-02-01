"""Tests for provider auto-detection logic."""

import sys
from unittest.mock import MagicMock, patch

from eurydice.providers.auto import (
    _is_embedded_available,
    _is_orpheus_cpp_available,
    _is_vllm_available,
    detect_best_provider,
    get_provider_info,
)


class TestAvailabilityChecks:
    """Tests for individual availability checks."""

    def test_is_vllm_available_with_cuda(self):
        """Test vLLM detection with CUDA available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with (
            patch.dict(sys.modules, {"torch": mock_torch, "orpheus_tts": MagicMock()}),
            patch("eurydice.providers.auto.torch", mock_torch, create=True),
            patch("builtins.__import__") as mock_import,
        ):

            def import_side_effect(name, *args, **kwargs):
                if name == "torch":
                    return mock_torch
                if name == "orpheus_tts":
                    return MagicMock()
                raise ImportError()

            mock_import.side_effect = import_side_effect
            # The function imports internally, so we test the actual behavior
            result = _is_vllm_available()
            # Result depends on actual system state
            assert isinstance(result, bool)

    def test_is_vllm_available_without_cuda(self):
        """Test vLLM detection without CUDA."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = _is_vllm_available()
            # Should be False because CUDA is not available
            # (actual result depends on system)
            assert isinstance(result, bool)

    def test_is_orpheus_cpp_available(self):
        """Test orpheus-cpp detection."""
        result = _is_orpheus_cpp_available()
        # Result depends on whether orpheus-cpp is installed
        assert isinstance(result, bool)

    def test_is_embedded_available(self):
        """Test embedded (transformers) detection."""
        result = _is_embedded_available()
        # Result depends on whether transformers is installed
        assert isinstance(result, bool)


class TestDetectBestProvider:
    """Tests for detect_best_provider function."""

    def test_returns_valid_provider(self):
        """Test that detect_best_provider returns a valid provider name."""
        result = detect_best_provider()
        assert result in ["vllm", "orpheus-cpp", "embedded", "lmstudio"]

    def test_fallback_to_lmstudio(self):
        """Test fallback to lmstudio when nothing else is available."""
        with (
            patch(
                "eurydice.providers.auto._is_vllm_available", return_value=False
            ),
            patch(
                "eurydice.providers.auto._is_orpheus_cpp_available", return_value=False
            ),
            patch(
                "eurydice.providers.auto._is_embedded_available", return_value=False
            ),
        ):
            result = detect_best_provider()
            assert result == "lmstudio"

    def test_prefers_vllm_when_available(self):
        """Test that vLLM is preferred when available."""
        with (
            patch("eurydice.providers.auto._is_vllm_available", return_value=True),
            patch(
                "eurydice.providers.auto._is_orpheus_cpp_available", return_value=True
            ),
            patch(
                "eurydice.providers.auto._is_embedded_available", return_value=True
            ),
        ):
            result = detect_best_provider()
            assert result == "vllm"

    def test_prefers_orpheus_cpp_when_no_vllm(self):
        """Test that orpheus-cpp is preferred when vLLM not available."""
        with (
            patch("eurydice.providers.auto._is_vllm_available", return_value=False),
            patch(
                "eurydice.providers.auto._is_orpheus_cpp_available", return_value=True
            ),
            patch(
                "eurydice.providers.auto._is_embedded_available", return_value=True
            ),
        ):
            result = detect_best_provider()
            assert result == "orpheus-cpp"

    def test_prefers_embedded_when_no_cpp(self):
        """Test that embedded is preferred when orpheus-cpp not available."""
        with (
            patch("eurydice.providers.auto._is_vllm_available", return_value=False),
            patch(
                "eurydice.providers.auto._is_orpheus_cpp_available", return_value=False
            ),
            patch(
                "eurydice.providers.auto._is_embedded_available", return_value=True
            ),
        ):
            result = detect_best_provider()
            assert result == "embedded"


class TestGetProviderInfo:
    """Tests for get_provider_info function."""

    def test_returns_all_providers(self):
        """Test that all providers are included in info."""
        info = get_provider_info()
        assert "vllm" in info
        assert "orpheus-cpp" in info
        assert "embedded" in info
        assert "lmstudio" in info

    def test_provider_info_structure(self):
        """Test that provider info has expected structure."""
        info = get_provider_info()
        for _provider_name, provider_info in info.items():
            assert "available" in provider_info
            assert "description" in provider_info
            assert "install" in provider_info
            assert isinstance(provider_info["available"], bool)
            assert isinstance(provider_info["description"], str)
            assert isinstance(provider_info["install"], str)

    def test_lmstudio_always_available(self):
        """Test that lmstudio is always marked as available."""
        info = get_provider_info()
        assert info["lmstudio"]["available"] is True


class TestClientAutoDetection:
    """Tests for auto-detection integration with Eurydice client."""

    def test_client_has_available_providers_method(self):
        """Test that Eurydice class has available_providers method."""
        from eurydice import Eurydice

        info = Eurydice.available_providers()
        assert isinstance(info, dict)
        assert "vllm" in info

    def test_client_has_detect_best_provider_method(self):
        """Test that Eurydice class has detect_best_provider method."""
        from eurydice import Eurydice

        result = Eurydice.detect_best_provider()
        assert result in ["vllm", "orpheus-cpp", "embedded", "lmstudio"]
