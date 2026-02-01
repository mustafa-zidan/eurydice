"""Inference providers for eurydice."""

from eurydice.providers.auto import detect_best_provider, get_provider_info
from eurydice.providers.base import Provider
from eurydice.providers.embedded import EmbeddedProvider
from eurydice.providers.lmstudio import LMStudioProvider
from eurydice.providers.orpheus_cpp import OrpheusCppProvider
from eurydice.providers.vllm import VLLMProvider

__all__ = [
    "Provider",
    "EmbeddedProvider",
    "LMStudioProvider",
    "OrpheusCppProvider",
    "VLLMProvider",
    "detect_best_provider",
    "get_provider_info",
]
