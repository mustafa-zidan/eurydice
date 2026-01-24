"""Inference providers for orpheus-tts."""

from orpheus_tts.providers.base import Provider
from orpheus_tts.providers.lmstudio import LMStudioProvider

__all__ = [
    "Provider",
    "LMStudioProvider",
]
