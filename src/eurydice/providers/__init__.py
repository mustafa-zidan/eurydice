"""Inference providers for eurydice."""

from eurydice.providers.base import Provider
from eurydice.providers.lmstudio import LMStudioProvider

__all__ = [
    "Provider",
    "LMStudioProvider",
]
