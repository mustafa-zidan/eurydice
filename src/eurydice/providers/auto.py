"""Auto-detection logic for selecting the best available provider."""


def _is_vllm_available() -> bool:
    """Check if vLLM/orpheus-speech is available with CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        import orpheus_tts  # noqa: F401

        return True
    except ImportError:
        return False


def _is_orpheus_cpp_available() -> bool:
    """Check if orpheus-cpp is available."""
    try:
        import orpheus_cpp  # noqa: F401

        return True
    except ImportError:
        return False


def _is_embedded_available() -> bool:
    """Check if transformers/embedded provider is available."""
    try:
        import accelerate  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _is_audio_available() -> bool:
    """Check if audio dependencies are available (for token-based providers)."""
    try:
        import numpy  # noqa: F401
        import snac  # noqa: F401
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def detect_best_provider() -> str:
    """
    Detect the best available provider based on installed dependencies.

    Priority order:
    1. vllm - Fastest GPU inference (requires CUDA + orpheus-speech)
    2. orpheus-cpp - Fast CPU/Metal inference (requires orpheus-cpp)
    3. embedded - Local GPU inference (requires transformers)
    4. lmstudio - External server (always available as fallback)

    Returns:
        Provider name string ("vllm", "orpheus-cpp", "embedded", or "lmstudio")
    """
    # Check for vLLM (fastest GPU option)
    if _is_vllm_available():
        return "vllm"

    # Check for orpheus-cpp (fast CPU/Metal)
    if _is_orpheus_cpp_available():
        return "orpheus-cpp"

    # Check for embedded (transformers-based)
    if _is_embedded_available():
        return "embedded"

    # Fallback to lmstudio (external server)
    return "lmstudio"


def get_provider_info() -> dict[str, dict[str, bool | str]]:
    """
    Get information about all available providers.

    Returns:
        Dict mapping provider names to their availability info
    """
    return {
        "vllm": {
            "available": _is_vllm_available(),
            "description": "Fastest GPU inference via vLLM (CUDA required)",
            "install": "pip install eurydice-tts[vllm]",
        },
        "orpheus-cpp": {
            "available": _is_orpheus_cpp_available(),
            "description": "Fast CPU/Metal inference via llama.cpp",
            "install": "pip install eurydice-tts[cpp]",
        },
        "embedded": {
            "available": _is_embedded_available(),
            "description": "Local inference via transformers",
            "install": "pip install eurydice-tts[embedded]",
        },
        "lmstudio": {
            "available": True,  # Always available (external server)
            "description": "External LM Studio server",
            "install": "pip install eurydice-tts[audio]",
        },
    }
