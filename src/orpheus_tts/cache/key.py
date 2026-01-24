"""Cache key generation."""

import hashlib

from orpheus_tts.config import GenerationParams
from orpheus_tts.types import Voice


def generate_cache_key(
    text: str,
    voice: Voice,
    params: GenerationParams,
    model: str = "",
) -> str:
    """
    Generate a content-addressed cache key.

    The key is based on all parameters that affect audio output:
    - Input text
    - Voice selection
    - Generation parameters
    - Model identifier

    Args:
        text: Input text
        voice: Voice used
        params: Generation parameters
        model: Model identifier

    Returns:
        32-character hex string cache key
    """
    components = [
        text,
        voice.value,
        str(params.temperature),
        str(params.top_p),
        str(params.repetition_penalty),
        model,
    ]
    combined = "|".join(components)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:32]
