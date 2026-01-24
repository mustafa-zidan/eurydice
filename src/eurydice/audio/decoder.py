"""SNAC decoder for converting tokens to audio."""

from eurydice.exceptions import DependencyError

# Try to import SNAC dependencies
_snac_available = False
_snac_model = None
_snac_device = "cpu"

try:
    import numpy as np  # noqa: F401
    import torch  # noqa: F401
    from snac import SNAC  # noqa: F401

    _snac_available = True
except ImportError:
    pass


def is_audio_available() -> bool:
    """Check if audio decoding dependencies are available."""
    return _snac_available


class SNACDecoder:
    """Wrapper for SNAC audio decoder."""

    def __init__(self, device: str | None = None):
        """
        Initialize the SNAC decoder.

        Args:
            device: Device to use ("cpu", "cuda", "mps", or None for auto-detect)
        """
        if not _snac_available:
            raise DependencyError("torch, numpy, snac", "uv install eurydice[audio]")

        self._model = None
        self._device = device or self._detect_device()

    def _detect_device(self) -> str:
        """Detect the best available device."""
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _ensure_model(self) -> None:
        """Ensure the SNAC model is loaded."""
        if self._model is not None:
            return

        from snac import SNAC

        self._model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        self._model = self._model.to(self._device)

    @property
    def device(self) -> str:
        """Get the device being used."""
        return self._device

    def decode_frame(self, multiframe: list[int]) -> bytes | None:
        """
        Convert token frames to audio bytes.

        Args:
            multiframe: List of 28 token IDs (4 complete 7-token frames)

        Returns:
            Audio bytes as int16 PCM, or None if invalid
        """
        if len(multiframe) < 7:
            return None

        self._ensure_model()

        import numpy as np
        import torch

        codes_0 = torch.tensor([], device=self._device, dtype=torch.int32)
        codes_1 = torch.tensor([], device=self._device, dtype=torch.int32)
        codes_2 = torch.tensor([], device=self._device, dtype=torch.int32)

        num_frames = len(multiframe) // 7
        frame = multiframe[: num_frames * 7]

        for j in range(num_frames):
            i = 7 * j
            if codes_0.shape[0] == 0:
                codes_0 = torch.tensor([frame[i]], device=self._device, dtype=torch.int32)
            else:
                codes_0 = torch.cat(
                    [
                        codes_0,
                        torch.tensor([frame[i]], device=self._device, dtype=torch.int32),
                    ]
                )

            if codes_1.shape[0] == 0:
                codes_1 = torch.tensor([frame[i + 1]], device=self._device, dtype=torch.int32)
                codes_1 = torch.cat(
                    [
                        codes_1,
                        torch.tensor([frame[i + 4]], device=self._device, dtype=torch.int32),
                    ]
                )
            else:
                codes_1 = torch.cat(
                    [
                        codes_1,
                        torch.tensor([frame[i + 1]], device=self._device, dtype=torch.int32),
                    ]
                )
                codes_1 = torch.cat(
                    [
                        codes_1,
                        torch.tensor([frame[i + 4]], device=self._device, dtype=torch.int32),
                    ]
                )

            if codes_2.shape[0] == 0:
                codes_2 = torch.tensor([frame[i + 2]], device=self._device, dtype=torch.int32)
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor([frame[i + 3]], device=self._device, dtype=torch.int32),
                    ]
                )
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor([frame[i + 5]], device=self._device, dtype=torch.int32),
                    ]
                )
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor([frame[i + 6]], device=self._device, dtype=torch.int32),
                    ]
                )
            else:
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor([frame[i + 2]], device=self._device, dtype=torch.int32),
                    ]
                )
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor([frame[i + 3]], device=self._device, dtype=torch.int32),
                    ]
                )
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor([frame[i + 5]], device=self._device, dtype=torch.int32),
                    ]
                )
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor([frame[i + 6]], device=self._device, dtype=torch.int32),
                    ]
                )

        codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]

        # Validate token ranges
        if (
            torch.any(codes[0] < 0)
            or torch.any(codes[0] > 4096)
            or torch.any(codes[1] < 0)
            or torch.any(codes[1] > 4096)
            or torch.any(codes[2] < 0)
            or torch.any(codes[2] > 4096)
        ):
            return None

        with torch.inference_mode():
            audio_hat = self._model.decode(codes)

        audio_slice = audio_hat[:, :, 2048:4096]
        detached_audio = audio_slice.detach().cpu()
        audio_np = detached_audio.numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        return audio_bytes
