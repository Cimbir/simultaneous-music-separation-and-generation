from __future__ import annotations

import numpy as np
import torch
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

from ..base import Metric


class SISDR(Metric):
    """SI-SDR in the waveform domain (dB, higher is better). Phase-sensitive - don't use after a vocoder."""

    name = "si_sdr"

    def __init__(self, zero_mean: bool = True):
        self.zero_mean = zero_mean
        self._metric = ScaleInvariantSignalDistortionRatio(zero_mean=zero_mean)

    def compute(self, estimates: np.ndarray | torch.Tensor, references: np.ndarray | torch.Tensor, **_) -> dict[str, float]:
        estimates = torch.as_tensor(estimates, dtype=torch.float32)
        references = torch.as_tensor(references, dtype=torch.float32)
        return {self.name: self._metric(estimates, references).item()}