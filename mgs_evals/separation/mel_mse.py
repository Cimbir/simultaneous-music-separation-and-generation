from __future__ import annotations

import numpy as np

from ..base import Metric


class MelMSE(Metric):
    """MSE on mel-spectrograms - preferred over SI-SDR for vocoder-based models since mel is phase-agnostic."""

    name = "mel_mse"

    def compute(self, estimates: np.ndarray, references: np.ndarray, **_) -> dict[str, float]:
        estimates = np.asarray(estimates, dtype=np.float32)
        references = np.asarray(references, dtype=np.float32)
        if estimates.shape != references.shape:
            raise ValueError(f"Shape mismatch: estimates {estimates.shape} vs references {references.shape}")
        return {self.name: float(np.mean((estimates - references) ** 2))}