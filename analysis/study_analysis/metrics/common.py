from __future__ import annotations

import numpy as np
import pandas as pd

from study_analysis.config import MODELS

AXES: dict[str, str] = {"realism": "realism_rank", "coherence": "coherence_rank"}


def rank_matrix(real_trials: pd.DataFrame, axis: str) -> pd.DataFrame:
    wide = real_trials.pivot_table(
        index=["participant_id", "trial_number"],
        columns="model", values=AXES[axis], aggfunc="first",
    )
    present = [m for m in MODELS if m in wide.columns]
    return wide.reindex(columns=present).dropna(how="any")


def bootstrap_ci(values, statistic=np.mean, *, n_boot: int, seed: int, alpha: float = 0.05):
    values = np.asarray(values, dtype=float)
    point = float(statistic(values))
    if values.size < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    replicates = statistic(values[idx], axis=1)
    lo, hi = np.percentile(replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def holm_correction(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, (m - rank) * p_values[i])
        adjusted[i] = min(running_max, 1.0)
    return adjusted.tolist()
