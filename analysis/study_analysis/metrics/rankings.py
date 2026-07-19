from __future__ import annotations

import numpy as np
import pandas as pd

from study_analysis.config import Config, MODELS
from study_analysis.metrics.common import AXES, bootstrap_ci, rank_matrix


def mean_rank_table(real_trials: pd.DataFrame, config: Config) -> pd.DataFrame:
    records = []
    for axis in AXES:
        matrix = rank_matrix(real_trials, axis)
        for offset, model in enumerate(MODELS):
            ranks = matrix[model].to_numpy()
            point, lo, hi = bootstrap_ci(ranks, n_boot=config.n_bootstrap, seed=config.seed + offset)
            records.append({"axis": axis, "model": model, "mean_rank": point,
                            "ci_low": lo, "ci_high": hi, "n_trials": ranks.size})
    return pd.DataFrame(records)


def win_rate_table(real_trials: pd.DataFrame) -> pd.DataFrame:
    records = []
    for axis, column in AXES.items():
        for model in MODELS:
            ranks = real_trials.loc[real_trials["model"] == model, column].dropna()
            if ranks.empty:
                continue
            records.append({"axis": axis, "model": model,
                            "win_rate": float(np.mean(ranks == 1)),
                            "top2_rate": float(np.mean(ranks <= 2)),
                            "n_trials": int(ranks.size)})
    return pd.DataFrame(records)
