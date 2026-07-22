from __future__ import annotations

import numpy as np
import pandas as pd

from study_analysis.config import Config, MODELS
from study_analysis.metrics.common import AXES, bootstrap_ci, rank_matrix


def mean_rank_table(real_trials: pd.DataFrame, config: Config) -> pd.DataFrame:
    records = []
    for axis in AXES:
        matrix = rank_matrix(real_trials, axis)
        for offset, model in enumerate(matrix.columns):
            ranks = matrix[model].to_numpy()
            point, lo, hi = bootstrap_ci(ranks, n_boot=config.n_bootstrap, seed=config.seed + offset)
            records.append({"axis": axis, "model": model, "mean_rank": point,
                            "ci_low": lo, "ci_high": hi, "n_trials": ranks.size})
    return pd.DataFrame(records)


def win_rate_table(real_trials: pd.DataFrame) -> pd.DataFrame:
    records = []
    present = [m for m in MODELS if m in set(real_trials["model"])]
    for axis, column in AXES.items():
        for model in present:
            ranks = real_trials.loc[real_trials["model"] == model, column].dropna()
            if ranks.empty:
                continue
            records.append({"axis": axis, "model": model,
                            "win_rate": float(np.mean(ranks == 1)),
                            "top2_rate": float(np.mean(ranks <= 2)),
                            "n_trials": int(ranks.size)})
    return pd.DataFrame(records)


def clip_rating_summary(real_trials: pd.DataFrame) -> pd.DataFrame:
    records = []
    for axis, column in AXES.items():
        counts = (real_trials.groupby(["model", "clip_id"])[column]
                  .count().reset_index(name="n_ratings"))
        for model, group in counts.groupby("model"):
            records.append({
                "axis": axis,
                "model": model,
                "n_clips": int(group.shape[0]),
                "min_ratings": int(group["n_ratings"].min()),
                "mean_ratings": float(group["n_ratings"].mean()),
                "max_ratings": int(group["n_ratings"].max()),
            })
    return pd.DataFrame(records)
