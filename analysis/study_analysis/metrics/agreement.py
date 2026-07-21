from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from study_analysis.config import Config, GROUND_TRUTH
from study_analysis.metrics.common import AXES, bootstrap_ci, rank_matrix

GROUND_TRUTH_CHANCE_RATE = 1 / 5


def kendalls_w(real_trials: pd.DataFrame) -> pd.DataFrame:
    records = []
    for axis in AXES:
        matrix = rank_matrix(real_trials, axis)
        n_judges, n_items = matrix.shape
        if n_judges == 0 or n_items < 2:
            w = float("nan")
        else:
            deviation = matrix.sum(axis=0).to_numpy() - n_judges * (n_items + 1) / 2
            s = float(np.sum(deviation**2))
            w = 12 * s / (n_judges**2 * (n_items**3 - n_items))
        records.append({"axis": axis, "kendall_w": w,
                        "agreement": _agreement_label(w),
                        "n_judges": n_judges, "n_items": n_items})
    return pd.DataFrame(records)


def intra_rater_reliability(long: pd.DataFrame, config: Config) -> pd.DataFrame:
    records = []
    for axis, column in AXES.items():
        values = np.asarray(_repeat_correlations(long, column))
        point, lo, hi = bootstrap_ci(values, n_boot=config.n_bootstrap, seed=config.seed)
        records.append({"axis": axis, "mean_spearman": point, "ci_low": lo,
                        "ci_high": hi, "n_participants": int(values.size)})
    return pd.DataFrame(records)


def reliability_distribution(long: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {"axis": axis, "spearman": value}
        for axis, column in AXES.items()
        for value in _repeat_correlations(long, column)
    ])


def ground_truth_anchoring(long: pd.DataFrame) -> pd.DataFrame:
    catch_gt = long[long["is_catch"] & (long["model"] == GROUND_TRUTH)]
    records = []
    for axis, column in AXES.items():
        ranks = catch_gt[column].dropna().to_numpy()
        if ranks.size == 0:
            continue
        n_first = int(np.sum(ranks == 1))
        p = stats.binomtest(n_first, ranks.size, p=GROUND_TRUTH_CHANCE_RATE,
                            alternative="greater").pvalue
        records.append({"axis": axis, "n_catch_trials": int(ranks.size),
                        "gt_ranked_first": n_first, "gt_first_rate": n_first / ranks.size,
                        "mean_gt_rank": float(np.mean(ranks)),
                        "chance_rate": GROUND_TRUTH_CHANCE_RATE,
                        "p_value_vs_chance": float(p)})
    return pd.DataFrame(records)


def _repeat_correlations(long: pd.DataFrame, column: str) -> list[float]:
    correlations = []
    for participant_id, repeat_rows in long[long["is_repeat"]].groupby("participant_id"):
        source_number = repeat_rows["repeat_of"].iloc[0]
        if pd.isna(source_number):
            continue
        source_rows = long[(long["participant_id"] == participant_id)
                           & (long["trial_number"] == source_number) & ~long["is_repeat"]]
        merged = repeat_rows.merge(source_rows, on="clip_id", suffixes=("_repeat", "_source"))
        repeat_ranks = merged[f"{column}_repeat"].to_numpy(dtype=float)
        source_ranks = merged[f"{column}_source"].to_numpy(dtype=float)
        if merged.shape[0] < 3 or np.isnan(repeat_ranks).any() or np.isnan(source_ranks).any():
            continue
        rho = stats.spearmanr(repeat_ranks, source_ranks).statistic
        if not np.isnan(rho):
            correlations.append(float(rho))
    return correlations


def _agreement_label(w: float) -> str:
    if np.isnan(w):
        return "not_available"
    if w > 0.7:
        return "strong"
    if w >= 0.3:
        return "moderate"
    return "weak"
