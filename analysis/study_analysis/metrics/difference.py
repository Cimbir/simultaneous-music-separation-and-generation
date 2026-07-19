from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from study_analysis.config import KEY_PAIR, MODELS
from study_analysis.metrics.common import AXES, holm_correction, rank_matrix


def friedman(real_trials: pd.DataFrame) -> pd.DataFrame:
    records = []
    for axis in AXES:
        matrix = rank_matrix(real_trials, axis)
        stat, p = stats.friedmanchisquare(*[matrix[m] for m in MODELS])
        records.append({"axis": axis, "friedman_chi2": float(stat),
                        "df": len(MODELS) - 1, "p_value": float(p),
                        "n_blocks": int(matrix.shape[0])})
    return pd.DataFrame(records)


def nemenyi(real_trials: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    k = len(MODELS)
    records = []
    for axis in AXES:
        matrix = rank_matrix(real_trials, axis)
        n_blocks = matrix.shape[0]
        mean_ranks = matrix.mean()
        q_alpha = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
        critical_difference = q_alpha * np.sqrt(k * (k + 1) / (6 * n_blocks))
        for a, b in combinations(MODELS, 2):
            gap = float(abs(mean_ranks[a] - mean_ranks[b]))
            records.append({"axis": axis, "model_a": a, "model_b": b,
                            "mean_rank_gap": gap,
                            "critical_difference": float(critical_difference),
                            "significant": gap > critical_difference})
    return pd.DataFrame(records)


def pairwise_wilcoxon(real_trials: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for axis in AXES:
        matrix = rank_matrix(real_trials, axis)
        rows, raw_p = [], []
        for a, b in combinations(MODELS, 2):
            p = _wilcoxon_p(matrix[a].to_numpy(), matrix[b].to_numpy())
            rows.append({"axis": axis, "model_a": a, "model_b": b,
                         "median_rank_a": float(matrix[a].median()),
                         "median_rank_b": float(matrix[b].median()), "p_value": p})
            raw_p.append(p)
        for row, p_adj in zip(rows, holm_correction(raw_p)):
            row["p_value_holm"] = p_adj
            row["significant"] = p_adj < 0.05
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def key_pair_test(real_trials: pd.DataFrame) -> pd.DataFrame:
    a_name, b_name = KEY_PAIR
    records = []
    for axis in AXES:
        matrix = rank_matrix(real_trials, axis)
        a, b = matrix[a_name].to_numpy(), matrix[b_name].to_numpy()
        records.append({"axis": axis, "model_a": a_name, "model_b": b_name,
                        "mean_rank_a": float(np.mean(a)), "mean_rank_b": float(np.mean(b)),
                        "p_value": _wilcoxon_p(a, b), "n_trials": int(a.size)})
    return pd.DataFrame(records)


def _wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 if np.all(a == b) else float(stats.wilcoxon(a, b).pvalue)
