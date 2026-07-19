from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from study_analysis.config import Config, GROUND_TRUTH
from study_analysis.metrics.common import AXES, bootstrap_ci


def realism_vs_coherence(real_trials: pd.DataFrame, config: Config) -> pd.DataFrame:
    taus = per_trial_taus(real_trials)["tau"].to_numpy()
    point, lo, hi = bootstrap_ci(taus, n_boot=config.n_bootstrap, seed=config.seed)
    return pd.DataFrame([{
        "mean_kendall_tau": point, "ci_low": lo, "ci_high": hi,
        "frac_positive": float(np.mean(taus > 0)) if taus.size else float("nan"),
        "n_trials": int(taus.size),
    }])


def per_trial_taus(real_trials: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (participant, trial_number), trial in real_trials.groupby(["participant_id", "trial_number"]):
        realism = trial["realism_rank"].to_numpy(dtype=float)
        coherence = trial["coherence_rank"].to_numpy(dtype=float)
        if np.isnan(realism).any() or np.isnan(coherence).any():
            continue
        tau = stats.kendalltau(realism, coherence).statistic
        if not np.isnan(tau):
            records.append({"participant_id": participant, "trial_number": trial_number, "tau": tau})
    return pd.DataFrame(records)


def clip_mean_ranks(real_trials: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for axis, column in AXES.items():
        clip = (real_trials.groupby(["clip_id", "model"])[column].mean()
                .reset_index().rename(columns={column: "mean_human_rank"}))
        clip["axis"] = axis
        frames.append(clip)
    return pd.concat(frames, ignore_index=True)


def metric_human_correlation(
    real_trials: pd.DataFrame, metrics_csv: Path | None
) -> pd.DataFrame | None:
    if metrics_csv is None or not Path(metrics_csv).is_file():
        return None

    metrics = pd.read_csv(metrics_csv)
    metric_columns = [c for c in metrics.columns if c != "clip_id"]
    human = clip_mean_ranks(real_trials)
    human = human[human["model"] != GROUND_TRUTH]

    records = []
    for axis in AXES:
        merged = human[human["axis"] == axis].merge(metrics, on="clip_id", how="inner")
        for metric in metric_columns:
            valid = merged[["mean_human_rank", metric]].dropna()
            if valid.shape[0] < 3:
                continue
            result = stats.kendalltau(valid[metric], valid["mean_human_rank"])
            records.append({"axis": axis, "metric": metric,
                            "kendall_tau": float(result.statistic),
                            "abs_tau": abs(float(result.statistic)),
                            "p_value": float(result.pvalue), "n_clips": int(valid.shape[0])})

    table = pd.DataFrame(records)
    if not table.empty:
        table["proxy_rank"] = (table.groupby("axis")["abs_tau"]
                               .rank(ascending=False, method="min").astype(int))
        table = table.sort_values(["axis", "proxy_rank"]).reset_index(drop=True)
    return table
