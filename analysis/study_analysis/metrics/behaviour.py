from __future__ import annotations

import pandas as pd

from study_analysis.config import MODELS


def replays_by_model(real_trials: pd.DataFrame) -> pd.DataFrame:
    grouped = real_trials.dropna(subset=["replays"]).groupby("model")["replays"]
    table = grouped.agg(mean_replays="mean", median_replays="median", n="count").reset_index()
    return table.set_index("model").reindex(list(MODELS)).reset_index()


def replays_by_rank(real_trials: pd.DataFrame) -> pd.DataFrame:
    rows = real_trials.dropna(subset=["replays", "realism_rank"])
    return rows.groupby("realism_rank")["replays"].agg(mean_replays="mean", n="count").reset_index()


def duration_by_trial(long: pd.DataFrame) -> pd.DataFrame:
    per_trial = (long.drop_duplicates(subset=["participant_id", "trial_number"])
                 .dropna(subset=["duration_s"]))
    return (per_trial.groupby("trial_number")["duration_s"]
            .agg(median_duration_s="median", mean_duration_s="mean", n="count").reset_index())
