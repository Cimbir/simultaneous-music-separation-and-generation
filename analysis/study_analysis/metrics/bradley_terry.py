from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from study_analysis.config import Config, MODELS
from study_analysis.metrics.common import AXES


def pairwise_preferences(real_trials: pd.DataFrame) -> pd.DataFrame:
    records = []
    for axis, column in AXES.items():
        for block_id, ((participant, trial_number), trial) in enumerate(
            real_trials.groupby(["participant_id", "trial_number"], sort=False)
        ):
            ranked = trial.dropna(subset=[column])
            for a, b in combinations(ranked.itertuples(), 2):
                winner, loser = (a, b) if getattr(a, column) < getattr(b, column) else (b, a)
                records.append({
                    "axis": axis,
                    "participant_id": participant,
                    "trial_number": trial_number,
                    "block_id": block_id,
                    "winner": winner.model,
                    "loser": loser.model,
                })
    return pd.DataFrame(records)


def bradley_terry_scores(real_trials: pd.DataFrame, config: Config) -> pd.DataFrame:
    preferences = pairwise_preferences(real_trials)
    records = []
    for axis, axis_preferences in preferences.groupby("axis"):
        models = [model for model in MODELS if model in _models(axis_preferences)]
        point = _fit(axis_preferences, models)
        replicates = _bootstrap(axis_preferences, models, config)
        intervals = {
            model: np.percentile(replicates[:, i], [2.5, 97.5])
            for i, model in enumerate(models)
        }
        ranks = pd.Series(point, index=models).rank(ascending=False, method="min")
        for model, score in zip(models, point):
            lo, hi = intervals[model]
            records.append({
                "axis": axis,
                "model": model,
                "bt_score": float(score),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "bt_rank": int(ranks[model]),
                "n_blocks": int(axis_preferences["block_id"].nunique()),
                "n_pairwise_preferences": int(axis_preferences.shape[0]),
            })
    return pd.DataFrame(records)


def _models(preferences: pd.DataFrame) -> set[str]:
    return set(preferences["winner"]) | set(preferences["loser"])


def _fit(preferences: pd.DataFrame, models: list[str]) -> np.ndarray:
    index = {model: i for i, model in enumerate(models)}
    wins = np.zeros((len(models), len(models)), dtype=float)
    for row in preferences.itertuples():
        wins[index[row.winner], index[row.loser]] += 1
    return _mm_scores(wins)


def _mm_scores(wins: np.ndarray, *, max_iter: int = 10_000, tol: float = 1e-10) -> np.ndarray:
    abilities = np.ones(wins.shape[0], dtype=float)
    totals = wins + wins.T
    won = wins.sum(axis=1)
    for _ in range(max_iter):
        denom = np.divide(totals, abilities[:, None] + abilities[None, :]).sum(axis=1)
        updated = np.divide(won, denom, out=np.zeros_like(won), where=denom > 0)
        updated = np.maximum(updated, 1e-12)
        updated /= np.exp(np.mean(np.log(updated)))
        if np.max(np.abs(np.log(updated) - np.log(abilities))) < tol:
            abilities = updated
            break
        abilities = updated
    scores = np.log(abilities)
    return scores - scores.mean()


def _bootstrap(
    preferences: pd.DataFrame, models: list[str], config: Config
) -> np.ndarray:
    rng = np.random.default_rng(config.seed)
    block_ids = preferences["block_id"].unique()
    by_block = {block_id: block for block_id, block in preferences.groupby("block_id")}
    replicates = []
    for _ in range(config.n_bootstrap):
        sampled = rng.choice(block_ids, size=len(block_ids), replace=True)
        sample = pd.concat([by_block[block_id] for block_id in sampled], ignore_index=True)
        replicates.append(_fit(sample, models))
    return np.vstack(replicates)
