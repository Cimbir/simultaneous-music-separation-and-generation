from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import stats

from study_analysis.config import Config, GROUND_TRUTH, MODEL_ALIASES, MODELS
from study_analysis.metrics.common import AXES, bootstrap_ci

LOWER_IS_BETTER = {
    "mel_mse", "fad_vggish", "fad_clap", "kad", "kad_vggish", "kad_clap", "irs", "cbd",
}

HIGHER_IS_BETTER = {
    "si_sdr_i", "cbs", "cocola_both", "cocola_harmonic",
    "cocola_percussive", "ba", "ta",
}


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
        clip = (real_trials.groupby(["clip_id", "model"])[column]
                .agg(mean_human_rank="mean", n_ratings="count")
                .reset_index())
        clip["axis"] = axis
        clip["sample_id"] = clip["clip_id"].map(_sample_id)
        frames.append(clip)
    return pd.concat(frames, ignore_index=True)


def clip_metric_scores(metrics_csv: Path | None) -> pd.DataFrame | None:
    if metrics_csv is None or not Path(metrics_csv).is_file():
        return None

    metrics = _read_metrics_csv(Path(metrics_csv))
    return metrics[metrics["model"].isin(MODELS)].reset_index(drop=True)


def model_metric_table(model_metrics: Path | None) -> pd.DataFrame | None:
    if model_metrics is None or not Path(model_metrics).is_file():
        return None

    metrics = _read_model_metrics(Path(model_metrics))
    return metrics[metrics["model"].isin(MODELS)].reset_index(drop=True)


def clip_metric_spearman(
    real_trials: pd.DataFrame, metrics_csv: Path | None, config: Config
) -> pd.DataFrame | None:
    metrics = clip_metric_scores(metrics_csv)
    if metrics is None:
        return None

    metric_columns = _metric_columns(metrics)
    human = clip_mean_ranks(real_trials)
    human = human[human["model"] != GROUND_TRUTH]

    records = []
    for axis in AXES:
        merged = human[human["axis"] == axis].merge(
            metrics, on=["clip_id", "model", "sample_id"], how="inner"
        )
        for metric in metric_columns:
            valid = merged[["mean_human_rank", metric]].dropna()
            if valid.shape[0] < 3:
                continue
            direction = metric_direction(metric)
            metric_quality = valid[metric].to_numpy(dtype=float) * direction
            human_quality = -valid["mean_human_rank"].to_numpy(dtype=float)
            result = stats.spearmanr(metric_quality, human_quality)
            raw = stats.spearmanr(valid[metric], valid["mean_human_rank"])
            lo, hi = _bootstrap_spearman(
                metric_quality, human_quality, n_boot=config.n_bootstrap, seed=config.seed
            )
            records.append({
                "axis": axis,
                "metric": metric,
                "score_direction": metric_direction_label(metric),
                "spearman_rho": float(result.statistic),
                "ci_low": lo,
                "ci_high": hi,
                "p_value": float(result.pvalue),
                "raw_score_vs_rank_rho": float(raw.statistic),
                "n_clips": int(valid.shape[0]),
                "n_models": int(merged.loc[valid.index, "model"].nunique()),
            })

    table = pd.DataFrame(records)
    if not table.empty:
        table["abs_rho"] = table["spearman_rho"].abs()
        table["metric_rank"] = (table.groupby("axis")["abs_rho"]
                               .rank(ascending=False, method="min").astype(int))
        table = table.sort_values(["axis", "metric_rank", "metric"]).reset_index(drop=True)
    return table


def model_metric_scores(
    clip_metrics_csv: Path | None, model_metrics: Path | None
) -> pd.DataFrame | None:
    frames = []
    clip_metrics = clip_metric_scores(clip_metrics_csv)
    if clip_metrics is not None:
        grouped = clip_metrics.groupby("model")[_metric_columns(clip_metrics)].mean().reset_index()
        frames.append(_long_scores(grouped, "clip_mean"))

    model_table = model_metric_table(model_metrics)
    if model_table is not None:
        frames.append(_long_scores(model_table, "table"))

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def model_metric_alignment(
    bt_scores: pd.DataFrame,
    clip_metrics_csv: Path | None,
    model_metrics: Path | None,
    config: Config,
) -> pd.DataFrame | None:
    scores = model_metric_scores(clip_metrics_csv, model_metrics)
    if scores is None:
        return None

    records = []
    for (source, metric), metric_scores in scores.groupby(["source", "metric"]):
        for axis, human_scores in bt_scores.groupby("axis"):
            merged = human_scores.merge(metric_scores, on="model", how="inner").dropna(
                subset=["bt_score", "score"]
            )
            if merged.shape[0] < 3:
                continue
            direction = metric_direction(metric)
            metric_quality = merged["score"].to_numpy(dtype=float) * direction
            human_quality = merged["bt_score"].to_numpy(dtype=float)
            result = stats.spearmanr(metric_quality, human_quality)
            lo, hi = _bootstrap_spearman(
                metric_quality, human_quality, n_boot=config.n_bootstrap, seed=config.seed
            )
            inversions, n_pairs = _rank_inversions(metric_quality, human_quality)
            records.append({
                "axis": axis,
                "source": source,
                "metric": metric,
                "score_direction": metric_direction_label(metric),
                "spearman_rho": float(result.statistic),
                "ci_low": lo,
                "ci_high": hi,
                "top1_accuracy": int(_top(metric_quality, merged["model"]) == _top(human_quality, merged["model"])),
                "pairwise_rank_inversions": inversions,
                "n_model_pairs": n_pairs,
                "pairwise_rank_inversion_rate": inversions / n_pairs if n_pairs else float("nan"),
                "metric_top_model": _top(metric_quality, merged["model"]),
                "human_top_model": _top(human_quality, merged["model"]),
                "n_models": int(merged.shape[0]),
            })

    table = pd.DataFrame(records)
    if not table.empty:
        table["abs_rho"] = table["spearman_rho"].abs()
        table = table.sort_values(["axis", "source", "metric"]).reset_index(drop=True)
    return table


def metric_direction(metric: str) -> int:
    if metric in LOWER_IS_BETTER:
        return -1
    if metric in HIGHER_IS_BETTER:
        return 1
    return 1


def metric_direction_label(metric: str) -> str:
    return "lower_better" if metric_direction(metric) < 0 else "higher_better"


def _read_metrics_csv(path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(path)
    metrics = metrics.rename(columns={column: _metric_name(column) for column in metrics.columns})
    if "sample_id" not in metrics.columns and "clip_id" in metrics.columns:
        metrics["sample_id"] = metrics["clip_id"].map(_sample_id)
    metrics["model"] = metrics["model"].map(_model_name)
    metrics["sample_id"] = metrics["sample_id"].astype(str).str.zfill(2)
    metrics["clip_id"] = metrics.apply(lambda row: f"{row.model}/{row.sample_id}", axis=1)
    return metrics


def _read_model_metrics(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".md":
        metrics = _read_markdown_table(path)
    else:
        metrics = pd.read_csv(path)
    metrics = metrics.rename(columns={column: _metric_name(column) for column in metrics.columns})
    metrics = metrics.rename(columns={"model": "model"})
    metrics["model"] = metrics["model"].map(_model_name)
    for column in _score_columns(metrics):
        metrics[column] = metrics[column].map(_metric_value)
    return metrics


def _read_markdown_table(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and not all(set(cell) <= {"-", ":"} for cell in cells):
            rows.append(cells)
    return pd.DataFrame(rows[1:], columns=rows[0])


def _long_scores(scores: pd.DataFrame, source: str) -> pd.DataFrame:
    records = []
    for row in scores.itertuples(index=False):
        for metric in _metric_columns(scores):
            value = getattr(row, metric)
            records.append({
                "source": source,
                "model": row.model,
                "metric": metric,
                "score": float(value) if pd.notna(value) else float("nan"),
                "score_direction": metric_direction_label(metric),
            })
    return pd.DataFrame(records)


def _metric_columns(metrics: pd.DataFrame) -> list[str]:
    return [
        column for column in metrics.columns
        if column in _score_columns(metrics)
        and pd.api.types.is_numeric_dtype(metrics[column])
    ]


def _score_columns(metrics: pd.DataFrame) -> list[str]:
    return [
        column for column in metrics.columns
        if column not in {"model", "sample_id", "clip_id", "source"}
    ]


def _metric_name(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower())).strip("_")


def _metric_value(value) -> float:
    if pd.isna(value):
        return float("nan")
    cleaned = str(value).replace("*", "").strip()
    if cleaned.lower() in {"", "nan"}:
        return float("nan")
    return float(cleaned)


def _model_name(model: str) -> str:
    model = str(model).strip()
    if model.lower() == "ground_truth":
        return GROUND_TRUTH
    return MODEL_ALIASES.get(model, model)


def _sample_id(clip_id: str) -> str:
    return str(clip_id).split("/")[-1].zfill(2)


def _bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, *, n_boot: int, seed: int
) -> tuple[float, float]:
    if x.size < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        idx = rng.integers(0, x.size, size=x.size)
        if np.unique(x[idx]).size < 2 or np.unique(y[idx]).size < 2:
            continue
        rho = stats.spearmanr(x[idx], y[idx]).statistic
        if not np.isnan(rho):
            estimates.append(float(rho))
    if not estimates:
        return float("nan"), float("nan")
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    return float(lo), float(hi)


def _rank_inversions(metric_quality: np.ndarray, human_quality: np.ndarray) -> tuple[int, int]:
    inversions = 0
    n_pairs = 0
    for i, j in zip(*np.triu_indices(metric_quality.size, k=1)):
        human_order = np.sign(human_quality[i] - human_quality[j])
        metric_order = np.sign(metric_quality[i] - metric_quality[j])
        if human_order == 0 or metric_order == 0:
            continue
        n_pairs += 1
        inversions += int(human_order != metric_order)
    return inversions, n_pairs


def _top(values: np.ndarray, models: pd.Series) -> str:
    return str(models.iloc[int(np.argmax(values))])
