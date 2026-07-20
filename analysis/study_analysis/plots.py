from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study_analysis.config import GRID_COLOR, INK_PRIMARY, INK_SECONDARY, MODEL_COLORS
from study_analysis.metrics.common import AXES

_AXIS_TITLE = {"realism": "Realism", "coherence": "Instrument coherence"}
_REALISM_COLOR = "#2a78d6"
_COHERENCE_COLOR = "#eb6834"


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": INK_SECONDARY, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelcolor": INK_PRIMARY, "text.color": INK_PRIMARY,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "font.size": 10, "figure.dpi": 120,
    })


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _color(model: str) -> str:
    return MODEL_COLORS.get(model, INK_SECONDARY)


def _grid(ax, which: str = "y") -> None:
    getattr(ax, f"{which}axis").grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def _bar_labels(ax, bars, values, fmt="{:.2f}") -> None:
    for bar, value in zip(bars, values):
        ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, value),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)


def plot_mean_ranks(mean_rank_table: pd.DataFrame, out_dir: Path) -> Path:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, (axis_name, panel) in zip(axes, mean_rank_table.groupby("axis")):
        best_last = panel.sort_values("mean_rank", ascending=False)
        y = np.arange(len(best_last))
        errors = [best_last["mean_rank"] - best_last["ci_low"],
                  best_last["ci_high"] - best_last["mean_rank"]]
        ax.errorbar(best_last["mean_rank"], y, xerr=errors, fmt="none",
                    ecolor=INK_SECONDARY, elinewidth=1.2, capsize=3, zorder=2)
        ax.scatter(best_last["mean_rank"], y,
                   color=[_color(m) for m in best_last["model"]], s=90, zorder=3)
        for yi, (_, row) in zip(y, best_last.iterrows()):
            ax.annotate(f"{row['mean_rank']:.2f}", (row["mean_rank"], yi),
                        textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
        ax.set_yticks(y, best_last["model"])
        ax.set_title(_AXIS_TITLE[axis_name])
        ax.set_xlabel("Mean rank  (1 = best)")
        ax.set_xlim(0.7, len(best_last) + 0.3)
        _grid(ax, "x")
    fig.suptitle("Model preference by axis", fontweight="bold", y=1.02)
    return _save(fig, out_dir, "mean_ranks")


def plot_critical_difference(
    mean_rank_table: pd.DataFrame, nemenyi_table: pd.DataFrame, out_dir: Path
) -> Path:
    _style()
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.4))
    for ax, axis_name in zip(axes, AXES):
        mean_ranks = (mean_rank_table[mean_rank_table["axis"] == axis_name]
                      .set_index("model")["mean_rank"])
        critical_difference = float(
            nemenyi_table.loc[nemenyi_table["axis"] == axis_name, "critical_difference"].iloc[0]
        )
        _draw_cd_axis(ax, mean_ranks, critical_difference, _AXIS_TITLE[axis_name])
    fig.suptitle("Critical difference diagram", fontweight="bold", y=1.0)
    fig.tight_layout()
    return _save(fig, out_dir, "critical_difference")


def _cliques_within_cd(sorted_ranks: np.ndarray, critical_difference: float) -> list[tuple[int, int]]:
    spans = []
    for start in range(len(sorted_ranks)):
        end = start
        while (end + 1 < len(sorted_ranks)
               and sorted_ranks[end + 1] - sorted_ranks[start] <= critical_difference):
            end += 1
        if end > start:
            spans.append((start, end))
    return sorted({
        span for span in spans
        if not any(other != span and other[0] <= span[0] and span[1] <= other[1]
                   for other in spans)
    })


def _draw_cd_axis(ax, mean_ranks: pd.Series, critical_difference: float, title: str) -> None:
    ordered = mean_ranks.sort_values()
    ax.set_xlim(0.8, len(mean_ranks) + 0.2)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Mean rank  (1 = best)")
    _grid(ax, "x")
    for model, rank in ordered.items():
        ax.scatter(rank, 0.6, color=_color(model), s=80, zorder=3)
        ax.annotate(f"{model}\n{rank:.2f}", (rank, 0.6), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8.5)
    ranks = ordered.to_numpy()
    level = 0.34
    for start, end in _cliques_within_cd(ranks, critical_difference):
        ax.plot([ranks[start], ranks[end]], [level, level], color=INK_PRIMARY,
                linewidth=3, solid_capstyle="round", zorder=2)
        level -= 0.09
    ax.annotate(f"CD = {critical_difference:.2f}", (0.82, 0.06),
                fontsize=8.5, color=INK_SECONDARY)


def plot_realism_vs_coherence(mean_rank_table: pd.DataFrame, out_dir: Path) -> Path:
    _style()
    wide = mean_rank_table.pivot(index="model", columns="axis", values="mean_rank")
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    limit = (0.8, len(wide) + 0.2)
    ax.plot(limit, limit, color=GRID_COLOR, linewidth=1, zorder=1)
    for model, row in wide.iterrows():
        ax.scatter(row["realism"], row["coherence"], color=_color(model), s=110, zorder=3)
        ax.annotate(model, (row["realism"], row["coherence"]), textcoords="offset points",
                    xytext=(8, 4), fontsize=9)
    ax.set_xlim(*limit)
    ax.set_ylim(*limit)
    ax.set_xlabel("Mean realism rank  (1 = best)")
    ax.set_ylabel("Mean coherence rank  (1 = best)")
    ax.set_title("Realism vs coherence per model", fontweight="bold")
    _grid(ax, "x")
    _grid(ax, "y")
    return _save(fig, out_dir, "realism_vs_coherence")


def plot_kendall_w(kendall_w_table: pd.DataFrame, out_dir: Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    bars = ax.bar([_AXIS_TITLE[a] for a in kendall_w_table["axis"]],
                  kendall_w_table["kendall_w"], color=_REALISM_COLOR, width=0.55)
    _bar_labels(ax, bars, kendall_w_table["kendall_w"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Kendall's W  (0 = none, 1 = full)")
    ax.set_title("Between-rater consensus", fontweight="bold")
    _grid(ax)
    return _save(fig, out_dir, "kendall_w")


def plot_reliability(distribution: pd.DataFrame, out_dir: Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    bins = np.linspace(-1, 1, 17)
    for axis_name, color in (("realism", _REALISM_COLOR), ("coherence", _COHERENCE_COLOR)):
        ax.hist(distribution.loc[distribution["axis"] == axis_name, "spearman"],
                bins=bins, alpha=0.6, color=color, label=_AXIS_TITLE[axis_name])
    ax.axvline(0, color=INK_SECONDARY, linewidth=1, linestyle="--")
    ax.set_xlabel("Spearman (repeat vs source trial)")
    ax.set_ylabel("Participants")
    ax.set_title("Intra-rater reliability", fontweight="bold")
    ax.legend(frameon=False)
    _grid(ax)
    return _save(fig, out_dir, "reliability")


def plot_ground_truth(anchoring_table: pd.DataFrame, out_dir: Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    bars = ax.bar([_AXIS_TITLE[a] for a in anchoring_table["axis"]],
                  anchoring_table["gt_first_rate"], color="#008300", width=0.55)
    chance = float(anchoring_table["chance_rate"].iloc[0])
    ax.axhline(chance, color=INK_SECONDARY, linestyle="--", linewidth=1,
               label=f"Chance ({chance:.2f})")
    _bar_labels(ax, bars, anchoring_table["gt_first_rate"], fmt="{:.0%}")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Ground truth ranked #1")
    ax.set_title("Can listeners spot the real recording?", fontweight="bold", fontsize=10)
    ax.legend(frameon=False)
    _grid(ax)
    return _save(fig, out_dir, "ground_truth_anchoring")


def plot_axis_tau(per_trial_tau: pd.DataFrame, out_dir: Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.hist(per_trial_tau["tau"], bins=np.linspace(-1, 1, 17), color="#4a3aa7", alpha=0.8)
    mean_tau = per_trial_tau["tau"].mean()
    ax.axvline(mean_tau, color=INK_PRIMARY, linewidth=1.5, label=f"mean = {mean_tau:.2f}")
    ax.set_xlabel("Kendall tau (realism vs coherence, per trial)")
    ax.set_ylabel("Trials")
    ax.set_title("Are the two axes the same judgement?", fontweight="bold", fontsize=10)
    ax.legend(frameon=False)
    _grid(ax)
    return _save(fig, out_dir, "axis_tau")


def plot_metric_proxy(proxy_table: pd.DataFrame, out_dir: Path) -> Path:
    _style()
    metrics = list(proxy_table["metric"].unique())
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(metrics)), 4))
    for offset, (axis_name, color) in enumerate(
        (("realism", _REALISM_COLOR), ("coherence", _COHERENCE_COLOR))
    ):
        panel = proxy_table[proxy_table["axis"] == axis_name].set_index("metric")
        heights = [panel.loc[m, "abs_tau"] if m in panel.index else 0 for m in metrics]
        ax.bar(x + (offset - 0.5) * 0.4, heights, width=0.4, color=color,
               label=_AXIS_TITLE[axis_name])
    ax.set_xticks(x, metrics, rotation=20, ha="right")
    ax.set_ylabel("|Kendall tau| vs human rank")
    ax.set_title("Which metric tracks human taste?", fontweight="bold")
    ax.legend(frameon=False)
    _grid(ax)
    return _save(fig, out_dir, "metric_proxy")


def plot_replays_by_model(replay_table: pd.DataFrame, out_dir: Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    present = replay_table.dropna(subset=["mean_replays"])
    bars = ax.bar(present["model"], present["mean_replays"],
                  color=[_color(m) for m in present["model"]], width=0.6)
    _bar_labels(ax, bars, present["mean_replays"], fmt="{:.1f}")
    ax.set_ylabel("Mean replays per clip")
    ax.set_title("Re-listening by model", fontweight="bold")
    _grid(ax)
    return _save(fig, out_dir, "replays_by_model")


def plot_duration_by_trial(duration_table: pd.DataFrame, out_dir: Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(duration_table["trial_number"], duration_table["median_duration_s"],
            color=_REALISM_COLOR, linewidth=2, marker="o", markersize=6)
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Median duration (s)")
    ax.set_title("Time per trial over the session", fontweight="bold")
    ax.set_xticks(duration_table["trial_number"])
    _grid(ax)
    return _save(fig, out_dir, "duration_by_trial")
