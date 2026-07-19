from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import pandas as pd

from study_analysis import db, plots
from study_analysis.config import Config
from study_analysis.loader import StudyData, build_study_data
from study_analysis.metrics import agreement, behaviour, correlation, difference, rankings


def run_analysis(config: Config) -> None:
    data = _load(config)
    tables_dir = config.out_dir / "tables"
    figures_dir = config.out_dir / "figures"

    _report_quality(data)
    if data.long.empty:
        print("No responses in the database yet — nothing to analyse.")
        return

    tables = _compute_tables(data, config)
    _write_tables(tables, tables_dir)
    _make_figures(tables, figures_dir)
    _print_headlines(tables)

    print(f"\nTables → {tables_dir}\nFigures → {figures_dir}")


def _load(config: Config) -> StudyData:
    participants = db.fetch_participants(config)
    responses = db.fetch_responses(config)
    print(f"Fetched {len(participants)} participants, {len(responses)} responses.")
    return build_study_data(participants, responses, config)


def _compute_tables(data: StudyData, config: Config) -> dict[str, pd.DataFrame]:
    scoring_trials = data.clean().real_trials()
    all_trials = data.long

    tables = {
        "mean_rank": rankings.mean_rank_table(scoring_trials, config),
        "win_rate": rankings.win_rate_table(scoring_trials),
        "friedman": difference.friedman(scoring_trials),
        "nemenyi": difference.nemenyi(scoring_trials),
        "pairwise_wilcoxon": difference.pairwise_wilcoxon(scoring_trials),
        "key_pair": difference.key_pair_test(scoring_trials),
        "realism_vs_coherence": correlation.realism_vs_coherence(scoring_trials, config),
        "per_trial_tau": correlation.per_trial_taus(scoring_trials),
        "clip_mean_ranks": correlation.clip_mean_ranks(scoring_trials),
        "kendall_w": agreement.kendalls_w(scoring_trials),
        "reliability": agreement.intra_rater_reliability(all_trials, config),
        "reliability_distribution": agreement.reliability_distribution(all_trials),
        "ground_truth_anchoring": agreement.ground_truth_anchoring(all_trials),
        "replays_by_model": behaviour.replays_by_model(scoring_trials),
        "replays_by_rank": behaviour.replays_by_rank(scoring_trials),
        "duration_by_trial": behaviour.duration_by_trial(all_trials),
    }

    proxy = correlation.metric_human_correlation(scoring_trials, config.objective_metrics_csv)
    if proxy is not None:
        tables["metric_proxy"] = proxy
    return tables


def _write_tables(tables: dict[str, pd.DataFrame], tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(tables_dir / f"{name}.csv", index=False)


def _make_figures(tables: dict[str, pd.DataFrame], figures_dir: Path) -> None:
    plots.plot_mean_ranks(tables["mean_rank"], figures_dir)
    plots.plot_critical_difference(tables["mean_rank"], tables["nemenyi"], figures_dir)
    plots.plot_realism_vs_coherence(tables["mean_rank"], figures_dir)
    plots.plot_kendall_w(tables["kendall_w"], figures_dir)
    plots.plot_reliability(tables["reliability_distribution"], figures_dir)
    plots.plot_ground_truth(tables["ground_truth_anchoring"], figures_dir)
    plots.plot_axis_tau(tables["per_trial_tau"], figures_dir)
    plots.plot_replays_by_model(tables["replays_by_model"], figures_dir)
    plots.plot_duration_by_trial(tables["duration_by_trial"], figures_dir)
    if "metric_proxy" in tables:
        plots.plot_metric_proxy(tables["metric_proxy"], figures_dir)


def _report_quality(data: StudyData) -> None:
    participants = data.participants
    total = len(participants)
    if not total:
        print("No participants found.")
        return
    engaged = int((participants["n_trials"] > 0).sum())
    reached_catch = int(participants["reached_catch"].sum())
    passed = int(participants["catch_passed"].sum())
    pass_rate = f"{passed / reached_catch:.0%}" if reached_catch else "n/a"
    print(f"Participants: {total} claimed a session, {engaged} did at least one trial, "
          f"{reached_catch} reached the attention check.")
    print(f"Attention check: {passed}/{reached_catch} passed ({pass_rate}). "
          f"Analysis sample = {passed} participants.")


def _print_headlines(tables: dict[str, pd.DataFrame]) -> None:
    print("\n--- Headline results ---")
    for _, row in tables["friedman"].iterrows():
        print(f"Friedman [{row['axis']}]: chi2={row['friedman_chi2']:.1f}, "
              f"p={row['p_value']:.2e} over {row['n_blocks']} trials.")
    for _, row in tables["key_pair"].iterrows():
        better = row["model_a"] if row["mean_rank_a"] < row["mean_rank_b"] else row["model_b"]
        print(f"{row['model_a']} vs {row['model_b']} [{row['axis']}]: "
              f"p={row['p_value']:.3f}, better = {better}.")
    for _, row in tables["ground_truth_anchoring"].iterrows():
        print(f"Ground truth #1 [{row['axis']}]: {row['gt_first_rate']:.0%} "
              f"(chance {row['chance_rate']:.0%}, p={row['p_value_vs_chance']:.2e}).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse the music listening study.")
    parser.add_argument("--out", type=Path, help="output directory")
    parser.add_argument("--metrics-csv", type=Path,
                        help="objective metrics per clip (enables the proxy analysis)")
    args = parser.parse_args()

    config = Config.from_env()
    if args.out:
        config = dataclasses.replace(config, out_dir=args.out)
    if args.metrics_csv:
        config = dataclasses.replace(config, objective_metrics_csv=args.metrics_csv)

    run_analysis(config)


if __name__ == "__main__":
    main()
