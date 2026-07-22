# https://github.com/YangLabHKUST/SyncTrack/blob/main/eval_metrics/CBD.py
from __future__ import annotations

import numpy as np
from tqdm import tqdm

from madmom.evaluation.beats import find_closest_matches, calc_absolute_errors

from ..base import Metric
from ._common import extract_beats, collect_track_groups


def _relative_beat_error(ref_beats: np.ndarray, tgt_beats: np.ndarray) -> list[dict]:
    ref_beats = np.asarray(ref_beats)
    tgt_beats = np.asarray(tgt_beats)
    N = len(ref_beats)
    if N < 2 or len(tgt_beats) == 0:
        return []

    matches = find_closest_matches(ref_beats, tgt_beats)
    matched = tgt_beats[matches]
    errors = calc_absolute_errors(ref_beats, tgt_beats, matches)

    intervals = np.diff(ref_beats)
    prev_iv = np.concatenate([[intervals[0]], intervals])
    next_iv = np.concatenate([intervals, [intervals[-1]]])

    left  = ref_beats - prev_iv * 0.5
    right = ref_beats + next_iv * 0.5
    in_window = (matched >= left) & (matched < right)

    denom = np.where(matched >= ref_beats, next_iv * 0.5, prev_iv * 0.5)
    norm_errors = np.full(N, np.nan)
    norm_errors[in_window] = errors[in_window] / denom[in_window]

    return [
        {
            "idx":        i,
            "ref":        float(ref_beats[i]),
            "paired":     bool(in_window[i]),
            "error":      float(errors[i]) if in_window[i] else float("nan"),
            "norm_error": float(norm_errors[i]),
        }
        for i in range(N)
    ]


def _summarize(results: list[dict]) -> dict[str, float]:
    paired = [r["norm_error"] for r in results if r["paired"]]
    return {
        "avg_error":        float(np.nanmean(paired))   if paired else float("nan"),
        "avg_std_error":    float(np.nanstd(paired))    if paired else float("nan"),
        "avg_median_error": float(np.nanmedian(paired)) if paired else float("nan"),
    }


def _multi_track_consistency(beats_list: list[np.ndarray]) -> tuple[list[dict], dict]:
    N = len(beats_list)
    stats = []
    for i in range(N):
        all_results: list[dict] = []
        for j in range(N):
            if i == j:
                continue
            all_results.extend(_relative_beat_error(beats_list[i], beats_list[j]))
        stat = _summarize(all_results)
        stat["reference_track"] = i
        stats.append(stat)

    overall = {
        "avg_error":        float(np.nanmean([s["avg_error"]        for s in stats])),
        "avg_std_error":    float(np.nanmean([s["avg_std_error"]     for s in stats])),
        "avg_median_error": float(np.nanmean([s["avg_median_error"]  for s in stats])),
    }
    return stats, overall



class CBD(Metric):
    """Cross-track Beat Dispersion. Normalized pairwise cross-stem beat timing error. Lower is better."""

    name = "cbd"

    def __init__(self, stems: list[str] | None = None):
        self.stems = stems or ["stem_0", "stem_1", "stem_2", "stem_3"]

    def compute(self, folder: str, stems: list[str] | None = None, max_amount: int | None = None, **_) -> dict:
        stems = stems or self.stems
        track_groups = collect_track_groups(folder, stems, max_amount, label="CBD")

        all_avg_error:  list[float] = []
        all_avg_std:    list[float] = []
        all_avg_median: list[float] = []
        all_stats:      list[dict]  = []

        for wav_files in tqdm(track_groups, desc="CBD tracks"):
            beats_list = [extract_beats(f) for f in wav_files]
            _, overall = _multi_track_consistency(beats_list)
            all_stats.append(overall)
            all_avg_error.append(overall["avg_error"])
            all_avg_std.append(overall["avg_std_error"])
            all_avg_median.append(overall["avg_median_error"])

        return {
            "cbd_avg_error":        float(np.nanmean(all_avg_error)),
            "cbd_avg_std_error":    float(np.nanmean(all_avg_std)),
            "cbd_avg_median_error": float(np.nanmean(all_avg_median)),
            "cbd_track_stats":      all_stats,
        }
