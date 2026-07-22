# https://github.com/YangLabHKUST/SyncTrack/blob/main/eval_metrics/IRS.py
from __future__ import annotations

import os
import numpy as np
from tqdm import tqdm

from ..base import Metric
from ._common import extract_beats


def _beat_intervals(audio_file: str) -> tuple[list[float], list[float]]:
    beat_times = extract_beats(audio_file)
    if len(beat_times) < 2:
        return [], beat_times.tolist()
    return np.diff(beat_times).tolist(), beat_times.tolist()


class IRS(Metric):
    """Intra-Rhythmic Stability. Coefficient of variation (std/mean) of beat intervals per stem. Lower is better.

    CV is tempo-agnostic, so averaging across tracks at different tempos is valid.
    """

    name = "irs"

    def __init__(self, stems: list[str] | None = None):
        self.stems = stems or ["stem_0", "stem_1", "stem_2", "stem_3"]

    def compute(self, folder: str, stems: list[str] | None = None, max_amount: int | None = None, **_) -> dict:
        stems = stems or self.stems

        track_dirs = sorted(
            d for d in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, d))
        )

        if max_amount is not None:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(track_dirs), min(max_amount, len(track_dirs)), replace=False)
            track_dirs = [track_dirs[i] for i in idx]

        stem_cvs:   dict[str, list[float]] = {s: [] for s in stems}
        stem_means: dict[str, list[float]] = {s: [] for s in stems}
        stem_stds:  dict[str, list[float]] = {s: [] for s in stems}
        sample_results: list[float] = []

        for track in tqdm(track_dirs, desc="IRS tracks"):
            track_path = os.path.join(folder, track)
            track_cvs: list[float] = []
            for stem in stems:
                fpath = os.path.join(track_path, stem + ".wav")
                if not os.path.isfile(fpath):
                    continue
                intervals, _ = _beat_intervals(fpath)
                if len(intervals) >= 2:
                    m = float(np.mean(intervals))
                    s = float(np.std(intervals))
                    stem_stds[stem].append(s)
                    if m > 0:
                        cv = s / m
                        stem_cvs[stem].append(cv)
                        stem_means[stem].append(m)
                        track_cvs.append(cv)
            sample_results.append(float(np.mean(track_cvs)) if track_cvs else float("nan"))

        stem_results: dict = {}
        all_cvs: list[float] = []
        for stem in stems:
            cvs   = stem_cvs[stem]
            means = stem_means[stem]
            stds  = stem_stds[stem]
            avg_cv   = float(np.mean(cvs))   if cvs   else float("nan")
            avg_mean = float(np.mean(means))  if means else float("nan")
            avg_std  = float(np.mean(stds))   if stds  else float("nan")
            stem_results[stem] = {
                "avg_cv":             avg_cv,
                "avg_interval_mean":  avg_mean,
                "avg_interval_std":   avg_std,
                "file_count":         len(stds),
            }
            if not np.isnan(avg_cv):
                all_cvs.append(avg_cv)

        return {
            "irs":                float(np.mean(all_cvs)) if all_cvs else float("nan"),
            "irs_stem_results":   stem_results,
            "irs_sample_results": sample_results,
        }
