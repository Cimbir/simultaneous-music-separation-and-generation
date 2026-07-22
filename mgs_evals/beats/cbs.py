# https://github.com/YangLabHKUST/SyncTrack/blob/main/eval_metrics/CBS.py
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm
import librosa

from ..base import Metric
from ._common import extract_beats, collect_track_groups


def _compute_alignment(track_files: list[str], window_size: float) -> dict:
    track_beats = [extract_beats(f) for f in track_files]

    durations = []
    for f in track_files:
        y, sr_ = librosa.load(f, sr=None)
        durations.append(len(y) / sr_)
    max_len = max(durations)

    N = len(track_beats)
    valid = sum(len(b) > 0 for b in track_beats)
    if valid <= 1:
        return {"mean_beat_ratio": float("nan")}

    T = int(np.ceil((max_len - window_size) / window_size)) + 1
    windows = [(i * window_size, i * window_size + window_size) for i in range(T)]

    b = np.zeros((T, N), dtype=int)
    for j, beats in enumerate(track_beats):
        if len(beats) == 0:
            continue
        for i, (start, end) in enumerate(windows):
            if np.any((beats >= start) & (beats < end)):
                b[i, j] = 1

    p = np.sum(b, axis=1) / valid
    valid_mask = np.sum(b, axis=1) >= 1
    total_valid = int(np.sum(valid_mask))
    mean_beat_ratio = (
        float(np.sum(p[valid_mask]) / (total_valid + 1e-10)) if total_valid > 0 else 0.0
    )
    return {"mean_beat_ratio": mean_beat_ratio}



class CBS(Metric):
    """Cross-track Beat Synchronization. Fraction of stems that beat together per time window. Higher is better."""

    name = "cbs"

    def __init__(
        self,
        stems: list[str] | None = None,
        window_size: float = 0.07,
        num_workers: int = 4,
    ):
        self.stems = stems or ["stem_0", "stem_1", "stem_2", "stem_3"]
        self.window_size = window_size
        self.num_workers = num_workers

    def compute(
            self,
            folder: str,
            stems: list[str] | None = None,
            window_size: float | None = None,
            num_workers: int | None = None,
            max_amount: int | None = None,
            **_,
    ) -> dict:
        stems = stems or self.stems
        window_size = window_size if window_size is not None else self.window_size
        num_workers = num_workers if num_workers is not None else self.num_workers

        track_groups = collect_track_groups(folder, stems, max_amount, label="CBS")
        track_results: list[float] = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_compute_alignment, files, window_size): files
                for files in track_groups
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="CBS tracks"):
                res = fut.result()
                track_results.append(res["mean_beat_ratio"])

        avg = float(np.nanmean(track_results)) if track_results else float("nan")
        return {
            "cbs_mean_beat_ratio": avg,
            "cbs_track_results":   track_results,
        }
