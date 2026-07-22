from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor


def collect_track_groups(
    folder: str,
    stems: list[str],
    max_amount: int | None = None,
    label: str = "",
) -> list[list[str]]:
    track_dirs = sorted(
        d for d in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, d))
    )
    if max_amount is not None:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(track_dirs), min(max_amount, len(track_dirs)), replace=False)
        track_dirs = [track_dirs[i] for i in idx]

    prefix = f"[{label}] " if label else ""
    groups = []
    for track in track_dirs:
        track_path = os.path.join(folder, track)
        files = [os.path.join(track_path, stem + ".wav") for stem in stems]
        missing = [s for s, f in zip(stems, files) if not os.path.isfile(f)]
        present = [f for f in files if os.path.isfile(f)]
        if missing:
            print(f"{prefix}{track}: missing stems {missing}, ignoring those stems")
        if len(present) >= 2:
            groups.append(present)
        elif present:
            print(f"{prefix}{track}: only 1 stem present, skipping")
    return groups


def extract_beats(audio_file: str) -> np.ndarray:
    proc = RNNDownBeatProcessor()
    activation = proc(audio_file)
    dbn = DBNDownBeatTrackingProcessor(
        beats_per_bar=[3, 4], min_bpm=30, max_bpm=300,
        fps=150, transition_lambda=100,
    )
    try:
        return dbn(activation)[:, 0]
    except ValueError:
        warnings.warn(f"madmom DBN failed on {audio_file!r} - skipping beat tracking for this file")
        return np.array([])