from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np
import soundfile as sf
from numpy import ndarray


def _collect_track_dirs(folder: str, max_amount: int | None, seed: int) -> list[str]:
    dirs = sorted(
        d for d in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, d))
    )
    if max_amount is not None and max_amount < len(dirs):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(dirs), max_amount, replace=False)
        dirs = [dirs[i] for i in sorted(idx)]
    return dirs


def load_stems(
    track_dir: str,
    stems: list[str],
    label: str = "",
    skip_silent: bool = False,
    silence_threshold: float = 1e-6,
) -> tuple[None, None] | tuple[dict[str, ndarray[Any, Any]], int | None]:
    prefix = f"[{label}] " if label else ""
    wavs: dict[str, np.ndarray] = {}
    source_sr: int | None = None

    for stem in stems:
        path = os.path.join(track_dir, stem + ".wav")
        if not os.path.isfile(path):
            warnings.warn(f"{prefix}Missing {path} - ignoring stem")
            continue
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if skip_silent and np.max(np.abs(audio)) < silence_threshold:
            warnings.warn(f"{prefix}Silent audio {path} - ignoring stem")
            continue
        if source_sr is None:
            source_sr = sr
        wavs[stem] = audio

    if len(wavs) < 2:
        warnings.warn(f"{prefix}{track_dir}: fewer than 2 usable stems - skipping track")
        return None, None

    min_len = min(len(a) for a in wavs.values())
    wavs = {s: a[:min_len] for s, a in wavs.items()}
    return wavs, source_sr
