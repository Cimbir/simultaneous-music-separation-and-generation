from __future__ import annotations

import os
from typing import Any

import numpy as np
from tqdm import tqdm

from ..base import Metric
from ._preprocess import track_stems_bar_chroma
from .profiles import load_profiles, build_key_templates


def _max_correlation(x: np.ndarray, templates: np.ndarray) -> float:
    """Max Pearson correlation of x (12,) against any row in templates (K, 12). Returns 0.0 on zero variance."""
    x_c = x - x.mean()
    x_std = np.sqrt((x_c ** 2).sum())
    if x_std < 1e-10:
        return 0.0

    T_c = templates - templates.mean(axis=1, keepdims=True)  # (K, 12)
    T_std = np.sqrt((T_c ** 2).sum(axis=1))                  # (K,)

    denom = x_std * T_std  # (K,)
    corrs = np.where(denom < 1e-10, 0.0, T_c @ x_c / denom)  # (K,)
    return float(corrs.max())


def tonal_coherence(
    stems: dict[str, np.ndarray],
    profiles: np.ndarray,
    segment_size: int = 4,
) -> dict[str, Any]:
    """Mean max-correlation against key templates over non-overlapping segment_size-bar windows."""
    n_bars = min(c.shape[1] for c in stems.values()) if stems else 0

    segment_scores: list[float] = []
    for start in range(0, n_bars, segment_size):
        end = min(start + segment_size, n_bars)

        mix_chroma = np.zeros(12)
        for chroma in stems.values():
            seg = chroma[:, start:end]
            valid = ~np.any(np.isnan(seg), axis=0)
            if valid.any():
                mix_chroma += seg[:, valid].sum(axis=1)

        l1 = np.abs(mix_chroma).sum()
        if l1 < 1e-8:
            continue
        mix_chroma /= l1

        segment_scores.append(_max_correlation(mix_chroma, profiles))

    tc = float(np.mean(segment_scores)) if segment_scores else float("nan")
    return {"tc": tc, "n_segments": len(segment_scores)}


def tc_short_clip(
    stems: dict[str, np.ndarray],
    profiles: np.ndarray,
) -> float:
    """Whole-clip TC for clips too short for segment-based scoring: aggregate all bars, correlate once."""
    mix_chroma = np.zeros(12)
    for chroma in stems.values():
        valid = ~np.any(np.isnan(chroma), axis=0)
        if valid.any():
            mix_chroma += chroma[:, valid].sum(axis=1)

    l1 = np.abs(mix_chroma).sum()
    if l1 < 1e-8:
        return float("nan")
    mix_chroma /= l1

    return _max_correlation(mix_chroma, profiles)


class TC(Metric):
    """Tonal Coherence. How strongly the mix commits to any key, measured via max Pearson correlation.

    Profiles: "ks" (Krumhansl-Schmuckler), "temperley", or "corpus" (EM-derived from Slakh2100).
    use_short_clip_fallback=True is recommended for 10-15 s clips: clips shorter than segment_size
    bars skip the windowing and aggregate the whole clip instead.
    """

    name = "tc"

    def __init__(
        self,
        stems: list[str] | None = None,
        exclude: list[str] | None = None,
        beat_ref_stem: str | None = "drums",
        profile_set: str = "ks",
        segment_size: int = 4,
        hop_length: int = 512,
        use_short_clip_fallback: bool = True,
    ):
        self.stems = stems or ["guitar", "bass", "piano"]
        self.exclude = exclude or ["drums"]
        self.beat_ref_stem = beat_ref_stem
        self.profile_set = profile_set
        self.segment_size = segment_size
        self.hop_length = hop_length
        self.use_short_clip_fallback = use_short_clip_fallback

    def compute(
        self,
        folder: str,
        stems: list[str] | None = None,
        exclude: list[str] | None = None,
        beat_ref_stem: str | None = "KEEP",
        **_,
    ) -> dict:
        stems = stems or self.stems
        exclude = exclude if exclude is not None else self.exclude
        beat_ref_stem = self.beat_ref_stem if beat_ref_stem == "KEEP" else beat_ref_stem
        active = [s for s in stems if s not in exclude]

        major, minor = load_profiles(self.profile_set)
        templates = build_key_templates(major, minor)

        track_dirs = sorted(
            d for d in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, d))
        )

        track_scores: list[float] = []
        track_n_segs: list[int]   = []
        short_clip_count = 0

        for track in tqdm(track_dirs, desc="TC tracks"):
            track_path = os.path.join(folder, track)
            stem_file_dict = {
                stem: os.path.join(track_path, stem + ".wav")
                for stem in active
                if os.path.isfile(os.path.join(track_path, stem + ".wav"))
            }
            if not stem_file_dict:
                continue

            beat_ref = None
            if beat_ref_stem is not None:
                ref_file = os.path.join(track_path, beat_ref_stem + ".wav")
                if os.path.isfile(ref_file):
                    beat_ref = ref_file
            track_stems = track_stems_bar_chroma(stem_file_dict, self.hop_length, beat_ref)

            if not track_stems:
                continue

            n_bars = min(c.shape[1] for c in track_stems.values())

            if self.use_short_clip_fallback and n_bars < self.segment_size:
                score = tc_short_clip(track_stems, templates)
                if not np.isnan(score):
                    track_scores.append(score)
                    track_n_segs.append(1)
                    short_clip_count += 1
            else:
                result = tonal_coherence(track_stems, templates, self.segment_size)
                if not np.isnan(result["tc"]):
                    track_scores.append(result["tc"])
                    track_n_segs.append(result["n_segments"])

        tc = float(np.mean(track_scores)) if track_scores else float("nan")
        avg_segs = float(np.mean(track_n_segs)) if track_n_segs else float("nan")

        return {
            "tc":          tc,
            "profile_set": self.profile_set,
            "n_segments":  avg_segs,
            "short_clip":  short_clip_count,
        }
