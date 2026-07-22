from __future__ import annotations

import itertools
import os

import numpy as np
from tqdm import tqdm

from ..base import Metric
from ._preprocess import track_stems_bar_chroma


_CONSONANCE_MAP: dict[int, float] = {
    0: 1.00,   # unison
    7: 0.95,   # perfect 5th
    5: 0.85,   # perfect 4th
    4: 0.80,   # major 3rd
    3: 0.75,   # minor 3rd
    9: 0.70,   # major 6th
    8: 0.65,   # minor 6th
    2: 0.40,   # major 2nd
    10: 0.35,  # minor 7th
    1: 0.10,   # minor 2nd
    11: 0.10,  # major 7th
    6: 0.05,   # tritone
}


def _build_consonance_matrix() -> np.ndarray:
    W = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            W[i, j] = _CONSONANCE_MAP[(j - i) % 12]
    return W

# W[i, j] = consonance of interval from pitch class i to j
W: np.ndarray = _build_consonance_matrix()
# mean-centered: random chroma pairs score ~0; positive = above-chance consonance
W_C: np.ndarray = W - W.mean()


def _pair_score(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Centered consonance score for one stem pair. Typical Slakh2100 value: ~0.05."""
    valid = ~(np.any(np.isnan(a), axis=0) | np.any(np.isnan(b), axis=0))
    if not valid.any():
        return float("nan")

    av, bv = a[:, valid], b[:, valid]

    raw_c  = np.einsum("it,ij,jt->t", av, W_C, bv)
    self_a = np.einsum("it,ij,jt->t", av, W, av)
    self_b = np.einsum("it,ij,jt->t", bv, W, bv)
    scores = raw_c / np.sqrt(self_a * self_b + eps)
    return float(np.mean(scores))


def harmonic_compatibility(stems: dict[str, np.ndarray], eps: float = 1e-8) -> dict[str, float]:
    """Pairwise consonance-weighted HC scores. stems: {name: (12, n_bars)} L1-normalized chroma, nan = silent bar."""
    names = list(stems.keys())
    pair_scores: dict[str, float] = {}

    for a_name, b_name in itertools.combinations(names, 2):
        a = stems[a_name]
        b = stems[b_name]
        n = min(a.shape[1], b.shape[1])
        key = f"hc_{a_name}_{b_name}"
        pair_scores[key] = _pair_score(a[:, :n], b[:, :n], eps)

    valid_vals = [v for v in pair_scores.values() if not np.isnan(v)]
    hc = float(np.mean(valid_vals)) if valid_vals else float("nan")

    return {**pair_scores, "hc": hc, "n_pairs": len(pair_scores)}


class HC(Metric):
    """Harmonic Compatibility. Consonance-weighted pairwise cross-stem score at bar level. Higher is better.

    Uses a 12x12 consonance matrix rather than cosine similarity, so it scores intervals
    between simultaneously active pitch classes rather than chroma shape similarity.
    Beat tracking is shared across stems (via drums by default) for consistent bar grids.
    """

    name = "hc"

    def __init__(
        self,
        stems: list[str] | None = None,
        exclude: list[str] | None = None,
        beat_ref_stem: str | None = "drums",
        hop_length: int = 512,
        eps: float = 1e-8,
    ):
        self.stems = stems or ["guitar", "bass", "piano"]
        self.exclude = exclude or ["drums"]
        self.beat_ref_stem = beat_ref_stem
        self.hop_length = hop_length
        self.eps = eps

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

        track_dirs = sorted(
            d for d in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, d))
        )

        pair_accumulator: dict[str, list[float]] = {}

        for track in tqdm(track_dirs, desc="HC tracks"):
            track_path = os.path.join(folder, track)
            stem_file_dict = {
                stem: os.path.join(track_path, stem + ".wav")
                for stem in active
                if os.path.isfile(os.path.join(track_path, stem + ".wav"))
            }
            if len(stem_file_dict) < 2:
                continue

            beat_ref = None
            if beat_ref_stem is not None:
                ref_file = os.path.join(track_path, beat_ref_stem + ".wav")
                if os.path.isfile(ref_file):
                    beat_ref = ref_file
            track_stems = track_stems_bar_chroma(stem_file_dict, self.hop_length, beat_ref)

            result = harmonic_compatibility(track_stems, eps=self.eps)
            for key, val in result.items():
                if key.startswith("hc_") and not np.isnan(val):
                    pair_accumulator.setdefault(key, []).append(val)

        if not pair_accumulator:
            return {"hc": float("nan"), "n_pairs": 0}

        pair_means  = {k: float(np.mean(v))  for k, v in pair_accumulator.items()}
        pair_counts = {f"n_{k}": len(v)      for k, v in pair_accumulator.items()}
        hc = float(np.mean(list(pair_means.values())))
        return {**pair_means, **pair_counts, "hc": hc, "n_pairs": len(pair_means)}
