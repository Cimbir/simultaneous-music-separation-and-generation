from __future__ import annotations

import numpy as np
import pytest

from mgs_evals.harmonic.hc import harmonic_compatibility, W
from .conftest import one_hot, chord, tile_bars


def single_bar_stems(**kwargs) -> dict[str, np.ndarray]:
    """Wrap (12,) vectors into (12, 1) for single-bar tests"""
    return {k: v.reshape(12, 1) for k, v in kwargs.items()}


def test_unison_all_stems():
    """All stems playing the same pitch -> every pair HC = 1.0 - W.mean() (centered scale)"""
    C = one_hot(0)
    stems = single_bar_stems(a=C.copy(), b=C.copy(), c=C.copy())
    result = harmonic_compatibility(stems)
    expected = 1.0 - W.mean()
    for key, val in result.items():
        if key.startswith("hc_"):
            assert np.isclose(val, expected, atol=1e-6), f"{key} = {val}, expected {expected:.4f}"


def test_unison_any_pitch():
    """Unison is 1.0 - W.mean() regardless of which pitch class both stems share"""
    expected = 1.0 - W.mean()
    for pc in [0, 4, 7, 11]:
        v = one_hot(pc)
        stems = single_bar_stems(a=v.copy(), b=v.copy())
        r = harmonic_compatibility(stems)
        assert np.isclose(r["hc_a_b"], expected, atol=1e-6)


def test_perfect_fifth():
    """C vs G (interval 7 = perfect 5th) -> HC = 0.95 - W.mean() ~ 0.39"""
    stems = single_bar_stems(a=one_hot(0), b=one_hot(7))
    r = harmonic_compatibility(stems)
    assert np.isclose(r["hc_a_b"], 0.95 - W.mean(), atol=1e-6)


def test_tritone():
    """C vs F# (interval 6 = tritone) -> HC = 0.05 - W.mean() ~ -0.51 (below-average consonance)"""
    stems = single_bar_stems(a=one_hot(0), b=one_hot(6))
    r = harmonic_compatibility(stems)
    assert np.isclose(r["hc_a_b"], 0.05 - W.mean(), atol=1e-6)


def test_c_major_triad():
    """
    guitar=E(4), bass=C(0), piano=G(7) - C major triad
    All pairs should have above-average consonance (centered score > 0.0)

    Pair keys follow insertion order of the stems dict:
        hc_guitar_bass  = 0.65 - W.mean()  (min 6th, interval 8)
        hc_guitar_piano = 0.75 - W.mean()  (min 3rd, interval 3)
        hc_bass_piano   = 0.95 - W.mean()  (perf 5th, interval 7)
    """
    stems = single_bar_stems(guitar=one_hot(4), bass=one_hot(0), piano=one_hot(7))
    r = harmonic_compatibility(stems)
    assert r["hc"] > 0.0
    for key in ("hc_guitar_bass", "hc_guitar_piano", "hc_bass_piano"):
        assert r[key] > 0.0, f"{key} = {r[key]}"


def test_random_scores_in_range(random_stems_factory):
    """Random chroma -> all pair scores in [-1.0, 1.0] (centered scale)"""
    stems = random_stems_factory(n_bars=16)
    r = harmonic_compatibility(stems)
    for key, val in r.items():
        if key.startswith("hc_"):
            assert -1.0 <= val <= 1.0 + 1e-7, f"{key} = {val} out of range"


def test_silent_bar_skipped():
    """Bars where a stem is all-zero (or nan) are excluded from the mean"""
    a = np.stack([one_hot(0), np.full(12, np.nan)], axis=1)  # C, nan
    b = np.stack([one_hot(7), np.full(12, np.nan)], axis=1)  # G, nan
    stems = {"x": a, "y": b}
    r = harmonic_compatibility(stems)
    expected = 0.95 - W.mean()
    assert np.isclose(r["hc_x_y"], expected, atol=1e-6), f"Expected {expected:.4f}, got {r['hc_x_y']}"


def test_all_silent_returns_nan():
    """If all bars are silent for a pair, the pair score is nan"""
    nan_col = np.full((12, 4), np.nan)
    r = harmonic_compatibility({"a": nan_col, "b": nan_col.copy()})
    assert np.isnan(r["hc_a_b"])
    assert np.isnan(r["hc"])


def test_multi_bar_mean():
    """
    Two bars: bar 0 = unison (score 1.0 - W.mean()), bar 1 = tritone (score 0.05 - W.mean())
    Expected mean = 0.525 - W.mean()
    """
    a = np.stack([one_hot(0), one_hot(0)], axis=1)  # C, C
    b = np.stack([one_hot(0), one_hot(6)], axis=1)  # C, F#
    r = harmonic_compatibility({"a": a, "b": b})
    assert np.isclose(r["hc_a_b"], 0.525 - W.mean(), atol=1e-6)


def test_pair_count():
    """n_pairs is correct for 2-stem and 3-stem dicts"""
    v = one_hot(0).reshape(12, 1)
    r2 = harmonic_compatibility({"a": v, "b": v.copy()})
    r3 = harmonic_compatibility({"a": v, "b": v.copy(), "c": v.copy()})
    assert r2["n_pairs"] == 1
    assert r3["n_pairs"] == 3
