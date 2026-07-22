from __future__ import annotations

import numpy as np
import pytest

from mgs_evals.harmonic.tc import tonal_coherence, tc_short_clip
from mgs_evals.harmonic.profiles import (
    KS_MAJOR, KS_MINOR,
    TEMPERLEY_MAJOR, TEMPERLEY_MINOR,
    load_profiles,
    build_key_templates,
)
from .conftest import one_hot, chord, tile_bars

import mgs_evals.harmonic.profiles as _profiles_module

# Pre-built templates reused across tests
_KS_TEMPLATES = build_key_templates(*load_profiles("ks"))
_TEMPERLEY_TEMPLATES = build_key_templates(*load_profiles("temperley"))



def uniform_stems(names=("a", "b", "c"), n_bars=4) -> dict[str, np.ndarray]:
    """All bins equal energy (1/12) for every bar"""
    col = np.full(12, 1.0 / 12)
    return {name: np.tile(col[:, None], (1, n_bars)) for name in names}


def ks_major_stems(n_bars: int = 4, n_stems: int = 3) -> dict[str, np.ndarray]:
    """All stems carry exactly the KS major profile in every bar"""
    col = KS_MAJOR / KS_MAJOR.sum()
    names = [chr(ord("a") + i) for i in range(n_stems)]
    return {name: np.tile(col[:, None], (1, n_bars)) for name in names}


def diatonic_c_major_stems(n_bars: int = 4) -> dict[str, np.ndarray]:
    """
    Energy only on the 7 notes of the C major scale (C D E F G A B)
    Pitch class indices: 0, 2, 4, 5, 7, 9, 11
    """
    v = chord([0, 2, 4, 5, 7, 9, 11])
    return {"a": np.tile(v[:, None], (1, n_bars)), "b": np.tile(v[:, None], (1, n_bars))}


def chromatic_stems(n_bars: int = 4) -> dict[str, np.ndarray]:
    """All 12 bins equal energy"""
    return uniform_stems(n_bars=n_bars)  # same as uniform


def test_perfect_ks_profile_scores_near_1():
    """Chroma matching KS major perfectly -> Pearson correlation = 1.0"""
    stems = ks_major_stems(n_bars=4)
    r = tonal_coherence(stems, _KS_TEMPLATES, segment_size=4)
    assert np.isclose(r["tc"], 1.0, atol=1e-6), f"Expected 1.0, got {r['tc']}"


def test_uniform_chroma_scores_near_0():
    """Uniform chroma has zero variance -> Pearson = 0 -> TC ~ 0.0"""
    stems = uniform_stems(n_bars=8)
    r = tonal_coherence(stems, _KS_TEMPLATES, segment_size=4)
    assert np.isclose(r["tc"], 0.0, atol=1e-6), f"Expected ~0.0, got {r['tc']}"


def test_diatonic_c_major_high_tc():
    """7 diatonic notes of C major -> TC clearly above 0.6"""
    stems = diatonic_c_major_stems(n_bars=8)
    r = tonal_coherence(stems, _KS_TEMPLATES, segment_size=4)
    assert r["tc"] > 0.6, f"Expected TC > 0.6, got {r['tc']}"


def test_chromatic_low_tc():
    """All 12 bins equal -> TC ~ 0.0 (same as uniform)"""
    stems = chromatic_stems(n_bars=8)
    r = tonal_coherence(stems, _KS_TEMPLATES, segment_size=4)
    assert r["tc"] < 0.1, f"Expected TC < 0.1, got {r['tc']}"


def test_all_profile_sets_same_direction(corpus_profiles_path):
    """KS, Temperley, and corpus all rank tonal input higher than uniform"""
    tonal  = ks_major_stems(n_bars=8)
    random = uniform_stems(n_bars=8)

    for ps in ("ks", "temperley", "corpus"):
        kwargs = {"corpus_path": corpus_profiles_path} if ps == "corpus" else {}
        t = build_key_templates(*load_profiles(ps, **kwargs))
        tc_tonal  = tonal_coherence(tonal,  t, segment_size=4)["tc"]
        tc_random = tonal_coherence(random, t, segment_size=4)["tc"]
        assert tc_tonal > tc_random, (
            f"profile_set={ps!r}: tonal ({tc_tonal:.3f}) not > random ({tc_random:.3f})"
        )


def test_segment_size_affects_n_segments():
    """Halving segment_size roughly doubles n_segments"""
    stems = ks_major_stems(n_bars=8)
    r2 = tonal_coherence(stems, _KS_TEMPLATES, segment_size=2)
    r4 = tonal_coherence(stems, _KS_TEMPLATES, segment_size=4)
    assert r2["n_segments"] >= r4["n_segments"] * 2 - 1


def test_partial_final_segment_included():
    """5 bars with segment_size=4 -> 2 segments (4-bar + 1-bar partial)"""
    stems = ks_major_stems(n_bars=5)
    r = tonal_coherence(stems, _KS_TEMPLATES, segment_size=4)
    assert r["n_segments"] == 2


def test_short_track_one_bar():
    """1 bar (less than default segment_size=4) still returns a score"""
    stems = ks_major_stems(n_bars=1)
    r = tonal_coherence(stems, _KS_TEMPLATES, segment_size=4)
    assert not np.isnan(r["tc"]), "Expected a valid TC score for 1-bar track"


def test_corpus_missing_raises_file_not_found(tmp_path, monkeypatch):
    """Requesting corpus profiles with no .npz file -> FileNotFoundError"""
    monkeypatch.setattr(_profiles_module, "_CORPUS_PATH", tmp_path / "nonexistent.npz")

    with pytest.raises(FileNotFoundError, match="corpus_profiles.npz"):
        load_profiles("corpus")


def test_tc_short_clip_perfect_match():
    """tc_short_clip on stems exactly matching KS major -> near 1.0"""
    major, minor = load_profiles("ks")
    templates = build_key_templates(major, minor)

    col = KS_MAJOR / KS_MAJOR.sum()
    stems = {"a": np.tile(col[:, None], (1, 2))}
    score = tc_short_clip(stems, templates)
    assert np.isclose(score, 1.0, atol=1e-6), f"Expected 1.0, got {score}"


def test_tc_short_clip_tonal_beats_uniform():
    """tc_short_clip ranks tonal stems above uniform for any key template set"""
    major, minor = load_profiles("ks")
    templates = build_key_templates(major, minor)

    tonal  = {"a": np.tile((KS_MAJOR / KS_MAJOR.sum())[:, None], (1, 3))}
    random = {"a": np.tile(np.full(12, 1.0 / 12)[:, None], (1, 3))}

    assert tc_short_clip(tonal, templates) > tc_short_clip(random, templates)


def test_tc_short_clip_silent_returns_nan():
    """tc_short_clip on all-zero chroma -> nan"""
    major, minor = load_profiles("ks")
    templates = build_key_templates(major, minor)
    stems = {"a": np.zeros((12, 2))}
    assert np.isnan(tc_short_clip(stems, templates))
