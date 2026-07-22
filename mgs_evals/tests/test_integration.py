from __future__ import annotations

import tempfile
import os

import numpy as np
import pytest
import soundfile as sf

from mgs_evals.harmonic.hc import harmonic_compatibility
from mgs_evals.harmonic.tc import tonal_coherence, tc_short_clip
from mgs_evals.harmonic.profiles import load_profiles, build_key_templates
from mgs_evals.harmonic._preprocess import stem_bar_chroma
from .conftest import one_hot, chord, tile_bars


def _good_stems(n_repeats: int = 2) -> dict[str, np.ndarray]:
    """C-F-G-C progression, consonant chord tones throughout"""
    return {
        "guitar": tile_bars([one_hot(4), one_hot(9), one_hot(11), one_hot(4)], n_repeats),
        "bass":   tile_bars([one_hot(0), one_hot(5), one_hot(7),  one_hot(0)], n_repeats),
        "piano":  tile_bars(
            [chord([0, 4, 7]), chord([5, 9, 0]), chord([7, 11, 2]), chord([0, 4, 7])],
            n_repeats,
        ),
    }


def _tritone_bass_stems(n_repeats: int = 2) -> dict[str, np.ndarray]:
    """Same as good_stems but bass plays tritone substitutions"""
    stems = _good_stems(n_repeats)
    stems["bass"] = tile_bars([one_hot(6), one_hot(11), one_hot(1), one_hot(6)], n_repeats)
    return stems


def _random_stems(n_bars: int = 8, seed: int = 0) -> dict[str, np.ndarray]:
    """Uniform random L1-normalized chroma - no tonal center"""
    rng = np.random.default_rng(seed)
    return {
        name: rng.dirichlet(np.ones(12), size=n_bars).T
        for name in ("guitar", "bass", "piano")
    }


def test_good_stems_hc_above_threshold():
    """Consonant C major progression -> HC > 0.1 (centered scale; typical Slakh ~0.05)"""
    r = harmonic_compatibility(_good_stems())
    assert r["hc"] > 0.1, f"Expected HC > 0.1, got {r['hc']:.4f}"


def test_good_stems_tc_above_threshold():
    """C major progression -> TC > 0.60 with KS profiles"""
    templates = build_key_templates(*load_profiles("ks"))
    r = tonal_coherence(_good_stems(), templates, segment_size=4)
    assert r["tc"] > 0.60, f"Expected TC > 0.60, got {r['tc']:.4f}"


def test_tritone_bass_hc_lower_than_good():
    """Tritone substitutions in the bass -> lower guitar-bass HC than good stems"""
    good_r    = harmonic_compatibility(_good_stems())
    tritone_r = harmonic_compatibility(_tritone_bass_stems())
    key = "hc_guitar_bass"
    assert tritone_r[key] < good_r[key], (
        f"Expected tritone {key} ({tritone_r[key]:.3f}) < good ({good_r[key]:.3f})"
    )


def test_random_stems_tc_lower_than_good():
    """Random pitch distribution -> lower TC than C major progression"""
    templates = build_key_templates(*load_profiles("ks"))
    good_tc   = tonal_coherence(_good_stems(),   templates, segment_size=4)["tc"]
    random_tc = tonal_coherence(_random_stems(), templates, segment_size=4)["tc"]
    assert random_tc < good_tc, (
        f"Expected random TC ({random_tc:.3f}) < good TC ({good_tc:.3f})"
    )


def test_short_clip_tc_returns_float():
    """tc_short_clip on a 2-bar chroma dict returns a finite float"""
    major, minor = load_profiles("ks")
    templates = build_key_templates(major, minor)
    good_2bar = {k: v[:, :2] for k, v in _good_stems().items()}
    score = tc_short_clip(good_2bar, templates)
    assert isinstance(score, float)
    assert np.isfinite(score), f"Expected finite score, got {score}"


def test_tritone_bass_shifts_interval_quality():
    """
    Tritone-subbed bass shifts guitar-bass from minor-6th (positive) to major-2nd (negative) -
    crosses below average consonance, a clear measurable drop
    """
    tritone_r = harmonic_compatibility(_tritone_bass_stems())
    good_r    = harmonic_compatibility(_good_stems())
    assert tritone_r["hc_guitar_bass"] < 0.0, (
        f"Expected < 0.0, got {tritone_r['hc_guitar_bass']:.3f}"
    )
    assert tritone_r["hc_guitar_bass"] < good_r["hc_guitar_bass"], (
        f"Tritone ({tritone_r['hc_guitar_bass']:.3f}) should be below good ({good_r['hc_guitar_bass']:.3f})"
    )


def _write_sinusoid_wav(freqs_hz: list[float], duration_s: float, sr: int, path: str):
    """Write a sum-of-sinusoids WAV file"""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    audio = sum(np.sin(2 * np.pi * f * t) for f in freqs_hz).astype(np.float32)
    peak = np.abs(audio).max()
    if peak > 0:
        audio /= peak
    sf.write(path, audio, sr, subtype="PCM_16")


def test_pipeline_smoke():
    """
    Full pipeline: write a 10-second sinusoid WAV, extract bar chroma, and compute HC
    Asserts no exceptions are raised and the output shapes are valid
    """
    sr = 16_000
    with tempfile.TemporaryDirectory() as tmpdir:
        freqs = {"bass": [65.41], "guitar": [164.81, 195.99], "piano": [261.63, 329.63, 392.0]}
        stems_dict: dict[str, np.ndarray] = {}

        for name, f_list in freqs.items():
            fpath = os.path.join(tmpdir, f"{name}.wav")
            _write_sinusoid_wav(f_list, duration_s=10.0, sr=sr, path=fpath)
            bar_chroma = stem_bar_chroma(fpath)
            assert bar_chroma.shape[0] == 12, "Chroma should have 12 pitch classes"
            assert bar_chroma.shape[1] >= 1, "Should have at least one bar"
            stems_dict[name] = bar_chroma

        result = harmonic_compatibility(stems_dict)
        assert "hc" in result
        assert result["n_pairs"] == 3
