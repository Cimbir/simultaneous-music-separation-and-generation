from __future__ import annotations

import numpy as np
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_slakh: needs data/slakh2100/test/")
    config.addinivalue_line("markers", "requires_model_outputs: needs inference/ model outputs")


def one_hot(pc: int) -> np.ndarray:
    """(12,) vector with 1.0 at pitch class pc, zero elsewhere. Already L1-norm=1"""
    if not (0 <= pc < 12):
        raise ValueError(f"Pitch class must be in [0, 11], got {pc}")
    v = np.zeros(12)
    v[pc] = 1.0
    return v


def chord(pcs: list[int]) -> np.ndarray:
    """(12,) equal-energy vector over pitch classes pcs, L1-normalized"""
    v = np.zeros(12)
    for pc in pcs:
        if not (0 <= pc < 12):
            raise ValueError(f"Pitch class must be in [0, 11], got {pc}")
        v[pc] = 1.0
    return v / (v.sum() + 1e-8)


def tile_bars(bar_vectors: list[np.ndarray], n_repeats: int = 1) -> np.ndarray:
    """Stack a list of (12,) vectors n_repeats times into a (12, n_bars) array"""
    cols = bar_vectors * n_repeats
    return np.stack(cols, axis=1)


@pytest.fixture
def c_major_stems():
    """
    8-bar (2× repeat) C-F-G-C progression

    guitar: E  A  B  E  (upper chord tone per bar)
    bass:   C  F  G  C  (root)
    piano:  C+E+G  F+A+C  G+B+D  C+E+G  (full chord)
    """
    guitar_pat = [one_hot(4), one_hot(9), one_hot(11), one_hot(4)]
    bass_pat   = [one_hot(0), one_hot(5), one_hot(7),  one_hot(0)]
    piano_pat  = [chord([0, 4, 7]), chord([5, 9, 0]), chord([7, 11, 2]), chord([0, 4, 7])]
    return {
        "guitar": tile_bars(guitar_pat, n_repeats=2),
        "bass":   tile_bars(bass_pat,   n_repeats=2),
        "piano":  tile_bars(piano_pat,  n_repeats=2),
    }


@pytest.fixture
def tritone_bass_stems(c_major_stems):
    """
    Same as c_major_stems but bass plays tritone substitutions

    bass: F#(6)  B(11)  C#(1)  F#(6)  (tritone of each chord root)
    """
    bass_tritone = [one_hot(6), one_hot(11), one_hot(1), one_hot(6)]
    stems = dict(c_major_stems)
    stems["bass"] = tile_bars(bass_tritone, n_repeats=2)
    return stems


@pytest.fixture
def random_stems_factory():
    def make(
        names: list[str] | None = None,
        n_bars: int = 8,
        seed: int = 42,
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        names = names or ["guitar", "bass", "piano"]
        return {
            name: rng.dirichlet(np.ones(12), size=n_bars).T  # (12, n_bars)
            for name in names
        }
    return make


@pytest.fixture
def sinusoid_audio_factory():
    def make(
        freqs_hz: list[float],
        duration_s: float = 2.0,
        sr: int = 16_000,
    ) -> tuple[np.ndarray, int]:
        t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
        audio = sum(np.sin(2 * np.pi * f * t) for f in freqs_hz)
        audio = audio.astype(np.float32)
        peak = np.abs(audio).max()
        if peak > 0:
            audio /= peak
        return audio, sr
    return make


@pytest.fixture
def corpus_profiles_path(tmp_path, monkeypatch):
    """
    Write placeholder corpus profiles (Temperley values) to a temp .npz file and monkeypatch profiles._CORPUS_PATH to point to it
    Returns the Path to the .npz file
    """
    from mgs_evals.harmonic import TEMPERLEY_MAJOR, TEMPERLEY_MINOR
    import mgs_evals.harmonic.profiles as _profiles_module

    npz_path = tmp_path / "corpus_profiles.npz"
    np.savez(npz_path, major=TEMPERLEY_MAJOR, minor=TEMPERLEY_MINOR)
    monkeypatch.setattr(_profiles_module, "_CORPUS_PATH", npz_path)
    return npz_path
