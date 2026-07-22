from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from mgs_evals.harmonic.hc import harmonic_compatibility
from mgs_evals.harmonic.tc import tonal_coherence, tc_short_clip
from mgs_evals.harmonic.profiles import load_profiles, build_key_templates
from mgs_evals.harmonic._preprocess import stem_bar_chroma


REPO_ROOT = Path(__file__).parents[2]
SLAKH_TEST_DIR = REPO_ROOT / "data" / "slakh2100" / "test"
MSGLD_GEN_DIR = REPO_ROOT / "inference" / "MSG-LD" / "gen"
PITCHED_STEMS = ["guitar", "bass", "piano"]
PROFILE_SET = "ks"
SEGMENT_SIZE = 4
HOP_LENGTH = 512
MAX_TRACKS = 1  # cap for speed (set to None to use all)


def _load_track_stems(
    track_dir: Path,
    stems: list[str] = PITCHED_STEMS,
) -> dict[str, np.ndarray] | None:
    """Load bar-level chroma for each pitched stem in a track directory"""
    result = {}
    for stem in stems:
        fpath = track_dir / f"{stem}.wav"
        if fpath.exists():
            result[stem] = stem_bar_chroma(str(fpath), hop_length=HOP_LENGTH)
    return result if len(result) >= 2 else None


def _eval_folder(
    folder: Path,
    templates: np.ndarray,
    max_tracks: int | None = MAX_TRACKS,
) -> dict[str, list[float]]:
    """
    Compute HC and TC per track in a folder 
    
    Returns {metric: [per-track scores]}
    """
    track_dirs = sorted(d for d in folder.iterdir() if d.is_dir())
    if max_tracks is not None:
        track_dirs = track_dirs[:max_tracks]

    hc_vals: list[float] = []
    tc_vals: list[float] = []

    for td in track_dirs:
        stems = _load_track_stems(td)
        if stems is None:
            continue

        hc_r = harmonic_compatibility(stems)
        if not np.isnan(hc_r["hc"]):
            hc_vals.append(hc_r["hc"])

        n_bars = min(c.shape[1] for c in stems.values())
        if n_bars < SEGMENT_SIZE:
            tc_score = tc_short_clip(stems, templates)
        else:
            tc_score = tonal_coherence(stems, templates, SEGMENT_SIZE)["tc"]

        if not np.isnan(tc_score):
            tc_vals.append(tc_score)

    return {"hc": hc_vals, "tc": tc_vals}


def _random_baseline(
    templates: np.ndarray,
    n_tracks: int = 20,
    n_bars: int = 8,
    seed: int = 0,
) -> dict[str, list[float]]:
    """Compute HC and TC on uniform random chroma arrays"""
    rng = np.random.default_rng(seed)

    hc_vals, tc_vals = [], []
    for _ in range(n_tracks):
        stems = {
            name: rng.dirichlet(np.ones(12), size=n_bars).T
            for name in PITCHED_STEMS
        }
        hc_r = harmonic_compatibility(stems)
        if not np.isnan(hc_r["hc"]):
            hc_vals.append(hc_r["hc"])

        tc_r = tonal_coherence(stems, templates, SEGMENT_SIZE)
        if not np.isnan(tc_r["tc"]):
            tc_vals.append(tc_r["tc"])

    return {"hc": hc_vals, "tc": tc_vals}


def _mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else float("nan")


def _print_table(rows: list[tuple[str, dict[str, list[float]]]]):
    print("Eval Start")
    print(f"{'Source':<25}  {'HC':>8}  {'TC':>8}  {'n_tracks':>9}")
    for label, data in rows:
        hc = _mean(data["hc"])
        tc = _mean(data["tc"])
        n  = len(data["hc"])
        print(f"{label:<25}  {hc:>8.4f}  {tc:>8.4f}  {n:>9}")
    print("Eval End")


def test_comparison_table():
    """
    Compute and print HC/TC for all available sources
    Always asserts: random < real Slakh (when data present)
    """
    major, minor = load_profiles(PROFILE_SET)
    templates = build_key_templates(major, minor)

    rows: list[tuple[str, dict[str, list[float]]]] = []


    slakh_available = SLAKH_TEST_DIR.exists()
    slakh_data = None
    if slakh_available:
        slakh_data = _eval_folder(SLAKH_TEST_DIR, templates)
        rows.append(("Slakh2100 (real)", slakh_data))
    else:
        pytest.skip(f"Slakh test data not found at {SLAKH_TEST_DIR}")


    if MSGLD_GEN_DIR.exists():
        msgld_data = _eval_folder(MSGLD_GEN_DIR, templates)
        rows.append(("MSG-LD generated", msgld_data))


    random_data = _random_baseline(templates, n_tracks=MAX_TRACKS or 20)
    rows.append(("Random baseline", random_data))

    _print_table(rows)


    if slakh_data is not None and slakh_data["hc"]:
        assert _mean(slakh_data["hc"]) > _mean(random_data["hc"]), (
            f"Real Slakh HC ({_mean(slakh_data['hc']):.4f}) should exceed "
            f"random HC ({_mean(random_data['hc']):.4f})"
        )
    if slakh_data is not None and slakh_data["tc"]:
        assert _mean(slakh_data["tc"]) > _mean(random_data["tc"]), (
            f"Real Slakh TC ({_mean(slakh_data['tc']):.4f}) should exceed "
            f"random TC ({_mean(random_data['tc']):.4f})"
        )
