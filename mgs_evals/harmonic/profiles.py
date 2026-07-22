from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

# Krumhansl & Schmuckler (1990) profiles - derived from listener experiments
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Temperley (2001) profiles - derived from popular music scores
TEMPERLEY_MAJOR = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
TEMPERLEY_MINOR = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])

ProfileSet = Literal["ks", "temperley", "corpus"]

_CORPUS_PATH = Path(__file__).parent / "corpus_profiles.npz"


def load_profiles(profile_set: ProfileSet = "ks", corpus_path: Path = _CORPUS_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Returns (major, minor) shape (12,) in pitch-class order from C. "corpus" requires running corpus_em.py first."""
    if profile_set == "ks":
        return KS_MAJOR.copy(), KS_MINOR.copy()
    if profile_set == "temperley":
        return TEMPERLEY_MAJOR.copy(), TEMPERLEY_MINOR.copy()
    if profile_set == "corpus":
        if not corpus_path.exists():
            raise FileNotFoundError(
                f"Corpus profiles not found at {corpus_path}"
            )
        data = np.load(corpus_path)
        return data["major"].copy(), data["minor"].copy()
    raise ValueError(
        f"Unknown profile_set {profile_set!r}. Valid options: 'ks', 'temperley', 'corpus'."
    )


def build_key_templates(major: np.ndarray, minor: np.ndarray) -> np.ndarray:
    """Returns (24, 12) ordered [C_maj, C_min, C#_maj, C#_min, ..., B_maj, B_min]."""
    templates = []
    for root in range(12):
        templates.append(np.roll(major, root))
        templates.append(np.roll(minor, root))
    return np.stack(templates)  # (24, 12)
