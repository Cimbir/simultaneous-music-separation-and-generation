from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np
import librosa
import soundfile as sf

try:
    from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
    _MADMOM_AVAILABLE = True
except ImportError:
    print("madmom not found, falling back to librosa beat tracking for downbeat estimation. "
          "For better accuracy, install madmom: pip install madmom (python 3.9)")
    _MADMOM_AVAILABLE = False


def chroma_frames(audio_file: str, hop_length: int = 512) -> tuple[Any, int | float]:
    """Returns (chroma, sr) where chroma is (12, T), unnormalized."""
    y, sr = librosa.load(audio_file, sr=None, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, norm=None)
    return chroma, sr


def track_beats(audio_file: str) -> np.ndarray:
    """Bar-start (downbeat) times in seconds. Falls back to librosa (every 4th beat) when madmom is unavailable."""
    if _MADMOM_AVAILABLE:
        activation = RNNDownBeatProcessor()(audio_file)
        dbn = DBNDownBeatTrackingProcessor(
            beats_per_bar=[3, 4], min_bpm=30, max_bpm=300,
            fps=150, transition_lambda=100,
        )
        beats = dbn(activation)  # (n_beats, 2): [time, position_in_bar]
        # position == 1 marks the downbeat (bar start)
        bar_starts = beats[beats[:, 1] == 1, 0]
        return bar_starts if len(bar_starts) > 0 else beats[:, 0][:1]
    else:
        y, sr = librosa.load(audio_file, sr=None, mono=True)
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        bar_frames = beat_frames[::4]  # approximate: every 4th beat = bar start
        return librosa.frames_to_time(bar_frames, sr=sr, hop_length=512)


def aggregate_to_bars(
    chroma: np.ndarray,
    beat_times: np.ndarray,
    hop_length: int,
    sr: int,
) -> np.ndarray:
    """Average chroma frames within each bar window. Returns (12, n_bars) L1-normalized; silent bars are all-nan."""
    if len(beat_times) == 0:
        return np.full((12, 1), np.nan)

    n_frames = chroma.shape[1]
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)

    bar_starts = beat_times
    bar_ends = np.append(beat_times[1:], frame_times[-1] + 1e-6)

    bars: list[np.ndarray] = []
    for start, end in zip(bar_starts, bar_ends):
        mask = (frame_times >= start) & (frame_times < end)
        if not mask.any():
            bars.append(np.full(12, np.nan))
            continue
        mean_chroma = chroma[:, mask].mean(axis=1)
        l1 = np.abs(mean_chroma).sum()
        if l1 < 1e-6:
            bars.append(np.full(12, np.nan))
        else:
            bars.append(mean_chroma / l1)

    return np.stack(bars, axis=1) if bars else np.full((12, 1), np.nan)


def _mixture_to_tmp_wav(stem_files: dict[str, str]) -> str:
    """
    Sum all stems to a mono mixture and write to a temp wav file
    Caller is responsible for deleting the returned path
    """
    mixture: np.ndarray | None = None
    ref_sr: int | None = None
    for fpath in stem_files.values():
        y, sr = librosa.load(fpath, sr=None, mono=True)
        if mixture is None:
            mixture = np.zeros(len(y), dtype=np.float32)
            ref_sr = sr
        n = min(len(mixture), len(y))
        mixture[:n] += y[:n]

    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(tmp_path, mixture, ref_sr)
    return tmp_path


def track_stems_bar_chroma(
    stem_files: dict[str, str],
    hop_length: int = 512,
    beat_ref_file: str | None = None,
) -> dict[str, np.ndarray]:
    """Bar-level chroma for multiple stems with shared beat tracking.

    Beat times come from beat_ref_file (e.g. drums) or the mixture when None.
    All returned arrays share the same n_bars so HC and TC see consistent grids.
    """
    if not stem_files:
        return {}

    tmp_path: str | None = None
    try:
        if beat_ref_file is not None:
            beat_times = track_beats(beat_ref_file)
        else:
            tmp_path = _mixture_to_tmp_wav(stem_files)
            beat_times = track_beats(tmp_path)
    finally:
        if tmp_path is not None:
            os.unlink(tmp_path)

    result: dict[str, np.ndarray] = {}
    for name, fpath in stem_files.items():
        chroma, sr = chroma_frames(fpath, hop_length=hop_length)
        if len(beat_times) == 0:
            mean_c = chroma.mean(axis=1)
            l1 = np.abs(mean_c).sum()
            result[name] = (mean_c / l1).reshape(12, 1) if l1 >= 1e-6 else np.full((12, 1), np.nan)
        else:
            result[name] = aggregate_to_bars(chroma, beat_times, hop_length=hop_length, sr=sr)

    return result


def stem_bar_chroma(
    audio_file: str,
    hop_length: int = 512,
) -> np.ndarray:
    """audio file -> (12, n_bars) L1-normalized chroma. Silent bars are all-nan; no downbeats = whole clip as one bar."""
    chroma, sr = chroma_frames(audio_file, hop_length=hop_length)
    beat_times = track_beats(audio_file)

    if len(beat_times) == 0:
        mean_c = chroma.mean(axis=1)
        l1 = np.abs(mean_c).sum()
        return (mean_c / l1).reshape(12, 1) if l1 >= 1e-6 else np.full((12, 1), np.nan)

    return aggregate_to_bars(chroma, beat_times, hop_length=hop_length, sr=sr)
