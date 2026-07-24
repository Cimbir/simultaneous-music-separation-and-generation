from __future__ import annotations

import os

import numpy as np
from tqdm import tqdm
import torch
import librosa
from scipy.ndimage import median_filter

from ..base import Metric
from ._common import _collect_track_dirs, load_stems

# pip install git+https://github.com/maswang32/cocola-stem-gen.git@main
from contrastive_model.contrastive_model import CoCola
from contrastive_model import constants

CHUNK_SIZE = 80000   # 5 s at 16 kHz
HOP_SIZE   = 40000   # 50% overlap
TARGET_SR  = 16000   # COCOLA was trained at 16 kHz
_SILENCE_THRESHOLD = 1e-6

def _extract_hpss_features(chunks: np.ndarray) -> np.ndarray:
    """(B, S) -> (B, 2, n_mels, T) HPSS mel-dB features. Channel 0 = harmonic, 1 = percussive."""
    B = chunks.shape[0]

    mags = np.stack([
        np.abs(librosa.stft(chunks[i], n_fft=1024, win_length=400, hop_length=160))
        for i in range(B)
    ])  # (B, 513, T)

    harm_mag = median_filter(mags, size=(1, 1, 31), mode='reflect')
    perc_mag = median_filter(mags, size=(1, 31, 1), mode='reflect')
    total = harm_mag + perc_mag + 1e-8
    harm_S = (mags * harm_mag / total) ** 2  # (B, 513, T)
    perc_S = (mags * perc_mag / total) ** 2

    mel_fb = librosa.filters.mel(
        sr=TARGET_SR, n_fft=1024, n_mels=64, fmin=60.0, fmax=7800.0
    )  # (64, 513)
    mel_h = np.einsum('mf,bft->bmt', mel_fb, harm_S)  # (B, 64, T)
    mel_p = np.einsum('mf,bft->bmt', mel_fb, perc_S)

    def _to_db(S: np.ndarray) -> np.ndarray:
        log_S = 10.0 * np.log10(np.maximum(S, 1e-10))
        ref = log_S.reshape(B, -1).max(axis=1)[:, np.newaxis, np.newaxis]
        return log_S - ref

    return np.stack([_to_db(mel_h), _to_db(mel_p)], axis=1).astype(np.float32) # (B, 2, 64, T)


def _chunk_audio(audio: np.ndarray, chunk_size: int, hop: int) -> np.ndarray:
    n = len(audio)
    if n <= chunk_size:
        padded = np.zeros(chunk_size, dtype=np.float32)
        padded[:n] = audio
        return padded[np.newaxis, :]  # (1, chunk_size)

    num_frames = (n - chunk_size) // hop + 1
    out = np.empty((num_frames, chunk_size), dtype=np.float32)
    for i in range(num_frames):
        start = i * hop
        out[i] = audio[start:start + chunk_size]
    return out



def _resample_if_needed(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=source_sr, target_sr=target_sr)


_MODES = ("both", "harmonic", "percussive")


def _nanmean(xs: list[float]) -> float:
    return float(np.nanmean(xs)) if xs else float("nan")


class COCOLA(Metric):
    """COCOLA coherence: each stem scored against the sum of the other 3 using a pretrained HPSS contrastive model.

    Returns harmonic, percussive, and combined scores alongside a random-pairing baseline. Higher is better.
    Checkpoint: https://drive.google.com/file/d/1S-_OvnDwNFLNZD5BmI1Ouck_prutRVWZ/view
    """

    name = "cocola"

    def __init__(
        self,
        checkpoint_path: str,
        device: str | None = None,
        seed: int = 42,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        model = CoCola.load_from_checkpoint(checkpoint_path, map_location=self._device, weights_only=False)
        model.eval()
        model.to(self._device)

        self._model = model
        self._mode_map = {
            "both":       constants.EmbeddingMode.BOTH,
            "harmonic":   constants.EmbeddingMode.HARMONIC,
            "percussive": constants.EmbeddingMode.PERCUSSIVE,
        }
        self._seed = seed

    def _score_pair_all_modes(
        self, context: np.ndarray, pred: np.ndarray
    ) -> dict[str, float]:
        # features extracted once, reused across all three modes
        ctx_chunks  = _chunk_audio(context, CHUNK_SIZE, HOP_SIZE)
        pred_chunks = _chunk_audio(pred,    CHUNK_SIZE, HOP_SIZE)

        min_chunks = min(len(ctx_chunks), len(pred_chunks))
        if min_chunks == 0:
            return {m: float("nan") for m in _MODES}
        ctx_chunks  = ctx_chunks[:min_chunks]
        pred_chunks = pred_chunks[:min_chunks]

        feat_ctx  = _extract_hpss_features(ctx_chunks)
        feat_pred = _extract_hpss_features(pred_chunks)

        with torch.no_grad():
            t_ctx  = torch.from_numpy(feat_ctx).to(self._device)
            t_pred = torch.from_numpy(feat_pred).to(self._device)
            scores: dict[str, float] = {}
            for mode_name, mode_enum in self._mode_map.items():
                self._model.set_embedding_mode(mode_enum)
                scores[mode_name] = float(self._model.score(t_ctx, t_pred).mean().item())

        return scores

    def compute(
        self,
        folder: str,
        stems: list[str] | None = None,
        max_amount: int | None = None,
        **_,
    ) -> dict:
        stems = stems or ["bass", "drums", "guitar", "piano"]
        track_names = _collect_track_dirs(folder, max_amount, self._seed)

        stem_scores: dict[str, dict[str, list[float]]] = {
            s: {m: [] for m in _MODES} for s in stems
        }
        sample_scores: list[dict[str, float]] = []
        random_scores: list[dict[str, float]] = []

        for i, track_name in enumerate(tqdm(track_names, desc="COCOLA")):
            track_path = os.path.join(folder, track_name)
            wavs, source_sr = load_stems(track_path, stems, label="COCOLA", skip_silent=True)
            if wavs is None:
                continue

            present_stems = list(wavs.keys())
            sample_stem_vals: dict[str, list[float]] = {m: [] for m in _MODES}
            for stem_name in present_stems:
                ctx_stems = [s for s in present_stems if s != stem_name]
                if not ctx_stems:
                    continue
                pred    = _resample_if_needed(wavs[stem_name], source_sr, TARGET_SR)
                ctx_sum = sum(_resample_if_needed(wavs[s], source_sr, TARGET_SR) for s in ctx_stems)
                pair_scores = self._score_pair_all_modes(ctx_sum, pred)
                for m in _MODES:
                    stem_scores[stem_name][m].append(pair_scores[m])
                    sample_stem_vals[m].append(pair_scores[m])

            sample_scores.append({m: float(np.nanmean(sample_stem_vals[m])) for m in _MODES})

            # Random baseline: accompaniment from the next sample (circular)
            rand_name = track_names[(i + 1) % len(track_names)]
            rand_wavs, rand_sr = load_stems(os.path.join(folder, rand_name), stems, label="COCOLA", skip_silent=True)
            if rand_wavs is not None:
                rand_stem_vals: dict[str, list[float]] = {m: [] for m in _MODES}
                rand_stems = [s for s in present_stems if s in rand_wavs]
                for stem_name in rand_stems:
                    ctx_stems = [s for s in rand_stems if s != stem_name]
                    if not ctx_stems:
                        continue
                    pred     = _resample_if_needed(wavs[stem_name], source_sr, TARGET_SR)
                    ctx_rand = sum(_resample_if_needed(rand_wavs[s], rand_sr, TARGET_SR) for s in ctx_stems)
                    pair_scores = self._score_pair_all_modes(ctx_rand, pred)
                    for m in _MODES:
                        rand_stem_vals[m].append(pair_scores[m])
                random_scores.append({m: float(np.nanmean(rand_stem_vals[m])) for m in _MODES})

        return {
            "cocola_both":              _nanmean([s["both"]        for s in sample_scores]),
            "cocola_harmonic":          _nanmean([s["harmonic"]    for s in sample_scores]),
            "cocola_percussive":        _nanmean([s["percussive"]  for s in sample_scores]),
            "cocola_random_both":       _nanmean([s["both"]        for s in random_scores]),
            "cocola_random_harmonic":   _nanmean([s["harmonic"]    for s in random_scores]),
            "cocola_random_percussive": _nanmean([s["percussive"]  for s in random_scores]),
            "cocola_stem_scores": {
                s: {m: _nanmean(stem_scores[s][m]) for m in _MODES}
                for s in stems
            },
            "cocola_sample_scores": sample_scores,
        }
