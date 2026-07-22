from __future__ import annotations

import os
import tempfile
import warnings

import numpy as np
from tqdm import tqdm
import torch
import soundfile as sf
from madmom.evaluation.beats import BeatEvaluation

from ..base import Metric
from ._common import _collect_track_dirs, load_stems

from beat_this.inference import File2Beats


_MADMOM_KEYS = ["fmeasure", "cemgil", "cmlc", "cmlt", "amlc", "amlt"]



def _safe_nanmean(xs: list[float]) -> float:
    return float(np.nanmean(xs)) if xs else float("nan")


class BeatAlignment(Metric):
    """Beat alignment of each stem against the sum of the other stems, scored with madmom BeatEvaluation F-measure.

    The accompaniment sum is written to a temp file because beat_this.File2Beats only accepts file paths.
    """
    name = "beat_alignment"

    def __init__(
        self,
        checkpoint_path: str = "final0",
        device: str | None = None,
        dbn: bool = False,
        seed: int = 42,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._file2beats = File2Beats(
            checkpoint_path=checkpoint_path,
            device=device,
            dbn=dbn,
        )
        self._seed = seed

    def compute(
        self,
        folder: str,
        stems: list[str] | None = None,
        max_amount: int | None = None,
        **_,
    ) -> dict:
        stems = stems or ["bass", "drums", "guitar", "piano"]
        track_names = _collect_track_dirs(folder, max_amount, self._seed)

        all_scores: dict[str, list[float]] = {k: [] for k in _MADMOM_KEYS}
        stem_fmeasure: dict[str, list[float]] = {s: [] for s in stems}
        sample_fmeasure: list[float] = []
        n_skipped = 0

        for track_name in tqdm(track_names, desc="Beat Alignment"):
            track_path = os.path.join(folder, track_name)
            wavs, source_sr = load_stems(track_path, stems, label="BA")
            if wavs is None:
                n_skipped += 1
                continue

            track_stem_fmeasures: list[float] = []
            present_stems = list(wavs.keys())

            for stem_name in present_stems:
                pred_path = os.path.join(track_path, stem_name + ".wav")

                context = np.clip(
                    sum(wavs[s] for s in present_stems if s != stem_name),
                    -1.0, 1.0,
                ).astype(np.float32)

                tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                os.close(tmp_fd)
                try:
                    sf.write(tmp_path, context, source_sr, subtype="PCM_16")
                    context_beats, _ = self._file2beats(tmp_path)
                    pred_beats, _    = self._file2beats(pred_path)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

                if len(context_beats) < 2 or len(pred_beats) < 2:
                    n_skipped += 1
                    continue

                try:
                    ev = BeatEvaluation(context_beats, pred_beats)
                    for key in _MADMOM_KEYS:
                        all_scores[key].append(float(getattr(ev, key)))
                    stem_fmeasure[stem_name].append(float(ev.fmeasure))
                    track_stem_fmeasures.append(float(ev.fmeasure))
                except Exception as exc:
                    warnings.warn(f"[BA] BeatEvaluation failed for {track_path}/{stem_name}: {exc}")
                    n_skipped += 1

            if track_stem_fmeasures:
                sample_fmeasure.append(float(np.mean(track_stem_fmeasures)))

        return {
            "ba_fmeasure":         _safe_nanmean(all_scores["fmeasure"]),
            "ba_cemgil":           _safe_nanmean(all_scores["cemgil"]),
            "ba_cmlc":             _safe_nanmean(all_scores["cmlc"]),
            "ba_cmlt":             _safe_nanmean(all_scores["cmlt"]),
            "ba_amlc":             _safe_nanmean(all_scores["amlc"]),
            "ba_amlt":             _safe_nanmean(all_scores["amlt"]),
            "ba_stem_fmeasure":    {s: _safe_nanmean(v) for s, v in stem_fmeasure.items()},
            "ba_sample_fmeasure":  sample_fmeasure,
            "ba_n_skipped":        n_skipped,
        }
