from __future__ import annotations

from typing import Sequence

import numpy as np

from ..base import Metric
from ._embed import CLAPEmbedder


class CLAPScore(Metric):
    """Audio-text cosine similarity in CLAP space. Higher is better"""

    name = "clap_score"

    def __init__(self, device: str = "cpu", ckpt_path: str | None = None):
        self._embedder = CLAPEmbedder(device=device, ckpt_path=ckpt_path)

    def compute(
        self,
        audios: Sequence[np.ndarray],
        texts: Sequence[str],
        sr: int = 16_000,
        **_,
    ) -> dict[str, float]:
        if len(audios) != len(texts):
            raise ValueError(
                f"audios and texts must have the same length "
                f"({len(audios)} vs {len(texts)})"
            )

        audio_embs = self._embedder.embed(audios, sr) # (N, D)
        text_embs = self._embedder.embed_text(texts)  # (N, D)

        # Row-wise cosine similarity
        norms_a = np.linalg.norm(audio_embs, axis=1, keepdims=True) + 1e-8
        norms_t = np.linalg.norm(text_embs, axis=1, keepdims=True) + 1e-8
        cosines = np.sum((audio_embs / norms_a) * (text_embs / norms_t), axis=1)
        return {self.name: float(cosines.mean())}
