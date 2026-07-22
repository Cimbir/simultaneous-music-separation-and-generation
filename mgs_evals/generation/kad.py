from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial.distance import cdist

from ..base import Metric
from ._embed import AudioEmbedder, get_embedder


class KAD(Metric):
    """Kernel Audio Distance (MMD with RBF kernel). Preferred over FAD when embeddings are non-Gaussian.

    Lower is better; 0 = identical distributions. Backends: "vggish" (128-d) or "clap" (512-d).
    Sigma defaults to the median pairwise distance (median heuristic) when not set.
    Accepts raw waveforms, file paths (batched), or pre-computed embeddings.
    """

    def __init__(
        self,
        backend: str = "vggish",
        sigma: float | None = None,
        device: str = "cpu",
        embedder: AudioEmbedder | None = None,
        max_embeddings: int = 5000,
        **embedder_kwargs,
    ):
        self.backend = backend
        self.sigma = sigma
        self.device = device
        self.max_embeddings = max_embeddings
        self._embedder = embedder or get_embedder(backend, device=device, **embedder_kwargs)

    @property
    def name(self) -> str:
        return f"kad_{self.backend}"

    def _get_embeddings(
        self,
        audios: Sequence[np.ndarray] | None,
        files: Sequence[str] | None,
        embeddings: np.ndarray | None,
        sr: int,
        batch_size: int,
        normalize_rms: bool,
        label: str,
    ) -> np.ndarray:
        if embeddings is not None:
            return np.asarray(embeddings, dtype=np.float64)
        if files is not None:
            return self._embedder.embed_files(files, sr=sr, batch_size=batch_size,
                                              normalize_rms=normalize_rms).astype(np.float64)
        if audios is None:
            raise ValueError(
                f"Provide {label} as waveforms, file paths ({label}_files), or pre-computed embeddings"
            )
        return self._embedder.embed(audios, sr).astype(np.float64)

    @staticmethod
    def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
        XX = np.sum(X ** 2, axis=1, keepdims=True)
        YY = np.sum(Y ** 2, axis=1, keepdims=True)
        sq_dists = XX + YY.T - 2.0 * (X @ Y.T)
        return np.exp(-sq_dists / (2.0 * sigma ** 2))

    @staticmethod
    def _median_sigma(X: np.ndarray, Y: np.ndarray, subsample: int = 200) -> float:
        rng = np.random.default_rng(0)
        idx_x = rng.choice(len(X), min(subsample, len(X)), replace=False)
        idx_y = rng.choice(len(Y), min(subsample, len(Y)), replace=False)
        dists = cdist(X[idx_x], Y[idx_y])
        median = float(np.median(dists))
        return median if median > 0 else 1.0

    def compute(
        self,
        generated: Sequence[np.ndarray] | None = None,
        reference: Sequence[np.ndarray] | None = None,
        generated_files: Sequence[str] | None = None,
        reference_files: Sequence[str] | None = None,
        gen_embeddings: np.ndarray | None = None,
        ref_embeddings: np.ndarray | None = None,
        sr: int = 16_000,
        batch_size: int = 32,
        normalize_rms: bool = True,
        **_,
    ) -> dict[str, float]:
        gen_emb = self._get_embeddings(generated, generated_files, gen_embeddings, sr, batch_size, normalize_rms, "generated")
        ref_emb = self._get_embeddings(reference, reference_files, ref_embeddings, sr, batch_size, normalize_rms, "reference")

        rng = np.random.default_rng(42)
        if self.max_embeddings and len(gen_emb) > self.max_embeddings:
            gen_emb = gen_emb[rng.choice(len(gen_emb), self.max_embeddings, replace=False)]
        if self.max_embeddings and len(ref_emb) > self.max_embeddings:
            ref_emb = ref_emb[rng.choice(len(ref_emb), self.max_embeddings, replace=False)]

        sigma = self.sigma or self._median_sigma(gen_emb, ref_emb)

        K_xx = self._rbf_kernel(gen_emb, gen_emb, sigma)
        K_yy = self._rbf_kernel(ref_emb, ref_emb, sigma)
        K_xy = self._rbf_kernel(gen_emb, ref_emb, sigma)

        mmd2 = float(K_xx.mean() - 2.0 * K_xy.mean() + K_yy.mean())
        return {self.name: float(np.sqrt(max(mmd2, 0.0)))}
