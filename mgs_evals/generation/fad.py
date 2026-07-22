from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.linalg import sqrtm

from ..base import Metric
from ._embed import AudioEmbedder, get_embedder



class FAD(Metric):
    """Frechet Audio Distance. Fits a Gaussian to embeddings from each set and returns the Frechet distance.

    Lower is better; 0 = identical distributions. Backends: "vggish" (128-d) or "clap" (512-d).
    Accepts raw waveforms, file paths (batched), or pre-computed embeddings.
    """

    def __init__(
        self,
        backend: str = "vggish",
        device: str = "cpu",
        embedder: AudioEmbedder | None = None,
        **embedder_kwargs,
    ):
        self.backend = backend
        self.device = device
        self._embedder = embedder or get_embedder(backend, device=device, **embedder_kwargs)

    @property
    def name(self) -> str:
        return f"fad_{self.backend}"

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
    def _frechet_distance(mu1: np.ndarray, s1: np.ndarray,
                          mu2: np.ndarray, s2: np.ndarray) -> float:
        diff = mu1 - mu2
        cov_mean, _ = sqrtm(s1 @ s2, disp=False)
        if np.iscomplexobj(cov_mean):
            if np.max(np.abs(cov_mean.imag)) > 1e-3:
                raise ValueError("sqrtm produced large imaginary component; check embeddings.")
            cov_mean = cov_mean.real
        fad = float(diff @ diff + np.trace(s1 + s2 - 2.0 * cov_mean))
        return fad

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

        if len(gen_emb) < 2 or len(ref_emb) < 2:
            raise ValueError("FAD requires at least 2 samples per set for covariance estimation.")

        mean_g = gen_emb.mean(0)
        mean_r = ref_emb.mean(0)
        sigma_g = np.cov(gen_emb, rowvar=False)
        sigma_r = np.cov(ref_emb, rowvar=False)

        return {self.name: self._frechet_distance(mean_r, sigma_r, mean_g, sigma_g)}
