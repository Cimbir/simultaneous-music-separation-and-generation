from __future__ import annotations

from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
import torchaudio


def mel_mse(
    mel_pred: torch.Tensor,
    mel_target: torch.Tensor,
    *,
    per_stem: bool = False,
) -> torch.Tensor:
    if per_stem and mel_pred.dim() >= 3:
        B, S = mel_pred.shape[0], mel_pred.shape[1]
        pred_flat = mel_pred.reshape(B, S, -1)
        tgt_flat = mel_target.reshape(B, S, -1)
        return F.mse_loss(pred_flat, tgt_flat, reduction="none").mean(dim=(0, 2))
    return F.mse_loss(mel_pred, mel_target)


def _frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


class FADEvaluator:
    def __init__(
        self,
        embedding_fn: Optional[Callable] = None,
        sample_rate: int = 16000,
        device: str = "cpu",
    ) -> None:
        self.sr     = sample_rate
        self.device = device
        self._embed = embedding_fn if embedding_fn is not None else self._build_vggish()
        self._real_embs: List[np.ndarray] = []
        self._fake_embs: List[np.ndarray] = []

    def update_real(self, audios: List[Union[np.ndarray, torch.Tensor]]) -> None:
        self._real_embs.extend(self._extract_many(audios))

    def update_fake(self, audios: List[Union[np.ndarray, torch.Tensor]]) -> None:
        self._fake_embs.extend(self._extract_many(audios))

    def compute(self) -> float:
        if len(self._real_embs) < 2 or len(self._fake_embs) < 2:
            raise ValueError(
                f"FAD needs >= 2 clips per set. "
                f"Got {len(self._real_embs)} real, {len(self._fake_embs)} fake."
            )
        mu1, sigma1 = _embedding_statistics(self._real_embs)
        mu2, sigma2 = _embedding_statistics(self._fake_embs)
        return _frechet_distance(mu1, sigma1, mu2, sigma2)

    def reset(self) -> None:
        self._real_embs.clear()
        self._fake_embs.clear()


    def _extract_many(
        self, audios: List[Union[np.ndarray, torch.Tensor]]
    ) -> List[np.ndarray]:
        embeddings = []
        for audio in audios:
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
            audio = np.asarray(audio, dtype=np.float32).squeeze()
            emb = self._embed(audio, self.sr)
            embeddings.append(np.asarray(emb, dtype=np.float32).flatten())
        return embeddings

    def _build_vggish(self) -> Callable:
        try:
            bundle = torchaudio.pipelines.VGGISH
            model  = bundle.get_model().to(self.device).eval()

            @torch.no_grad()
            def embed(audio: np.ndarray, sr: int) -> np.ndarray:
                waveform = torch.from_numpy(audio).unsqueeze(0) # (1, T)
                if sr != bundle.sample_rate:
                    waveform = torchaudio.functional.resample(
                        waveform, sr, bundle.sample_rate
                    )
                waveform = waveform.to(self.device)
                emb = model(waveform) # (N_windows, 128)
                return emb.mean(dim=0).cpu().numpy() # (128,)

            return embed

        except Exception as exc:
            raise ImportError(
                "Default VGGish backend requires torchaudio >= 0.13 with "
                "torchaudio.pipelines.VGGISH. Install it or pass a custom "
                "embedding_fn to FADEvaluator."
            ) from exc


def _embedding_statistics(
    embeddings: List[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    mat = np.stack(embeddings, axis=0) # (N, D)
    return mat.mean(axis=0), np.cov(mat, rowvar=False)
