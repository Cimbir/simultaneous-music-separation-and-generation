from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import soundfile as sf
import torch
import torchaudio


class AudioEmbedder(ABC):
    """Encode a list of waveforms into a 2-D embedding matrix"""

    @abstractmethod
    def embed(self, audios: Sequence[np.ndarray], sr: int) -> np.ndarray:
        """Returns (N, D) - one row per clip."""
        ...

    @abstractmethod
    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        """Encode text prompts into the same embedding space (CLAP only). Returns (N, D)."""
        ...

    def embed_files(
        self,
        paths: Sequence[str],
        sr: int = 16000,
        batch_size: int = 32,
        normalize_rms: bool = True,
        target_rms: float = 0.1,
    ) -> np.ndarray:
        """Encode audio files in batches so peak RAM is O(batch_size * clip_len) not O(N * clip_len).

        normalize_rms prevents amplitude differences between datasets (e.g. quiet stems vs
        louder generated audio) from distorting the embedding distribution.
        """
        rows = []
        for i in range(0, len(paths), batch_size):
            batch_wavs = []
            for p in paths[i : i + batch_size]:
                audio, file_sr = sf.read(str(p), dtype="float32", always_2d=False)
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                if file_sr != sr:
                    wav = torch.from_numpy(audio).unsqueeze(0)
                    audio = torchaudio.functional.resample(wav, file_sr, sr).squeeze(0).numpy()
                if normalize_rms:
                    rms = np.sqrt(np.mean(audio ** 2))
                    if rms > 1e-8:
                        audio = audio * (target_rms / rms)
                batch_wavs.append(audio)
            rows.append(self.embed(batch_wavs, sr))
        return np.concatenate(rows, axis=0)


class VGGishEmbedder(AudioEmbedder):
    """VGGish 128-d embeddings. Uses raw ReLU activations (postprocess=False) - no PCA, no quantization.

    postprocess=True quantizes to [0, 255], inflating FAD by ~4000x; we skip it to match MSDM
    and the frechet-audio-distance library (use_pca=False, use_activation=False).
    Each clip yields one row per 0.96-second VGGish window.
    """

    TARGET_SR = 16000

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None
        self._vggish_input = None

    def _load(self):
        if self._model is not None:
            return
        for k in list(sys.modules):
            if k == "torchvggish" or k.startswith("torchvggish."):
                del sys.modules[k]
        from torchvggish.torchvggish import vggish  # pip install torchvggish>=0.2
        from torchvggish import vggish_input
        self._vggish_input = vggish_input
        self._model = vggish(postprocess=False).eval().to(self.device)

    def embed(self, audios: Sequence[np.ndarray], sr: int) -> np.ndarray:
        self._load()
        rows = []
        for audio in audios:
            audio = audio.astype(np.float32)
            if sr != self.TARGET_SR:
                wav = torch.from_numpy(audio).unsqueeze(0)
                audio = torchaudio.functional.resample(wav, sr, self.TARGET_SR).squeeze(0).numpy()
            examples = self._vggish_input.waveform_to_examples(audio, self.TARGET_SR)
            examples = examples.float().to(self.device)
            with torch.no_grad():
                emb = self._model.forward(examples)
            rows.append(emb.cpu().numpy())
        return np.concatenate(rows, axis=0)

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError("VGGish has no text encoder. Use CLAP for text embeddings")


class CLAPEmbedder(AudioEmbedder):
    """CLAP 512-d joint audio-text embeddings."""

    TARGET_SR = 48000

    def __init__(self, ckpt_path: str | None = None, device: str = "cpu",
                 enable_fusion: bool = False):
        self.ckpt_path = ckpt_path
        self.device = device
        self.enable_fusion = enable_fusion
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        saved_argv, sys.argv = sys.argv, sys.argv[:1]
        try:
            import laion_clap
            model = laion_clap.CLAP_Module(enable_fusion=self.enable_fusion)
        except ImportError:
            raise ImportError("CLAP backend requires laion-clap: pip install laion-clap")
        finally:
            sys.argv = saved_argv
        if self.ckpt_path:
            model.load_ckpt(self.ckpt_path)
        else:
            model.load_ckpt()
        model.eval().to(self.device)
        self._model = model

    def _resample(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if sr == self.TARGET_SR:
            return audio
        wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        wav = torchaudio.functional.resample(wav, sr, self.TARGET_SR)
        return wav.squeeze(0).numpy()

    def embed(self, audios: Sequence[np.ndarray], sr: int) -> np.ndarray:
        self._load()
        rows = []
        for audio in audios:
            audio = self._resample(audio.astype(np.float32), sr)
            emb = self._model.get_audio_embedding_from_data(x=audio.reshape(1, -1))
            rows.append(emb)
        return np.concatenate(rows, axis=0)

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        self._load()
        return self._model.get_text_embedding(list(texts))


_BACKENDS: dict[str, type[AudioEmbedder]] = {
    "vggish": VGGishEmbedder,
    "clap": CLAPEmbedder,
}


def get_embedder(backend: str, device: str = "cpu", **kwargs) -> AudioEmbedder:
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}. Choose from {list(_BACKENDS)}")
    return _BACKENDS[backend](device=device, **kwargs)
