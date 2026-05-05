import numpy as np
import librosa
from modules.abstractions.extractor import MelExtractor
import torch

from utilities.audio.stft import TacotronSTFT

class MsgLdMelExtractor(MelExtractor):
    """Mel-spectrogram extractor using MSG-LD's TacotronSTFT."""

    def __init__(
        self,
        filter_length: int = 1024,
        hop_length: int = 160,
        win_length: int = 1024,
        n_mel_channels: int = 64,
        sampling_rate: int = 16000,
        mel_fmin: float = 0,
        mel_fmax: float = 8000,
    ):
        self.stft = TacotronSTFT(
            filter_length, hop_length, win_length,
            n_mel_channels, sampling_rate, mel_fmin, mel_fmax,
        )
        self.sampling_rate = sampling_rate

    def audio_to_mel(self, audio: np.ndarray, sr: int | None = None) -> torch.Tensor:
        if sr is not None and sr != self.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate)
        audio = np.clip(audio, -1.0, 1.0)
        audio_t = torch.FloatTensor(audio).unsqueeze(0) # (1, T)
        mel, _, _ = self.stft.mel_spectrogram(audio_t)  # (1, n_mels, T')
        return mel  # (1, n_mels, T')