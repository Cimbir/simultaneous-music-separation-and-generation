import numpy as np
import torch
from abc import ABC, abstractmethod

class MelExtractor(ABC):
    """Interface for audio → mel-spectrogram conversion."""

    @property
    @abstractmethod
    def sampling_rate(self) -> int:
        """Sample rate expected by this extractor."""
        ...

    @property
    @abstractmethod
    def hop_length(self) -> int:
        """Audio samples between adjacent mel frames."""
        ...

    @abstractmethod
    def audio_to_mel(self, audio: np.ndarray, sr: int | None) -> torch.Tensor:
        """
        Convert raw audio waveform -> mel-spectrogram
        
        Args:
            audio: 1-D numpy array of audio samples (float32, range [-1, 1])
            sr: Sample rate of the audio.
        Returns:
            Mel-spectrogram tensor (1, n_mels, T)
        """
        ...
