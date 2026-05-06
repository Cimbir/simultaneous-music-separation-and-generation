import numpy as np
import torch
from abc import ABC, abstractmethod

class Vocoder(ABC):
    """Interface for a model that converts mel-spectrograms to audio."""

    @abstractmethod
    def mel_to_audio(self, mel: torch.Tensor) -> np.ndarray:
        """
        Convert mel-spectrogram -> audio waveform
        
        Args:
            mel: (B, n_mels, T) mel-spectrogram (linear or log scale depending on model)
        Returns:
            np.ndarray of int16 audio samples, shape (B, num_samples) or (num_samples,)
        """
        ...

    @abstractmethod
    def to(self, device): ...

    @abstractmethod  
    def eval(self): ...