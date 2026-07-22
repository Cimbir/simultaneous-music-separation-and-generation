import torch
from abc import ABC, abstractmethod

class VAE(ABC):
    """Interface for a VAE that operates on mel-spectrograms."""

    @abstractmethod
    def encode(self, mel: torch.Tensor) -> dict:
        """
        Encode mel-spectogram -> latent distribution parameters
        
        Args:
            mel: (B, 1, n_mel, T) mel-spectrogram tensor
        Returns:
            {
                'mean' - Tensor of shape (B, C, H, W) representing the mean of the latent distribution
                'logvar' - Tensor of shape (B, C, H, W) representing the log-variance of the latent distribution
                'posterior' (optional) - The posterior distribution object if available
            }
        """
        ...

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent samples -> mel-spectrogram
        
        Args:
            z: (B, C, H, W) latent tensor
        Returns:
            Reconstructed mel-spectrogram (B, 1, n_mel, T)
        """
        ...

    @abstractmethod
    def to(self, device): ...

    @abstractmethod
    def eval(self): ...