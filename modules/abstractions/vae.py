import torch
from abc import ABC, abstractmethod


class DeterministicPosterior:
    """Posterior-compatible wrapper for encoders that return deterministic latents."""

    def __init__(self, mean: torch.Tensor, logvar_value: float = -30.0):
        self.mean = mean
        self.logvar = torch.full_like(mean, logvar_value)

    def sample(self) -> torch.Tensor:
        return self.mean

    def mode(self) -> torch.Tensor:
        return self.mean


def deterministic_encode_output(mean: torch.Tensor, logvar_value: float = -30.0) -> dict:
    posterior = DeterministicPosterior(mean, logvar_value)
    return {"mean": posterior.mean, "logvar": posterior.logvar, "posterior": posterior}


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
