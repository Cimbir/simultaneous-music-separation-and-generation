import torch
from modules.abstractions.vae import VAE, deterministic_encode_output
from dacvae import DACVAE

class DACVAEBackend(VAE):
    """Continuous Gaussian latent, no RVQ. Better fit for latent diffusion."""

    def __init__(self, device: str = "cpu"):
        self.model = DACVAE.load("facebook/dacvae-watermarked")
        self._device = device

    @property
    def sample_rate(self) -> int:
        return self.model.sample_rate

    def to(self, device):
        self._device = device
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        return self

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> dict:
        z = self.model.encode(audio.to(self._device)) # [B, D, T']
        return deterministic_encode_output(z)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.decode(z.to(self._device)) # [B, 1, T]
