from modules.abstractions.vae import VAE
import torch
import dac

class DACBackend(VAE):
    """Post-RVQ quantized continuous latent z [B, 64, T']. Hop: 512 at 44 kHz."""

    def __init__(self, model_type: str = "44khz", device: str = "cpu"):
        model_path = dac.utils.download(model_type=model_type)
        self.model = dac.DAC.load(model_path)
        self._sr = {"44khz": 44100, "24khz": 24000, "16khz": 16000}[model_type]
        self._device = device

    @property
    def sample_rate(self) -> int:
        return self._sr

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
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.model.preprocess(audio.to(self._device), self._sr)
        z, codes, latents, _, _ = self.model.encode(x)
        return z # [B, 64, T']

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.decode(z.to(self._device)) # [B, 1, T]