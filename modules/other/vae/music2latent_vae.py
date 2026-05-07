import numpy as np
import torch
from modules.abstractions.vae import VAE
from music2latent import EncoderDecoder


class Music2LatentVAE(VAE):
    def __init__(self, device: str = "cpu"):
        self._device = device
        self.model = EncoderDecoder(device=torch.device(device))

    @property
    def sample_rate(self) -> int:
        return 44100

    def to(self, device):
        self._device = device
        self.model.device = torch.device(device)
        self.model.gen.to(device)
        return self

    def eval(self):
        self.model.gen.eval()
        for p in self.model.gen.parameters():
            p.requires_grad_(False)
        return self

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: [B, 1, T] -> latents: [B, D, L]
        latents = []
        for sample in audio:
            wav = sample.squeeze(0).cpu().numpy() # [T]
            z = self.model.encode(wav)             # [1, D, L]
            latents.append(torch.from_numpy(z))
        return torch.cat(latents, dim=0).to(self._device) # [B, D, L]

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, D, L] -> audio: [B, 1, T]
        waveforms = []
        for latent in z:
            lat = latent.unsqueeze(0).cpu().numpy() # [1, D, L]
            wav = self.model.decode(lat)             # [T, 1]
            wav_t = torch.from_numpy(wav).permute(1, 0).unsqueeze(0) # [1, 1, T]
            waveforms.append(wav_t)
        return torch.cat(waveforms, dim=0).to(self._device) # [B, 1, T]
