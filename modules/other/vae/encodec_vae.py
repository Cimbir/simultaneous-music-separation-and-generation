import torch
from modules.abstractions.vae import VAE
from encodec import EncodecModel

class EncodecVAE(VAE):

    def __init__(self, bandwidth=6.0, device="cpu"):
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)
        self._device = device

    @property
    def sample_rate(self) -> int:
        return 24000

    def to(self, device):
        self._device = device
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        frames = self.model.encode(audio.to(self._device))
        return frames[0][0].float()  # [B, n_q, T']

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.decode([(z.long(), None)])