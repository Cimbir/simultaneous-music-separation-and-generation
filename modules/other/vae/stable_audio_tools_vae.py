import torch
from modules.abstractions.vae import VAE
from stable_audio_tools import get_pretrained_model


class StableAudioVAE(VAE):

    def __init__(self, model_name="stabilityai/stable-audio-open-1.0", device="cpu"):
        model, model_config = get_pretrained_model(model_name)
        self.autoencoder = model.pretransform.model
        self._sr = model_config["sample_rate"]
        self._device = device

    @property
    def sample_rate(self) -> int:
        return self._sr

    def to(self, device):
        self._device = device
        self.autoencoder.to(device)
        return self

    def eval(self):
        self.autoencoder.eval()
        return self

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(audio.to(self._device))

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(z.to(self._device))