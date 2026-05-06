import bigvgan
from modules.abstractions.vocoder import Vocoder
import torch
import numpy as np

class BigVGANVocoder(Vocoder):
    """BigVGAN: higher quality than HiFi-GAN, same mel-to-audio interface."""

    def __init__(self, model_name="nvidia/bigvgan_v2_24khz_100band_256x", device="cpu"):
        self.model = bigvgan.BigVGAN.from_pretrained(model_name)
        self.model.remove_weight_norm()
        self.model.eval()
        self.sample_rate = 24000 # adjust per model variant
        self._device = device

    def to(self, device):
        self._device = device
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    @torch.no_grad()
    def mel_to_audio(self, mel: torch.Tensor) -> np.ndarray:
        wav = self.model(mel.to(self._device)).squeeze(1)
        return (wav.cpu().numpy() * 32768).astype("int16")
    