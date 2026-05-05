import numpy as np
import torch
from modules.abstractions.vocoder import Vocoder

import hifigan

class MsgLdHiFiGAN(Vocoder):
    """Wraps the HiFi-GAN vocoder from MSG-LD."""

    def __init__(self, ckpt_path: str, config: dict, device="cpu"):

        assert ckpt_path is not None, "Checkpoint path must be provided for HiFi-GAN"
        assert config is not None, "Config must be provided for HiFi-GAN"

        # HiFi-GAN config
        h = hifigan.AttrDict(config)

        # Initialize the model
        self.model = hifigan.Generator(h)
        self.sample_rate = h.get("sampling_rate", 16000)
        self._device = device

        # Load checkpoint
        print(f"Loading HiFi-GAN weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt["generator"])

        self.model.eval()
        self.model.remove_weight_norm()

    def to(self, device):
        self._device = device
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    @torch.no_grad()
    def mel_to_audio(self, mel: torch.Tensor) -> np.ndarray:
        """mel: (B, n_mels, T) -> int16 numpy audio (B, num_samples)"""
        mel = mel.to(self._device)
        wav = self.model(mel).squeeze(1)
        wav = (wav.cpu().numpy() * 32768).astype("int16")
        return wav