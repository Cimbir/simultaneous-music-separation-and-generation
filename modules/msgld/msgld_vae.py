from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from modules.abstractions.vae import VAE

from latent_diffusion.modules.diffusionmodules.model import Encoder, Decoder
from latent_diffusion.modules.distributions.distributions import DiagonalGaussianDistribution

class MsgLdVAE(VAE):
    """Wraps the AutoencoderKL from MSG-LD."""

    def __init__(self, vae_ckpt: str, device="cpu",
                 embed_dim: int = 8, ddconfig: dict | None = None):

        # Distribution class from MSG-LD, used for encoding the latent space
        self.DiagonalGaussianDistribution = DiagonalGaussianDistribution

        # The default config
        if ddconfig is None:
            ddconfig = dict(
                double_z=True, z_channels=8, resolution=256,
                in_channels=1, out_ch=1, ch=128,
                ch_mult=[1, 2, 4], num_res_blocks=2,
                attn_resolutions=[], dropout=0.0,
            )

        # Encoder -> quantization -> decoder
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quant_conv = nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

        # Load checkpoint
        print(f"Loading VAE weights from {vae_ckpt}")
        ckpt = torch.load(vae_ckpt, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        # Filter to only keys that belong to our sub-modules
        prefix_map = {
            "encoder.": self.encoder,
            "decoder.": self.decoder,
            "quant_conv.": self.quant_conv,
            "post_quant_conv.": self.post_quant_conv,
        }
        for prefix, module in prefix_map.items():
            sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            if sub:
                module.load_state_dict(sub, strict=False)
                print(f"  Loaded {len(sub)} params for {prefix[:-1]}")

        self._device = device

    def to(self, device):
        self._device = device
        self.encoder.to(device)
        self.decoder.to(device)
        self.quant_conv.to(device)
        self.post_quant_conv.to(device)
        return self

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        self.quant_conv.eval()
        self.post_quant_conv.eval()
        return self

    @torch.no_grad()
    def encode(self, mel: torch.Tensor) -> dict:
        """mel: (B, 1, n_mels, T) -> dict(mean, logvar, posterior)"""
        mel = mel.permute(0, 1, 3, 2).to(self._device)  # (B, 1, T, n_mels)
        h = self.encoder(mel)
        moments = self.quant_conv(h)
        posterior = self.DiagonalGaussianDistribution(moments)
        return {"mean": posterior.mean, "logvar": posterior.logvar, "posterior": posterior}

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, C, H, W) -> reconstructed mel (B, 1, n_mels, T)"""
        z = z.to(self._device)
        z = self.post_quant_conv(z)
        mel = self.decoder(z)
        mel = mel.permute(0, 1, 3, 2)  # (B, 1, n_mels, T)
        return mel