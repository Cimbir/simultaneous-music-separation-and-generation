from latent_diffusion.modules.diffusionmodules.openaimodel import UNetModel
import torch
import torch.nn as nn
from typing import List, Optional, Callable, Any
from modules.abstractions.diffusion_model import DiffusionModel
from modules.abstractions.vae import VAE
from modules.abstractions.extractor import MelExtractor

class MsgLdDiffusionWrapper(nn.Module):
    """
    Model wrapper for the model conditioning
    MSG-LD concatenated the mixture latent along the stem axis (dim=2 for 3D UNet)
    """

    def __init__(self, unet: nn.Module, conditioning_key: str):
        super().__init__()
        self.diffusion_model = unet
        self.conditioning_key = conditioning_key

    def forward(self, x, t, c: Optional[List[torch.Tensor]] = None):
        x = x.contiguous()
        t = t.contiguous()

        if self.conditioning_key is None:
            return self.diffusion_model(x, t)
        elif self.conditioning_key == "concat" and c is not None:
            xc = torch.cat([x] + c, dim=2)
            return self.diffusion_model(xc, t)
        elif self.conditioning_key == "crossattn" and c is not None:
            cc = torch.cat(c, 1)
            return self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == "film" and c is not None:
            cc = c[0].squeeze(1)
            return self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError(f"conditioning_key={self.conditioning_key}")


class MsgLdCondModel(nn.Module):
    """
    Patch conditioning model: encodes the mixture then replicates across all stems for conditioning tensor
    Exactly reproduces Patch_Cond_Model from the paper.
    """

    def __init__(self, num_stems: int, z_channels: int = 8, training: bool = False, unconditional_prob: float = 0.0):
        super().__init__()
        self.num_stems = num_stems
        self.z_channels = z_channels
        self.unconditional_prob = unconditional_prob
        self.training = training
        # These will be patched in by the diffusion model
        self.encode_first_stage: Optional[Callable[[Any], Any]] = None
        self.get_first_stage_encoding: Optional[Callable[[Any], Any]] = None
        self.device_fn: Optional[Callable[[], torch.device]] = None
        self.embed_mode = "fbank"

    def get_unconditional_condition(self, batchsize):
        assert self.device_fn is not None, "device_fn not set"
        device = self.device_fn() if callable(self.device_fn) else self.device_fn
        return torch.zeros(batchsize, self.num_stems, self.z_channels, 256, 16, device=device)

    def forward(self, batch):
        assert self.encode_first_stage is not None, "encode_first_stage not set"
        assert self.get_first_stage_encoding is not None, "get_first_stage_encoding not set"
        
        encoder_posterior = self.encode_first_stage(batch) # through vae get posterior
        embed = self.get_first_stage_encoding(encoder_posterior).detach() # and sample

        extra_dims = list(embed.shape[1:])
        expanded = torch.zeros(embed.size(0), self.num_stems, *extra_dims, device=embed.device)
        for i in range(embed.size(0)):
            if self.training and float(torch.rand(1)) < self.unconditional_prob:
                expanded[i] = 0.0
            else:
                expanded[i] = embed[i].repeat(self.num_stems, 1, 1, 1)

        return expanded.detach()


class MsgLdDiffusionModel(DiffusionModel):
    """
    Full MSG-LD latent diffusion model
    """

    def __init__(
        self,
        vae: VAE,
        mel_extractor: MelExtractor,
        # UNet
        model_channels: int = 128,
        channel_mult: List[int] = [1, 2, 3, 5],
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [8, 4, 2],
        num_head_channels: int = 32,
        # Diffusion
        num_stems: int = 4,
        z_channels: int = 8,
        timesteps: int = 1000,
        linear_start: float = 0.0015,
        linear_end: float = 0.0195,
        parameterization: str = "eps",
        conditioning_key: str = "concat",
        # Conditioning
        unconditional_prob: float = 0.0,
        # Scale
        scale_by_std: bool = True,
        scale_factor: float = 1.0,
        latent_t_size: int = 256,
        latent_f_size: int = 16,
        device: str = "cpu",
    ):
        self.vae = vae
        self.mel_extractor = mel_extractor
        self.num_stems = num_stems
        self.z_channels = z_channels
        self.parameterization = parameterization
        self.conditioning_key = conditioning_key
        self.scale_by_std = scale_by_std
        self.scale_factor = scale_factor
        self.latent_t_size = latent_t_size
        self.latent_f_size = latent_f_size
        self._device = device
        self._scale_computed = False

        # Register noise schedule
        self.setup_noise_schedule(
            beta_schedule="linear",
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
        )

        # The 3D UNet model
        in_channels = num_stems
        out_channels = num_stems

        self.unet = UNetModel(
            image_size=64,
            in_channels=in_channels,
            out_channels=out_channels,
            model_channels=model_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            channel_mult=channel_mult,
            num_head_channels=num_head_channels,
            dims=3,
            no_condition=False,
            extra_film_use_concat=True,
            use_spatial_transformer=False,
        )

        self.model = MsgLdDiffusionWrapper(self.unet, conditioning_key)

        # Conditioning model
        self.cond_stage_model = MsgLdCondModel(
            num_stems=num_stems,
            z_channels=z_channels,
            unconditional_prob=unconditional_prob,
        )
        self.cond_stage_model.encode_first_stage = self._encode_first_stage_raw
        self.cond_stage_model.get_first_stage_encoding = self._get_first_stage_encoding
        self.cond_stage_model.device_fn = lambda: torch.device(self._device)

        # Logvar
        self.logvar = nn.Parameter(torch.full((timesteps,), 0.0))

    @property
    def device(self) -> torch.device:
        return torch.device(self._device)

    def build_model(self) -> nn.Module:
        return self.model

    def encode_to_latent(self, batch: dict) -> torch.Tensor:
        """Encode fbank_stems to latent: (B, 1, num_stems, T, F) -> (B, num_stems, z_ch, 256, 16)."""
        x = batch["fbank_stems"]
        if x.dim() == 4:
            x = x.unsqueeze(1)
        x = x.to(self._device)

        # Reshape by putting each stem in as separate batch item
        batch_size, _, num_channels, *dims = x.shape
        x_flat = x.view(batch_size * num_channels, 1, *dims) # (B*num_stems, 1, T, F)
        # And then encode through the VAE        
        enc = self.vae.encode(x_flat)
        posterior = enc["posterior"]
        z = posterior.sample() # (B*num_stems, z_ch, T_latent, F_latent)

        # Compute scale factor on first batch
        if self.scale_by_std and not self._scale_computed:
            self.scale_factor = 1.0 / z.flatten().std().item()
            self._scale_computed = True

        # Normalize
        z = z * self.scale_factor
        # And finally reshape back to (B, num_stems, z_ch, T_latent, F_latent) to be ready for LDM
        z = z.view(-1, self.num_stems, self.z_channels, self.latent_t_size, self.latent_f_size)
        return z.detach() # (B, num_stems, z_ch, T_latent, F_latent)

    def get_conditioning(self, batch: dict):
        """Encode mixture fbank as conditioning through the same VAE."""
        x_mix = batch["fbank"]
        if isinstance(x_mix, torch.Tensor):
            x_mix = x_mix.to(self._device)
            if x_mix.dim() == 3:
                x_mix = x_mix.unsqueeze(1)  # (B, 1, n_mels, T)
        return self.cond_stage_model(x_mix)

    def apply_model(self, x_noisy, t, cond):
        """Run the denoising network. Wraps cond into the expected dict."""
        if not isinstance(cond, list):
            cond = [cond]
        return self.model(x_noisy, t, cond)

    def state_dict(self):
        sd = {}
        sd["unet"] = self.unet.state_dict()
        sd["cond_stage_model"] = {
            k: v for k, v in self.cond_stage_model.state_dict().items()
        }
        sd["logvar"] = self.logvar
        sd["scale_factor"] = self.scale_factor
        return sd

    def load_state_dict(self, sd, strict=True):
        self.unet.load_state_dict(sd["unet"], strict=strict)
        self.cond_stage_model.load_state_dict(sd.get("cond_stage_model", {}), strict=False)
        if "logvar" in sd:
            self.logvar = sd["logvar"]
        if "scale_factor" in sd:
            self.scale_factor = sd["scale_factor"]
            self._scale_computed = True

    def to(self, device):
        self._device = device
        super().to(device)
        self.model.to(device)
        self.unet.to(device)
        self.cond_stage_model.to(device)
        self.logvar = self.logvar.to(device)
        self.vae.to(device)
        return self

    def parameters(self):
        """Only the trainable denoising network parameters (VAE is frozen)."""
        return self.model.parameters()

    def train(self, mode=True):
        self.model.train(mode)
        self.cond_stage_model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        self.cond_stage_model.eval()
        return self
    
    def _encode_first_stage_raw(self, x):
        """Encode through VAE (returns posterior)."""
        return self.vae.encode(x)["posterior"]

    def _get_first_stage_encoding(self, posterior):
        """Sample from posterior and scale."""
        z = posterior.sample()
        return z * self.scale_factor