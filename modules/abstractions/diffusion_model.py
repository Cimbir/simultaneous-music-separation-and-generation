from functools import partial
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class DiffusionModel(ABC):
    """
    Abstraction for an LDM
    Noise addition and q sampling are shared
    """

    # Abstract interface

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Return the diffusion model wrapped with conditioning routing."""
        ...

    @abstractmethod
    def apply_model(
        self, x_noisy: torch.Tensor, t: torch.Tensor, cond
    ) -> torch.Tensor:
        """
        Run the diffusion model on noisy input x_noisy at timestep t with conditioning cond.
        Returns the model prediction (noise, x0, or velocity depending on parameterization).
        """
        ...

    @abstractmethod
    def get_conditioning(self, batch: dict) -> object:
        """Given a data batch, return the conditioning tensor(s) that apply_model expects."""
        ...

    @abstractmethod
    def encode_to_latent(self, batch: dict) -> torch.Tensor:
        """Given a data batch, return the latent tensor x0 that the diffusion operates on."""
        ...

    @abstractmethod
    def state_dict(self) -> dict:
        """Return full model state dict for serialization."""
        ...

    @abstractmethod
    def load_state_dict(self, sd: dict, strict: bool = True):
        """Load model state dict."""
        ...

    # Shared implementation

    def setup_noise_schedule(
        self,
        beta_schedule: str = "linear",
        timesteps: int = 1000,
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
    ):
        """
        Precompute noise schedule and related tensors for the diffusion process.

        After calling this, the following attributes are available on self:
            betas - how much noise to add at each timestep 
            alphas - how much of the original signal remains at each timestep
            posterior - parameters for the reverse diffusion process
            lvlb_weights - weights for the variational lower bound loss at each timestep
        """
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end

        to_t = partial(torch.tensor, dtype=torch.float32)
        
        if beta_schedule == "linear":
            betas = (
                torch.linspace(linear_start**0.5, linear_end**0.5, timesteps, dtype=torch.float64) ** 2
            ).numpy()
        elif beta_schedule == "cosine":
            s = 8e-3
            steps = np.arange(timesteps + 1, dtype=np.float64) / timesteps + s
            alphas = np.cos(steps / (1 + s) * np.pi / 2) ** 2
            alphas = alphas / alphas[0]
            betas = np.clip(1 - alphas[1:] / alphas[:-1], 0, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # For forward process (noising)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.betas = to_t(betas)
        self.alphas_cumprod = to_t(alphas_cumprod)
        self.alphas_cumprod_prev = to_t(alphas_cumprod_prev)

        self.sqrt_alphas_cumprod = to_t(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_t(np.sqrt(1.0 - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = to_t(np.log(1.0 - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_t(np.sqrt(1.0 / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_t(np.sqrt(1.0 / alphas_cumprod - 1))

        # For reverse process (denoising)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = to_t(posterior_variance)
        self.posterior_log_variance_clipped = to_t(np.log(np.maximum(posterior_variance, 1e-20)))
        self.posterior_mean_coef1 = to_t(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.posterior_mean_coef2 = to_t((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod))

        # VLB weights for each timestep training
        lvlb_weights = (
            self.betas ** 2
            / (2 * self.posterior_variance * to_t(alphas) * (1 - self.alphas_cumprod))
        )
        lvlb_weights[0] = lvlb_weights[1]
        self.lvlb_weights = lvlb_weights

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Forward diffusion: x_t = sqrt(alpha) * x_0 + sqrt(1-alpha) * eps"""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_origin = self.sqrt_alphas_cumprod.to(x_0.device)
        sqrt_noise = self.sqrt_one_minus_alphas_cumprod.to(x_0.device)
        return (
            self._extract(sqrt_origin, t, x_0.shape) * x_0
            + self._extract(sqrt_noise, t, x_0.shape) * noise
        )

    def get_velocity(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Velocity target: v = sqrt(alpha) * eps - sqrt(1-alpha) * x_0."""
        sqrt_origin = self.sqrt_alphas_cumprod.to(x_0.device)
        sqrt_noise = self.sqrt_one_minus_alphas_cumprod.to(x_0.device)
        return (
            self._extract(sqrt_origin, t, x_0.shape) * noise
            - self._extract(sqrt_noise, t, x_0.shape) * x_0
        )

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        """For each batch item in a grab the correct alpha by t for the forward process"""
        b, *_ = t.shape
        out = a.gather(-1, t).contiguous()
        return out.reshape(b, *((1,) * (len(shape) - 1))).contiguous()

    def to(self, device):
        """Move all preprocessed data to device"""
        for attr in [
            "betas", "alphas_cumprod", "alphas_cumprod_prev",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
            "log_one_minus_alphas_cumprod", "sqrt_recip_alphas_cumprod",
            "sqrt_recipm1_alphas_cumprod", "posterior_variance",
            "posterior_log_variance_clipped", "posterior_mean_coef1",
            "posterior_mean_coef2", "lvlb_weights",
        ]:
            if hasattr(self, attr):
                setattr(self, attr, getattr(self, attr).to(device))
        return self
