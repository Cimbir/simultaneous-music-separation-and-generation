from __future__ import annotations

import torch
from tqdm import tqdm

from latent_diffusion.modules.diffusionmodules.util import make_ddim_timesteps
from modules.abstractions.sampler import Sampler
from modules.abstractions.diffusion_model import DiffusionModel

class HeunSampler(Sampler):
    """
    Deterministic Heun sampler for diffusion models from EDM paper pg. 5

    On each step denoise future point and move based on both denoising eps
    """

    def __init__(self, diffusion_model: DiffusionModel):
        self.dm = diffusion_model

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        conditioning: torch.Tensor,
        steps: int = 200,
        eta: float = 1.0,
        verbose: bool = False,
        *,
        batch_size: int = 1,
        cfg_scale: float = 1.0,
        ddim_discretize: str = "uniform",
        callback=None,
        **kwargs,
    ) -> torch.Tensor:
        if kwargs:
            unsupported = ", ".join(sorted(kwargs.keys()))
            raise NotImplementedError(f"{self.name} sampler does not support: {unsupported}")

        if verbose and eta != 0.0:
            print(f"{self.name} sampler is deterministic and ignores eta.")

        dm = self.dm
        device = dm.betas.device
        size = (batch_size, *shape)

        timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=steps,
            num_ddpm_timesteps=dm.num_timesteps,
            verbose=verbose,
        )
        timesteps = torch.as_tensor(timesteps[::-1].copy(), device=device, dtype=torch.long)

        alphas = dm.alphas_cumprod.to(device=device, dtype=torch.float32)[timesteps]
        sigmas = torch.sqrt((1.0 - alphas) / alphas)
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])

        x_t = torch.randn(size, device=device)

        x = x_t / alphas[0].sqrt()

        uncond = dm.cond_stage_model.get_unconditional_condition(batch_size)
        if cfg_scale == 1.0:
            uncond = None

        iterator = range(len(timesteps))
        if verbose:
            iterator = tqdm(iterator, desc=f"{self.name} Sampler", total=len(timesteps))

        for i in iterator:
            t = torch.full((batch_size,), timesteps[i].item(), device=device, dtype=torch.long)
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            alpha = alphas[i]

            denoised = self._predict_x0(x, alpha, t, conditioning, uncond, cfg_scale)
            d = self._to_derivative(x, sigma, denoised)
            dt = sigma_next - sigma

            if sigma_next.item() == 0.0:
                x = x + d * dt
            else:
                x_euler = x + d * dt
                t_next = torch.full(
                    (batch_size,),
                    timesteps[i + 1].item(),
                    device=device,
                    dtype=torch.long,
                )
                alpha_next = alphas[i + 1]
                denoised_next = self._predict_x0(
                    x_euler,
                    alpha_next,
                    t_next,
                    conditioning,
                    uncond,
                    cfg_scale,
                )
                d_next = self._to_derivative(x_euler, sigma_next, denoised_next)
                x = x + 0.5 * (d + d_next) * dt

            if callback:
                callback(i)

        return x

    def _predict_x0(
        self,
        x: torch.Tensor,
        alpha: torch.Tensor,
        t: torch.Tensor,
        conditioning: torch.Tensor,
        unconditional_conditioning: torch.Tensor | None,
        cfg_scale: float,
    ) -> torch.Tensor:
        sqrt_alpha = alpha.sqrt()
        sqrt_one_minus_alpha = (1.0 - alpha).sqrt()
        x_t = x * sqrt_alpha
        model_output = self._apply_model_cfg(
            x_t,
            t,
            conditioning,
            unconditional_conditioning,
            cfg_scale,
        )

        if self.dm.parameterization == "eps":
            return (x_t - sqrt_one_minus_alpha * model_output) / sqrt_alpha
        if self.dm.parameterization == "x0":
            return model_output
        if self.dm.parameterization == "v":
            return sqrt_alpha * x_t - sqrt_one_minus_alpha * model_output
        raise NotImplementedError(f"{self.name} sampler does not support parameterization={self.dm.parameterization}")

    def _apply_model_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditioning: torch.Tensor,
        unconditional_conditioning: torch.Tensor | None,
        cfg_scale: float,
    ) -> torch.Tensor:
        if unconditional_conditioning is None or cfg_scale == 1.0:
            return self.dm.apply_model(x, t, conditioning)

        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        c_in = torch.cat([unconditional_conditioning, conditioning])
        e_t_uncond, e_t = self.dm.apply_model(x_in, t_in, c_in).chunk(2)
        return e_t_uncond + cfg_scale * (e_t - e_t_uncond)

    @staticmethod
    def _to_derivative(
        x: torch.Tensor,
        sigma: torch.Tensor,
        denoised: torch.Tensor,
    ) -> torch.Tensor:
        return (x - denoised) / sigma

    def get_diffusion_model(self):
        return self.dm
