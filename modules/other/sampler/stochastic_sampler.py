import torch
import math
from tqdm import tqdm

from modules.abstractions.sampler import Sampler
from modules.abstractions.diffusion_model import DiffusionModel

class StochasticSampler(Sampler):
    """
    Stochastic sampler for diffusion models from EDM paper pg. 5

    Notes:
    Heun method - 2nd order correcting for more stable denoising
    Timestep manipulation - Dense sampling on noisy parts to change it more
    Churn - additional noise to push samples out of predefined parts
    """

    def __init__(self, diffusion_model: DiffusionModel):
        self.dm = diffusion_model
        self.name = "Stochastic"

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
        rho: float = 7.0,
        s_churn: float = 0.0,
        s_min: float = 0.0,
        s_max: float = float("inf"),
        s_noise: float = 1.0,
        callback=None,
        **kwargs,
    ) -> torch.Tensor:
        if kwargs:
            unsupported = ", ".join(sorted(kwargs.keys()))
            raise NotImplementedError(f"{self.name} sampler does not support: {unsupported}")
        if verbose and eta != 0.0:
            print(f"{self.name} sampler ignores eta; use s_churn/s_noise for stochasticity.")

        dm = self.dm
        device = dm.betas.device
        size = (batch_size, *shape)

        train_sigmas = self._training_sigmas(device)
        sigma_min = train_sigmas[0].item()
        sigma_max = train_sigmas[-1].item()
        if sigma_min >= sigma_max:
            raise ValueError(f"sigma_min must be < sigma_max, got {sigma_min} >= {sigma_max}")

        step_indices = torch.arange(steps, device=device, dtype=torch.float32)
        t_steps = (
            sigma_max ** (1.0 / rho)
            + step_indices / max(steps - 1, 1) * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
        ) ** rho
        t_steps = self._round_sigmas(t_steps, train_sigmas)
        t_steps = torch.cat([t_steps, t_steps.new_zeros(1)])

        x_next = torch.randn(size, device=device) * t_steps[0]

        uncond = dm.cond_stage_model.get_unconditional_condition(batch_size)
        if cfg_scale == 1.0:
            uncond = None

        iterator = range(steps)
        if verbose:
            iterator = tqdm(iterator, desc=f"{self.name} Sampler", total=steps)

        for i in iterator:
            print(f"step {i}")
            sigma_cur = t_steps[i]
            sigma_next = t_steps[i + 1]
            x_cur = x_next

            gamma = min(s_churn / steps, math.sqrt(2.0) - 1.0) if s_min <= sigma_cur.item() <= s_max else 0.0
            sigma_hat = self._round_sigmas(sigma_cur * (1.0 + gamma), train_sigmas)
            if gamma > 0.0:
                noise_scale = (sigma_hat.square() - sigma_cur.square()).clamp(min=0.0).sqrt()
                x_hat = x_cur + noise_scale * s_noise * torch.randn_like(x_cur)
            else:
                x_hat = x_cur

            denoised = self._predict_x0_sigma(
                x_hat,
                sigma_hat,
                train_sigmas,
                conditioning,
                uncond,
                cfg_scale,
            )
            d = self._to_derivative(x_hat, sigma_hat, denoised)
            dt = sigma_next - sigma_hat
            x_next = x_hat + d * dt

            if sigma_next.item() != 0.0:
                denoised_next = self._predict_x0_sigma(
                    x_next,
                    sigma_next,
                    train_sigmas,
                    conditioning,
                    uncond,
                    cfg_scale,
                )
                d_next = self._to_derivative(x_next, sigma_next, denoised_next)
                x_next = x_hat + 0.5 * dt * (d + d_next)

            if callback:
                callback(i)

        return x_next

    def _predict_x0_sigma(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        train_sigmas: torch.Tensor,
        conditioning: torch.Tensor,
        unconditional_conditioning: torch.Tensor | None,
        cfg_scale: float,
    ) -> torch.Tensor:
        t_index = torch.argmin((train_sigmas - sigma).abs())
        alpha = self.dm.alphas_cumprod.to(device=x.device, dtype=torch.float32)[t_index]
        t = torch.full((x.shape[0],), int(t_index.item()), device=x.device, dtype=torch.long)
        return self._predict_x0(
            x,
            alpha,
            t,
            conditioning,
            unconditional_conditioning,
            cfg_scale,
        )

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
        model_output = self._apply_model_cfg(x_t, t, conditioning, unconditional_conditioning, cfg_scale)

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

    def _training_sigmas(self, device: torch.device) -> torch.Tensor:
        alphas = self.dm.alphas_cumprod.to(device=device, dtype=torch.float32)
        return torch.sqrt((1.0 - alphas) / alphas)

    @staticmethod
    def _round_sigmas(sigmas: torch.Tensor, train_sigmas: torch.Tensor) -> torch.Tensor:
        original_shape = sigmas.shape
        flat_sigmas = sigmas.reshape(-1)
        nearest = (flat_sigmas[:, None] - train_sigmas[None, :]).abs().argmin(dim=1)
        return train_sigmas[nearest].reshape(original_shape)
    
    def get_diffusion_model(self):
        return self.dm