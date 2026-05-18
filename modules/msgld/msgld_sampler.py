import torch
import math
from tqdm import tqdm

from latent_diffusion.models.ddim import DDIMSampler
from latent_diffusion.modules.diffusionmodules.util import make_ddim_timesteps
from modules.abstractions.sampler import Sampler


class MsgLdHeunSampler(Sampler):
    """
    Deterministic Heun sampler for MSG-LD eps/x0/v diffusion models.

    The sampler integrates in sigma-space using the same discrete training
    timesteps selected by the DDIM discretization helper.
    """

    def __init__(self, diffusion_model, *, use_correction: bool = True, name: str = "Heun"):
        self.dm = diffusion_model
        self.use_correction = use_correction
        self.name = name

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
        x_T: torch.Tensor | None = None,
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

        if x_T is None:
            x_t = torch.randn(size, device=device)
        else:
            x_t = x_T.to(device)

        # Convert VP noised latent x_t into K-diffusion form:
        # x = x0 + sigma * eps.
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

            if not self.use_correction or sigma_next.item() == 0.0:
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
        """Return the diffusion model used for sampling."""
        return self.dm


class MsgLdEulerSampler(MsgLdHeunSampler):
    """Deterministic first-order Euler sampler for the same sigma-space path."""

    def __init__(self, diffusion_model):
        super().__init__(diffusion_model, use_correction=False, name="Euler")


class MsgLdEdmSampler(MsgLdHeunSampler):
    """
    EDM-style Heun sampler for MSG-LD models trained on the VP forward process.

    Uses Karras et al.'s rho-spaced sigma schedule and optional churn, while
    rounding each continuous sigma to the closest discrete training timestep
    before calling the existing timestep-conditioned UNet.
    """

    def __init__(self, diffusion_model):
        super().__init__(diffusion_model, use_correction=True, name="EDM")

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
        x_T: torch.Tensor | None = None,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
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
        supported_min = train_sigmas[0].item()
        supported_max = train_sigmas[-1].item()
        sigma_min = supported_min if sigma_min is None else max(float(sigma_min), supported_min)
        sigma_max = supported_max if sigma_max is None else min(float(sigma_max), supported_max)
        if sigma_min >= sigma_max:
            raise ValueError(f"sigma_min must be < sigma_max, got {sigma_min} >= {sigma_max}")

        step_indices = torch.arange(steps, device=device, dtype=torch.float32)
        t_steps = (
            sigma_max ** (1.0 / rho)
            + step_indices / max(steps - 1, 1) * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
        ) ** rho
        t_steps = self._round_sigmas(t_steps, train_sigmas)
        t_steps = torch.cat([t_steps, t_steps.new_zeros(1)])

        if x_T is None:
            x_next = torch.randn(size, device=device) * t_steps[0]
        else:
            x_next = x_T.to(device) * t_steps[0]

        uncond = dm.cond_stage_model.get_unconditional_condition(batch_size)
        if cfg_scale == 1.0:
            uncond = None

        iterator = range(steps)
        if verbose:
            iterator = tqdm(iterator, desc=f"{self.name} Sampler", total=steps)

        for i in iterator:
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
            d_cur = self._to_derivative(x_hat, sigma_hat, denoised)
            x_next = x_hat + (sigma_next - sigma_hat) * d_cur

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
                x_next = x_hat + 0.5 * (sigma_next - sigma_hat) * (d_cur + d_next)

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

    def _training_sigmas(self, device: torch.device) -> torch.Tensor:
        alphas = self.dm.alphas_cumprod.to(device=device, dtype=torch.float32)
        return torch.sqrt((1.0 - alphas) / alphas)

    @staticmethod
    def _round_sigmas(sigmas: torch.Tensor, train_sigmas: torch.Tensor) -> torch.Tensor:
        original_shape = sigmas.shape
        flat_sigmas = sigmas.reshape(-1)
        nearest = (flat_sigmas[:, None] - train_sigmas[None, :]).abs().argmin(dim=1)
        return train_sigmas[nearest].reshape(original_shape)


class MsgLdSampler(Sampler):
    """
    Wraps MSG-LD's sampler options to satisfy the Sampler interface.
    
    Not included steps or eta, because could be variable (originally 200 steps and eta 1.0)

    Args:
        diffusion_model: MsgLdDiffusionModel instance — provides the noise
                         schedule, apply_model, and cond_stage_model.
    """

    def __init__(self, diffusion_model):
        self.dm = diffusion_model
        self._ddim = DDIMSampler(diffusion_model)
        self._euler = MsgLdEulerSampler(diffusion_model)
        self._heun = MsgLdHeunSampler(diffusion_model)
        self._edm = MsgLdEdmSampler(diffusion_model)

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
        sampler_type: str = "ddim",
        **kwargs,
    ) -> torch.Tensor:
        sampler_type = sampler_type.lower()

        if sampler_type == "edm":
            return self._edm.sample(
                shape=shape,
                conditioning=conditioning,
                steps=steps,
                eta=eta,
                verbose=verbose,
                batch_size=batch_size,
                cfg_scale=cfg_scale,
                **kwargs,
            )

        if sampler_type in {"euler", "heun"}:
            ode_sampler = self._euler if sampler_type == "euler" else self._heun
            return ode_sampler.sample(
                shape=shape,
                conditioning=conditioning,
                steps=steps,
                eta=eta,
                verbose=verbose,
                batch_size=batch_size,
                cfg_scale=cfg_scale,
                ddim_discretize=ddim_discretize,
                **kwargs,
            )

        if sampler_type != "ddim":
            raise ValueError(f"Unknown sampler_type={sampler_type}. Expected 'ddim', 'euler', 'heun', or 'edm'.")

        uncond = self.dm.cond_stage_model.get_unconditional_condition(batch_size)
        samples, _ = self._ddim.sample(
            S=steps,
            eta=eta,
            ddim_discretize=ddim_discretize,
            verbose=verbose,
            batch_size=batch_size,
            shape=shape,
            conditioning=conditioning,
            unconditional_guidance_scale=cfg_scale,
            unconditional_conditioning=uncond if cfg_scale != 1.0 else None,
            **kwargs,
        )
        return samples

    def default_shape(self) -> tuple:
        """Return the standard (num_stems, z_channels, T, F) shape for this model."""
        dm = self.dm
        return (dm.num_stems, dm.z_channels, dm.latent_t_size, dm.latent_f_size)

    def get_diffusion_model(self):
        """Return the diffusion model used for sampling."""
        return self.dm
