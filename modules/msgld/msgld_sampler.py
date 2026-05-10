import torch

from latent_diffusion.models.ddim import DDIMSampler
from modules.abstractions.sampler import Sampler


class MsgLdSampler(Sampler):
    """
    Wraps MSG-LD's DDIMSampler to satisfy the Sampler interface.
    
    Not included steps or eta, because could be variable (originally 200 steps and eta 1.0)

    Args:
        diffusion_model: MsgLdDiffusionModel instance — provides the noise
                         schedule, apply_model, and cond_stage_model.
    """

    def __init__(self, diffusion_model):
        self.dm = diffusion_model
        self._ddim = DDIMSampler(diffusion_model)

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
        **kwargs,
    ) -> torch.Tensor:
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
