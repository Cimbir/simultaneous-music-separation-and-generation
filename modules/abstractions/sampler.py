import torch
from abc import ABC, abstractmethod


class Sampler(ABC):
    """Interface for reverse-diffusion samplers over a multi-stem latent space."""

    @abstractmethod
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
        **kwargs,
    ) -> torch.Tensor:
        """
        Run reverse diffusion from Gaussian noise to a clean latent.

        Args:
            shape       : per-sample latent shape, e.g. (num_stems, z_channels, T, F).
                          Does not include the batch dimension.
            conditioning: conditioning tensor, shape (B, ...).
                          Pass the null/unconditional token for pure generation.
            steps       : number of diffusion steps to run.
            eta         : noise scale for stochastic sampling.
            verbose     : whether to print progress during sampling.
            batch_size  : number of independent samples to draw in parallel.
            cfg_scale   : classifier-free guidance scale.
                          1.0  = no guidance boost (conditioning only).
                          >1.0 = amplify conditioning signal vs. null baseline.

        Returns:
            Tensor of shape (B, *shape) in *scaled* latent space.
            Divide by scale_factor before passing to the VAE decoder.
        """
        ...

    @abstractmethod
    def get_diffusion_model(self):
        """
        Get the diffusion model used for sampling
        
        Returns:
            The diffusion model instance used by this sampler.
        """
        ...