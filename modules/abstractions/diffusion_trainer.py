from __future__ import annotations

import pytorch_lightning as pl
import torch
from abc import ABC, abstractmethod
from modules.abstractions.diffusion_model import DiffusionModel

class DiffusionTrainer(pl.LightningModule, ABC):
    """
    Abstract trainer for a latent diffusion model.
    Warmup and training/validation steps are shared
    """

    def __init__(self, learning_rate: float = 1e-4, warmup_steps: int = 1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self._initial_lr: float | None = None

    # Abstract interface

    @abstractmethod
    def build_diffusion_model(self) -> DiffusionModel:
        """Construct and return the diffusion model instance."""
        ...

    @abstractmethod
    def compute_loss(self, batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Given a data batch, compute (scalar_loss, {metric_name: value})."""
        ...

    # Shared implementation

    def warmup_step(self):
        """Linear LR warmup from 0 -> initial_lr over warmup_steps steps."""
        if self._initial_lr is None:
            self._initial_lr = self.learning_rate
        if self.global_step <= self.warmup_steps:
            frac = self.global_step / max(self.warmup_steps, 1)
            self.trainer.optimizers[0].param_groups[0]["lr"] = frac * self._initial_lr
        else:
            self.trainer.optimizers[0].param_groups[0]["lr"] = self._initial_lr

    def training_step(self, batch, batch_idx):
        """Warmup -> compute loss -> log metrics"""
        self.warmup_step()
        loss, loss_dict = self.compute_loss(batch)
        self.log_dict(
            {k: float(v) for k, v in loss_dict.items()},
            prog_bar=True, logger=True, on_step=True, on_epoch=True,
        )
        self.log("global_step", float(self.global_step), prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("lr_abs", float(self.trainer.optimizers[0].param_groups[0]["lr"]), prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        """Compute loss -> log metrics"""
        loss, loss_dict = self.compute_loss(batch)
        self.log_dict(
            {k: float(v) for k, v in loss_dict.items()},
            prog_bar=True, logger=True, on_step=False, on_epoch=True,
        )
        return loss