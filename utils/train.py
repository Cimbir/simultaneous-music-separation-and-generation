from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from modules.abstractions.diffusion_trainer import DiffusionTrainer


def run_training(
    trainer: DiffusionTrainer,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    *,
    # Stopping
    max_steps: int = 200000,
    # Hardware
    devices: list[int] | None = None,
    precision: str = "16-mixed",
    # Optimization
    gradient_clip_val: float = 1.0,
    accumulate_grad_batches: int = 1,
    # Validation
    val_check_interval: int = 2000,
    # Logging
    log_dir: str = "lightning_logs",
    experiment_name: str = "experiment",
    log_every_n_steps: int = 50,
    # Checkpointing
    save_every_n_steps: int = 5000,
    # Resume
    resume_from: Optional[str] = None,
    # Any extra PL callbacks
    extra_callbacks: list | None = None,
) -> None:
    """
    Run training using PyTorch Lightning.

    Args:
        trainer: A concrete DiffusionTrainer subclass. It is responsible for:
            - compute_loss(batch) -> (loss, metrics_dict)
            - configure_optimizers()
            - setup(stage) for device management of frozen components
            - on_save/load_checkpoint for non-nn.Module state
        train_loader: Yields batches consumed by trainer.compute_loss.
        val_loader: Optional; if None, validation is skipped.
        devices: GPU indices. None falls back to CPU.
        extra_callbacks: Additional pl.Callback instances to register.
    """
    if devices is None:
        devices = [0] if torch.cuda.is_available() else None
    accelerator = "gpu" if devices else "cpu"

    ckpt_dir = Path(log_dir) / experiment_name / "checkpoints"
    callbacks: list[pl.Callback] = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="step={step:07d}",
            every_n_train_steps=save_every_n_steps,
            save_top_k=-1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        *(extra_callbacks or []),
    ]

    pl_trainer = pl.Trainer(
        max_steps=max_steps,
        devices=devices,
        accelerator=accelerator,
        precision=precision,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=val_check_interval if val_loader is not None else 1.0,
        callbacks=callbacks,
        logger=TensorBoardLogger(log_dir, name=experiment_name),
        log_every_n_steps=log_every_n_steps,
    )

    pl_trainer.fit(
        trainer,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_from,
    )
