from modules.abstractions.diffusion_trainer import DiffusionTrainer
from modules.abstractions.diffusion_model import DiffusionModel
from modules.msgld.msgld_diffusion_model import MsgLdDiffusionModel
import torch
import torch.nn as nn
from torch.optim import AdamW

class MsgLdDiffusionTrainer(DiffusionTrainer):
    """
    MSG-LD trainer loop
    """

    def __init__(
        self,
        diffusion_model: MsgLdDiffusionModel,
        learning_rate: float = 3e-5,
        warmup_steps: int = 1000,
        l_simple_weight: float = 1.0,
        original_elbo_weight: float = 0.0,
        loss_type: str = "l2",
        learn_logvar: bool = False,
    ):
        super().__init__(learning_rate=learning_rate, warmup_steps=warmup_steps)
        self.diffusion_model = diffusion_model
        self.l_simple_weight = l_simple_weight
        self.original_elbo_weight = original_elbo_weight
        self.loss_type = loss_type
        self.learn_logvar = learn_logvar

        self.denoiser = diffusion_model.model
        self.cond_stage = diffusion_model.cond_stage_model

    def setup(self, stage: str):
        device = str(self.device)
        self.diffusion_model.to(device)
        self.diffusion_model.vae.to(device)

    def on_save_checkpoint(self, checkpoint: dict):
        checkpoint["dm_scale_factor"] = self.diffusion_model.scale_factor
        checkpoint["dm_scale_computed"] = self.diffusion_model._scale_computed
        lv = self.diffusion_model.logvar
        checkpoint["dm_logvar"] = lv.data if isinstance(lv, nn.Parameter) else lv

    def on_load_checkpoint(self, checkpoint: dict):
        if "dm_scale_factor" in checkpoint:
            self.diffusion_model.scale_factor = checkpoint["dm_scale_factor"]
            self.diffusion_model._scale_computed = checkpoint.get("dm_scale_computed", True)
        if "dm_logvar" in checkpoint:
            lv = self.diffusion_model.logvar
            if isinstance(lv, nn.Parameter):
                lv.data.copy_(checkpoint["dm_logvar"])
            else:
                self.diffusion_model.logvar = checkpoint["dm_logvar"]

    def build_diffusion_model(self) -> DiffusionModel:
        return self.diffusion_model

    def _get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
        elif self.loss_type == "l2":
            if mean:
                return torch.nn.functional.mse_loss(target, pred)
            loss = torch.nn.functional.mse_loss(target, pred, reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        return loss.mean() if mean else loss

    def compute_loss(self, batch):
        dm = self.diffusion_model

        # 1. Encode stems to latent space
        z = dm.encode_to_latent(batch)

        # 2. Get conditioning
        c = dm.get_conditioning(batch)

        # 3. Sample random timestep
        t = torch.randint(0, dm.num_timesteps, (z.shape[0],), device=z.device).long()

        # 4. Forward diffusion
        noise = torch.randn_like(z)
        x_noisy = dm.q_sample(z, t, noise=noise)

        # 5. Predict noise
        model_output = dm.apply_model(x_noisy, t, c)

        # 6. Compute target
        if dm.parameterization == "eps":
            target = noise
        elif dm.parameterization == "x0":
            target = z
        else:
            raise NotImplementedError(f"parameterization={dm.parameterization}")

        # 7. Loss computation
        prefix = "train" if self.training else "val"
        loss_dict = {}

        loss_simple = self._get_loss(model_output, target, mean=False).mean([1, 2, 3, 4])
        loss_dict[f"{prefix}/loss_simple"] = loss_simple.mean()

        logvar_t = dm.logvar[t].to(z.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t

        if self.learn_logvar:
            loss_dict[f"{prefix}/loss_gamma"] = loss.mean()
            loss_dict["logvar"] = dm.logvar.data.mean()

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self._get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3, 4))
        loss_vlb = (dm.lvlb_weights[t].to(z.device) * loss_vlb).mean()
        loss_dict[f"{prefix}/loss_vlb"] = loss_vlb

        loss += self.original_elbo_weight * loss_vlb
        loss_dict[f"{prefix}/loss"] = loss

        return loss, loss_dict

    def configure_optimizers(self):
        params = list(self.diffusion_model.parameters())
        if self.learn_logvar and isinstance(self.diffusion_model.logvar, torch.nn.Parameter):
            params.append(self.diffusion_model.logvar)
        return AdamW(params, lr=self.learning_rate)