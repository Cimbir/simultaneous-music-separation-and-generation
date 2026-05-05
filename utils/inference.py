from latent_diffusion.models.ddim import DDIMSampler
import torch
import soundfile as sf
from pathlib import Path

STEM_NAMES = ["bass", "drums", "guitar", "piano"]


@torch.no_grad()
def _run_ddim(
    diffusion_model,
    conditioning, # (B, num_stems, z_ch, T, F) - mixture VAE latent replicated per stem
    *,
    ddim_steps=200,
    ddim_eta=1.0, # 0 = deterministic, 1 = full stochastic (DDPM-like)
    cfg_scale=1.0, # 1.0 = no guidance, >1 amplifies conditioning vs zero baseline
    batch_size=1,
):
    """
    Core DDIM sampling.

    Returns: samples (B, num_stems, z_channels, T, F) in scaled latent space.
    """
    dm = diffusion_model
    shape = (dm.num_stems, dm.z_channels, dm.latent_t_size, dm.latent_f_size)
    uncond = dm.cond_stage_model.get_unconditional_condition(batch_size)

    sampler = DDIMSampler(dm)
    samples, _ = sampler.sample(
        S=ddim_steps,
        batch_size=batch_size,
        shape=shape,
        conditioning=conditioning,
        eta=ddim_eta,
        verbose=False,
        unconditional_guidance_scale=cfg_scale,
        unconditional_conditioning=uncond if cfg_scale != 1.0 else None,
    )
    return samples


@torch.no_grad()
def _decode_samples(samples, dm, vae, vocoder):
    """
    Decode DDIM latents -> per-stem audio.

    Flow:
      (B, S, C, T, F)  --[/ scale_factor]--> reshape (B*S, C, T, F)
        --[VAE decode]--> mel  (B*S, 1, n_mels, T_mel)
        --[HiFiGAN]   --> wav  (B*S, num_samples) int16
        --[reshape]   --> (B, S, num_samples)

    Returns:
        audio : int16 numpy   (B, num_stems, num_samples)
        mels  : float tensor  (B, num_stems, 1, n_mels, T_mel)
    """
    B, S, C, T, F = samples.shape

    z = (samples / dm.scale_factor).reshape(B * S, C, T, F)
    mels = vae.decode(z)
    wav = vocoder.mel_to_audio(mels.squeeze(1))

    audio = wav.reshape(B, S, -1)
    mels = mels.reshape(B, S, *mels.shape[1:])
    return audio, mels


@torch.no_grad()
def generate_stems(
    dm,
    vae,
    vocoder,
    *,
    ddim_steps=200,
    ddim_eta=1.0,
):
    """
    Generate all 4 stems from scratch (unconditional).

    Conditions on an all-zero latent — the null token the model saw during
    training with unconditional_prob > 0 and in the paper's zero-conditioning mode.

    Returns:
        audio : int16 numpy (num_stems, num_samples)
        mels : float tensor (num_stems, 1, n_mels, T_mel)
    """
    dm.eval()
    cond = dm.cond_stage_model.get_unconditional_condition(1) # zeros (1, S, C, T, F)
    samples = _run_ddim(dm, cond, ddim_steps=ddim_steps, ddim_eta=ddim_eta, cfg_scale=1.0)
    audio, mels = _decode_samples(samples, dm, vae, vocoder)
    return audio[0], mels[0]


@torch.no_grad()
def separate_mixture(
    mixture_audio,
    sr,
    dm,
    vae,
    vocoder,
    mel_extractor,
    *,
    ddim_steps=200,
    ddim_eta=1.0,
    cfg_scale=3.0,
    target_length=1024,
):
    """
    Separate a mixture audio clip into individual stems using MSG-LD.

    1. Encode mixture via VAE -> replicate latent across stems as conditioning.
    2. Reverse-diffuse from Gaussian noise guided by that mixture latent (DDIM + CFG).
    3. Decode each stem latent via VAE + HiFiGAN.

    Args:
        mixture_audio : 1-D float32 numpy array in [-1, 1], any length
        sr : sample rate
        cfg_scale : guidance strength; 1 = none, 3-5 = recommended for separation
        target_length : mel frames to use — must match latent_t_size * VAE time stride (= 1024 for default config at 16 kHz, hop=160)

    Returns:
        audio : int16 numpy   (num_stems, num_samples)
        mels  : float tensor  (num_stems, 1, n_mels, T_mel)
    """
    dm.eval()

    # 1. Build conditioning from mixture
    mel = mel_extractor.audio_to_mel(mixture_audio, sr)   # (1, n_mels, T)
    T = mel.shape[-1]
    if T < target_length:
        mel = torch.nn.functional.pad(mel, (0, target_length - T))
    elif T > target_length:
        mel = mel[..., :target_length]

    cond = dm.get_conditioning({"fbank": mel.unsqueeze(0)})

    # 2. DDIM reverse diffusion
    samples = _run_ddim(
        dm, cond,
        ddim_steps=ddim_steps,
        ddim_eta=ddim_eta,
        cfg_scale=cfg_scale,
    )

    # 3. Decode
    audio, mels = _decode_samples(samples, dm, vae, vocoder)
    return audio[0], mels[0]


# def save_stems(audio, sr, out_dir, prefix="stem"):
#     """Write per-stem wav files.  audio: (num_stems, num_samples) int16."""
#     out = Path(out_dir)
#     out.mkdir(parents=True, exist_ok=True)
#     for i, name in enumerate(STEM_NAMES):
#         path = out / f"{prefix}_{name}.wav"
#         sf.write(str(path), audio[i].astype("float32") / 32768.0, sr)
#         print(f"  Saved {path}")
