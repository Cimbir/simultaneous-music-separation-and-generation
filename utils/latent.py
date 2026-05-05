import numpy as np
import torch
from modules.abstractions.extractor import MelExtractor
from modules.abstractions.vocoder import Vocoder
from modules.abstractions.vae import VAE

# audio -> codec.encode -> z
def encode_audio(
    audio: np.ndarray,
    sr: int,
    extractor: MelExtractor,
    vae: VAE,
    target_length: int = 1024,
) -> dict:
    """
    Full encoding pipeline: audio -> latent distribution params + sample.
    
    Returns dict with keys: mean, logvar, z_sample, mel (all torch tensors).
    """
    # 1. Audio -> mel-spectrogram
    mel = extractor.audio_to_mel(audio, sr)  # (1, n_mels, T')

    # 2. Pad/crop time dimension to target_length
    T = mel.shape[-1]
    if T < target_length:
        print(f"Padding mel from {T} to {target_length} frames")
        mel = torch.nn.functional.pad(mel, (0, target_length - T))
    elif T > target_length:
        print(f"Cropping mel from {T} to {target_length} frames")
        mel = mel[:, :, :target_length]

    # 3. Add batch dimension for VAE: (1, n_mels, T) -> (B, 1, n_mels, T)
    mel_input = mel.unsqueeze(0)  # (1, 1, n_mels, T)

    # 4. Encode
    enc = vae.encode(mel_input)

    # 5. Sample from posterior to get latent z
    mean, logvar = enc["mean"], enc["logvar"]
    std = torch.exp(0.5 * logvar)
    z_sample = mean + std * torch.randn_like(std)

    return {
        "mean": mean.cpu(),
        "logvar": logvar.cpu(),
        "z_sample": z_sample.cpu(),
        "mel": mel.cpu(),               # (1, n_mels, T') before VAE
        "mel_input": mel_input.cpu(),   # (1, 1, n_mels, T') as fed to VAE
    }

# z -> codec.decode -> audio
def decode_latent(
    z: torch.Tensor,
    vae: VAE,
    vocoder: Vocoder,
) -> dict:
    """
    Full decoding pipeline: latent sample -> mel -> audio.
    
    Args:
        z: (B, C, H, W) latent tensor.
    Returns:
        dict with keys: mel_recon (B, 1, n_mels, T), audio (int16 numpy).
    """
    # 1. Decode latent -> mel
    mel_recon = vae.decode(z)  # (B, 1, n_mels, T)

    # 2. Mel -> audio via vocoder
    mel_for_vocoder = mel_recon.squeeze(1)  # (B, n_mels, T)
    audio = vocoder.mel_to_audio(mel_for_vocoder)

    return {
        "mel_recon": mel_recon.cpu(),
        "audio": audio,
    }