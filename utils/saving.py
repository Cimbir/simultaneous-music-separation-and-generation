import numpy as np
import soundfile as sf
from pathlib import Path
from modules.abstractions.vae import VAE
from modules.abstractions.extractor import MelExtractor
import torch
from utils.latent import encode_audio
from typing import List, Tuple
import os

STEM_NAMES = ["bass", "drums", "guitar", "piano"]


def save_encoded(enc: dict, path: str):
    np.savez_compressed(
        path,
        mean=enc["mean"].numpy(),
        logvar=enc["logvar"].numpy(),
        z_sample=enc["z_sample"].numpy(),
        mel=enc["mel"].numpy(),
    )
    print(f"Saved encoded latent to {path}")


def load_encoded(path: str) -> dict:
    data = np.load(path)
    return {
        "mean": torch.from_numpy(data["mean"]),
        "logvar": torch.from_numpy(data["logvar"]),
        "z_sample": torch.from_numpy(data["z_sample"]),
        "mel": torch.from_numpy(data["mel"]),
    }


def encode_and_save_dataset(
    audio_dir_path: str,
    output_dir_path: str,
    mel_extractor: MelExtractor,
    vae: VAE,
    target_length: int = 1024,
    glob_pattern: str = "**/*.wav",
):
    audio_dir = Path(audio_dir_path)
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(audio_dir.glob(glob_pattern))
    print(f"Found {len(wav_files)} wav files in {audio_dir}")

    for wav_path in wav_files:
        rel = wav_path.relative_to(audio_dir)
        out_path = output_dir / rel.with_suffix(".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            continue

        audio, file_sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        enc = encode_audio(audio, file_sr, mel_extractor, vae, target_length)
        save_encoded(enc, str(out_path))

    print(f"Done. Encoded latents saved to {output_dir}")
    
    
def save_sep_originals(
    mix_clips: List[np.ndarray],
    original_stems_batch: List[List[np.ndarray]],
    mel_extractor: MelExtractor,
    originals_path: str,
    start_from: int = 0,
):
    for sample_idx, (mix_clip, stems_clip) in enumerate(zip(mix_clips, original_stems_batch)):
        sample_dir = os.path.join(originals_path, f"sample_{start_from + sample_idx:04d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        mix_path = os.path.join(sample_dir, "mixture.wav")
        sf.write(mix_path, mix_clip, mel_extractor.sampling_rate)
        
        mix_mel = mel_extractor.extract_mel(mix_clip)
        np.save(os.path.join(sample_dir, "mixture_mel.npy"), mix_mel)
        
        for stem_idx, (stem_audio, stem_name) in enumerate(zip(stems_clip, STEM_NAMES)):
            stem_path = os.path.join(sample_dir, f"{stem_name}.wav")
            sf.write(stem_path, stem_audio, mel_extractor.sampling_rate)
            
            stem_mel = mel_extractor.extract_mel(stem_audio)
            np.save(os.path.join(sample_dir, f"{stem_name}_mel.npy"), stem_mel)
        
        print(f"Saved originals for sample {start_from + sample_idx:04d}")
        
def load_sep_originals(
    sample_dir: str,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    mix_mel = np.load(os.path.join(sample_dir, "mixture_mel.npy"))
    stem_mels = []
    for stem_name in STEM_NAMES:
        stem_mel_path = os.path.join(sample_dir, f"{stem_name}_mel.npy")
        if os.path.exists(stem_mel_path):
            stem_mels.append(np.load(stem_mel_path))
        else:
            print(f"Warning: missing mel for {stem_name} in {sample_dir}")
            stem_mels.append(None)
    return mix_mel, stem_mels