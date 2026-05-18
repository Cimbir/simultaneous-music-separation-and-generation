from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from modules.abstractions.extractor import MelExtractor
import librosa
import soundfile as sf
from IPython.display import Audio, display


STEM_NAMES = ("bass", "drums", "guitar", "piano")

@dataclass(frozen=True)
class StemTrack:
    track_dir: Path
    stem_paths: dict[str, Path]


def duration_seconds(path: Path) -> float:
    """
    Get the duration of the audio in seconds

    Args:
        path (Path): Path to the audio file

    Raises:
        ImportError: If neither soundfile nor librosa is available or if the file format is unsupported

    Returns:
        float: Duration of the audio in seconds
    """
    if sf is not None:
        try:
            info = sf.info(path)
            return float(info.frames) / float(info.samplerate)
        except (RuntimeError, OSError):
            pass

    if librosa is not None:
        try:
            return float(librosa.get_duration(path=str(path)))
        except TypeError:
            return float(librosa.get_duration(filename=str(path)))
    raise ImportError(f"Reading {path.suffix} audio requires soundfile or librosa")


def resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """
    Resample the audio to the target sampling rate

    Args:
        audio (np.ndarray): Audio data as a 1D numpy array
        sr (int): Original sampling rate of the audio
        target_sr (int): Target sampling rate to resample the audio to

    Returns:
        np.ndarray: Resampled audio data as a 1D numpy array with the target sampling rate
    """
    if audio.shape[0] == 0:
        return audio.astype(np.float32, copy=False)
    if sr == target_sr:
        return audio.astype(np.float32, copy=False)
    if librosa is not None:
        return librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)

    target_len = max(1, int(round(audio.shape[0] * target_sr / sr)))
    source_x = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    target_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(target_x, source_x, audio).astype(np.float32)


def read_audio_segment(
    path: Path,
    target_sr: int,
    *,
    offset: float,
    duration: float,
) -> np.ndarray:
    """
    Read the audio from offset for duration

    Args:
        path (Path): Path to the audio file
        target_sr (int): Target sampling rate to resample the audio to
        offset (float): Offset in seconds to start reading the audio from
        duration (float): Duration in seconds to read the audio for

    Raises:
        ImportError: If neither soundfile nor librosa is available or if the file format is unsupported

    Returns:
        np.ndarray: Audio data as a 1D numpy array with the target sampling rate
    """
    if sf is not None:
        try:
            info = sf.info(path)
            start = max(0, int(round(offset * info.samplerate)))
            frames = max(1, int(math.ceil(duration * info.samplerate)))
            audio, sr = sf.read(path, start=start, frames=frames, dtype="float32")
            if audio.ndim > 1:
                # If stereo
                audio = audio.mean(axis=1)
            audio = np.asarray(audio, dtype=np.float32)
            return resample(audio, sr, target_sr)
        except (RuntimeError, OSError):
            pass

    if librosa is not None:
        audio, sr = librosa.load(
            str(path),
            sr=target_sr,
            mono=True,
            offset=max(0.0, offset),
            duration=duration,
        )
        return np.asarray(audio, dtype=np.float32)

    raise ImportError(f"Reading {path.suffix} audio requires soundfile or librosa")


def pad_or_trim_audio(audio: np.ndarray, target_samples: int) -> np.ndarray:
    """
    Pad or trim the audio to the target number of samples

    Args:
        audio (np.ndarray): Audio data as a 1D numpy array
        target_samples (int): Target number of samples

    Returns:
        np.ndarray: Audio data as a 1D numpy array with the target number of samples
    """
    if audio.shape[0] < target_samples:
        audio = np.pad(audio, (0, target_samples - audio.shape[0]))
    elif audio.shape[0] > target_samples:
        audio = audio[:target_samples]
    return audio.astype(np.float32, copy=False)


def read_fixed_audio_segment(
    path: Path,
    target_sr: int,
    *,
    offset: float,
    duration: float,
    target_samples: int,
) -> np.ndarray:
    """
    Read the audio from offset for duration and return the target number of samples

    Args:
        path (Path): Path to the audio file
        target_sr (int): Target sampling rate to resample the audio to
        offset (float): Offset in seconds to start reading the audio from
        duration (float): Duration in seconds to read the audio for
        target_samples (int): Target number of samples to return after padding or trimming the audio

    Returns:
        np.ndarray: Audio data as a 1D numpy array with the target sampling rate and target number of samples
    """
    audio = read_audio_segment(path, target_sr, offset=offset, duration=duration)
    return pad_or_trim_audio(audio, target_samples)


def fix_mel_length(mel: torch.Tensor, target_length: int) -> torch.Tensor:
    if mel.shape[-1] < target_length:
        return F.pad(mel, (0, target_length - mel.shape[-1]))
    if mel.shape[-1] > target_length:
        return mel[..., :target_length]
    return mel


class StemAudioDataset(Dataset):
    """
    Each item returns:
        fbank: (1, n_mels, target_length)
        fbank_stems: (num_stems, n_mels, target_length)
    """

    def __init__(
        self,
        root_dir: str | Path,
        mel_extractor: MelExtractor,
        *,
        stem_names: Sequence[str] = STEM_NAMES,
        target_length: int = 1024,
        sum_stems_for_missing_mixture: bool = True,
        max_search_depth: int = 2,
        return_paths: bool = False,
    ):
        assert target_length >= 1, "target_length must be >= 1"

        self.root_dir = Path(root_dir)
        self.mel_extractor = mel_extractor
        self.stem_names = tuple(stem_names)
        self.target_length = int(target_length)
        self.sum_stems_for_missing_mixture = bool(sum_stems_for_missing_mixture)
        self.max_search_depth = int(max_search_depth)
        self.return_paths = bool(return_paths)

        self.sampling_rate = int(mel_extractor.sampling_rate)
        self.hop_length = int(mel_extractor.hop_length)
        self.segment_samples = max(1, (self.target_length - 1) * self.hop_length)
        self.segment_duration = self.segment_samples / float(self.sampling_rate)
        
        self.tracks = self._discover_tracks()
        assert len(self.tracks) != 0, "No tracks were discovered"

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, index: int) -> dict:
        track = self.tracks[index]
        start = self._get_start(track)

        stem_audio = [
            read_fixed_audio_segment(
                track.stem_paths[stem],
                self.sampling_rate,
                offset=start,
                duration=self.segment_duration,
                target_samples=self.segment_samples,
            ) 
            if stem in track.stem_paths 
            else np.zeros(self.segment_samples, dtype=np.float32)
            for stem in self.stem_names
        ]

        mix_audio = np.sum(np.stack(stem_audio, axis=0), axis=0)

        fbank = self._audio_to_mel(mix_audio)
        fbank_stems = torch.stack(
            [self._audio_to_mel(audio).squeeze(0) for audio in stem_audio],
            dim=0,
        )
        
        item = {
            "fbank": fbank.float(),
            "fbank_stems": fbank_stems.float(),
        }
        if self.return_paths:
            item.update(
                {
                    "track": str(track.track_dir),
                    "stem_paths": [str(track.stem_paths[stem]) for stem in self.stem_names if stem in track.stem_paths],
                }
            )
        return item

    def _audio_to_mel(self, audio: np.ndarray) -> torch.Tensor:
        audio = np.clip(audio, -1.0, 1.0)
        mel = self.mel_extractor.audio_to_mel(audio, self.sampling_rate)
        return fix_mel_length(mel, self.target_length)

    def _get_start(self, track: StemTrack) -> float:
        paths = list(track.stem_paths.values())
        if not paths:  # No stems available
            return 0.0
        duration = min(duration_seconds(path) for path in paths)
        if duration <= self.segment_duration:
            return 0.0
        return random.uniform(0.0, duration - self.segment_duration)

    def _discover_tracks(self) -> list[StemTrack]:
        tracks: list[StemTrack] = []
        for track_dir in sorted(self.root_dir.iterdir()):
            stem_paths: dict[str, Path] = {}
            
            for stem in self.stem_names:
                stem_path_name = track_dir / f"{stem}.wav"
                if stem_path_name.exists():
                    stem_paths[stem] = stem_path_name
            
            tracks.append(StemTrack(track_dir=track_dir, stem_paths=stem_paths))   
            
        return tracks


def create_stem_dataloader(
    root_dir: str | Path,
    mel_extractor: MelExtractor,
    *,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    pin_memory: bool | None = None,
    **dataset_kwargs,
) -> DataLoader:
    dataset = StemAudioDataset(root_dir, mel_extractor, **dataset_kwargs)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
