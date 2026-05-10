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


DEFAULT_STEM_NAMES = ("bass", "drums", "guitar", "piano")
DEFAULT_MIXTURE_NAMES = ("mixture", "mix")
DEFAULT_AUDIO_EXTENSIONS = (".flac", ".wav")
DEFAULT_STEM_ALIASES: dict[str, tuple[str, ...]] = {
    "bass": ("bass",),
    "drums": ("drums", "drum"),
    "guitar": ("guitar", "guitars"),
    "piano": ("piano", "keys", "keyboard"),
}


@dataclass(frozen=True)
class StemTrack:
    track_dir: Path
    stem_paths: tuple[Path, ...]
    mixture_path: Path | None


def _normalise_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def _iter_audio_files(root: Path, extensions: Sequence[str], max_depth: int) -> list[Path]:
    exts = {ext.lower() for ext in extensions}
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        if len(path.relative_to(root).parts) <= max_depth:
            files.append(path)
    return sorted(files)


def _find_named_audio(
    root: Path,
    names: Sequence[str],
    extensions: Sequence[str],
    max_depth: int,
) -> Path | None:
    wanted = {_normalise_name(name) for name in names}
    for path in _iter_audio_files(root, extensions, max_depth):
        if _normalise_name(path.stem) in wanted:
            return path
    return None


def _duration_seconds(path: Path) -> float:
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


def _resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
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


def _read_mono_audio(
    path: Path,
    target_sr: int,
    *,
    offset: float,
    duration: float,
) -> np.ndarray:
    if sf is not None:
        try:
            info = sf.info(path)
            start = max(0, int(round(offset * info.samplerate)))
            frames = max(1, int(math.ceil(duration * info.samplerate)))
            audio, sr = sf.read(path, start=start, frames=frames, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = np.asarray(audio, dtype=np.float32)
            return _resample(audio, sr, target_sr)
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


def _pad_or_trim_audio(audio: np.ndarray, target_samples: int) -> np.ndarray:
    if audio.shape[0] < target_samples:
        audio = np.pad(audio, (0, target_samples - audio.shape[0]))
    elif audio.shape[0] > target_samples:
        audio = audio[:target_samples]
    return audio.astype(np.float32, copy=False)


def _load_fixed_mono_clip(
    path: Path,
    target_sr: int,
    *,
    offset: float,
    duration: float,
    target_samples: int,
) -> np.ndarray:
    audio = _read_mono_audio(path, target_sr, offset=offset, duration=duration)
    return _pad_or_trim_audio(audio, target_samples)


def _fix_mel_length(mel: torch.Tensor, target_length: int) -> torch.Tensor:
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
        stem_names: Sequence[str] = DEFAULT_STEM_NAMES,
        stem_aliases: Mapping[str, Sequence[str]] | None = None,
        mixture_names: Sequence[str] = DEFAULT_MIXTURE_NAMES,
        audio_extensions: Sequence[str] = DEFAULT_AUDIO_EXTENSIONS,
        target_length: int = 1024,
        random_crop: bool = True,
        sum_stems_for_missing_mixture: bool = True,
        max_search_depth: int = 2,
        return_paths: bool = False,
    ):
        self.root_dir = Path(root_dir)
        if target_length < 1:
            raise ValueError("target_length must be >= 1")
        self.mel_extractor = mel_extractor
        self.stem_names = tuple(stem_names)
        self.stem_aliases = {
            **DEFAULT_STEM_ALIASES,
            **{k: tuple(v) for k, v in (stem_aliases or {}).items()},
        }
        self.mixture_names = tuple(mixture_names)
        self.audio_extensions = tuple(audio_extensions)
        self.target_length = int(target_length)
        self.random_crop = bool(random_crop)
        self.sum_stems_for_missing_mixture = bool(sum_stems_for_missing_mixture)
        self.max_search_depth = int(max_search_depth)
        self.return_paths = bool(return_paths)

        self.sampling_rate = int(mel_extractor.sampling_rate)
        self.hop_length = int(mel_extractor.hop_length)
        self.segment_samples = max(1, (self.target_length - 1) * self.hop_length)
        self.segment_duration = self.segment_samples / float(self.sampling_rate)
        self.tracks = self._discover_tracks()

        if not self.tracks:
            stems = ", ".join(self.stem_names)
            raise ValueError(f"No usable tracks found in {self.root_dir} with stems: {stems}")

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, index: int) -> dict:
        track = self.tracks[index]
        start = self._sample_start(track)

        stem_audio = [
            _load_fixed_mono_clip(
                path,
                self.sampling_rate,
                offset=start,
                duration=self.segment_duration,
                target_samples=self.segment_samples,
            )
            for path in track.stem_paths
        ]

        if track.mixture_path is None:
            mix_audio = np.sum(np.stack(stem_audio, axis=0), axis=0)
        else:
            mix_audio = _load_fixed_mono_clip(
                track.mixture_path,
                self.sampling_rate,
                offset=start,
                duration=self.segment_duration,
                target_samples=self.segment_samples,
            )

        fbank = self._audio_to_fixed_mel(mix_audio)
        fbank_stems = torch.stack(
            [self._audio_to_fixed_mel(audio).squeeze(0) for audio in stem_audio],
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
                    "mixture_path": str(track.mixture_path) if track.mixture_path else "",
                    "stem_paths": [str(path) for path in track.stem_paths],
                }
            )
        return item

    def _audio_to_fixed_mel(self, audio: np.ndarray) -> torch.Tensor:
        audio = np.clip(audio, -1.0, 1.0)
        mel = self.mel_extractor.audio_to_mel(audio, self.sampling_rate)
        return _fix_mel_length(mel, self.target_length)

    def _sample_start(self, track: StemTrack) -> float:
        paths = list(track.stem_paths)
        if track.mixture_path is not None:
            paths.append(track.mixture_path)
        duration = min(_duration_seconds(path) for path in paths)
        if not self.random_crop or duration <= self.segment_duration:
            return 0.0
        return random.uniform(0.0, duration - self.segment_duration)

    def _discover_tracks(self) -> list[StemTrack]:
        candidates = [self.root_dir]
        candidates.extend(path for path in sorted(self.root_dir.rglob("*")) if path.is_dir())

        tracks: list[StemTrack] = []
        for candidate in candidates:
            stem_paths: list[Path] = []
            for stem in self.stem_names:
                aliases = self.stem_aliases.get(stem, (stem,))
                path = _find_named_audio(
                    candidate,
                    aliases,
                    self.audio_extensions,
                    self.max_search_depth,
                )
                if path is None:
                    break
                stem_paths.append(path)
            else:
                mixture_path = _find_named_audio(
                    candidate,
                    self.mixture_names,
                    self.audio_extensions,
                    self.max_search_depth,
                )
                if mixture_path is not None or self.sum_stems_for_missing_mixture:
                    tracks.append(StemTrack(candidate, tuple(stem_paths), mixture_path))

        unique: list[StemTrack] = []
        seen_stems: set[tuple[Path, ...]] = set()
        tracks = sorted(
            tracks,
            key=lambda item: (item.mixture_path is not None, len(item.track_dir.parts)),
            reverse=True,
        )
        for track in tracks:
            key = tuple(sorted(track.stem_paths))
            if key in seen_stems:
                continue
            seen_stems.add(key)
            unique.append(track)
        return sorted(unique, key=lambda item: str(item.track_dir))


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


def create_train_val_dataloaders(
    train_dir: str | Path,
    val_dir: str | Path | None,
    mel_extractor: MelExtractor,
    *,
    batch_size: int = 2,
    num_workers: int = 0,
    **dataset_kwargs,
) -> tuple[DataLoader, DataLoader | None]:
    train_loader = create_stem_dataloader(
        train_dir,
        mel_extractor,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        **dataset_kwargs,
    )
    val_loader = None
    if val_dir is not None:
        val_dataset_kwargs = {**dataset_kwargs, "random_crop": False}
        val_loader = create_stem_dataloader(
            val_dir,
            mel_extractor,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            **val_dataset_kwargs,
        )
    return train_loader, val_loader
