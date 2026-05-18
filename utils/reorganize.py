import shutil
from pathlib import Path

INSTRUMENTS = ["bass", "drums", "guitar", "piano"]
SPLIT_MAP = {
    "train": "train",
    "validation": "val",
    "test": "test",
}


def reorganize_main_dataset(src_root: Path, dst_root: Path, dry_run: bool = False) -> None:
    if dry_run:
        print("Running in dry-run mode. No files will be copied, but actions will be printed.")
    
    total_copied = 0
    errors = []

    for instrument in INSTRUMENTS:
        instr_dir = src_root / f"{instrument}_22050"
        if not instr_dir.exists():
            print(f"[WARN] Missing instrument dir: {instr_dir}")
            continue

        for src_split, dst_split in SPLIT_MAP.items():
            split_dir = instr_dir / src_split
            if not split_dir.exists():
                print(f"[WARN] Missing split dir: {split_dir}")
                continue

            wav_files = sorted(split_dir.glob("*.wav"))
            if not wav_files:
                nested = split_dir / src_split
                if nested.exists():
                    wav_files = sorted(nested.glob("*.wav"))
                    if wav_files:
                        split_dir = nested
            print(f"{instrument}/{src_split} -> {dst_split}/ : {len(wav_files)} files")

            for wav in wav_files:
                track_name = wav.stem
                dst_track_dir = dst_root / dst_split / track_name
                dst_file = dst_track_dir / f"{instrument}.wav"

                if not dry_run:
                    dst_track_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(wav, dst_file)
                        total_copied += 1
                    except Exception as e:
                        errors.append((wav, dst_file, str(e)))
                else:
                    print(f"    {wav} -> {dst_file}")
                    total_copied += 1

    if errors:
        print(f"{len(errors)} errors:")
        for src, dst, err in errors:
            print(f"{src} -> {dst}: {err}")
