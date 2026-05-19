import json
from pathlib import Path

import numpy as np
from datasets import Audio, load_dataset
from tqdm import tqdm

from audio_preprocessing import process_and_save
from config import AudioConfig, DataConfig

DATASETS_DIR = Path("./datasets")


def _write_index(cache: Path, subdir: str) -> None:
    files = sorted(str(p) for p in (cache / subdir).glob("*.npz"))
    (cache / f"{subdir}_index.json").write_text(json.dumps(files))


def _local_parquet_files(root: Path, pattern: str) -> list[str]:
    return sorted(str(p) for p in root.glob(pattern))


def _stream_parquet(files: list[str], target_sr: int):
    ds = load_dataset(
        "parquet", data_files={"train": files}, split="train", streaming=True
    )
    return ds.cast_column("audio", Audio(sampling_rate=target_sr))


def process_nsynth(
    parquet_root: Path,
    parquet_glob: str,
    out_dir: Path,
    target_sr: int,
    budget_s: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _local_parquet_files(parquet_root, parquet_glob)
    if not files:
        raise FileNotFoundError(f"No NSynth parquet files under {parquet_root}")

    ds = _stream_parquet(files, target_sr)

    budget = {"remaining_s": budget_s}
    pbar = tqdm(total=int(budget_s), unit="s", desc="NSynth")
    # NSynth notes are 4s. Filter for vocal-range pitches (C2..C6).
    for i, ex in enumerate(ds):
        if budget["remaining_s"] <= 0:
            break
        pitch = ex.get("pitch", 60)
        if pitch < 36 or pitch > 84:
            continue
        wav = ex["audio"]["array"].astype(np.float32)
        src_sr = ex["audio"]["sampling_rate"]
        sec = process_and_save(wav, src_sr, out_dir, f"n_{i:07d}", budget)
        pbar.update(sec)
    pbar.close()


def process_vctk(
    parquet_root: Path,
    parquet_glob: str,
    out_dir: Path,
    target_sr: int,
    budget_s: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _local_parquet_files(parquet_root, parquet_glob)
    if not files:
        raise FileNotFoundError(f"No VCTK parquet files under {parquet_root}")

    ds = _stream_parquet(files, target_sr)

    budget = {"remaining_s": budget_s}
    pbar = tqdm(total=int(budget_s), unit="s", desc="VCTK")
    for i, ex in enumerate(ds):
        if budget["remaining_s"] <= 0:
            break
        wav = ex["audio"]["array"].astype(np.float32)
        src_sr = ex["audio"]["sampling_rate"]
        spk = ex.get("speaker_id", f"unk_{i}")
        sec = process_and_save(wav, src_sr, out_dir, f"v_{spk}_{i:06d}", budget)
        pbar.update(sec)
    pbar.close()


def process_opensinger(
    parquet_root: Path,
    parquet_glob: str,
    out_dir: Path,
    target_sr: int,
    budget_s: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _local_parquet_files(parquet_root, parquet_glob)
    if not files:
        raise FileNotFoundError(f"No process_singer parquet files under {parquet_root}")

    ds = _stream_parquet(files, target_sr)

    budget = {"remaining_s": budget_s}
    pbar = tqdm(total=int(budget_s), unit="s", desc="VCTK")
    for i, ex in enumerate(ds):
        if budget["remaining_s"] <= 0:
            break
        wav = ex["audio"]["array"].astype(np.float32)
        src_sr = ex["audio"]["sampling_rate"]
        spk = ex.get("id", f"unk_{i}")
        gender = ex.get("gender", "")
        sec = process_and_save(
            wav, src_sr, out_dir, f"o_{gender}_{spk}_{i:06d}", budget
        )
        pbar.update(sec)
    pbar.close()


PROCESSORS = {
    "opensinger_male": process_opensinger,
    "nsynth": process_nsynth,
    "vctk": process_vctk,
    "opensinger_female": process_opensinger,
}


def main() -> None:
    data_cfg = DataConfig()
    audio_cfg = AudioConfig()

    cache = Path(data_cfg.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    print("Processing datasets...")
    for key in data_cfg.datasets_to_load:
        spec = data_cfg.datasets[key]
        PROCESSORS[key](
            DATASETS_DIR / spec.subdir,
            spec.parquet_glob,
            cache / spec.subdir,
            audio_cfg.target_sr,
            spec.length_hr * 3600.0,
        )
        _write_index(cache, spec.subdir)


if __name__ == "__main__":
    main()
