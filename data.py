import json
import random
import math
import numpy as np
import torch
from typing import List
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from config import DataConfig, TrainConfig, AudioConfig

data_cfg = DataConfig()
audio_cfg = AudioConfig()
train_cfg = TrainConfig()


def _mel_centers_hz(n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    # HTK mel scale — matches torchaudio.transforms.MelSpectrogram default.
    mel_min = 2595.0 * math.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * math.log10(1.0 + fmax / 700.0)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    freqs = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    return freqs[1:-1]


_MEL_CENTERS = _mel_centers_hz(
    audio_cfg.n_mels, float(audio_cfg.fmin), float(audio_cfg.fmax)
)


def mel_pitch_shift(mel: torch.Tensor, semitones: float) -> torch.Tensor:
    if semitones == 0:
        return mel

    n_mels = mel.shape[-2]
    assert n_mels == _MEL_CENTERS.shape[0], (
        f"mel.shape[-2]={n_mels} doesn't match AudioConfig.n_mels={_MEL_CENTERS.shape[0]}"
    )

    factor = 2.0 ** (-semitones / 12.0)
    src_freqs = _MEL_CENTERS * factor

    idx_hi = np.searchsorted(_MEL_CENTERS, src_freqs)
    idx_lo = np.clip(idx_hi - 1, 0, n_mels - 1)
    idx_hi = np.clip(idx_hi, 0, n_mels - 1)
    pick_hi = np.abs(_MEL_CENTERS[idx_hi] - src_freqs) <= np.abs(
        _MEL_CENTERS[idx_lo] - src_freqs
    )
    src_idx = np.where(pick_hi, idx_hi, idx_lo)
    in_range = (src_freqs >= _MEL_CENTERS[0]) & (src_freqs <= _MEL_CENTERS[-1])

    src_idx_t = torch.from_numpy(src_idx).long().to(mel.device)
    in_range_t = torch.from_numpy(in_range).to(mel.device)
    floor = mel.min()

    out = mel.index_select(-2, src_idx_t)
    mask_shape = [1] * (out.ndim - 2) + [n_mels, 1]
    return torch.where(in_range_t.view(*mask_shape), out, floor)


class PitchDataset(Dataset):
    def __init__(
        self, split: str = "train", subdirs: List[str] = data_cfg.datasets_to_load
    ):
        files = []
        for subdir in subdirs:
            with open(Path(data_cfg.cache_dir) / f"{subdir}_index.json") as f:
                files.extend(json.load(f))

        random.Random(0).shuffle(files)
        n_val = int(len(files) * data_cfg.val_split)
        self.files = files[n_val:] if split == "train" else files[:n_val]
        self.perturb_st = train_cfg.perturb_st
        self.is_train = split == "train"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        d = np.load(self.files[index])
        mel = torch.from_numpy(d["mel"])
        f0 = torch.from_numpy(d["f0"])

        # Self-Supervised perturbation
        if self.is_train:
            semis = random.uniform(-self.perturb_st, self.perturb_st)
        else:
            semis = 0.0

        mel_in = mel_pitch_shift(mel, semis)

        return {
            "mel_in": mel_in.unsqueeze(0),  # [1, 80, T]
            "mel_tgt": mel.unsqueeze(0),  # [1, 80, T]
            "f0": f0,  # [T]
            "pitch": -semis,
        }

    @staticmethod
    def make_loader(split="train"):
        ds = PitchDataset(split)
        return DataLoader(
            ds,
            batch_size=train_cfg.batch_size,
            shuffle=(split == "train"),
            num_workers=train_cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )
