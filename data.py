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

BINS_PER_OCT = audio_cfg.n_mels / math.log2(audio_cfg.fmax / max(audio_cfg.fmin, 20.0))
BINS_PER_SEMITONE = BINS_PER_OCT / 12.0


def mel_pitch_shift(mel: torch.Tensor, semitones: float) -> torch.Tensor:
    n_bin = round(BINS_PER_SEMITONE * semitones)
    if n_bin == 0:
        return mel

    out = torch.roll(mel, shifts=n_bin, dims=-2)
    if n_bin > 0:
        out[:n_bin] = mel.min()
    else:
        out[n_bin:] = mel.min()
    return out


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
