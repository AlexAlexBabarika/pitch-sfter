import json
import random
import math
import numpy as np
import torch
import torchaudio
import librosa
from typing import List
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from config import DataConfig, TrainConfig, AudioConfig

data_cfg = DataConfig()
audio_cfg = AudioConfig()
train_cfg = TrainConfig()

_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=audio_cfg.target_sr,
    n_fft=audio_cfg.n_fft,
    hop_length=audio_cfg.hop_length,
    win_length=audio_cfg.win_length,
    n_mels=audio_cfg.n_mels,
    f_min=audio_cfg.fmin,
    f_max=audio_cfg.fmax,
    power=1.0,
    center=True,
)


def _audio_to_log_mel(audio: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(audio).float().unsqueeze(0)
    m = _mel(t).squeeze(0)
    return torch.log(m.clamp(min=1e-5))


class PitchDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        dataset_keys: List[str] = data_cfg.datasets_to_load,
    ):
        files = []
        for key in dataset_keys:
            spec = data_cfg.datasets[key]
            with open(Path(data_cfg.cache_dir) / f"{spec.subdir}_index.json") as f:
                files.extend(json.load(f))

        random.Random(0).shuffle(files)
        n_val = int(len(files) * data_cfg.val_split)
        self.files = files[n_val:] if split == "train" else files[:n_val]
        self.perturb_st = train_cfg.perturb_st
        self.is_train = split == "train"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with np.load(self.files[index]) as d:
            audio = d["audio"].astype(np.float32, copy=True)
            mel_tgt = torch.from_numpy(d["mel"].copy())  # log-mel of the original clip
            f0_raw = torch.from_numpy(d["f0"].copy())  # Hz, aligned to mel frames
            conf = torch.from_numpy(d["conf"].copy())

        if self.is_train:
            semis = random.uniform(-self.perturb_st, self.perturb_st)
        else:
            rng = random.Random(index)
            semis = rng.uniform(-self.perturb_st, self.perturb_st)
            if abs(semis) < 0.5:
                semis = 0.5 if semis >= 0 else -0.5

        # Real audio-domain pitch shift, then recompute mel — mel_in
        # carries the artifact profile of a phase-vocoder shift rather than a
        # mel-bin warp.
        if abs(semis) > 1e-4:
            audio_shifted = librosa.effects.pitch_shift(
                audio, sr=audio_cfg.target_sr, n_steps=semis
            )
        else:
            audio_shifted = audio
        mel_in = _audio_to_log_mel(audio_shifted)

        # pitch_shift can leave a +-1 frame offset after mel; clamp to target.
        T = mel_tgt.shape[-1]
        if mel_in.shape[-1] > T:
            mel_in = mel_in[..., :T]
        elif mel_in.shape[-1] < T:
            mel_in = torch.nn.functional.pad(
                mel_in, (0, T - mel_in.shape[-1]), value=math.log(1e-5)
            )

        f0_hz_in = f0_raw * (2.0 ** (semis / 12.0))
        voiced = (conf > 0.5).to(f0_raw.dtype)

        # Drop voicing when the shifted fundamental leaves the analysable
        # range (no harmonics left in the mel band).
        in_range = f0_hz_in < (audio_cfg.fmax / 2.0)
        voiced = voiced * in_range.to(voiced.dtype)
        f0_norm = torch.log2(f0_hz_in.clamp(min=0.0) + 1.0)
        f0_feat = torch.stack([f0_norm, voiced], dim=0)  # [2, T]

        return {
            "mel_in": mel_in.unsqueeze(0),  # [1, 80, T]
            "mel_tgt": mel_tgt.unsqueeze(0),  # [1, 80, T]
            "f0": f0_feat,  # [2, T]
            "shift": torch.tensor(-semis, dtype=torch.float32),
        }

    @staticmethod
    def make_loader(split="train"):
        ds = PitchDataset(split)
        is_train = split == "train"
        nw = train_cfg.num_workers if is_train else min(2, train_cfg.num_workers)
        return DataLoader(
            ds,
            batch_size=train_cfg.batch_size,
            shuffle=is_train,
            num_workers=nw,
            pin_memory=True,
            drop_last=is_train,
            persistent_workers=(nw > 0),
            multiprocessing_context="spawn" if nw > 0 else None,
        )
