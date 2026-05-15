import pyloudnorm
import librosa
import torchaudio
import torch
import numpy as np
import crepe
import math
from typing import Generator
from config import AudioConfig
from pathlib import Path

cfg = AudioConfig()

SUPPORTED_SRS = [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000]
CREPE_MODEL_CAPACITIES = ["tiny", "small", "medium", "large", "full"]
CLIP_LEN = int(cfg.clip_seconds * cfg.target_sr)

meter = pyloudnorm.Meter(rate=cfg.target_sr)
mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=cfg.target_sr,
    n_fft=cfg.n_fft,
    hop_length=cfg.hop_length,
    win_length=cfg.win_length,
    n_mels=cfg.n_mels,
    f_min=cfg.fmin,
    f_max=cfg.fmax,
    power=1.0,
    center=True,
)


def load_audio_file(file_path: str) -> tuple[np.ndarray, int | float]:
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr


def resample_audio_to_target_sr(
    audio: np.ndarray, original_sr: float | int
) -> np.ndarray:
    if cfg.target_sr not in SUPPORTED_SRS:
        raise ValueError(
            f"The target rate of {cfg.target_sr} isn't supported. Supported rates:\n{SUPPORTED_SRS}"
        )
    return librosa.resample(audio, orig_sr=original_sr, target_sr=cfg.target_sr)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    try:
        loud = meter.integrated_loudness(audio)
        if math.isfinite(loud):
            audio = pyloudnorm.normalize.loudness(audio, loud, cfg.target_lufs)
    except Exception:
        pass
    peak = np.max(np.abs(audio)) + 1e-9
    if peak > 0.99:
        audio = audio * (0.99 / peak)
    return audio.astype(np.float32)


def segment_audio(audio: np.ndarray) -> Generator[np.ndarray]:
    n = len(audio)
    for start in range(0, n, CLIP_LEN):
        clip = audio[start : start + CLIP_LEN]
        if len(clip) < CLIP_LEN // 2:
            continue
        if len(clip) < CLIP_LEN:
            clip = np.pad(clip, (0, CLIP_LEN - len(clip)))
        yield clip


def compute_mel_spectrogram(
    audio: np.ndarray,
) -> np.ndarray:
    t = torch.from_numpy(audio).float().unsqueeze(0)
    m = mel(t).squeeze(0).numpy()

    return np.log(np.maximum(m, 1e-5)).astype(np.float32)


def extract_f0(
    audio: np.ndarray,
):
    if cfg.crepe_model_capacity not in CREPE_MODEL_CAPACITIES:
        raise ValueError(
            f"Model capacity {cfg.crepe_model_capacity} isn't supported by CREPE. Supported capacities:\n{CREPE_MODEL_CAPACITIES}"
        )

    w16 = librosa.resample(audio, orig_sr=cfg.target_sr, target_sr=16000)
    time, freq, conf, _ = crepe.predict(
        w16,
        16000,
        viterbi=True,
        step_size=1000 * cfg.hop_length / cfg.target_sr,
        verbose=0,
        model_capacity=cfg.crepe_model_capacity,
    )

    return time, freq, conf


def process_and_save(
    audio: np.ndarray, src_sr: int, out_dir: Path, base_id: str, budget: dict
) -> float:
    if budget["remaining_s"] <= 0:
        return 0.0
    if src_sr != cfg.target_sr:
        audio = resample_audio_to_target_sr(audio, src_sr)
    audio, _ = librosa.effects.trim(audio, top_db=30)
    if len(audio) < cfg.target_lufs * 0.5:
        return 0.0
    audio = normalize_audio(audio)

    written = 0.0
    for i, clip in enumerate(segment_audio(audio)):
        if budget["remaining_s"] <= 0:
            break
        mel = compute_mel_spectrogram(clip)  # [80, T]
        print("computed mel")
        f0 = extract_f0(clip)  # [T_f0]
        print("extracted f0")
        # align F0 length to mel length
        if len(f0) != mel.shape[1]:
            f0 = np.interp(
                np.linspace(0, 1, mel.shape[1]),
                np.linspace(0, 1, len(f0)),
                f0,
            ).astype(np.float32)
        out_path = out_dir / f"{base_id}_{i:04d}.npz"
        np.savez(out_path, mel=mel, f0=f0)
        seconds = CLIP_LEN / cfg.target_sr
        written += seconds
        budget["remaining_s"] -= seconds
    return written
