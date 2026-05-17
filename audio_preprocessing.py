import pyloudnorm
import librosa
import torchaudio
import torch
import torchcrepe
import numpy as np
import math
from typing import Generator
from config import AudioConfig
from pathlib import Path

cfg = AudioConfig()

SUPPORTED_SRS = [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000]
CREPE_MODEL_CAPACITIES = ["tiny", "full"]
CLIP_LEN = int(cfg.clip_seconds * cfg.target_sr)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

meter = pyloudnorm.Meter(rate=cfg.target_sr)
MEL_TRANSFORM = torchaudio.transforms.MelSpectrogram(
    sample_rate=cfg.target_sr,
    n_fft=cfg.n_fft,
    hop_length=cfg.hop_length,
    win_length=cfg.win_length,
    n_mels=cfg.n_mels,
    f_min=cfg.fmin,
    f_max=cfg.fmax,
    power=1.0,
    center=True,
).to(DEVICE)


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


def segment_audio(audio: np.ndarray) -> Generator[tuple[np.ndarray, int]]:
    n = len(audio)
    for start in range(0, n, CLIP_LEN):
        clip = audio[start : start + CLIP_LEN]
        real_len = len(clip)
        if real_len < CLIP_LEN // 2:
            continue
        if real_len < CLIP_LEN:
            clip = np.pad(clip, (0, CLIP_LEN - real_len))
        yield clip, real_len


def compute_mel_spectrogram(clips: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(clips).float().to(DEVICE)
    m = MEL_TRANSFORM(t).cpu().numpy()
    return np.log(np.maximum(m, 1e-5)).astype(np.float32)


def extract_f0(clips: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if cfg.crepe_model_capacity not in CREPE_MODEL_CAPACITIES:
        raise ValueError(
            f"Model capacity {cfg.crepe_model_capacity} isn't supported by torchcrepe. Supported capacities:\n{CREPE_MODEL_CAPACITIES}"
        )

    audio_t = torch.from_numpy(clips).float().to(DEVICE)
    audio_16k = torchaudio.functional.resample(audio_t, cfg.target_sr, 16000)
    # CREPE frame interval is 186/16000 ≈ 11.625 ms; mel frame interval is 256/22050 ≈ 11.610 ms. The np.interp later masks it
    # Across 220 frames it's gonna be around 3ms. Has to be examined
    hop = round(16000 * cfg.hop_length / cfg.target_sr)

    pitch, periodicity = torchcrepe.predict(
        audio_16k,
        sample_rate=16000,
        hop_length=hop,
        model=cfg.crepe_model_capacity,
        batch_size=2048,
        device=DEVICE,
        decoder=torchcrepe.decode.viterbi,
        return_periodicity=True,
    )

    return pitch.cpu().numpy().astype(np.float32), periodicity.cpu().numpy().astype(np.float32)


def process_and_save(
    audio: np.ndarray, src_sr: int, out_dir: Path, base_id: str, budget: dict
) -> float:
    if budget["remaining_s"] <= 0:
        return 0.0
    if src_sr != cfg.target_sr:
        audio = resample_audio_to_target_sr(audio, src_sr)
    audio, _ = librosa.effects.trim(audio, top_db=30)
    if len(audio) < cfg.target_sr * 0.5:
        return 0.0
    audio = normalize_audio(audio)

    # Collect clips up to the budget, then run mel + CREPE on the whole batch
    clips: list[np.ndarray] = []
    real_lens: list[int] = []
    remaining = budget["remaining_s"]
    for clip, real_len in segment_audio(audio):
        if remaining <= 0:
            break
        clips.append(clip)
        real_lens.append(real_len)
        remaining -= real_len / cfg.target_sr
    if not clips:
        return 0.0

    batch = np.stack(clips, axis=0)
    mels = compute_mel_spectrogram(batch)  # (B, n_mels, T_mel)
    f0s, confs = extract_f0(batch)  # (B, T_f0)

    written = 0.0
    for i, (clip, real_len, mel, f0, conf) in enumerate(
        zip(clips, real_lens, mels, f0s, confs)
    ):
        if f0.shape[0] != mel.shape[1]:
            # Nearest-neighbor: linear interp would smear CREPE's 0-Hz unvoiced
            # frames into bogus low pitches across voiced -> unvoiced transitions.
            idx = np.round(
                np.linspace(0, f0.shape[0] - 1, mel.shape[1])
            ).astype(np.int64)
            f0 = f0[idx]
            conf = conf[idx]
        out_path = out_dir / f"{base_id}_{i:04d}.npz"
        np.savez(
            out_path,
            audio=clip.astype(np.float32),
            mel=mel,
            f0=f0,
            conf=conf,
        )
        seconds = real_len / cfg.target_sr
        written += seconds
        budget["remaining_s"] -= seconds
    return written
