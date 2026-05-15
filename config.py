from dataclasses import dataclass


@dataclass
class AudioConfig:
    target_lufs: int = -23
    target_sr: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    win_length: int = 1024
    clip_seconds: float = 2.56
    crepe_model_capacity: str = "large"
    fmin: int = 0
    fmax: int = 8000


@dataclass
class CacheConfig:
    cache_dir: str = "./cache"
