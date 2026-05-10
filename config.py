from dataclasses import dataclass


@dataclass
class Config:
    target_lufs: int = -23
    target_sr: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    window_size: int = 3
    crepe_model_capacity: str = "tiny"
