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
    crepe_model_capacity: str = "full"
    fmin: int = 0
    fmax: int = 8000


@dataclass
class DataConfig:
    cache_dir: str = "./cache"

    nsynth_hf_dataset_name: str = "jg583/NSynth"
    nsynth_subdir_name: str = "nsynth"
    nsynth_length_hr: int = 2

    vctk_hf_dataset_name: str = "sanchit-gandhi/vctk"
    vctk_subdir_name: str = "vctk"
    vctk_length_hr: int = 2

    datasets_to_load = ["nsynth", "vctk"]
    val_split: float = 0.02

    assert val_split < 1


@dataclass
class TrainConfig:
    perturb_st: int = 2
    batch_size: int = 32
    num_workers: int = 6
    warmup_steps: int = 1000
    max_steps: int = 200000
    lr: float = 2.0e-4
    weight_decay: float = 0.01
    betas = (0.8, 0.99)
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    cond_dropout: float = 0.1
    skip_dropout: float = 0.2
    log_every: int = 100
    val_every: int = 5000
    ckpt_dir: str = "./checkpoints"


@dataclass
class ModelConfig:
    base_channels: int = 64
    channel_mults = [1, 2, 4, 6]  # 64, 128, 256, 384
    shift_emb_dim: int = 64
    f0_emb_dim: int = 64
    attention_in_bottleneck: bool = True
