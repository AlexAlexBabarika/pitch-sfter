from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    target_lufs: int = -23
    target_sr: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    win_length: int = 1024
    clip_seconds: float = 2.56
    crepe_model_capacity: str = "medium"
    fmin: int = 0
    fmax: int = 8000


@dataclass(frozen=True)
class DatasetSpec:
    hf_name: str
    subdir: str
    length_hr: int
    parquet_glob: str


@dataclass
class DataConfig:
    cache_dir: str = "./cache"
    val_split: float = 0.02

    datasets: dict[str, DatasetSpec] = field(
        default_factory=lambda: {
            "nsynth": DatasetSpec(
                hf_name="jg583/NSynth",
                subdir="nsynth",
                length_hr=2,
                parquet_glob="data/train/*.parquet",
            ),
            "vctk": DatasetSpec(
                hf_name="sanchit-gandhi/vctk",
                subdir="vctk",
                length_hr=2,
                parquet_glob="data/train-*.parquet",
            ),
        }
    )
    datasets_to_load: list[str] = field(default_factory=lambda: ["nsynth", "vctk"])

    def __post_init__(self):
        assert self.val_split < 1
        unknown = set(self.datasets_to_load) - self.datasets.keys()
        assert not unknown, f"unknown datasets in datasets_to_load: {unknown}"


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
    save_every: int = 100
    keep_last: int = 5
    ckpt_dir: str = "./checkpoints"
    tb_dir: str = "./runs"


@dataclass
class ModelConfig:
    base_channels: int = 64
    channel_mults = [1, 2, 4, 6]  # 64, 128, 256, 384
    shift_emb_dim: int = 64
    f0_emb_dim: int = 64
    attention_in_bottleneck: bool = True
