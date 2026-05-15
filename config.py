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
