import json
from types import SimpleNamespace

import torch
from huggingface_hub import hf_hub_download

from inference.hifigan.config import V1_CONFIG
from inference.hifigan.models import Generator


class HiFiGANVocoder:
    def __init__(
        self,
        repo_id: str,
        device: torch.device,
        ckpt_filename: str = "g_02500000",
        config_filename: str | None = "config.json",
    ):
        if config_filename is not None:
            cfg_path = hf_hub_download(repo_id=repo_id, filename=config_filename)
            with open(cfg_path) as f:
                cfg = json.load(f)
        else:
            cfg = dict(V1_CONFIG)
        h = SimpleNamespace(**cfg)

        ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_filename)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)

        self.generator = Generator(h).to(device)
        self.generator.load_state_dict(state["generator"])
        self.generator.eval()
        self.generator.remove_weight_norm()
        self.device = device
        self.sample_rate = int(getattr(h, "sampling_rate", V1_CONFIG["sampling_rate"]))

    @torch.inference_mode()
    def mel_to_audio(self, log_mel: torch.Tensor) -> torch.Tensor:
        # log_mel: [B, n_mels, T] in natural-log magnitude space.
        # Returns: [B, samples] float32 audio, peak-clamped to [-1, 1].
        x = log_mel.to(self.device)
        audio = self.generator(x).squeeze(1)
        return audio.clamp(-1.0, 1.0).float()
