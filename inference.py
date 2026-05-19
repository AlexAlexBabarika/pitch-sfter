import argparse
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import librosa
import numpy as np
import torch
import torchaudio

from config import AudioConfig
from model import PitchUNet


DEMUCS_SR = 44100
DEFAULT_VOCODER_REPO = "alexalexbabarika/hifigan-universal-v1"
_STEP_RE = re.compile(r"^step_(\d+)\.pt$")


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(arg)


def _find_latest_ckpt(ckpt_dir: Path) -> Path | None:
    final = Path("final.pt")
    if final.exists():
        return final
    if not ckpt_dir.exists():
        return None
    candidates = []
    for p in ckpt_dir.glob("step_*.pt"):
        m = _STEP_RE.match(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _load_unet(ckpt_path: Path, device: torch.device, use_ema: bool) -> PitchUNet:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PitchUNet().to(device)
    model.load_state_dict(ckpt["model"])
    if use_ema:
        shadow = ckpt["ema"]["shadow"]
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in shadow:
                    p.copy_(shadow[n].to(device))
    model.eval()
    return model


def _peak_normalize(audio: np.ndarray, headroom: float = 0.99) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) + 1e-9
    if peak > headroom:
        audio = audio * (headroom / peak)
    return audio.astype(np.float32, copy=False)


def _default_output(input_path: Path, semis: float) -> Path:
    sign = "+" if semis >= 0 else "-"
    mag = f"{abs(semis):g}"
    return input_path.with_name(f"{input_path.stem}_shifted_{sign}{mag}st.wav")


def _match_length(x: np.ndarray, length: int) -> np.ndarray:
    if x.shape[-1] >= length:
        return x[..., :length]
    pad = [(0, 0)] * (x.ndim - 1) + [(0, length - x.shape[-1])]
    return np.pad(x, pad)


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    if audio.ndim == 1:
        tensor = torch.from_numpy(audio).unsqueeze(0)
    else:
        tensor = torch.from_numpy(audio)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), tensor, sr)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Pitch-shift a song through PitchUNet + HiFi-GAN."
    )
    p.add_argument("input", type=Path, help="Input audio file (wav/mp3/flac/...).")
    p.add_argument(
        "-s",
        "--semitones",
        type=float,
        required=True,
        help="Semitone shift. Positive = up, negative = down.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output wav path. Default: <input_stem>_shifted_<+N>st.wav next to input.",
    )
    p.add_argument(
        "--no-separate",
        action="store_true",
        help="Skip Demucs and shift the whole mix.",
    )
    p.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="PitchUNet checkpoint. Default: ./final.pt if present, "
        "else latest step_*.pt under ./checkpoints/.",
    )
    p.add_argument(
        "--no-ema",
        action="store_true",
        help="Load raw model weights instead of the EMA shadow.",
    )
    p.add_argument(
        "--vocoder-repo",
        default=DEFAULT_VOCODER_REPO,
        help="HF repo id hosting HiFi-GAN UNIVERSAL_V1.",
    )
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Clip batch size for the UNet+vocoder loop.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 2

    device = _resolve_device(args.device)
    audio_cfg = AudioConfig()

    ckpt = args.ckpt or _find_latest_ckpt(Path("./checkpoints"))
    if ckpt is None:
        print(
            "No checkpoint found in ./checkpoints/ and --ckpt not given.",
            file=sys.stderr,
        )
        return 2
    if args.verbose:
        print(f"Loading PitchUNet from {ckpt}")
    unet = _load_unet(ckpt, device, use_ema=not args.no_ema)

    # Local imports keep cold startup snappy if -h is the only ask.
    from inference.pipeline import shift_audio_through_model
    from inference.vocoder import HiFiGANVocoder

    if args.verbose:
        print(f"Loading vocoder from {args.vocoder_repo}")
    vocoder = HiFiGANVocoder(args.vocoder_repo, device=device)

    default_name = _default_output(args.input, args.semitones).name
    if args.output is None:
        output = _default_output(args.input, args.semitones)
    elif args.output.is_dir() or str(args.output).endswith("/"):
        output = args.output / default_name
    else:
        output = args.output

    if args.no_separate:
        if args.verbose:
            print(f"Loading {args.input} (mono @ {audio_cfg.target_sr} Hz)")
        audio, _ = librosa.load(args.input, sr=audio_cfg.target_sr, mono=True)
        if args.verbose:
            print(f"Shifting by {args.semitones:+g} semitones")
        shifted = shift_audio_through_model(
            audio,
            args.semitones,
            unet,
            vocoder,
            device,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )
        shifted = _peak_normalize(shifted)
        _write_wav(output, shifted, audio_cfg.target_sr)
        if args.verbose:
            print(f"Wrote {output} @ {audio_cfg.target_sr} Hz")
        return 0

    from inference.separation import separate_stems

    if args.verbose:
        print(f"Loading {args.input} (stereo @ {DEMUCS_SR} Hz)")
    audio, _ = librosa.load(args.input, sr=DEMUCS_SR, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    audio_t = torch.from_numpy(audio.astype(np.float32))

    if args.verbose:
        print("Running Demucs separation")
    stems = separate_stems(audio_t, DEMUCS_SR, device)

    stems_dir = output.with_name(f"{output.stem}_stems")
    stems_dir.mkdir(parents=True, exist_ok=True)
    for name, tensor in stems.items():
        _write_wav(stems_dir / f"{name}.wav", tensor.numpy(), DEMUCS_SR)
    if args.verbose:
        print(f"Saved stems to {stems_dir}")

    vocals_mono_44k = stems["vocals"].mean(dim=0).numpy()
    vocals_22k = librosa.resample(
        vocals_mono_44k, orig_sr=DEMUCS_SR, target_sr=audio_cfg.target_sr
    )
    if args.verbose:
        print(f"Shifting vocals by {args.semitones:+g} semitones")
    shifted_22k = shift_audio_through_model(
        vocals_22k,
        args.semitones,
        unet,
        vocoder,
        device,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    shifted_44k_mono = librosa.resample(
        shifted_22k, orig_sr=audio_cfg.target_sr, target_sr=DEMUCS_SR
    )
    shifted_44k = np.stack([shifted_44k_mono, shifted_44k_mono], axis=0)

    target_len = max(shifted_44k.shape[-1], int(stems["drums"].shape[-1]))
    mix = _match_length(shifted_44k, target_len).copy()
    for name in ("drums", "bass", "other"):
        s = stems[name].numpy()
        mix = mix + _match_length(s, target_len)

    mix = _peak_normalize(mix)
    _write_wav(output, mix, DEMUCS_SR)
    if args.verbose:
        print(f"Wrote {output} @ {DEMUCS_SR} Hz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
