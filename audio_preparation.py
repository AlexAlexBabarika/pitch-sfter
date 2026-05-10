import librosa
import numpy as np
import pyloudnorm

SUPPORTED_SRS = [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000]

def load_audio_file(file_path: str) -> tuple[np.ndarray, float | int]:
    audio, sr = librosa.load(file_path, sr=None)

    return audio, sr

def resample_audio(audio: np.ndarray, original_sr: float | int, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    if target_sr not in SUPPORTED_SRS:
        raise ValueError(f"The target rate of {target_sr} isn't supported. Supported rates:\n{SUPPORTED_SRS}")
    return [librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr), target_sr]

def normalize_audio(audio: np.ndarray, sr: int, target_lufs: int = -23) -> np.ndarray:
    meter = pyloudnorm.Meter(rate=sr)
    loudness = meter.integrated_loudness(audio)
    audio = pyloudnorm.normalize.loudness(audio, loudness, target_lufs)
    return audio

def segment_audio(audio: np.ndarray, sr: int, segment_length: int) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    samples_per_segment: int = sr * segment_length

    for i in range(0, len(audio), samples_per_segment):
        segments.append(audio[i:i+samples_per_segment])
    
    return segments

def compute_mel_spectrogram(audio: np.ndarray, sr: int | float, n_fft: int =1024, hop_length: int = 256, n_mels: int = 80) -> np.ndarray:
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    return mel_spectrogram

def mel_log_compression(mel_spectrogram: np.ndarray) -> np.ndarray:
    return np.log(mel_spectrogram + 1e-5)