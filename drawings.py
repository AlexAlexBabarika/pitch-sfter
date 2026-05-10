import altair as alt
import librosa
import numpy as np
import polars as pl

alt.data_transformers.disable_max_rows()

def waveform_chart(audio: np.ndarray, sr: int, title: str, n_buckets: int = 1500, color: str = "#6366f1") -> alt.Chart:
    n = len(audio)
    bucket_size = max(1, n // n_buckets)
    usable = bucket_size * (n // bucket_size)
    buckets = audio[:usable].reshape(-1, bucket_size)
    times = np.arange(buckets.shape[0]) * bucket_size / sr
    df = pl.DataFrame({
        "time": times.astype(np.float64, copy=False),
        "min": buckets.min(axis=1).astype(np.float64, copy=False),
        "max": buckets.max(axis=1).astype(np.float64, copy=False),
        "rms": np.sqrt((buckets ** 2).mean(axis=1)).astype(np.float64, copy=False),
    })
    base = alt.Chart(df).properties(width=760, height=200, title=title)
    envelope = base.mark_area(opacity=0.45, color=color).encode(
        x=alt.X("time:Q", title="Time (s)"),
        y=alt.Y("min:Q", title="Amplitude", scale=alt.Scale(domain=[-1, 1])),
        y2="max:Q",
    )
    rms_top = base.mark_line(color="#ec4899", strokeWidth=1.2).encode(x="time:Q", y="rms:Q")
    rms_bot = base.transform_calculate(neg="-datum.rms").mark_line(color="#ec4899", strokeWidth=1.2).encode(
        x="time:Q", y=alt.Y("neg:Q"),
    )
    return (envelope + rms_top + rms_bot).configure_view(strokeOpacity=0).configure_axis(grid=False)


def comparison_chart(stages: list[tuple[np.ndarray, int | float, str]], n_buckets: int = 1500) -> alt.Chart:
    frames = []
    for audio, sr, label in stages:
        n = len(audio)
        bucket_size = max(1, n // n_buckets)
        usable = bucket_size * (n // bucket_size)
        buckets = audio[:usable].reshape(-1, bucket_size)
        frames.append(pl.DataFrame({
            "time": (np.arange(buckets.shape[0]) * bucket_size / sr).astype(np.float64, copy=False),
            "min": buckets.min(axis=1).astype(np.float64, copy=False),
            "max": buckets.max(axis=1).astype(np.float64, copy=False),
            "stage": label,
        }))
    df = pl.concat(frames)
    return alt.Chart(df).mark_area(opacity=0.55).encode(
        x=alt.X("time:Q", title="Time (s)"),
        y=alt.Y("min:Q", title="Amplitude", scale=alt.Scale(domain=[-1, 1])),
        y2="max:Q",
        color=alt.Color("stage:N", legend=alt.Legend(title=None), scale=alt.Scale(scheme="set2")),
        row=alt.Row("stage:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
    ).properties(width=760, height=140).configure_view(strokeOpacity=0).configure_axis(grid=False)


def spectrogram_chart(
    mel: np.ndarray,
    sr: int,
    title: str,
    hop_length: int = 256,
    max_frames: int = 800,
    scheme: str = "magma",
) -> alt.Chart:
    n_mels, n_frames = mel.shape
    if n_frames > max_frames:
        bucket = n_frames // max_frames
        usable = bucket * (n_frames // bucket)
        mel = mel[:, :usable].reshape(n_mels, -1, bucket).mean(axis=2)
        n_frames = mel.shape[1]
        frame_hop = hop_length * bucket
    else:
        frame_hop = hop_length

    dt = frame_hop / sr
    t0 = np.arange(n_frames) * dt
    t1 = t0 + dt

    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmax=sr / 2)
    edges = np.empty(n_mels + 1)
    edges[1:-1] = (mel_freqs[:-1] + mel_freqs[1:]) / 2
    edges[0] = max(0.0, 2 * mel_freqs[0] - edges[1])
    edges[-1] = sr / 2
    f0 = edges[:-1]
    f1 = edges[1:]

    T0, F0 = np.meshgrid(t0, f0)
    T1, F1 = np.meshgrid(t1, f1)
    df = pl.DataFrame({
        "t0": T0.ravel().astype(np.float64, copy=False),
        "t1": T1.ravel().astype(np.float64, copy=False),
        "f0": F0.ravel().astype(np.float64, copy=False),
        "f1": F1.ravel().astype(np.float64, copy=False),
        "value": mel.ravel().astype(np.float64, copy=False),
    })

    return alt.Chart(df).mark_rect().encode(
        x=alt.X("t0:Q", title="Time (s)"),
        x2="t1:Q",
        y=alt.Y("f0:Q", title="Frequency (Hz)"),
        y2="f1:Q",
        color=alt.Color("value:Q", title="log-mel", scale=alt.Scale(scheme=scheme)),
    ).properties(width=760, height=260, title=title).configure_view(strokeOpacity=0)