"""
Audio I/O utilities.

Handles decoding from bytes, mono conversion, resampling and optional
normalisation for ingestion and query pipelines.
"""

from __future__ import annotations

import io
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment


def load_audio_from_bytes(
    contents: bytes,
    filename: str,
    target_sr: int,
    mono: bool = True,
    normalise: bool = True,
    max_duration_seconds: float | None = None,
    *,
    normalize: bool | None = None,
) -> tuple[np.ndarray, int]:
    """Decode audio bytes into a waveform.

    Args:
        contents: Raw file bytes.
        filename: Original filename (used to infer format).
        target_sr: Target sampling rate.
        mono: Whether to downmix to mono.
        normalise: Whether to peak-normalise the waveform.
        max_duration_seconds: Optional maximum allowed duration.

    Returns:
        Tuple of (waveform, sample_rate).
    """
    if not contents:
        raise ValueError("Uploaded file is empty.")

    ext = Path(filename).suffix.lower()
    if ext == ".wav":
        waveform, sr = sf.read(io.BytesIO(contents), dtype="float32")
    elif ext == ".mp3":
        audio = AudioSegment.from_file(io.BytesIO(contents), format="mp3")
        sr = audio.frame_rate
        channels = audio.channels
        sample_width = audio.sample_width
        samples = np.array(audio.get_array_of_samples())
        if channels > 1:
            samples = samples.reshape((-1, channels))
        max_val = float(1 << (8 * sample_width - 1))
        waveform = samples.astype(np.float32) / max_val
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if mono and waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if sr != target_sr:
        waveform = librosa.resample(y=waveform, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    duration = waveform.size / sr if sr else 0.0
    if max_duration_seconds is not None and duration > max_duration_seconds:
        raise ValueError(
            f"Audio duration {duration:.2f}s exceeds "
            f"max {max_duration_seconds:.2f}s."
        )

    if normalize is not None:
        normalise = normalize

    if normalise:
        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        if peak > 0:
            waveform = waveform / peak

    return waveform.astype(np.float32), sr
