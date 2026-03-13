"""
Audio transformation utilities for evaluation and contrastive training.

Reusable pitch, tempo, noise, crop, band-pass filter, and compound transforms.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

import librosa


def pitch_shift(waveform: np.ndarray, sr: int, semitones: int) -> np.ndarray:
    shifted = librosa.effects.pitch_shift(waveform, sr=sr, n_steps=semitones)
    return shifted.astype(np.float32)


def time_stretch(waveform: np.ndarray, rate: float) -> np.ndarray:
    stretched = librosa.effects.time_stretch(waveform, rate=rate)
    return stretched.astype(np.float32)


def add_noise(
    waveform: np.ndarray, snr_db: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(42)
    signal_power = float(np.mean(waveform**2)) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=waveform.shape).astype(np.float32)
    return (waveform + noise).astype(np.float32)


def crop_end(waveform: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    crop_samples = int(round(seconds * sr))
    if crop_samples <= 0:
        return waveform
    if crop_samples >= waveform.shape[0]:
        return waveform
    return waveform[:-crop_samples].astype(np.float32)


def bandpass_filter(
    waveform: np.ndarray, sr: int, low_hz: float, high_hz: float, order: int = 5
) -> np.ndarray:
    nyq = sr / 2.0
    low = max(low_hz / nyq, 1e-6)
    high = min(high_hz / nyq, 1.0 - 1e-6)
    sos = butter(order, [low, high], btype="band", output="sos")
    filtered = sosfilt(sos, waveform).astype(np.float32)
    return filtered


BANDPASS_PRESETS: dict[str, tuple[float, float]] = {
    "phone": (300.0, 4000.0),
    "laptop": (200.0, 8000.0),
    "tv": (100.0, 10000.0),
}


def normalise(waveform: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 0:
        return (waveform / peak).astype(np.float32)
    return waveform.astype(np.float32)


TRANSFORM_PRESETS = [
    ("pitch", -2),
    ("pitch", 2),
    ("tempo", 0.9),
    ("tempo", 1.1),
    ("noise", 15.0),
    ("crop", 0.5),
]


def apply_transform(
    waveform: np.ndarray,
    sr: int,
    kind: str,
    value: float | int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if kind == "pitch":
        return pitch_shift(waveform, sr, int(value))
    if kind == "tempo":
        return time_stretch(waveform, rate=float(value))
    if kind == "noise":
        return add_noise(waveform, snr_db=float(value), rng=rng)
    if kind == "crop":
        return crop_end(waveform, sr, seconds=float(value))
    if kind == "bandpass":
        preset_name = str(value)
        if preset_name not in BANDPASS_PRESETS:
            raise ValueError(f"Unknown bandpass preset: {preset_name}")
        low_hz, high_hz = BANDPASS_PRESETS[preset_name]
        return bandpass_filter(waveform, sr, low_hz, high_hz)
    raise ValueError(f"Unknown transform: {kind}")
