"""
Audio transformation utilities for evaluation and contrastive training.

Reusable pitch, tempo, noise, and crop transforms.
"""

from __future__ import annotations

import numpy as np

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
    raise ValueError(f"Unknown transform: {kind}")
