"""
Audio transformation utilities for evaluation and contrastive training.

Reusable pitch, tempo, noise, crop, band-pass filter, lossy compression, and compound transforms.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from pydub import AudioSegment
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

# MP3 encode/decode round-trip bitrates (kbps) for lossy compression stress tests.
LOSSY_MP3_BITRATES: tuple[int, ...] = (64, 96)


def lossy_mp3_roundtrip(waveform: np.ndarray, sr: int, bitrate_kbps: int) -> np.ndarray:
    """
    Encode mono float32 audio to MP3 at the given constant bitrate and decode back.

    Requires ffmpeg on PATH (used by pydub). Simulates re-encoding through messaging
    apps, low-quality exports, or aggressive streaming bitrates.
    """
    if waveform.size == 0:
        return waveform.astype(np.float32)
    samples_int16 = np.clip(waveform * 32767.0, -32768, 32767).astype(np.int16)
    audio = AudioSegment(
        data=samples_int16.tobytes(),
        sample_width=2,
        frame_rate=sr,
        channels=1,
    )
    tmp_mp3_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_mp3_path = Path(tmp.name)
        audio.export(str(tmp_mp3_path), format="mp3", bitrate=f"{bitrate_kbps}k")
        decoded, _ = librosa.load(str(tmp_mp3_path), sr=sr, mono=True)
        return decoded.astype(np.float32)
    finally:
        if tmp_mp3_path is not None:
            tmp_mp3_path.unlink(missing_ok=True)


def normalise(waveform: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 0:
        return (waveform / peak).astype(np.float32)
    return waveform.astype(np.float32)


TRANSFORM_PRESETS = [
    ("pitch", -4),
    ("pitch", -2),
    ("pitch", 2),
    ("pitch", 4),
    ("tempo", 0.85),
    ("tempo", 1.15),
    ("noise", 8.0),
    ("noise", 15.0),
    ("noise", 25.0),
    ("crop", 0.5),
    ("bandpass", "phone"),
    ("bandpass", "laptop"),
    ("lossy", 64),
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
    if kind == "lossy":
        if isinstance(value, str) and value.startswith("mp3_"):
            token = value.removeprefix("mp3_").replace("k", "")
            kbps = int(token)
        else:
            kbps = int(value)
        if kbps <= 0:
            raise ValueError(f"MP3 bitrate must be positive, got {kbps}")
        return lossy_mp3_roundtrip(waveform, sr, bitrate_kbps=kbps)
    raise ValueError(f"Unknown transform: {kind}")
