"""
Generate transformed audio clips for robustness evaluation.

Usage:
    python -m scripts.evaluate_transformations --input "path/to/audio.wav" --output-dir "data/query/eval"

Lossy MP3 transforms require ffmpeg on PATH (used by pydub).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

from waveid_backend.services.transforms import LOSSY_MP3_BITRATES, lossy_mp3_roundtrip


def _load_mono(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    waveform, sr = librosa.load(path.as_posix(), sr=target_sr, mono=True)
    return waveform.astype(np.float32), sr


def _normalise(waveform: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 0:
        return (waveform / peak).astype(np.float32)
    return waveform.astype(np.float32)


def _write_clip(output_path: Path, waveform: np.ndarray, sr: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path.as_posix(), _normalise(waveform), sr)


def _pitch_shift(waveform: np.ndarray, sr: int, semitones: int) -> np.ndarray:
    shifted = librosa.effects.pitch_shift(waveform, sr=sr, n_steps=semitones)
    return shifted.astype(np.float32)


def _time_stretch(waveform: np.ndarray, rate: float) -> np.ndarray:
    stretched = librosa.effects.time_stretch(waveform, rate=rate)
    return stretched.astype(np.float32)


def _add_noise(waveform: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal_power = float(np.mean(waveform**2)) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=waveform.shape).astype(np.float32)
    return (waveform + noise).astype(np.float32)


def _crop(waveform: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    crop_samples = int(round(seconds * sr))
    if crop_samples <= 0:
        return waveform
    if crop_samples >= waveform.shape[0]:
        return waveform
    return waveform[:-crop_samples].astype(np.float32)


def _bandpass_filter(
    waveform: np.ndarray, sr: int, low_hz: float, high_hz: float, order: int = 5
) -> np.ndarray:
    nyq = sr / 2.0
    low = max(low_hz / nyq, 1e-6)
    high = min(high_hz / nyq, 1.0 - 1e-6)
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfilt(sos, waveform).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create transformed clips for evaluation.")
    parser.add_argument("--input", required=True, help="Input WAV/MP3 path.")
    parser.add_argument("--output-dir", required=True, help="Directory for transformed outputs.")
    parser.add_argument("--sr", type=int, default=16_000, help="Target sample rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for noise generation.")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Optional max duration to process (useful for quick tests).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    output_dir = Path(args.output_dir)
    base_name = input_path.stem
    rng = np.random.default_rng(args.seed)

    waveform, sr = _load_mono(input_path, target_sr=args.sr)
    if args.max_seconds is not None and args.max_seconds > 0:
        max_samples = int(round(args.max_seconds * sr))
        if 0 < max_samples < waveform.shape[0]:
            waveform = waveform[:max_samples]

    # Baseline copy
    _write_clip(output_dir / f"{base_name}_orig.wav", waveform, sr)

    # Pitch shifts (semitones)
    for semitones in (-4, -2, 2, 4):
        transformed = _pitch_shift(waveform, sr, semitones=semitones)
        sign = "m" if semitones < 0 else "p"
        _write_clip(output_dir / f"{base_name}_pitch_{sign}{abs(semitones)}.wav", transformed, sr)

    # Tempo changes
    for rate in (0.85, 1.15):
        transformed = _time_stretch(waveform, rate=rate)
        label = str(rate).replace(".", "_")
        _write_clip(output_dir / f"{base_name}_tempo_{label}.wav", transformed, sr)

    # Noise injection at two SNR levels
    for snr_db in (20.0, 10.0):
        transformed = _add_noise(waveform, snr_db=snr_db, rng=rng)
        label = str(int(snr_db))
        _write_clip(output_dir / f"{base_name}_noise_snr{label}.wav", transformed, sr)

    # End-crop severities
    for seconds in (1.0, 2.0):
        transformed = _crop(waveform, sr, seconds=seconds)
        label = str(seconds).replace(".", "_")
        _write_clip(output_dir / f"{base_name}_crop_{label}s.wav", transformed, sr)

    # Band-pass filter (simulates playback through low-quality devices)
    bandpass_presets = {
        "phone": (300.0, 4000.0),
        "laptop": (200.0, 8000.0),
    }
    for preset_name, (low_hz, high_hz) in bandpass_presets.items():
        transformed = _bandpass_filter(waveform, sr, low_hz, high_hz)
        _write_clip(output_dir / f"{base_name}_bandpass_{preset_name}.wav", transformed, sr)

    # Compound: band-pass + noise (simulates recording from a device in a noisy environment)
    for preset_name, (low_hz, high_hz) in bandpass_presets.items():
        filtered = _bandpass_filter(waveform, sr, low_hz, high_hz)
        transformed = _add_noise(filtered, snr_db=15.0, rng=rng)
        _write_clip(output_dir / f"{base_name}_compound_{preset_name}_snr15.wav", transformed, sr)

    # Lossy compression: MP3 encode/decode at low bitrates (requires ffmpeg on PATH)
    for kbps in LOSSY_MP3_BITRATES:
        transformed = lossy_mp3_roundtrip(waveform, sr, bitrate_kbps=kbps)
        _write_clip(output_dir / f"{base_name}_lossy_mp3_{kbps}k.wav", transformed, sr)

    print(f"Generated transformed clips in: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
