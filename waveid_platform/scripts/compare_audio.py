"""
Compare two audio files by checksum and basic statistics.

Usage:
    python -m scripts.compare_audio --file-a "path/to/a.wav" --file-b "path/to/b.wav"
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_stats(path: Path) -> tuple[int, float, float, float]:
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    duration = data.shape[0] / sr if sr else 0.0
    return sr, duration, float(np.mean(data)), float(np.std(data))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two audio files.")
    parser.add_argument("--file-a", required=True, help="Path to first audio file.")
    parser.add_argument("--file-b", required=True, help="Path to second audio file.")
    args = parser.parse_args()

    path_a = Path(args.file_a)
    path_b = Path(args.file_b)
    if not path_a.exists():
        print(f"File not found: {path_a}")
        return 1
    if not path_b.exists():
        print(f"File not found: {path_b}")
        return 1

    print(f"file_a={path_a}")
    print(f"sha256_a={sha256(path_a)}")
    sr_a, dur_a, mean_a, std_a = audio_stats(path_a)
    print(f"sr_a={sr_a} duration_a={dur_a:.2f} mean_a={mean_a:.6f} std_a={std_a:.6f}")
    print("")
    print(f"file_b={path_b}")
    print(f"sha256_b={sha256(path_b)}")
    sr_b, dur_b, mean_b, std_b = audio_stats(path_b)
    print(f"sr_b={sr_b} duration_b={dur_b:.2f} mean_b={mean_b:.6f} std_b={std_b:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
