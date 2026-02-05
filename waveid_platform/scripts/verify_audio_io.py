"""
Quick sanity check for audio decoding and segmentation.

Usage:
    python -m scripts.verify_audio_io --file "path/to/audio.wav"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from waveid_backend.config import (
    HOP_SECONDS,
    MAX_DURATION_SECONDS,
    MONO,
    NORMALIZE,
    SAMPLE_RATE,
    SEGMENT_SECONDS,
)
from waveid_backend.services.audio_io import load_audio_from_bytes
from waveid_backend.services.segmentation import segment_audio


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify audio decoding pipeline.")
    parser.add_argument("--file", required=True, help="Path to WAV or MP3 file.")
    parser.add_argument("--sr", type=int, default=SAMPLE_RATE, help="Target sample rate.")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return 1

    contents = path.read_bytes()
    waveform, sr = load_audio_from_bytes(
        contents,
        filename=path.name,
        target_sr=args.sr,
        mono=MONO,
        normalize=NORMALIZE,
        max_duration_seconds=MAX_DURATION_SECONDS,
    )
    duration = waveform.size / sr if sr else 0.0
    segments = segment_audio(waveform, sr, SEGMENT_SECONDS, HOP_SECONDS)

    print(f"sample_rate={sr}")
    print(f"duration_sec={duration:.2f}")
    print(f"waveform_shape={waveform.shape}")
    print(f"segments={len(segments)}")
    if segments:
        print(
            "first_segment="
            f"{segments[0].start_time:.2f}-{segments[0].end_time:.2f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
