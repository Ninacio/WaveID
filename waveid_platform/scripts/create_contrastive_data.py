"""
Create positive/negative pairs for contrastive training.

Usage:
    python -m scripts.create_contrastive_data --dataset-dir "path/to/gtzan/blues" --output-dir "data/embeddings/contrastive_pairs" --max-tracks 10 --pairs-per-track 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np

from waveid_backend.config import (
    ALLOWED_EXTENSIONS,
    HOP_SECONDS,
    SAMPLE_RATE,
    SEGMENT_SECONDS,
)
from waveid_backend.services.segmentation import segment_audio
from waveid_backend.services.transforms import (
    TRANSFORM_PRESETS,
    apply_transform,
    normalise,
)


def _load_mono(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    waveform, sr = librosa.load(path.as_posix(), sr=target_sr, mono=True)
    return waveform.astype(np.float32), sr


def _fix_length(waveform: np.ndarray, target_len: int) -> np.ndarray:
    if waveform.shape[0] >= target_len:
        return waveform[:target_len].astype(np.float32)
    pad = np.zeros(target_len - waveform.shape[0], dtype=np.float32)
    return np.concatenate([waveform, pad]).astype(np.float32)


def _iter_audio_files(root: Path) -> list[Path]:
    """Find audio files in directory. Uses top-level glob to avoid rglob traversal issues."""
    files = []
    for ext in ALLOWED_EXTENSIONS:
        files.extend(root.glob(f"*{ext}"))
    return sorted(p for p in files if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description="Create contrastive training pairs.")
    parser.add_argument("--dataset-dir", required=True, help="Dataset root (e.g. GTZAN genre folder).")
    parser.add_argument("--output-dir", default="data/embeddings/contrastive_pairs", help="Output directory.")
    parser.add_argument("--max-tracks", type=int, default=20, help="Max tracks to process.")
    parser.add_argument("--pairs-per-track", type=int, default=30, help="Pairs to generate per track.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not dataset_dir.exists():
        print(f"Dataset dir not found: {dataset_dir}")
        print("Use an absolute path, e.g. C:\\Users\\...\\datasets\\GTZAN\\genres_original\\blues")
        return 1
    if not dataset_dir.is_dir():
        print(f"Dataset path is not a directory: {dataset_dir}")
        return 1

    rng = np.random.default_rng(args.seed)
    target_len = int(SEGMENT_SECONDS * SAMPLE_RATE)
    files = _iter_audio_files(dataset_dir)[: args.max_tracks]
    if len(files) < 2:
        print("Need at least 2 audio files for contrastive pairs.")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    anchors_list: list[np.ndarray] = []
    positives_list: list[np.ndarray] = []
    negatives_list: list[np.ndarray] = []

    all_segments: list[tuple[Path, np.ndarray]] = []
    for path in files:
        try:
            waveform, sr = _load_mono(path, SAMPLE_RATE)
            segments = segment_audio(waveform, sr, SEGMENT_SECONDS, HOP_SECONDS)
            for seg in segments:
                all_segments.append((path, seg.samples))
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

    if len(all_segments) < 2:
        print("Not enough segments.")
        return 1

    n = 0
    for _ in range(args.pairs_per_track * len(files)):
        if n >= args.pairs_per_track * len(files):
            break
        idx_a = rng.integers(0, len(all_segments))
        path_a, seg_a = all_segments[idx_a]
        anchor = normalise(seg_a)

        kind, value = TRANSFORM_PRESETS[rng.integers(0, len(TRANSFORM_PRESETS))]
        positive = normalise(
            _fix_length(
                apply_transform(seg_a.copy(), SAMPLE_RATE, kind, value, rng=rng),
                target_len,
            )
        )

        idx_neg = rng.integers(0, len(all_segments))
        while all_segments[idx_neg][0] == path_a:
            idx_neg = rng.integers(0, len(all_segments))
        negative = normalise(_fix_length(all_segments[idx_neg][1], target_len))

        anchors_list.append(_fix_length(anchor, target_len))
        positives_list.append(positive)
        negatives_list.append(negative)
        n += 1

    anchors_arr = np.stack(anchors_list, axis=0)
    positives_arr = np.stack(positives_list, axis=0)
    negatives_arr = np.stack(negatives_list, axis=0)

    np.save(output_dir / "anchors.npy", anchors_arr)
    np.save(output_dir / "positives.npy", positives_arr)
    np.save(output_dir / "negatives.npy", negatives_arr)

    print(f"Created {len(anchors_list)} pairs in {output_dir}")
    print(f"  anchors.npy: {anchors_arr.shape}")
    print(f"  positives.npy: {positives_arr.shape}")
    print(f"  negatives.npy: {negatives_arr.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
