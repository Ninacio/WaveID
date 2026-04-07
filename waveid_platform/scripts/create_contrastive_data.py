"""
Create positive/negative pairs for contrastive training.

Usage (single genre):
    python -m scripts.create_contrastive_data ^
        --dataset-dirs "../datasets/GTZAN/genres_original/blues" ^
        --output-dir "data/embeddings/contrastive_pairs" ^
        --max-tracks-per-dir 10 --pairs-per-track 50

Usage (all genres):
    python -m scripts.create_contrastive_data ^
        --dataset-dirs "../datasets/GTZAN/genres_original/blues" ^
                       "../datasets/GTZAN/genres_original/classical" ^
                       ... ^
        --output-dir "data/embeddings/contrastive_multi"
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
    """Find audio files in directory (top-level only)."""
    files = []
    for ext in ALLOWED_EXTENSIONS:
        files.extend(root.glob(f"*{ext}"))
    return sorted(p for p in files if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description="Create contrastive training pairs.")
    parser.add_argument(
        "--dataset-dirs", nargs="+", required=True, type=Path,
        help="One or more directories containing audio files (e.g. one per GTZAN genre).",
    )
    parser.add_argument("--output-dir", default="data/embeddings/contrastive_pairs", help="Output directory.")
    parser.add_argument("--max-tracks-per-dir", type=int, default=20, help="Max tracks to load per directory.")
    parser.add_argument("--pairs-per-track", type=int, default=30, help="Triplets to generate per track.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    rng = np.random.default_rng(args.seed)
    target_len = int(SEGMENT_SECONDS * SAMPLE_RATE)

    all_segments: list[tuple[Path, np.ndarray]] = []
    total_tracks = 0

    for dataset_dir in args.dataset_dirs:
        dataset_dir = dataset_dir.resolve()
        if not dataset_dir.is_dir():
            print(f"Warning: skipping non-directory {dataset_dir}")
            continue
        files = _iter_audio_files(dataset_dir)[: args.max_tracks_per_dir]
        loaded = 0
        for path in files:
            try:
                waveform, sr = _load_mono(path, SAMPLE_RATE)
                segments = segment_audio(waveform, sr, SEGMENT_SECONDS, HOP_SECONDS)
                for seg in segments:
                    all_segments.append((path, seg.samples))
                loaded += 1
            except Exception as e:
                print(f"  Skipping {path.name}: {e}")
        total_tracks += loaded
        print(f"Loaded {loaded} tracks from {dataset_dir.name} ({len(files)} found)")

    if len(all_segments) < 2:
        print("Not enough segments across all directories.")
        return 1

    total_pairs = args.pairs_per_track * total_tracks
    print(f"\nTotal tracks: {total_tracks}, segments: {len(all_segments)}, "
          f"generating {total_pairs} triplets...")

    output_dir.mkdir(parents=True, exist_ok=True)
    anchors_list: list[np.ndarray] = []
    positives_list: list[np.ndarray] = []
    negatives_list: list[np.ndarray] = []

    for i in range(total_pairs):
        idx_a = rng.integers(0, len(all_segments))
        path_a, seg_a = all_segments[idx_a]
        anchor = normalise(seg_a)

        kind, value = TRANSFORM_PRESETS[rng.integers(0, len(TRANSFORM_PRESETS))]
        try:
            positive = normalise(
                _fix_length(
                    apply_transform(seg_a.copy(), SAMPLE_RATE, kind, value, rng=rng),
                    target_len,
                )
            )
        except Exception:
            kind, value = "noise", 15.0
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

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{total_pairs} triplets generated...")

    anchors_arr = np.stack(anchors_list, axis=0)
    positives_arr = np.stack(positives_list, axis=0)
    negatives_arr = np.stack(negatives_list, axis=0)

    np.save(output_dir / "anchors.npy", anchors_arr)
    np.save(output_dir / "positives.npy", positives_arr)
    np.save(output_dir / "negatives.npy", negatives_arr)

    print(f"\nCreated {len(anchors_list)} triplets in {output_dir}")
    print(f"  anchors.npy:   {anchors_arr.shape}")
    print(f"  positives.npy: {positives_arr.shape}")
    print(f"  negatives.npy: {negatives_arr.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
