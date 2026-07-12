"""
Create anchor/positive/negative triplets for contrastive training.

Two input modes:

1. Flat per-directory genre folders (GTZAN-style) - the original mode:

    python -m scripts.create_contrastive_data ^
        --dataset-dirs "../datasets/GTZAN/genres_original/blues" ^
        --output-dir "data/embeddings/contrastive_pairs" ^
        --max-tracks-per-dir 10 --pairs-per-track 50

2. A flat (path, genre) manifest CSV - for large, non-genre-folder
   datasets like FMA (see scripts/build_fma_manifest.py):

    python -m scripts.create_contrastive_data ^
        --manifest "data/manifests/fma_small.csv" ^
        --output-dir "data/embeddings/contrastive_fma_small" ^
        --total-pairs 40000 --hard-negative-ratio 0.5

Audio is decoded in shuffled chunks (a "shuffle buffer") rather than either
loading the whole dataset upfront or doing fully random single-track access,
and output triplets are written straight to disk-backed memmap arrays
instead of being accumulated in a Python list. Together this keeps both RAM
and decode work bounded regardless of dataset size (GTZAN's ~1k tracks or
FMA's 8k-100k+), so the same script scales from a laptop run to a full pass
over a much larger dataset.

Long runs checkpoint progress after every chunk. If the process is
interrupted, rerun the exact same command with --resume added to pick up
where it left off instead of starting over.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

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

TrackEntry = tuple[Path, str]  # (audio_path, genre_or_dirname)


def _fix_length(waveform: np.ndarray, target_len: int) -> np.ndarray:
    """Trim or zero-pad a clip so it is exactly target_len samples long."""
    if waveform.shape[0] >= target_len:
        return waveform[:target_len].astype(np.float32)
    pad = np.zeros(target_len - waveform.shape[0], dtype=np.float32)
    return np.concatenate([waveform, pad]).astype(np.float32)


def _iter_audio_files(root: Path) -> list[Path]:
    """Find audio files under a directory, recursively (handles nested
    dataset layouts like FMA's 000/, 001/, ... shards as well as flat
    GTZAN-style genre folders)."""
    files: list[Path] = []
    for ext in ALLOWED_EXTENSIONS:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(p for p in files if p.is_file())


def _load_manifest(manifest_path: Path) -> list[TrackEntry]:
    entries: list[TrackEntry] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "path" not in reader.fieldnames or "genre" not in reader.fieldnames:
            raise SystemExit(f"Manifest {manifest_path} must have 'path' and 'genre' columns.")
        for row in reader:
            path = Path(row["path"])
            if path.is_file():
                entries.append((path, row["genre"]))
    return entries


def _load_dataset_dirs(dataset_dirs: list[Path], max_tracks_per_dir: int) -> list[TrackEntry]:
    entries: list[TrackEntry] = []
    for dataset_dir in dataset_dirs:
        dataset_dir = dataset_dir.resolve()
        if not dataset_dir.is_dir():
            print(f"Warning: skipping non-directory {dataset_dir}")
            continue
        files = _iter_audio_files(dataset_dir)[:max_tracks_per_dir]
        for path in files:
            entries.append((path, dataset_dir.name))
        print(f"Found {len(files)} tracks in {dataset_dir.name}")
    return entries


def _decode_audio(path: Path, target_sr: int) -> np.ndarray | None:
    """Decode one file to mono float32 at target_sr.

    soundfile's libsndfile has native MP3 decoding and is ~80x faster than
    librosa's default audioread fallback for MP3 on this platform. librosa
    is used only for the (cheap) resample step, and as a last-resort fallback
    for the handful of malformed files known to exist in FMA.
    """
    try:
        raw, native_sr = sf.read(path.as_posix(), dtype="float32", always_2d=False)
        if raw.ndim > 1:
            raw = raw.mean(axis=1)
        if native_sr != target_sr:
            raw = librosa.resample(raw, orig_sr=native_sr, target_sr=target_sr)
        return raw.astype(np.float32)
    except Exception:
        try:
            waveform, _ = librosa.load(path.as_posix(), sr=target_sr, mono=True)
            return waveform.astype(np.float32)
        except Exception as exc:
            print(f"  Skipping unreadable file {path.name}: {exc}")
            return None


def _decode_chunk(
    chunk_indices: list[int], entries: list[TrackEntry], target_sr: int, workers: int
) -> dict[int, np.ndarray]:
    """Decode a batch of tracks in parallel (I/O-bound: overlaps disk reads).

    Returns only successfully-decoded entries, keyed by their index into
    `entries` so the caller can look up genre labels alongside the audio.
    """
    decoded: dict[int, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_decode_audio, entries[i][0], target_sr): i for i in chunk_indices}
        for future in futures:
            idx = futures[future]
            waveform = future.result()
            if waveform is not None:
                decoded[idx] = waveform
    return decoded


def _open_scratch_memmap(path: Path, rows: int, cols: int) -> np.memmap:
    """Pre-allocate a disk-backed array for triplet outputs.

    Writing directly to a memmap (instead of accumulating Python lists and
    stacking at the end) keeps RAM usage flat regardless of how many
    triplets are generated - the whole point of the streaming rewrite would
    otherwise be undone by holding every output triplet in memory at once
    (e.g. 120k triplets x 3 arrays x 125KB/segment is tens of GB).
    """
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(rows, cols))


def _finalize_memmap(tmp_path: Path, final_path: Path, generated: int, total_rows: int) -> tuple[int, ...]:
    """Close out a scratch memmap: rename it if fully used, else copy the
    used prefix into a right-sized file (in bounded-size chunks, never
    loading the whole array into RAM) and drop the oversized scratch file."""
    if generated == total_rows:
        if final_path.exists():
            final_path.unlink()
        tmp_path.rename(final_path)
        shape = np.load(final_path, mmap_mode="r").shape
        return shape

    src = np.lib.format.open_memmap(tmp_path, mode="r")
    dst = _open_scratch_memmap(final_path.with_suffix(".partial.npy"), generated, src.shape[1])
    batch = 2000
    for start in range(0, generated, batch):
        end = min(start + batch, generated)
        dst[start:end] = src[start:end]
    dst.flush()
    shape = dst.shape
    del dst, src
    gc.collect()
    if final_path.exists():
        final_path.unlink()
    final_path.with_suffix(".partial.npy").rename(final_path)
    tmp_path.unlink()
    return shape


def _random_segment(
    waveform: np.ndarray, sr: int, segment_seconds: float, hop_seconds: float, rng: np.random.Generator
) -> np.ndarray | None:
    segments = segment_audio(waveform, sr, segment_seconds, hop_seconds)
    if not segments:
        return None
    return segments[int(rng.integers(0, len(segments)))].samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Create contrastive training triplets (streamed).")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset-dirs", nargs="+", type=Path,
        help="One or more directories containing audio files (GTZAN-style, one dir per genre).",
    )
    source.add_argument(
        "--manifest", type=Path,
        help="CSV with 'path,genre' columns (see scripts/build_fma_manifest.py) for large flat datasets like FMA.",
    )
    parser.add_argument("--output-dir", default="data/embeddings/contrastive_pairs", help="Output directory.")
    parser.add_argument("--max-tracks-per-dir", type=int, default=20, help="[--dataset-dirs only] Max tracks to load per directory.")
    parser.add_argument("--pairs-per-track", type=int, default=30, help="Triplets per track (ignored if --total-pairs is set).")
    parser.add_argument("--total-pairs", type=int, default=None, help="Total triplets to generate, overriding pairs-per-track x track count.")
    parser.add_argument("--max-tracks", type=int, default=None, help="Cap the number of tracks used from the manifest/dirs (for quick test runs).")
    parser.add_argument(
        "--hard-negative-ratio", type=float, default=0.0,
        help="Probability [0-1] of drawing the negative from the SAME genre as the anchor "
             "(different track) instead of a uniformly random genre. Requires genre labels "
             "(always available with --dataset-dirs; from the CSV with --manifest).",
    )
    parser.add_argument(
        "--cache-size", type=int, default=500,
        help="Shuffle-buffer size: tracks are processed in chunks of this many, decoded once per "
             "chunk and reused for all their triplets before moving on. This is what keeps peak "
             "memory bounded on huge datasets while avoiding re-decoding the same audio repeatedly "
             "(random access across the full dataset would make caching useless).",
    )
    parser.add_argument("--workers", type=int, default=8, help="Parallel decode threads per chunk (I/O-bound, so > CPU count is fine).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the checkpoint left in --output-dir by a previous (interrupted) run with "
             "identical arguments, instead of starting over. Progress is checkpointed after every chunk.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    rng = np.random.default_rng(args.seed)
    target_len = int(SEGMENT_SECONDS * SAMPLE_RATE)

    if args.manifest is not None:
        entries = _load_manifest(args.manifest)
        print(f"Loaded {len(entries)} tracks from manifest {args.manifest}")
    else:
        entries = _load_dataset_dirs(args.dataset_dirs, args.max_tracks_per_dir)

    if args.max_tracks is not None:
        rng.shuffle(entries)  # type: ignore[arg-type]
        entries = entries[: args.max_tracks]

    if len(entries) < 2:
        print("Not enough tracks found across all sources.")
        return 1

    # Genre -> indices into `entries`, used for hard-negative sampling.
    genre_to_indices: dict[str, list[int]] = {}
    for idx, (_, genre) in enumerate(entries):
        genre_to_indices.setdefault(genre, []).append(idx)

    total_tracks = len(entries)
    total_pairs = args.total_pairs if args.total_pairs is not None else args.pairs_per_track * total_tracks
    chunk_size = max(2, args.cache_size)
    print(
        f"\nTotal tracks: {total_tracks}, genres: {len(genre_to_indices)}, "
        f"generating up to {total_pairs} triplets in shuffle-buffers of {chunk_size} "
        f"(hard_negative_ratio={args.hard_negative_ratio}, workers={args.workers})..."
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process tracks in shuffled chunks: decode each chunk once (in parallel),
    # then draw every anchor/negative for that chunk's triplets from the
    # already-decoded set. This is what makes a full pass over a large,
    # non-repeating dataset like FMA tractable - random access across the
    # whole dataset would mean almost every triplet needs 2 fresh decodes.
    order = list(range(total_tracks))
    rng.shuffle(order)  # type: ignore[arg-type]

    tmp_anchors_path = output_dir / "_scratch_anchors.npy"
    tmp_positives_path = output_dir / "_scratch_positives.npy"
    tmp_negatives_path = output_dir / "_scratch_negatives.npy"
    checkpoint_path = output_dir / "_checkpoint.json"

    generated = 0
    decoded_tracks = 0
    unreadable_tracks = 0
    resume_chunk_start = 0

    can_resume = (
        args.resume
        and checkpoint_path.exists()
        and tmp_anchors_path.exists()
        and tmp_positives_path.exists()
        and tmp_negatives_path.exists()
    )
    if can_resume:
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("total_pairs") == total_pairs and checkpoint.get("total_tracks") == total_tracks:
            generated = checkpoint["generated"]
            decoded_tracks = checkpoint["decoded_tracks"]
            unreadable_tracks = checkpoint["unreadable_tracks"]
            resume_chunk_start = checkpoint["next_chunk_start"]
            anchors_mm = np.lib.format.open_memmap(tmp_anchors_path, mode="r+")
            positives_mm = np.lib.format.open_memmap(tmp_positives_path, mode="r+")
            negatives_mm = np.lib.format.open_memmap(tmp_negatives_path, mode="r+")
            print(
                f"Resuming from checkpoint: {generated}/{total_pairs} triplets already generated, "
                f"continuing at track {resume_chunk_start}/{total_tracks}."
            )
        else:
            print("Checkpoint doesn't match current arguments (dataset/pair-count changed); starting over.")
            can_resume = False

    if not can_resume:
        anchors_mm = _open_scratch_memmap(tmp_anchors_path, total_pairs, target_len)
        positives_mm = _open_scratch_memmap(tmp_positives_path, total_pairs, target_len)
        negatives_mm = _open_scratch_memmap(tmp_negatives_path, total_pairs, target_len)

    start_time = time.time()

    for chunk_start in range(resume_chunk_start, total_tracks, chunk_size):
        if generated >= total_pairs:
            break
        chunk_indices = order[chunk_start : chunk_start + chunk_size]
        print(f"  chunk {chunk_start // chunk_size + 1}: decoding {len(chunk_indices)} tracks...", flush=True)
        waveforms = _decode_chunk(chunk_indices, entries, SAMPLE_RATE, args.workers)
        decoded_tracks += len(waveforms)
        unreadable_tracks += len(chunk_indices) - len(waveforms)

        available = list(waveforms.keys())
        if len(available) < 2:
            continue  # whole chunk was unreadable, nothing to pair here

        # Chunk-local genre pools, for hard-negative mining without leaving the buffer.
        chunk_genre_to_indices: dict[str, list[int]] = {}
        for idx in available:
            chunk_genre_to_indices.setdefault(entries[idx][1], []).append(idx)

        for idx_a in available:
            if generated >= total_pairs:
                break
            path_a, genre_a = entries[idx_a]
            waveform_a = waveforms[idx_a]

            for _ in range(args.pairs_per_track):
                if generated >= total_pairs:
                    break
                seg_a = _random_segment(waveform_a, SAMPLE_RATE, SEGMENT_SECONDS, HOP_SECONDS, rng)
                if seg_a is None:
                    break  # this track has no usable segments at all; skip it
                anchor = normalise(_fix_length(seg_a, target_len))

                kind, value = TRANSFORM_PRESETS[rng.integers(0, len(TRANSFORM_PRESETS))]
                try:
                    positive = normalise(
                        _fix_length(apply_transform(seg_a.copy(), SAMPLE_RATE, kind, value, rng=rng), target_len)
                    )
                except Exception:
                    positive = normalise(
                        _fix_length(apply_transform(seg_a.copy(), SAMPLE_RATE, "noise", 15.0, rng=rng), target_len)
                    )

                # Negative: same genre (hard) or any genre (easy), always a different track,
                # always drawn from the currently-decoded chunk.
                use_hard_negative = (
                    args.hard_negative_ratio > 0
                    and rng.random() < args.hard_negative_ratio
                    and len(chunk_genre_to_indices.get(genre_a, [])) > 1
                )
                pool = chunk_genre_to_indices[genre_a] if use_hard_negative else available

                idx_neg = idx_a
                for _ in range(10):
                    candidate = int(pool[int(rng.integers(0, len(pool)))])
                    if candidate != idx_a:
                        idx_neg = candidate
                        break
                if idx_neg == idx_a:
                    continue  # chunk of size 1 for this genre and no easy fallback; skip

                seg_neg = _random_segment(waveforms[idx_neg], SAMPLE_RATE, SEGMENT_SECONDS, HOP_SECONDS, rng)
                if seg_neg is None:
                    continue
                negative = normalise(_fix_length(seg_neg, target_len))

                anchors_mm[generated] = anchor
                positives_mm[generated] = positive
                negatives_mm[generated] = negative
                generated += 1

        # Flush + checkpoint after every chunk so an interrupted run (this environment has
        # occasionally killed long-lived background processes) can resume with --resume
        # instead of losing progress and starting the whole pass over.
        anchors_mm.flush()
        positives_mm.flush()
        negatives_mm.flush()
        checkpoint_path.write_text(json.dumps({
            "total_pairs": total_pairs,
            "total_tracks": total_tracks,
            "generated": generated,
            "decoded_tracks": decoded_tracks,
            "unreadable_tracks": unreadable_tracks,
            "next_chunk_start": chunk_start + chunk_size,
        }))

        elapsed = time.time() - start_time
        rate = generated / elapsed if elapsed > 0 else 0
        print(
            f"  chunk {chunk_start // chunk_size + 1}: {decoded_tracks} tracks decoded "
            f"({unreadable_tracks} unreadable), {generated}/{total_pairs} triplets "
            f"({rate:.1f}/s, {elapsed:.0f}s elapsed)..."
        )

    if generated < total_pairs:
        print(f"Warning: only generated {generated}/{total_pairs} triplets (ran out of tracks).")

    anchors_mm.flush()
    positives_mm.flush()
    negatives_mm.flush()
    del anchors_mm, positives_mm, negatives_mm
    gc.collect()  # release the memmap file handles before renaming (required on Windows)

    if generated == 0:
        print("No triplets were generated - check that your audio files are readable.")
        for tmp in (tmp_anchors_path, tmp_positives_path, tmp_negatives_path):
            tmp.unlink(missing_ok=True)
        return 1

    anchors_shape = _finalize_memmap(tmp_anchors_path, output_dir / "anchors.npy", generated, total_pairs)
    positives_shape = _finalize_memmap(tmp_positives_path, output_dir / "positives.npy", generated, total_pairs)
    negatives_shape = _finalize_memmap(tmp_negatives_path, output_dir / "negatives.npy", generated, total_pairs)
    checkpoint_path.unlink(missing_ok=True)

    print(f"\nCreated {generated} triplets in {output_dir}")
    print(f"  anchors.npy:   {anchors_shape}")
    print(f"  positives.npy: {positives_shape}")
    print(f"  negatives.npy: {negatives_shape}")
    print(f"  Tracks decoded: {decoded_tracks} ({unreadable_tracks} unreadable/skipped)")
    print(f"  Total time: {time.time() - start_time:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
