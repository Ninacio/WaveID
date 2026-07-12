"""
Build a flat (path, genre) manifest CSV from an FMA dataset dump.

FMA stores audio nested as fma_small/000/000002.mp3, fma_small/001/...,
grouped by the first 3 digits of a zero-padded track id — not in
per-genre folders like GTZAN. Genre labels live separately in
fma_metadata/tracks.csv (a 2-row-header CSV keyed by track id).

This script joins the two so create_contrastive_data.py can do
genre-aware (hard-negative) sampling over FMA without reorganising
100k+ audio files on disk.

Usage:
    python -m scripts.build_fma_manifest ^
        --audio-dir "D:/WaveID-datasets/fma_small" ^
        --tracks-csv "D:/WaveID-datasets/fma_metadata/tracks.csv" ^
        --output "data/manifests/fma_small.csv"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


def _load_genre_map(tracks_csv: Path) -> dict[int, str]:
    """Map track_id -> top-level genre name from FMA's tracks.csv."""
    # tracks.csv has two header rows: (category, field). "track_id" is the index.
    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
    genre_col = ("track", "genre_top")
    if genre_col not in tracks.columns:
        raise SystemExit(
            f"Expected column {genre_col} not found in {tracks_csv}. "
            "Is this the official FMA tracks.csv?"
        )
    genres = tracks[genre_col].dropna()
    return {int(track_id): str(genre) for track_id, genre in genres.items()}


def _track_id_from_filename(path: Path) -> int | None:
    # FMA filenames are zero-padded track ids, e.g. "000002.mp3".
    stem = path.stem
    if not stem.isdigit():
        return None
    return int(stem)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an FMA (path, genre) manifest.")
    parser.add_argument("--audio-dir", required=True, type=Path, help="Root of the extracted fma_small/medium/large folder.")
    parser.add_argument("--tracks-csv", required=True, type=Path, help="Path to fma_metadata/tracks.csv.")
    parser.add_argument("--output", required=True, type=Path, help="Output manifest CSV path.")
    parser.add_argument("--min-per-genre", type=int, default=0, help="Drop genres with fewer than this many tracks found on disk.")
    args = parser.parse_args()

    if not args.audio_dir.is_dir():
        raise SystemExit(f"Audio directory not found: {args.audio_dir}")
    if not args.tracks_csv.is_file():
        raise SystemExit(f"tracks.csv not found: {args.tracks_csv}")

    print(f"Reading genre metadata from {args.tracks_csv} ...")
    genre_map = _load_genre_map(args.tracks_csv)
    print(f"  {len(genre_map)} tracks have a top-level genre label.")

    print(f"Scanning audio files under {args.audio_dir} ...")
    rows: list[tuple[str, str]] = []
    genre_counts: dict[str, int] = {}
    skipped_no_genre = 0
    skipped_bad_name = 0

    for path in args.audio_dir.rglob("*.mp3"):
        track_id = _track_id_from_filename(path)
        if track_id is None:
            skipped_bad_name += 1
            continue
        genre = genre_map.get(track_id)
        if genre is None:
            skipped_no_genre += 1
            continue
        rows.append((str(path.resolve()), genre))
        genre_counts[genre] = genre_counts.get(genre, 0) + 1

    if args.min_per_genre > 0:
        keep_genres = {g for g, n in genre_counts.items() if n >= args.min_per_genre}
        before = len(rows)
        rows = [(p, g) for p, g in rows if g in keep_genres]
        print(f"  Dropped {before - len(rows)} tracks from genres with < {args.min_per_genre} tracks.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "genre"])
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} tracks to {args.output}")
    print(f"  Skipped (no genre label): {skipped_no_genre}")
    print(f"  Skipped (unexpected filename): {skipped_bad_name}")
    print(f"  Genres: {len(genre_counts)}")
    for genre, count in sorted(genre_counts.items(), key=lambda kv: -kv[1])[:20]:
        print(f"    {genre:<20} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
