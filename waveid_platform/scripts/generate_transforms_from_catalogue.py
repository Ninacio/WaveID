"""
Generate transformed clips from all ingested reference tracks in the catalogue.

Reads the catalogue, finds each reference audio file on disk, and runs
evaluate_transformations to create pitch/tempo/noise/crop variants.
Useful for creating query clips to test robustness against preloaded tracks.

Usage:
    python -m scripts.generate_transforms_from_catalogue
    python -m scripts.generate_transforms_from_catalogue --limit 3 --max-seconds 5
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Resolve paths relative to waveid_platform
PLATFORM_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = PLATFORM_ROOT / "data" / "index"
REFERENCE_DIR = PLATFORM_ROOT / "data" / "reference"
EVAL_BASE = PLATFORM_ROOT / "data" / "query" / "eval"
CATALOGUE_PATH = INDEX_DIR / "catalogue.json"


def _sanitize(s: str) -> str:
    """Make a safe folder name from a filename stem."""
    return re.sub(r"[^\w\-.]", "_", s).replace(".", "_").strip("_") or "track"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate transformed clips from all catalogue reference tracks."
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=EVAL_BASE,
        help="Base directory for transformed outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of tracks to process (default: all).",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Max duration per track (for faster runs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without running.",
    )
    args = parser.parse_args()

    if not CATALOGUE_PATH.exists():
        print(f"Catalogue not found: {CATALOGUE_PATH}")
        print("Ingest reference tracks first (CLI or API).")
        return 1

    import json

    data = json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))
    tracks = data.get("tracks", {})
    if not tracks:
        print("Catalogue is empty. Ingest reference tracks first.")
        return 1

    # Build list of (track_id, filename) for tracks that have a reference file
    to_process: list[tuple[str, str]] = []
    for track_id, meta in tracks.items():
        filename = meta.get("filename")
        if not filename:
            continue
        suffix = Path(filename).suffix.lower()
        ref_path = REFERENCE_DIR / f"{track_id}{suffix}"
        if ref_path.exists():
            to_process.append((track_id, filename))
        else:
            print(f"Skip {track_id}: reference file not found: {ref_path}")

    if args.limit is not None:
        to_process = to_process[: args.limit]

    if not to_process:
        print("No reference files found in data/reference/.")
        return 1

    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    cmd_base = [
        sys.executable,
        "-m",
        "scripts.evaluate_transformations",
    ]
    if args.max_seconds is not None:
        cmd_base.extend(["--max-seconds", str(args.max_seconds)])

    for i, (track_id, filename) in enumerate(to_process, 1):
        suffix = Path(filename).suffix.lower()
        ref_path = REFERENCE_DIR / f"{track_id}{suffix}"
        stem = Path(filename).stem
        out_name = f"{_sanitize(stem)}_{track_id[:8]}"
        out_dir = output_base / out_name

        cmd = cmd_base + ["--input", str(ref_path), "--output-dir", str(out_dir)]

        if args.dry_run:
            print(f"[{i}/{len(to_process)}] Would run: {' '.join(cmd)}")
            continue

        print(f"[{i}/{len(to_process)}] {filename} -> {out_dir}")
        try:
            subprocess.run(cmd, check=True, cwd=str(PLATFORM_ROOT))
        except subprocess.CalledProcessError as e:
            print(f"  Error: {e}")
            return 1

    if not args.dry_run:
        print(f"\nDone. Transformed clips in: {output_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
