"""
Batch ingestion script for reference datasets.

Usage:
    python -m scripts.ingest_dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path

from waveid_backend.services.dataset_loader import ingest_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch ingest an audio dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/gtzan/genres_original"),
        help="Path to dataset root directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of tracks to ingest.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive directory scanning.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first file processing error.",
    )
    args = parser.parse_args()

    summary = ingest_dataset(
        args.dataset,
        recursive=not args.no_recursive,
        limit=args.limit,
        skip_errors=not args.fail_fast,
    )
    print(f"Ingested {summary['tracks_ingested']} tracks")
    print(f"Segments stored: {summary['segments_ingested']}")
    if summary["errors"]:
        print(f"Errors: {len(summary['errors'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
