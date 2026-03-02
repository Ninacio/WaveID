"""
Run evaluation pipeline over multiple reference tracks and combine results.

Usage:
    python -m scripts.run_evaluation_sweep --references-dir "path/to/gtzan/blues"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

from scripts.run_eval_pipeline import main as run_pipeline_main
from scripts.summarise_evaluation import main as summarise_main


def _iter_audio_files(root: Path, limit: int | None) -> list[Path]:
    files = sorted(
        [path for path in root.glob("*.wav") if path.is_file()]
        + [path for path in root.glob("*.mp3") if path.is_file()]
    )
    if limit is not None:
        return files[:limit]
    return files


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run evaluation pipeline over multiple references.")
    parser.add_argument("--references-dir", required=True, help="Directory with reference WAV/MP3 files.")
    parser.add_argument(
        "--limit-references",
        type=int,
        default=3,
        help="Optional cap on number of references for the sweep.",
    )
    parser.add_argument("--max-seconds", type=float, default=5.0, help="Max seconds per clip.")
    parser.add_argument("--max-query-segments", type=int, default=1, help="Query segment cap.")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k for hit metrics.")
    parser.add_argument(
        "--limit-queries",
        type=int,
        default=3,
        help="Optional max transformed queries per reference.",
    )
    parser.add_argument(
        "--fresh-index",
        action="store_true",
        help="Use isolated in-memory catalogue/index for each reference run.",
    )
    parser.add_argument(
        "--combined-csv",
        default="data/index/eval_results_sweep.csv",
        help="Combined detailed CSV output.",
    )
    parser.add_argument(
        "--summary-csv",
        default="data/index/eval_summary_sweep.csv",
        help="Combined transform summary CSV.",
    )
    parser.add_argument(
        "--severity-summary-csv",
        default="data/index/eval_summary_sweep_severity.csv",
        help="Combined transform+severity summary CSV.",
    )
    parser.add_argument(
        "--report-md",
        default="data/index/eval_report_sweep.md",
        help="Combined markdown report output.",
    )
    parser.add_argument(
        "--model-version",
        choices=["baseline-v1", "contrastive-v1"],
        default=None,
        help="Embedding model: baseline-v1 or contrastive-v1.",
    )
    args = parser.parse_args()

    references_dir = Path(args.references_dir)
    if not references_dir.exists():
        print(f"References directory not found: {references_dir}")
        return 1

    refs = _iter_audio_files(references_dir, args.limit_references)
    if not refs:
        print(f"No reference WAV/MP3 files found in: {references_dir}")
        return 1

    combined_rows: list[dict[str, str]] = []
    import sys

    old_argv = sys.argv
    try:
        for index, ref in enumerate(refs, start=1):
            work_dir = Path("data/query/eval/sweep") / ref.stem
            detail_csv = Path("data/index") / f"eval_results_{ref.stem}.csv"
            summary_csv = Path("data/index") / f"eval_summary_{ref.stem}.csv"
            pipeline_argv = [
                "scripts.run_eval_pipeline",
                "--reference",
                str(ref),
                "--work-dir",
                str(work_dir),
                "--max-seconds",
                str(args.max_seconds),
                "--max-query-segments",
                str(args.max_query_segments),
                "--top-k",
                str(args.top_k),
                "--limit-queries",
                str(args.limit_queries),
                "--eval-csv",
                str(detail_csv),
                "--summary-csv",
                str(summary_csv),
            ]
            if args.fresh_index:
                pipeline_argv.append("--fresh-index")
            if args.model_version is not None:
                pipeline_argv.extend(["--model-version", args.model_version])
            print(f"[{index}/{len(refs)}] Running pipeline for: {ref.name}")
            sys.argv = pipeline_argv
            code = run_pipeline_main()
            if code != 0:
                print(f"Pipeline failed for: {ref}")
                return code

            rows = _read_rows(detail_csv)
            for row in rows:
                row["reference_file"] = ref.name
            combined_rows.extend(rows)
    finally:
        sys.argv = old_argv

    if not combined_rows:
        print("No evaluation rows produced.")
        return 1

    combined_path = Path(args.combined_csv)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(combined_rows[0].keys())
    with combined_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)
    print(f"Saved combined CSV: {combined_path}")

    # Summarise combined results
    old_argv = sys.argv
    try:
        summary_argv = [
            "scripts.summarise_evaluation",
            "--input-csv",
            str(combined_path),
            "--output-csv",
            args.summary_csv,
            "--severity-output-csv",
            args.severity_summary_csv,
            "--report-md",
            args.report_md,
        ]
        sys.argv = summary_argv
        code = summarise_main()
        if code != 0:
            return code
    finally:
        sys.argv = old_argv

    print("Sweep complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
