"""
Run the full evaluation pipeline in one command:
1) Generate transformed clips
2) Run evaluation and write detailed CSV
3) Summarise metrics and write grouped CSV

Usage:
    python -m scripts.run_eval_pipeline --reference "path/to/ref.wav"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.evaluate_transformations import main as transformations_main
from scripts.run_evaluation import main as evaluation_main
from scripts.summarise_evaluation import main as summary_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full WaveID evaluation pipeline.")
    parser.add_argument("--reference", required=True, help="Reference WAV/MP3 path.")
    parser.add_argument(
        "--work-dir",
        default="data/query/eval/pipeline_run",
        help="Directory for generated transformed clips.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=5.0,
        help="Max seconds to process for quick runs.",
    )
    parser.add_argument(
        "--max-query-segments",
        type=int,
        default=1,
        help="Segment cap for faster query evaluation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k tracks for hit calculation.",
    )
    parser.add_argument(
        "--limit-queries",
        type=int,
        default=3,
        help="Optional maximum number of transformed queries to evaluate.",
    )
    parser.add_argument(
        "--fresh-index",
        action="store_true",
        help="Use isolated in-memory catalogue/index during evaluation.",
    )
    parser.add_argument(
        "--eval-csv",
        default="data/index/eval_results_pipeline.csv",
        help="Detailed evaluation CSV output.",
    )
    parser.add_argument(
        "--summary-csv",
        default="data/index/eval_summary_pipeline.csv",
        help="Grouped summary CSV output.",
    )
    parser.add_argument(
        "--model-version",
        choices=["baseline-v1", "contrastive-v1"],
        default=None,
        help="Embedding model: baseline-v1 or contrastive-v1.",
    )
    args = parser.parse_args()

    reference_path = Path(args.reference)
    if not reference_path.exists():
        print(f"Reference not found: {reference_path}")
        return 1

    work_dir = Path(args.work_dir)

    # 1) Transform generation
    transform_argv = [
        "scripts.evaluate_transformations",
        "--input",
        str(reference_path),
        "--output-dir",
        str(work_dir),
    ]
    if args.max_seconds is not None and args.max_seconds > 0:
        transform_argv.extend(["--max-seconds", str(args.max_seconds)])
    import sys

    old_argv = sys.argv
    try:
        sys.argv = transform_argv
        code = transformations_main()
        if code != 0:
            return code

        # 2) Evaluation run
        eval_argv = [
            "scripts.run_evaluation",
            "--reference",
            str(reference_path),
            "--queries-dir",
            str(work_dir),
            "--output-csv",
            args.eval_csv,
            "--max-query-segments",
            str(args.max_query_segments),
            "--top-k",
            str(args.top_k),
            "--limit-queries",
            str(args.limit_queries),
        ]
        if args.fresh_index:
            eval_argv.append("--fresh-index")
        if args.model_version is not None:
            eval_argv.extend(["--model-version", args.model_version])
        sys.argv = eval_argv
        code = evaluation_main()
        if code != 0:
            return code

        # 3) Summary
        summary_argv = [
            "scripts.summarise_evaluation",
            "--input-csv",
            args.eval_csv,
            "--output-csv",
            args.summary_csv,
        ]
        sys.argv = summary_argv
        code = summary_main()
        if code != 0:
            return code
    finally:
        sys.argv = old_argv

    print("")
    print("Pipeline complete.")
    print(f"Detailed CSV: {args.eval_csv}")
    print(f"Summary CSV: {args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
