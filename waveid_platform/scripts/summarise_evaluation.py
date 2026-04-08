"""
Summarise evaluation CSV results by transform type.

Usage:
    python -m scripts.summarise_evaluation --input-csv "data/index/eval_results.csv"
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _rate(num: int, den: int) -> float:
    return (num / den) if den else 0.0


def _build_group_stats(
    rows: list[dict[str, str]],
    key_field: str,
) -> dict[str, dict[str, int]]:
    """Count total queries, top-1 hits, and top-k hits for each unique value of key_field."""
    grouped: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "top1": 0, "topk": 0}
    )
    for row in rows:
        key = row.get(key_field, "unknown")
        hit_top1 = int(row.get("hit_top1", "0") or 0)
        hit_topk = int(row.get("hit_topk", "0") or 0)
        grouped[key]["n"] += 1
        grouped[key]["top1"] += hit_top1
        grouped[key]["topk"] += hit_topk
    return grouped


def _write_group_csv(
    output_path: Path,
    overall: dict[str, int],
    grouped: dict[str, dict[str, int]],
    key_name: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [key_name, "n", "top1_hits", "top1_rate", "topk_hits", "topk_rate"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                key_name: "overall",
                "n": overall["n"],
                "top1_hits": overall["top1"],
                "top1_rate": f"{_rate(overall['top1'], overall['n']):.6f}",
                "topk_hits": overall["topk"],
                "topk_rate": f"{_rate(overall['topk'], overall['n']):.6f}",
            }
        )
        for key in sorted(grouped.keys()):
            stats = grouped[key]
            n = stats["n"]
            writer.writerow(
                {
                    key_name: key,
                    "n": n,
                    "top1_hits": stats["top1"],
                    "top1_rate": f"{_rate(stats['top1'], n):.6f}",
                    "topk_hits": stats["topk"],
                    "topk_rate": f"{_rate(stats['topk'], n):.6f}",
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarise WaveID evaluation results.")
    parser.add_argument("--input-csv", required=True, help="Evaluation CSV path.")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to save transform-grouped summary as CSV.",
    )
    parser.add_argument(
        "--severity-output-csv",
        default=None,
        help="Optional path to save transform+severity summary as CSV.",
    )
    parser.add_argument(
        "--report-md",
        default=None,
        help="Optional path to save a markdown report.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"Input CSV not found: {input_path}")
        return 1

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = list(csv.DictReader(handle))

    if not reader:
        print("No rows found in input CSV.")
        return 1

    grouped = _build_group_stats(reader, "transform")
    severity_rows: list[dict[str, str]] = []
    overall = {"n": 0, "top1": 0, "topk": 0}
    for row in reader:
        overall["n"] += 1
        overall["top1"] += int(row.get("hit_top1", "0") or 0)
        overall["topk"] += int(row.get("hit_topk", "0") or 0)
        transform = row.get("transform", "unknown")
        severity = row.get("severity", "unknown")
        severity_rows.append(
            {
                **row,
                "transform_severity": f"{transform}:{severity}",
            }
        )
    severity_grouped = _build_group_stats(severity_rows, "transform_severity")

    print("Evaluation Summary")
    print("------------------")
    print(
        f"overall: n={overall['n']} "
        f"top1={overall['top1']}/{overall['n']} ({_rate(overall['top1'], overall['n']):.2%}) "
        f"topk={overall['topk']}/{overall['n']} ({_rate(overall['topk'], overall['n']):.2%})"
    )
    print("")
    print("By transform:")
    for transform in sorted(grouped.keys()):
        stats = grouped[transform]
        n = stats["n"]
        print(
            f"- {transform}: n={n} "
            f"top1={stats['top1']}/{n} ({_rate(stats['top1'], n):.2%}) "
            f"topk={stats['topk']}/{n} ({_rate(stats['topk'], n):.2%})"
        )

    if args.output_csv:
        output_path = Path(args.output_csv)
        _write_group_csv(output_path, overall, grouped, "transform")
        print("")
        print(f"Saved grouped summary CSV: {output_path}")

    if args.severity_output_csv:
        severity_path = Path(args.severity_output_csv)
        _write_group_csv(severity_path, overall, severity_grouped, "transform_severity")
        print(f"Saved severity summary CSV: {severity_path}")

    if args.report_md:
        report_path = Path(args.report_md)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        lines.append("# Evaluation Summary Report")
        lines.append("")
        lines.append(
            f"- Overall samples: {overall['n']}"
        )
        lines.append(
            f"- Top-1 accuracy: {_rate(overall['top1'], overall['n']):.2%} ({overall['top1']}/{overall['n']})"
        )
        lines.append(
            f"- Top-k accuracy: {_rate(overall['topk'], overall['n']):.2%} ({overall['topk']}/{overall['n']})"
        )
        lines.append("")
        lines.append("## By Transform")
        lines.append("")
        lines.append("| Transform | N | Top-1 | Top-k |")
        lines.append("| --- | ---: | ---: | ---: |")
        for transform in sorted(grouped.keys()):
            stats = grouped[transform]
            n = stats["n"]
            lines.append(
                f"| {transform} | {n} | {_rate(stats['top1'], n):.2%} | {_rate(stats['topk'], n):.2%} |"
            )
        lines.append("")
        lines.append("## By Transform Severity")
        lines.append("")
        lines.append("| Transform:Severity | N | Top-1 | Top-k |")
        lines.append("| --- | ---: | ---: | ---: |")
        for key in sorted(severity_grouped.keys()):
            stats = severity_grouped[key]
            n = stats["n"]
            lines.append(
                f"| {key} | {n} | {_rate(stats['top1'], n):.2%} | {_rate(stats['topk'], n):.2%} |"
            )
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Saved markdown report: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
