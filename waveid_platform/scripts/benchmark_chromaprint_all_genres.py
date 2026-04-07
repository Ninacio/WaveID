"""
Run Chromaprint (fpcalc + alignment) on the same transform suite as WaveID,
for N reference tracks per GTZAN genre (default 5 × 10 genres = 50 refs, 850 queries).

Requires: fpcalc (FPCALC env or PATH), ffmpeg for MP3 transforms in evaluate_transformations.

Usage (from waveid_platform):
    python -m scripts.benchmark_chromaprint_all_genres ^
        --gtzan-root "../datasets/GTZAN/genres_original" ^
        --tracks-per-genre 5
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from scripts.benchmark_chromaprint import chromaprint_query_rows

GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

EXPECTED_QUERIES = 17


def _generate_transforms(reference: Path, output_dir: Path, max_seconds: float) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable,
        "-m",
        "scripts.evaluate_transformations",
        "--input",
        str(reference),
        "--output-dir",
        str(output_dir),
        "--max-seconds",
        str(max_seconds),
    ]
    proc = subprocess.run(argv, cwd=Path(__file__).resolve().parents[1])
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Chromaprint benchmark across all GTZAN genres.")
    parser.add_argument("--gtzan-root", type=Path, required=True, help="Path to genres_original/.")
    parser.add_argument("--tracks-per-genre", type=int, default=5)
    parser.add_argument("--max-seconds", type=float, default=5.0)
    parser.add_argument(
        "--work-dir-root",
        type=Path,
        default=Path("data/query/eval/chromaprint_all_genres"),
        help="Where to write transformed WAVs per reference.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/index/benchmark_chromaprint_all_genres.csv"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("data/index/benchmark_chromaprint_all_genres_summary.csv"),
    )
    parser.add_argument(
        "--skip-existing-transforms",
        action="store_true",
        help="Skip evaluate_transformations if the work dir already has enough WAVs.",
    )
    parser.add_argument("--fpcalc", default=None, help="Path to fpcalc.exe (optional).")
    parser.add_argument("--max-length", type=int, default=120)
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()

    gtzan = args.gtzan_root.resolve()
    fpcalc = args.fpcalc

    all_rows: list[dict[str, object]] = []
    idx = 0

    for genre in GENRES:
        genre_dir = gtzan / genre
        if not genre_dir.is_dir():
            print(f"SKIP missing genre dir: {genre_dir}")
            continue
        wavs = sorted(genre_dir.glob("*.wav"))[: args.tracks_per_genre]
        for ref_path in wavs:
            idx += 1
            work_dir = args.work_dir_root / genre / ref_path.stem
            n_existing = len(list(work_dir.glob("*.wav")))
            if not args.skip_existing_transforms or n_existing < EXPECTED_QUERIES:
                code = _generate_transforms(ref_path, work_dir, args.max_seconds)
                if code != 0:
                    print(f"Transform generation failed for {ref_path}")
                    return code
            query_wavs = sorted(work_dir.glob("*.wav"))
            if len(query_wavs) < EXPECTED_QUERIES:
                print(f"Expected ~{EXPECTED_QUERIES} queries in {work_dir}, found {len(query_wavs)}")
                return 1

            try:
                rows = chromaprint_query_rows(
                    ref_path,
                    work_dir,
                    fpcalc=fpcalc,
                    max_length=args.max_length,
                    threshold=args.threshold,
                )
            except FileNotFoundError:
                print(
                    "fpcalc not found. Set --fpcalc or environment variable FPCALC to fpcalc.exe."
                )
                return 1
            except RuntimeError as exc:
                print(exc)
                return 1

            for r in rows:
                all_rows.append(
                    {
                        "genre": genre,
                        "reference_file": ref_path.name,
                        **{k: r[k] for k in r},
                    }
                )
            hits = sum(int(r["chromaprint_hit"]) for r in rows)
            print(f"[{idx}] {genre}/{ref_path.name}  Chromaprint hits: {hits}/{len(rows)}")

    if not all_rows:
        print("No rows produced.")
        return 1

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["genre", "reference_file"] + [
        k for k in all_rows[0] if k not in ("genre", "reference_file")
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows -> {args.output_csv}")

    # Summary: per genre, per transform, overall
    by_genre: dict[str, list] = defaultdict(list)
    by_transform: dict[str, list] = defaultdict(list)
    for row in all_rows:
        by_genre[str(row["genre"])].append(row)
        by_transform[str(row["transform"])].append(row)

    summary_rows: list[dict[str, object]] = []
    n_total = len(all_rows)
    h_total = sum(int(r["chromaprint_hit"]) for r in all_rows)
    sims = [float(r["chromaprint_similarity"]) for r in all_rows]
    summary_rows.append(
        {
            "group": "overall",
            "key": "all",
            "n": n_total,
            "hits": h_total,
            "hit_rate": round(100.0 * h_total / n_total, 2),
            "mean_sim": round(sum(sims) / len(sims), 4),
        }
    )
    for genre in GENRES:
        gr = by_genre.get(genre, [])
        if not gr:
            continue
        hg = sum(int(r["chromaprint_hit"]) for r in gr)
        sg = [float(r["chromaprint_similarity"]) for r in gr]
        summary_rows.append(
            {
                "group": "genre",
                "key": genre,
                "n": len(gr),
                "hits": hg,
                "hit_rate": round(100.0 * hg / len(gr), 2),
                "mean_sim": round(sum(sg) / len(sg), 4),
            }
        )
    for t in sorted(by_transform.keys()):
        tr = by_transform[t]
        ht = sum(int(r["chromaprint_hit"]) for r in tr)
        st = [float(r["chromaprint_similarity"]) for r in tr]
        summary_rows.append(
            {
                "group": "transform",
                "key": t,
                "n": len(tr),
                "hits": ht,
                "hit_rate": round(100.0 * ht / len(tr), 2),
                "mean_sim": round(sum(st) / len(st), 4),
            }
        )

    with args.summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Summary -> {args.summary_csv}")
    print(f"OVERALL Chromaprint hit rate: {100.0 * h_total / n_total:.1f}% ({h_total}/{n_total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
