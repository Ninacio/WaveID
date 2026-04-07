"""
Compare query clips to a reference track using Chromaprint (fpcalc + fingerprint alignment).

Requires:
  - fpcalc on PATH, or set environment variable FPCALC to the full path to fpcalc.exe
    (e.g. C:\\...\\WaveID\\tools\\Chromaprint\\chromaprint-fpcalc-1.6.0-windows-x86_64\\fpcalc.exe)
  - Same --reference path as in run_evaluation / run_eval_pipeline for a fair comparison to WaveID.

Similarity score follows pyacoustid's alignment of decompressed sub-fingerprints (0 = no match, 1 = strong match).

Usage:
    set FPCALC=C:\\...\\fpcalc.exe
    python -m scripts.benchmark_chromaprint --reference "../datasets/GTZAN/genres_original/blues/blues.00000.wav" --queries-dir "data/query/eval/sweep/blues.00000"

WaveID (baseline or contrastive) on the same clips: run the full pipeline per reference, e.g.::

    python -m scripts.run_eval_pipeline --reference "../datasets/GTZAN/genres_original/blues/blues.00000.wav" \\
        --work-dir "data/query/eval/bench_contrastive/blues.00000" --fresh-index --limit-queries 100 \\
        --model-version contrastive-v1 --eval-csv "data/index/benchmark_waveid_contrastive_blues.00000.csv" \\
        --summary-csv "data/index/benchmark_summary_contrastive_blues.00000.csv"

Repeat for blues.00001--blues.00004; aggregate the five detail CSVs for the WaveID vs Chromaprint comparison (dissertation Chapter~7).

All ten genres (850 queries)::

    python -m scripts.benchmark_chromaprint_all_genres --gtzan-root "../datasets/GTZAN/genres_original" \\
        --tracks-per-genre 5 --fpcalc "C:\\\\path\\\\to\\\\fpcalc.exe"
"""

from __future__ import annotations

import argparse
import base64
import csv
import os
import struct
import subprocess
from pathlib import Path

# Match pyacoustid defaults for chromaprint_compare-style alignment.
_MAX_ALIGN_OFFSET = 120
_MAX_BIT_ERROR = 2


def _parse_transform(stem: str) -> tuple[str, str]:
    if stem.endswith("_orig"):
        return ("orig", "none")
    for kind in ("compound", "bandpass", "lossy", "pitch", "tempo", "noise", "crop"):
        marker = f"_{kind}_"
        if marker in stem:
            return (kind, stem.split(marker, 1)[1])
    return ("unknown", "unknown")


def _b64decode_chromaprint(fp: bytes) -> bytes:
    fp = fp.strip()
    pad = (-len(fp)) % 4
    return base64.b64decode(fp + b"=" * pad, altchars=b"-_")


def _popcount(x: int) -> int:
    return bin(x).count("1")


def _decode_fingerprint_line(line: bytes) -> list[int]:
    """Decode FINGERPRINT= line from fpcalc (base64url, algorithm byte + uint32 LE stream)."""
    if not line.startswith(b"FINGERPRINT="):
        raise ValueError("expected FINGERPRINT=")
    raw = _b64decode_chromaprint(line.split(b"=", 1)[1])
    if len(raw) < 2:
        return []
    body = raw[1:]
    n = len(body) // 4
    return list(struct.unpack("<" + "I" * n, body[: n * 4]))


def _match_fingerprints(a: list[int], b: list[int]) -> float:
    if not a or not b:
        return 0.0
    asize, bsize = len(a), len(b)
    counts = [0] * (asize + bsize + 1)
    for i in range(asize):
        jbegin = max(0, i - _MAX_ALIGN_OFFSET)
        jend = min(bsize, i + _MAX_ALIGN_OFFSET)
        for j in range(jbegin, jend):
            if _popcount(a[i] ^ b[j]) <= _MAX_BIT_ERROR:
                counts[i - j + bsize] += 1
    topcount = max(counts)
    return float(topcount) / float(min(asize, bsize))


def _fpcalc_command() -> str:
    return os.environ.get("FPCALC", "fpcalc")


def chromaprint_query_rows(
    reference: Path,
    queries_dir: Path,
    fpcalc: str | None = None,
    max_length: int = 120,
    threshold: float = 0.35,
) -> list[dict[str, object]]:
    """
    Return one row per query WAV: query_file, transform, severity,
    chromaprint_similarity, chromaprint_hit, error.
    """
    fc = fpcalc or _fpcalc_command()
    ref_fp = fingerprint_file(reference.resolve(), fc, max_length)
    rows: list[dict[str, object]] = []
    for query_path in sorted(queries_dir.glob("*.wav")):
        transform, severity = _parse_transform(query_path.stem)
        try:
            q_fp = fingerprint_file(query_path.resolve(), fc, max_length)
            sim = _match_fingerprints(ref_fp, q_fp)
            err = ""
        except RuntimeError as exc:
            sim = 0.0
            err = str(exc)
        rows.append(
            {
                "query_file": query_path.name,
                "transform": transform,
                "severity": severity,
                "chromaprint_similarity": round(sim, 6),
                "chromaprint_hit": int(sim >= threshold),
                "error": err,
            }
        )
    return rows


def fingerprint_file(path: Path, fpcalc: str, max_length: int) -> list[int]:
    proc = subprocess.run(
        [fpcalc, "-length", str(max_length), str(path)],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace") if proc.stderr else ""
        raise RuntimeError(f"fpcalc failed ({proc.returncode}): {stderr}")
    for line in proc.stdout.splitlines():
        if line.startswith(b"FINGERPRINT="):
            return _decode_fingerprint_line(line)
    raise RuntimeError("fpcalc produced no FINGERPRINT=")


def main() -> int:
    parser = argparse.ArgumentParser(description="Chromaprint similarity vs reference for each query WAV.")
    parser.add_argument("--reference", required=True, type=Path, help="Reference WAV (same file as WaveID evaluation).")
    parser.add_argument("--queries-dir", required=True, type=Path, help="Directory of transformed query WAVs.")
    parser.add_argument(
        "--fpcalc",
        default=None,
        help="Path to fpcalc.exe (default: FPCALC env var, else 'fpcalc' on PATH).",
    )
    parser.add_argument("--max-length", type=int, default=120, help="fpcalc -length cap in seconds.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Similarity at or above this counts as chromaprint_hit (tune per corpus; default 0.35).",
    )
    parser.add_argument("--output-csv", type=Path, default=Path("data/index/chromaprint_benchmark.csv"))
    args = parser.parse_args()

    if not args.reference.exists():
        print(f"Reference not found: {args.reference}")
        return 1
    if not args.queries_dir.is_dir():
        print(f"Queries dir not found: {args.queries_dir}")
        return 1

    fpcalc = args.fpcalc or _fpcalc_command()
    query_files = sorted(args.queries_dir.glob("*.wav"))
    if not query_files:
        print(f"No WAV files in {args.queries_dir}")
        return 1

    try:
        rows = chromaprint_query_rows(
            args.reference,
            args.queries_dir,
            fpcalc=fpcalc,
            max_length=args.max_length,
            threshold=args.threshold,
        )
    except FileNotFoundError:
        print(
            "fpcalc not found. Set --fpcalc to fpcalc.exe or set environment variable FPCALC to its full path."
        )
        return 1
    except RuntimeError as exc:
        print(exc)
        return 1

    if not rows:
        print("No query rows produced.")
        return 1

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    hits = sum(int(r["chromaprint_hit"]) for r in rows)
    print(f"Reference: {args.reference}")
    print(f"Queries: {len(rows)}  Chromaprint hits (>={args.threshold}): {hits}/{len(rows)}")
    print(f"Wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
