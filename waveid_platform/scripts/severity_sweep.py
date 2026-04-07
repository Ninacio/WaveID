"""
Severity sweep: evaluate WaveID accuracy across a fine-grained grid of
transform severities to find breakpoints where identification degrades.

Usage:
    python -m scripts.severity_sweep ^
        --references ../datasets/GTZAN/genres_original/blues/blues.00000.wav ^
                     ../datasets/GTZAN/genres_original/blues/blues.00001.wav ^
        --output-csv data/index/severity_sweep.csv ^
        --max-seconds 5
"""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from waveid_backend.services.transforms import (
    pitch_shift,
    time_stretch,
    add_noise,
    crop_end,
    lossy_mp3_roundtrip,
    normalise,
)
from scripts.run_evaluation import (
    ingest_reference,
    query_track_matches,
)
from waveid_backend.services.catalogue import reset_state as reset_catalogue_state
from waveid_backend.services.search import reset_state as reset_search_state

SWEEP_GRID: dict[str, list[float]] = {
    "pitch": [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6],
    "tempo": [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30],
    "noise": [5, 8, 10, 12, 15, 20, 25, 30],
    "crop": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
    "lossy": [32, 48, 64, 80, 96, 128, 192],
}


def _load_mono(path: Path, sr: int) -> tuple[np.ndarray, int]:
    waveform, out_sr = librosa.load(path.as_posix(), sr=sr, mono=True)
    return waveform.astype(np.float32), out_sr


def _apply(waveform: np.ndarray, sr: int, kind: str, value: float,
           rng: np.random.Generator) -> np.ndarray:
    if kind == "pitch":
        return pitch_shift(waveform, sr, int(value))
    if kind == "tempo":
        return time_stretch(waveform, rate=value)
    if kind == "noise":
        return add_noise(waveform, snr_db=value, rng=rng)
    if kind == "crop":
        return crop_end(waveform, sr, seconds=value)
    if kind == "lossy":
        return lossy_mp3_roundtrip(waveform, sr, bitrate_kbps=int(value))
    raise ValueError(f"Unknown transform kind: {kind}")


def _severity_label(kind: str, value: float) -> str:
    if kind == "pitch":
        sign = "+" if value > 0 else ""
        return f"{sign}{int(value)}st"
    if kind == "tempo":
        return f"{value:.2f}x"
    if kind == "noise":
        return f"SNR{int(value)}dB"
    if kind == "crop":
        return f"{value:.1f}s"
    if kind == "lossy":
        return f"{int(value)}kbps"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Severity sweep across transform grid.")
    parser.add_argument("--references", nargs="+", required=True, type=Path,
                        help="One or more reference WAV paths.")
    parser.add_argument("--output-csv", type=Path, default=Path("data/index/severity_sweep.csv"))
    parser.add_argument("--sr", type=int, default=16_000)
    parser.add_argument("--max-seconds", type=float, default=5.0,
                        help="Truncate reference to this duration before transforming.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model-version", default="baseline-v1",
                        choices=["baseline-v1", "contrastive-v1"])
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    fieldnames = [
        "reference", "transform", "severity", "severity_value",
        "top_score", "hit_top1", "hit_topk", "rank",
    ]

    rows: list[dict] = []
    total = len(args.references) * sum(len(v) for v in SWEEP_GRID.values())
    done = 0

    for ref_path in args.references:
        if not ref_path.exists():
            print(f"Reference not found: {ref_path}")
            return 1

        waveform, sr = _load_mono(ref_path, args.sr)
        if args.max_seconds and args.max_seconds > 0:
            max_samples = int(round(args.max_seconds * sr))
            if 0 < max_samples < waveform.shape[0]:
                waveform = waveform[:max_samples]

        reset_catalogue_state(persist=False)
        reset_search_state(persist=False)
        ref_track_id = ingest_reference(ref_path, model_version=args.model_version)

        for kind, grid in SWEEP_GRID.items():
            for value in grid:
                done += 1
                label = _severity_label(kind, value)
                try:
                    transformed = _apply(waveform, sr, kind, value, rng)
                    transformed = normalise(transformed)
                except Exception as exc:
                    print(f"  [{done}/{total}] {ref_path.stem} {kind} {label}: ERROR {exc}")
                    rows.append({
                        "reference": ref_path.stem, "transform": kind,
                        "severity": label, "severity_value": value,
                        "top_score": "", "hit_top1": 0, "hit_topk": 0, "rank": "",
                    })
                    continue

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                try:
                    sf.write(tmp_path.as_posix(), transformed, sr)
                    matches = query_track_matches(
                        tmp_path, top_k=args.top_k,
                        max_query_segments=1, model_version=args.model_version,
                    )
                finally:
                    tmp_path.unlink(missing_ok=True)

                rank = None
                for idx, m in enumerate(matches, 1):
                    if str(m["track_id"]) == ref_track_id:
                        rank = idx
                        break
                top = matches[0] if matches else {}
                score = top.get("score", 0.0)

                rows.append({
                    "reference": ref_path.stem,
                    "transform": kind,
                    "severity": label,
                    "severity_value": value,
                    "top_score": round(float(score), 6) if score else "",
                    "hit_top1": int(rank == 1),
                    "hit_topk": int(rank is not None),
                    "rank": rank if rank is not None else "",
                })
                status = "HIT" if rank == 1 else "MISS"
                print(f"  [{done}/{total}] {ref_path.stem} {kind} {label}: "
                      f"score={score:.4f}  {status}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    hit1 = sum(int(r["hit_top1"]) for r in rows)
    print(f"\nSweep complete: {hit1}/{len(rows)} top-1 hits")
    print(f"Saved: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
