"""
Cross-genre evaluation: build a single shared index from multiple GTZAN
genres and evaluate transformed queries against the full catalogue.

Usage:
    python -m scripts.cross_genre_eval ^
        --gtzan-root "../datasets/GTZAN/genres_original" ^
        --tracks-per-genre 5 ^
        --model-version baseline-v1
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

import waveid_backend.config as cfg
from waveid_backend.services.catalogue import (
    reset_state as reset_catalogue,
)
from waveid_backend.services.search import (
    reset_state as reset_search,
)
from scripts.run_evaluation import (
    ingest_reference,
    query_track_matches,
    parse_transform,
)
from waveid_backend.services.transforms import lossy_mp3_roundtrip, LOSSY_MP3_BITRATES

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]


# ── transform generation (mirrors evaluate_transformations.py) ──────────

def _load_mono(path: Path, sr: int) -> tuple[np.ndarray, int]:
    w, s = librosa.load(path.as_posix(), sr=sr, mono=True)
    return w.astype(np.float32), s


def _norm(w: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(w))) if w.size else 0.0
    return (w / peak).astype(np.float32) if peak > 0 else w.astype(np.float32)


def _write(p: Path, w: np.ndarray, sr: int) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(p.as_posix(), _norm(w), sr)


def generate_transforms(
    src: Path, out_dir: Path, sr: int = 16_000, max_seconds: float = 5.0, seed: int = 42,
) -> list[Path]:
    rng = np.random.default_rng(seed)
    w, s = _load_mono(src, sr)
    max_samp = int(round(max_seconds * s))
    if 0 < max_samp < w.shape[0]:
        w = w[:max_samp]

    stem = src.stem
    paths: list[Path] = []

    def _save(tag: str, wav: np.ndarray) -> None:
        p = out_dir / f"{stem}_{tag}.wav"
        _write(p, wav, s)
        paths.append(p)

    _save("orig", w)

    for st in (-4, -2, 2, 4):
        sign = "m" if st < 0 else "p"
        _save(f"pitch_{sign}{abs(st)}", librosa.effects.pitch_shift(w, sr=s, n_steps=st).astype(np.float32))

    for rate in (0.85, 1.15):
        _save(f"tempo_{str(rate).replace('.','_')}", librosa.effects.time_stretch(w, rate=rate).astype(np.float32))

    for snr in (20.0, 10.0):
        sig_pow = float(np.mean(w ** 2)) + 1e-12
        noise = rng.normal(0.0, np.sqrt(sig_pow / (10 ** (snr / 10))), size=w.shape).astype(np.float32)
        _save(f"noise_snr{int(snr)}", (w + noise).astype(np.float32))

    for sec in (1.0, 2.0):
        cs = int(round(sec * s))
        if 0 < cs < w.shape[0]:
            _save(f"crop_{str(sec).replace('.','_')}s", w[:-cs].astype(np.float32))

    nyq = s / 2.0
    for name, (lo, hi) in [("phone", (300.0, 4000.0)), ("laptop", (200.0, 8000.0))]:
        sos = butter(5, [max(lo / nyq, 1e-6), min(hi / nyq, 1 - 1e-6)], btype="band", output="sos")
        _save(f"bandpass_{name}", sosfilt(sos, w).astype(np.float32))

    for name, (lo, hi) in [("phone", (300.0, 4000.0)), ("laptop", (200.0, 8000.0))]:
        sos = butter(5, [max(lo / nyq, 1e-6), min(hi / nyq, 1 - 1e-6)], btype="band", output="sos")
        filt = sosfilt(sos, w).astype(np.float32)
        sig_pow = float(np.mean(filt ** 2)) + 1e-12
        noise = rng.normal(0.0, np.sqrt(sig_pow / (10 ** (15.0 / 10))), size=filt.shape).astype(np.float32)
        _save(f"compound_{name}_snr15", (filt + noise).astype(np.float32))

    for kbps in LOSSY_MP3_BITRATES:
        _save(f"lossy_mp3_{kbps}k", lossy_mp3_roundtrip(w, s, bitrate_kbps=kbps))

    return paths


# ── main ────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-genre combined-index evaluation.")
    parser.add_argument("--gtzan-root", required=True, type=Path,
                        help="Path to genres_original/ containing genre subdirs.")
    parser.add_argument("--tracks-per-genre", type=int, default=5)
    parser.add_argument("--max-seconds", type=float, default=5.0)
    parser.add_argument("--max-query-segments", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model-version", choices=["baseline-v1", "contrastive-v1"], default=None)
    parser.add_argument("--output-csv", default="data/index/cross_genre_eval.csv")
    args = parser.parse_args()

    gtzan = args.gtzan_root.resolve()
    model = args.model_version or cfg.MODEL_VERSION

    # The default top-k is tuned for a small index. With 50 tracks each having
    # many segments, we need to retrieve more candidates before track-level aggregation.
    original_emb_top_k = cfg.QUERY_EMBEDDING_TOP_K
    cfg.QUERY_EMBEDDING_TOP_K = 50

    # ── Phase 1: reset index and ingest all references ──────────────
    print("=== Phase 1: Ingesting references into combined index ===")
    reset_catalogue(persist=False)
    reset_search(persist=False)

    track_map: dict[str, dict[str, str]] = {}  # track_id -> {filename, genre}
    total_ingested = 0

    for genre in GENRES:
        genre_dir = gtzan / genre
        if not genre_dir.exists():
            print(f"  WARNING: genre dir not found: {genre_dir}")
            continue
        wavs = sorted(genre_dir.glob("*.wav"))[:args.tracks_per_genre]
        for wav_path in wavs:
            track_id = ingest_reference(wav_path, model_version=model)
            track_map[track_id] = {"filename": wav_path.name, "genre": genre}
            total_ingested += 1
            print(f"  [{total_ingested}] {genre}/{wav_path.name} -> {track_id[:8]}")

    print(f"\nIngested {total_ingested} tracks across {len(GENRES)} genres.\n")

    # build reverse lookup: filename -> (track_id, genre)
    fname_to_meta: dict[str, dict[str, str]] = {}
    for tid, meta in track_map.items():
        fname_to_meta[meta["filename"]] = {"track_id": tid, "genre": meta["genre"]}

    # ── Phase 2: generate transforms and query ──────────────────────
    print("=== Phase 2: Generating transforms and querying ===")
    rows: list[dict[str, Any]] = []
    tmp_root = Path(tempfile.mkdtemp(prefix="crossgenre_"))

    try:
        ref_index = 0
        for genre in GENRES:
            genre_dir = gtzan / genre
            if not genre_dir.exists():
                continue
            wavs = sorted(genre_dir.glob("*.wav"))[:args.tracks_per_genre]
            for wav_path in wavs:
                ref_index += 1
                ref_meta = fname_to_meta[wav_path.name]
                ref_tid = ref_meta["track_id"]
                ref_genre = ref_meta["genre"]

                work_dir = tmp_root / wav_path.stem
                query_files = generate_transforms(
                    wav_path, work_dir, max_seconds=args.max_seconds,
                )

                for qf in query_files:
                    matches = query_track_matches(
                        qf, top_k=args.top_k, max_query_segments=args.max_query_segments,
                        model_version=model,
                    )
                    transform, severity = parse_transform(qf.stem)
                    # Find where the correct reference track appears in the ranked results
                    rank = None
                    for idx, m in enumerate(matches, 1):
                        if str(m["track_id"]) == ref_tid:
                            rank = idx
                            break
                    top = matches[0] if matches else {}
                    top_tid = str(top.get("track_id", ""))
                    top_meta = track_map.get(top_tid, {})

                    rows.append({
                        "reference_file": wav_path.name,
                        "ref_genre": ref_genre,
                        "query_file": qf.name,
                        "transform": transform,
                        "severity": severity,
                        "top_track_id": top_tid,
                        "top_filename": top.get("filename", ""),
                        "top_genre": top_meta.get("genre", ""),
                        "top_score": f"{top.get('score', 0):.4f}",
                        "hit_top1": int(rank == 1),
                        "hit_topk": int(rank is not None),
                        "rank": rank if rank is not None else "",
                        "num_matches": len(matches),
                    })

                status = "OK" if all(r["hit_top1"] for r in rows[-len(query_files):]) else "MISS"
                print(f"  [{ref_index}/{total_ingested}] {ref_genre}/{wav_path.name}  {status}")

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        cfg.QUERY_EMBEDDING_TOP_K = original_emb_top_k

    # ── Phase 3: write CSV and print summary ────────────────────────
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {out}\n")

    # overall
    n = len(rows)
    top1 = sum(int(r["hit_top1"]) for r in rows)
    topk = sum(int(r["hit_topk"]) for r in rows)
    print(f"{'='*60}")
    print(f"OVERALL  top-1: {top1}/{n} ({100*top1/n:.1f}%)   top-{args.top_k}: {topk}/{n} ({100*topk/n:.1f}%)")
    print(f"{'='*60}")

    # per genre
    print(f"\n{'Genre':<12} {'Top-1':>8} {'Rate':>7}")
    print("-" * 30)
    for genre in GENRES:
        gr = [r for r in rows if r["ref_genre"] == genre]
        if not gr:
            continue
        g1 = sum(int(r["hit_top1"]) for r in gr)
        print(f"{genre:<12} {g1:>4}/{len(gr):<4} {100*g1/len(gr):>6.1f}%")

    # per transform
    transforms = sorted(set(r["transform"] for r in rows))
    print(f"\n{'Transform':<12} {'Top-1':>8} {'Rate':>7}  {'Mean Score':>10}")
    print("-" * 45)
    for t in transforms:
        tr = [r for r in rows if r["transform"] == t]
        t1 = sum(int(r["hit_top1"]) for r in tr)
        scores = [float(r["top_score"]) for r in tr if r["hit_top1"]]
        ms = f"{sum(scores)/len(scores):.3f}" if scores else "n/a"
        print(f"{t:<12} {t1:>4}/{len(tr):<4} {100*t1/len(tr):>6.1f}%  {ms:>10}")

    # cross-genre mismatches
    misses = [r for r in rows if not int(r["hit_top1"])]
    if misses:
        print(f"\n{'='*60}")
        print(f"MISRANKED queries ({len(misses)}):")
        print(f"{'='*60}")
        for r in misses[:30]:
            print(f"  {r['ref_genre']}/{r['reference_file']}  {r['transform']}({r['severity']})  "
                  f"-> {r['top_genre']}/{r['top_filename']}  score={r['top_score']}  rank={r['rank']}")
        if len(misses) > 30:
            print(f"  ... and {len(misses)-30} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
