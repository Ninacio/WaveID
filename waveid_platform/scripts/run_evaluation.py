"""
Run query evaluation over transformed clips and write a CSV report.

Usage:
    python -m scripts.run_evaluation --reference "path/to/ref.wav" --queries-dir "data/query/eval/blues00000_short"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from waveid_backend.config import (
    HOP_SECONDS,
    MAX_DURATION_SECONDS,
    MIN_TRACK_SCORE,
    MODEL_VERSION,
    MONO,
    NORMALIZE,
    QUERY_EMBEDDING_TOP_K,
    QUERY_TRACK_TOP_K,
    REFERENCE_DIR,
    SAMPLE_RATE,
    SEGMENT_SECONDS,
)
from waveid_backend.services.audio_io import load_audio_from_bytes
from waveid_backend.services.catalogue import (
    add_segments,
    add_track,
    embedding_to_track_map,
    reset_state as reset_catalogue_state,
)
from waveid_backend.services.embedding import extract_embedding
from waveid_backend.services.search import (
    add_reference_embeddings,
    query_similar,
    reset_state as reset_search_state,
)
from waveid_backend.services.segmentation import segment_audio


def ingest_reference(path: Path, model_version: str | None = None) -> str:
    version = model_version or MODEL_VERSION
    contents = path.read_bytes()
    waveform, sr = load_audio_from_bytes(
        contents,
        filename=path.name,
        target_sr=SAMPLE_RATE,
        mono=MONO,
        normalize=NORMALIZE,
        max_duration_seconds=MAX_DURATION_SECONDS,
    )
    duration_seconds = float(waveform.size / sr) if sr else 0.0
    track_id = add_track(path.name, duration_seconds, sr, version)
    segments = segment_audio(waveform, sr, SEGMENT_SECONDS, HOP_SECONDS)
    embeddings = [extract_embedding(segment.samples, sr, model_version=version) for segment in segments]
    embedding_ids = add_reference_embeddings(embeddings)
    segment_records = [
        {
            "start_time": segment.start_time,
            "end_time": segment.end_time,
            "embedding_id": embedding_id,
        }
        for segment, embedding_id in zip(segments, embedding_ids)
    ]
    add_segments(track_id, segment_records)

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    reference_path = REFERENCE_DIR / f"{track_id}{path.suffix.lower()}"
    reference_path.write_bytes(contents)
    return track_id


def query_track_matches(
    path: Path,
    top_k: int,
    max_query_segments: int | None,
    model_version: str | None = None,
) -> list[dict[str, Any]]:
    contents = path.read_bytes()
    waveform, sr = load_audio_from_bytes(
        contents,
        filename=path.name,
        target_sr=SAMPLE_RATE,
        mono=MONO,
        normalize=NORMALIZE,
        max_duration_seconds=MAX_DURATION_SECONDS,
    )
    query_segments = segment_audio(waveform, sr, SEGMENT_SECONDS, HOP_SECONDS)
    if max_query_segments is not None and max_query_segments > 0:
        query_segments = query_segments[:max_query_segments]
    version = model_version or MODEL_VERSION
    if query_segments:
        segment_embeddings = [extract_embedding(segment.samples, sr, model_version=version) for segment in query_segments]
    else:
        segment_embeddings = [extract_embedding(waveform, sr, model_version=version)]

    track_scores: dict[str, dict[str, Any]] = {}
    for segment_embedding in segment_embeddings:
        segment_matches = query_similar(segment_embedding, top_k=QUERY_EMBEDDING_TOP_K)
        id_map = embedding_to_track_map([match["id"] for match in segment_matches])
        for match in segment_matches:
            emb_id = match["id"]
            score = float(match["score"])
            meta = id_map.get(emb_id)
            if meta is None:
                continue
            track_id = str(meta["track_id"])
            row = track_scores.setdefault(
                track_id,
                {
                    "track_id": track_id,
                    "filename": str(meta["filename"]),
                    "score_sum": 0.0,
                    "hits": 0,
                },
            )
            row["score_sum"] = float(row["score_sum"]) + score
            row["hits"] = int(row["hits"]) + 1

    ranked_matches: list[dict[str, Any]] = []
    for row in track_scores.values():
        hits = int(row["hits"])
        avg_score = float(row["score_sum"]) / max(hits, 1)
        if avg_score < MIN_TRACK_SCORE:
            continue
        ranked_matches.append(
            {
                "track_id": str(row["track_id"]),
                "filename": str(row["filename"]),
                "score": avg_score,
                "hits": hits,
            }
        )
    ranked_matches.sort(key=lambda item: (float(item["score"]), int(item["hits"])), reverse=True)
    return ranked_matches[:top_k]


def parse_transform(stem: str) -> tuple[str, str]:
    if stem.endswith("_orig"):
        return ("orig", "none")
    for kind in ("pitch", "tempo", "noise", "crop"):
        marker = f"_{kind}_"
        if marker in stem:
            severity = stem.split(marker, 1)[1]
            return (kind, severity)
    return ("unknown", "unknown")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate transformed query clips.")
    parser.add_argument("--reference", required=True, help="Path to reference WAV/MP3.")
    parser.add_argument("--queries-dir", required=True, help="Directory containing transformed WAV queries.")
    parser.add_argument("--output-csv", default="data/index/eval_results.csv", help="CSV output path.")
    parser.add_argument("--top-k", type=int, default=QUERY_TRACK_TOP_K, help="Track-level top-k.")
    parser.add_argument(
        "--max-query-segments",
        type=int,
        default=3,
        help="Cap query segments for faster evaluation.",
    )
    parser.add_argument(
        "--limit-queries",
        type=int,
        default=None,
        help="Optional maximum number of query files to evaluate.",
    )
    parser.add_argument(
        "--fresh-index",
        action="store_true",
        help="Use isolated in-memory catalogue/index for faster evaluation runs.",
    )
    parser.add_argument(
        "--model-version",
        choices=["baseline-v1", "contrastive-v1"],
        default=None,
        help="Embedding model: baseline-v1 (MFCC) or contrastive-v1 (CNN). Default from config.",
    )
    args = parser.parse_args()

    reference_path = Path(args.reference)
    queries_dir = Path(args.queries_dir)
    output_csv = Path(args.output_csv)
    if not reference_path.exists():
        print(f"Reference not found: {reference_path}")
        return 1
    if not queries_dir.exists():
        print(f"Queries dir not found: {queries_dir}")
        return 1

    query_files = sorted(queries_dir.glob("*.wav"))
    if args.limit_queries is not None:
        query_files = query_files[: args.limit_queries]
    if not query_files:
        print(f"No WAV files found in: {queries_dir}")
        return 1

    if args.fresh_index:
        reset_catalogue_state(persist=False)
        reset_search_state(persist=False)

    model_version = args.model_version
    reference_track_id = ingest_reference(reference_path, model_version=model_version)
    rows: list[dict[str, Any]] = []
    for query_file in query_files:
        matches = query_track_matches(
            query_file,
            top_k=max(1, args.top_k),
            max_query_segments=args.max_query_segments,
            model_version=model_version,
        )
        transform, severity = parse_transform(query_file.stem)
        rank = None
        for idx, match in enumerate(matches, start=1):
            if str(match["track_id"]) == reference_track_id:
                rank = idx
                break
        top = matches[0] if matches else {}
        rows.append(
            {
                "query_file": query_file.name,
                "transform": transform,
                "severity": severity,
                "top_track_id": top.get("track_id", ""),
                "top_filename": top.get("filename", ""),
                "top_score": top.get("score", ""),
                "top_hits": top.get("hits", ""),
                "hit_top1": int(rank == 1),
                "hit_topk": int(rank is not None),
                "rank": rank if rank is not None else "",
                "num_matches": len(matches),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    top1 = sum(int(row["hit_top1"]) for row in rows)
    topk = sum(int(row["hit_topk"]) for row in rows)
    print(f"Reference track id: {reference_track_id}")
    print(f"Queries evaluated: {len(rows)}")
    print(f"Top-1 hits: {top1}/{len(rows)}")
    print(f"Top-{max(1, args.top_k)} hits: {topk}/{len(rows)}")
    print(f"Saved CSV: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
