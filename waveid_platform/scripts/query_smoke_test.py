"""
Quick smoke test for the query pipeline.

Usage:
    python -m scripts.query_smoke_test --reference "path/to/ref.wav" --query "path/to/query.wav"
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
    get_track,
)
from waveid_backend.services.embedding import extract_embedding
from waveid_backend.services.search import add_reference_embeddings, query_similar
from waveid_backend.services.segmentation import segment_audio


def ingest_reference(path: Path) -> str:
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
    track_id = add_track(path.name, duration_seconds, sr, MODEL_VERSION)
    segments = segment_audio(waveform, sr, SEGMENT_SECONDS, HOP_SECONDS)
    embeddings = [extract_embedding(segment.samples, sr) for segment in segments]
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


def run_query(
    path: Path,
    top_k: int,
    max_query_segments: int | None = None,
) -> list[dict[str, float | int | str]]:
    contents = path.read_bytes()
    waveform, sr = load_audio_from_bytes(
        contents,
        filename=path.name,
        target_sr=SAMPLE_RATE,
        mono=MONO,
        normalize=NORMALIZE,
        max_duration_seconds=MAX_DURATION_SECONDS,
    )
    segments = segment_audio(waveform, sr, SEGMENT_SECONDS, HOP_SECONDS)
    if max_query_segments is not None and max_query_segments > 0:
        segments = segments[:max_query_segments]
    if segments:
        embeddings = [extract_embedding(segment.samples, sr) for segment in segments]
    else:
        embeddings = [extract_embedding(waveform, sr)]

    track_scores: dict[str, dict[str, float | int | str]] = {}
    for embedding in embeddings:
        segment_matches = query_similar(embedding, top_k=QUERY_EMBEDDING_TOP_K)
        id_map = embedding_to_track_map([match["id"] for match in segment_matches])
        for match in segment_matches:
            meta = id_map.get(match["id"])
            if meta is None:
                continue
            track_id = str(meta["track_id"])
            row = track_scores.setdefault(
                track_id,
                {"track_id": track_id, "filename": str(meta["filename"]), "score_sum": 0.0, "hits": 0},
            )
            row["score_sum"] = float(row["score_sum"]) + float(match["score"])
            row["hits"] = int(row["hits"]) + 1

    ranked: list[dict[str, float | int | str]] = []
    for row in track_scores.values():
        hits = int(row["hits"])
        score = float(row["score_sum"]) / max(hits, 1)
        if score < MIN_TRACK_SCORE:
            continue
        ranked.append(
            {
                "track_id": str(row["track_id"]),
                "filename": str(row["filename"]),
                "score": score,
                "hits": hits,
            }
        )
    ranked.sort(key=lambda item: (float(item["score"]), int(item["hits"])), reverse=True)
    return ranked[:top_k]


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for query pipeline.")
    parser.add_argument("--reference", required=True, help="Path to reference WAV/MP3.")
    parser.add_argument("--query", required=True, help="Path to query WAV/MP3.")
    parser.add_argument("--top-k", type=int, default=5, help="Matches to return.")
    parser.add_argument(
        "--max-query-segments",
        type=int,
        default=5,
        help="Optional cap on query segments for faster smoke tests.",
    )
    args = parser.parse_args()

    reference_path = Path(args.reference)
    query_path = Path(args.query)
    if not reference_path.exists():
        print(f"Reference file not found: {reference_path}")
        return 1
    if not query_path.exists():
        print(f"Query file not found: {query_path}")
        return 1

    track_id = ingest_reference(reference_path)
    matches = run_query(
        query_path,
        top_k=min(args.top_k, QUERY_TRACK_TOP_K),
        max_query_segments=args.max_query_segments,
    )

    print(f"ingested_track_id={track_id}")
    print(f"matches={len(matches)}")
    if not matches:
        print("No matches returned.")
        return 2

    top_track_ids = {str(match["track_id"]) for match in matches}
    hit = track_id in top_track_ids
    print(f"top_k_contains_reference={hit}")
    if not hit:
        track = get_track(track_id)
        ref_segments = track["num_segments"] if track else 0
        print(f"reference_segments={ref_segments}")
        print("Expected at least one match from the ingested reference track.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
