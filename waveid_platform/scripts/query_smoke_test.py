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
    MODEL_VERSION,
    MONO,
    NORMALIZE,
    REFERENCE_DIR,
    SAMPLE_RATE,
    SEGMENT_SECONDS,
)
from waveid_backend.services.audio_io import load_audio_from_bytes
from waveid_backend.services.catalogue import add_segments, add_track, get_track
from waveid_backend.services.embedding import extract_embedding
from waveid_backend.services.search import add_reference_embeddings, query_similar
from waveid_backend.services.segmentation import segment_audio


def ingest_reference(path: Path) -> tuple[str, set[str]]:
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

    return track_id, set(embedding_ids)


def run_query(path: Path, top_k: int) -> list[dict[str, float]]:
    contents = path.read_bytes()
    waveform, sr = load_audio_from_bytes(
        contents,
        filename=path.name,
        target_sr=SAMPLE_RATE,
        mono=MONO,
        normalize=NORMALIZE,
        max_duration_seconds=MAX_DURATION_SECONDS,
    )
    query_embedding = extract_embedding(waveform, sr)
    return query_similar(query_embedding, top_k=top_k)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for query pipeline.")
    parser.add_argument("--reference", required=True, help="Path to reference WAV/MP3.")
    parser.add_argument("--query", required=True, help="Path to query WAV/MP3.")
    parser.add_argument("--top-k", type=int, default=5, help="Matches to return.")
    args = parser.parse_args()

    reference_path = Path(args.reference)
    query_path = Path(args.query)
    if not reference_path.exists():
        print(f"Reference file not found: {reference_path}")
        return 1
    if not query_path.exists():
        print(f"Query file not found: {query_path}")
        return 1

    track_id, embedding_ids = ingest_reference(reference_path)
    matches = run_query(query_path, top_k=args.top_k)

    print(f"ingested_track_id={track_id}")
    print(f"matches={len(matches)}")
    if not matches:
        print("No matches returned.")
        return 2

    top_ids = {match["id"] for match in matches}
    hit = bool(top_ids & embedding_ids)
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
