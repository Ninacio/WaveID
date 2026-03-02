"""
WaveID Backend
================

This module implements a minimal FastAPI application for the WaveID
prototype. The application exposes endpoints for ingesting reference
tracks and identifying unknown audio clips. The heavy lifting (audio
preprocessing, embedding extraction, vector indexing and search) is
encapsulated in service modules. At this stage the implementation
provides stubs and placeholders where the full functionality can be
added later. The goal is to provide a working scaffold that can be
extended in subsequent iterations.

Run this application with:
    uvicorn waveid_backend.main:app --reload
"""

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
from pydantic import BaseModel

from .config import (
    ALLOWED_EXTENSIONS,
    HOP_SECONDS,
    MAX_DURATION_SECONDS,
    MAX_UPLOAD_MB,
    MODEL_VERSION,
    MIN_TRACK_SCORE,
    MONO,
    NORMALIZE,
    QUERY_DIR,
    QUERY_EMBEDDING_TOP_K,
    QUERY_TRACK_TOP_K,
    REFERENCE_DIR,
    SAMPLE_RATE,
    SEGMENT_SECONDS,
)
from .services.audio_io import load_audio_from_bytes
from .services.catalogue import (
    add_segments,
    add_track,
    embedding_to_track_map,
    get_track,
    list_tracks,
)
from .services.embedding import extract_embedding
from .services.search import add_reference_embeddings, query_similar
from .services.segmentation import segment_audio


app = FastAPI(title="WaveID Backend", version="0.1.0")


class IngestResponse(BaseModel):
    message: str
    track_id: str
    num_segments: int
    duration_seconds: float


class CatalogueTrack(BaseModel):
    track_id: str
    filename: str
    num_segments: int
    duration: float


class SegmentInfo(BaseModel):
    segment_id: str
    start_time: float
    end_time: float
    embedding_id: str


class TrackDetail(CatalogueTrack):
    segments: list[SegmentInfo]


def _validate_upload(file: UploadFile, contents: bytes) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(status_code=413, detail="Uploaded file is too large.")


@app.get("/health")
async def health() -> dict[str, str]:
    """Simple health check endpoint.

    Returns a JSON response with a status message. Use this endpoint
    to verify that the server is running.
    """
    return {"status": "ok"}


@app.post("/ingest-track", response_model=IngestResponse)
async def ingest_track(file: UploadFile = File(...)) -> IngestResponse:
    """Ingest a reference track into the catalogue.

    The uploaded audio file is read, preprocessed and segmented. Each
    segment is converted into an embedding and added to the vector index.
    At this stage the implementation uses a placeholder for the audio
    processing pipeline and returns a fixed number of segments.

    Args:
        file: Uploaded audio file (MP3/WAV).

    Returns:
        IngestResponse: Confirmation message with number of segments.
    """
    contents = await file.read()
    _validate_upload(file, contents)

    try:
        waveform, sr = load_audio_from_bytes(
            contents,
            filename=file.filename,
            target_sr=SAMPLE_RATE,
            mono=MONO,
            normalize=NORMALIZE,
            max_duration_seconds=MAX_DURATION_SECONDS,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    duration_seconds = float(waveform.size / sr) if sr else 0.0
    track_id = add_track(file.filename, duration_seconds, sr, MODEL_VERSION)
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
    reference_path = REFERENCE_DIR / f"{track_id}{Path(file.filename).suffix.lower()}"
    reference_path.write_bytes(contents)

    return IngestResponse(
        message=f"Ingested {file.filename}",
        track_id=track_id,
        num_segments=len(segments),
        duration_seconds=duration_seconds,
    )


class QueryMatch(BaseModel):
    track_id: str
    filename: str
    score: float
    hits: int


class QueryResponse(BaseModel):
    query_embedding: list[float]
    matches: list[QueryMatch]


@app.post("/query", response_model=QueryResponse)
async def query_clip(file: UploadFile = File(...)) -> QueryResponse:
    """Identify an unknown audio clip.

    This endpoint accepts a short audio clip, extracts its embedding
    and searches the catalogue for similar embeddings. The response
    contains the query embedding and a list of matched reference tracks
    along with their similarity scores.

    Args:
        file: Uploaded audio clip.

    Returns:
        QueryResponse: Query embedding and search results.
    """
    contents = await file.read()
    _validate_upload(file, contents)

    try:
        waveform, sr = load_audio_from_bytes(
            contents,
            filename=file.filename,
            target_sr=SAMPLE_RATE,
            mono=MONO,
            normalize=NORMALIZE,
            max_duration_seconds=MAX_DURATION_SECONDS,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    query_segments = segment_audio(waveform, sr, SEGMENT_SECONDS, HOP_SECONDS)
    if query_segments:
        segment_embeddings = [extract_embedding(segment.samples, sr) for segment in query_segments]
    else:
        segment_embeddings = [extract_embedding(waveform, sr)]
    query_embedding = np.mean(np.array(segment_embeddings, dtype=float), axis=0).tolist()

    track_scores: dict[str, dict[str, float | int | str]] = {}
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

    ranked_matches: list[QueryMatch] = []
    for row in track_scores.values():
        hits = int(row["hits"])
        avg_score = float(row["score_sum"]) / max(hits, 1)
        if avg_score < MIN_TRACK_SCORE:
            continue
        ranked_matches.append(
            QueryMatch(
                track_id=str(row["track_id"]),
                filename=str(row["filename"]),
                score=avg_score,
                hits=hits,
            )
        )
    ranked_matches.sort(key=lambda item: (item.score, item.hits), reverse=True)
    matches = ranked_matches[:QUERY_TRACK_TOP_K]

    QUERY_DIR.mkdir(parents=True, exist_ok=True)
    query_id = uuid4().hex
    query_path = QUERY_DIR / f"{query_id}{Path(file.filename).suffix.lower()}"
    query_path.write_bytes(contents)

    return QueryResponse(query_embedding=query_embedding, matches=matches)


@app.get("/catalogue", response_model=list[CatalogueTrack])
async def catalogue() -> list[CatalogueTrack]:
    """List ingested tracks and segment counts."""
    return list_tracks()


@app.get("/catalogue/{track_id}", response_model=TrackDetail)
async def catalogue_track(track_id: str) -> TrackDetail:
    """Get metadata and segment list for a track."""
    track = get_track(track_id)
    if track is None:
        raise HTTPException(status_code=404, detail="Track not found.")
    return track