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
from fastapi.staticfiles import StaticFiles
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
    reset_state as reset_catalogue,
)
from .services.embedding import extract_embedding
from .services.search import add_reference_embeddings, query_similar
from .services.search import reset_state as reset_search
from .services.segmentation import segment_audio


app = FastAPI(title="WaveID Backend", version="0.1.0")


@app.on_event("startup")
async def _startup_reset() -> None:
    """Wipe all persisted state so the catalogue starts empty on every boot."""
    from .config import INDEX_DIR
    for filename in ("catalogue.json", "embeddings.npy", "embedding_ids.json"):
        f = INDEX_DIR / filename
        if f.exists():
            f.unlink()
    reset_catalogue(persist=False)
    reset_search(persist=False)


# Serve frontend static files
_STATIC_DIR = Path(__file__).resolve().parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/")
    async def root():
        """Serve the minimal frontend UI."""
        index_path = _STATIC_DIR / "index.html"
        if index_path.exists():
            from fastapi.responses import FileResponse
            return FileResponse(
                index_path,
                headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
            )
        return {"message": "WaveID API", "docs": "/docs"}


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
    similarity: float
    coverage: float
    hits: int


class QueryResponse(BaseModel):
    query_embedding: list[float]
    matches: list[QueryMatch]
    confidence_gap: float
    confidence_label: str


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

    track_scores: dict[str, dict[str, float | int | str | set[int]]] = {}
    total_query_segments = max(len(segment_embeddings), 1)
    for segment_idx, segment_embedding in enumerate(segment_embeddings):
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
                    "matched_segments": set(),
                },
            )
            row["score_sum"] = float(row["score_sum"]) + score
            row["hits"] = int(row["hits"]) + 1
            cast_segments = row.get("matched_segments", set())
            if isinstance(cast_segments, set):
                cast_segments.add(segment_idx)
                row["matched_segments"] = cast_segments

    ranked_rows: list[dict[str, float | int | str]] = []
    for row in track_scores.values():
        hits = int(row["hits"])
        avg_score = float(row["score_sum"]) / max(hits, 1)
        if avg_score < MIN_TRACK_SCORE:
            continue
        matched_segments = row.get("matched_segments", set())
        coverage = (
            len(matched_segments) / total_query_segments
            if isinstance(matched_segments, set)
            else 0.0
        )
        density = hits / max(total_query_segments * QUERY_EMBEDDING_TOP_K, 1)
        composite_score = (0.65 * avg_score) + (0.3 * coverage) + (0.05 * density)
        ranked_rows.append(
            {
                "track_id": str(row["track_id"]),
                "filename": str(row["filename"]),
                "similarity": avg_score,
                "coverage": coverage,
                "hits": hits,
                "composite": composite_score,
            }
        )

    ranked_rows.sort(
        key=lambda item: (
            float(item["composite"]),
            float(item["similarity"]),
            int(item["hits"]),
        ),
        reverse=True,
    )
    top_rows = ranked_rows[:QUERY_TRACK_TOP_K]

    if top_rows:
        logits = np.array([float(item["composite"]) for item in top_rows], dtype=float)
        logits = logits - np.max(logits)
        temperature = 0.035
        probs = np.exp(logits / temperature)
        denom = float(np.sum(probs))
        confidences = probs / max(denom, 1e-12)
    else:
        confidences = np.array([], dtype=float)

    matches = [
        QueryMatch(
            track_id=str(item["track_id"]),
            filename=str(item["filename"]),
            score=float(confidences[idx]) if idx < len(confidences) else 0.0,
            similarity=float(item["similarity"]),
            coverage=float(item["coverage"]),
            hits=int(item["hits"]),
        )
        for idx, item in enumerate(top_rows)
    ]

    if len(confidences) >= 2:
        confidence_gap = float(confidences[0] - confidences[1])
    elif len(confidences) == 1:
        confidence_gap = float(confidences[0])
    else:
        confidence_gap = 0.0

    if confidence_gap >= 0.35:
        confidence_label = "high"
    elif confidence_gap >= 0.18:
        confidence_label = "medium"
    else:
        confidence_label = "low"

    QUERY_DIR.mkdir(parents=True, exist_ok=True)
    query_id = uuid4().hex
    query_path = QUERY_DIR / f"{query_id}{Path(file.filename).suffix.lower()}"
    query_path.write_bytes(contents)

    return QueryResponse(
        query_embedding=query_embedding,
        matches=matches,
        confidence_gap=confidence_gap,
        confidence_label=confidence_label,
    )


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


@app.post("/reset-catalogue")
async def reset_catalogue_endpoint() -> dict[str, str]:
    """Wipe all ingested tracks and embeddings, returning the catalogue to empty."""
    reset_catalogue(persist=True)
    reset_search(persist=True)
    for folder in (REFERENCE_DIR, QUERY_DIR):
        if folder.exists():
            for f in folder.iterdir():
                if f.is_file():
                    f.unlink()
    return {"message": "Catalogue cleared."}

