"""
WaveID Backend
================

This module implements a minimal FastAPI application for the WaveID
prototype. The application exposes endpoints for ingesting reference
tracks and identifying unknown audio clips. The heavy lifting (audio
preprocessing, embedding extraction, vector indexing and search) is
encapsulated in service modules.

Run this application with:
    uvicorn waveid_backend.main:app --reload
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .config import (
    API_KEY_CONFIGURED,
    CORS_ORIGINS,
    DUPLICATE_SAMPLE_SEGMENTS,
    DUPLICATE_THRESHOLD,
    HOP_SECONDS,
    MAX_DURATION_SECONDS,
    MAX_UPLOAD_MB,
    MIN_SIMILARITY_GAP,
    MIN_TRACK_SCORE,
    MODEL_VERSION,
    MONO,
    NORMALIZE,
    QUERY_DIR,
    QUERY_EMBEDDING_TOP_K,
    QUERY_TRACK_TOP_K,
    RATE_LIMIT_AUTH,
    RATE_LIMIT_DEFAULT,
    REFERENCE_DIR,
    REQUIRE_API_KEY,
    RESET_ON_STARTUP,
    SAMPLE_RATE,
    SEGMENT_SECONDS,
    STRONG_MATCH_SIMILARITY,
)
from .middleware.security_headers import SecurityHeadersMiddleware
from .security.auth import verify_api_key_value
from .security.validation import read_bounded_upload, validate_track_id, validate_upload
from .services.audio_io import load_audio_from_bytes
from .services.catalogue import (
    add_segments,
    add_track,
    delete_track,
    embedding_to_segment_map,
    embedding_to_track_map,
    get_track,
    list_tracks,
    reset_state as reset_catalogue,
    update_track_metadata,
)
from .services.embedding import extract_embedding
from .services.search import add_reference_embeddings, query_similar
from .services.search import remove_embeddings
from .services.search import reset_state as reset_search
from .services.segmentation import segment_audio

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT_DEFAULT])

app = FastAPI(
    title="WaveID Backend",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SecurityHeadersMiddleware)

if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )


@app.on_event("startup")
async def _startup() -> None:
    if API_KEY_CONFIGURED:
        logger.info("API key authentication is configured.")
    elif REQUIRE_API_KEY:
        logger.warning(
            "WAVEID_REQUIRE_API_KEY is true but WAVEID_API_KEY is not set. "
            "Protected routes will return 503."
        )
    else:
        logger.warning(
            "WAVEID_API_KEY is not set. Authentication routes are disabled; "
            "ingest/reset are open (set WAVEID_REQUIRE_API_KEY=true for production)."
        )

    if not RESET_ON_STARTUP:
        return

    from .config import INDEX_DIR

    for filename in ("catalogue.json", "embeddings.npy", "embedding_ids.json"):
        f = INDEX_DIR / filename
        if f.exists():
            f.unlink()
    reset_catalogue(persist=False)
    reset_search(persist=False)


_STATIC_DIR = Path(__file__).resolve().parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/")
    @limiter.limit(RATE_LIMIT_DEFAULT)
    async def root(request: Request):
        """Serve the minimal frontend UI."""
        index_path = _STATIC_DIR / "index.html"
        if index_path.exists():
            from fastapi.responses import FileResponse

            return FileResponse(
                index_path,
                headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
            )
        return {"message": "WaveID API", "docs": "/docs"}


class DuplicateInfo(BaseModel):
    track_id: str
    filename: str
    similarity: float


class IngestResponse(BaseModel):
    message: str
    track_id: str
    num_segments: int
    duration_seconds: float
    duplicate_of: DuplicateInfo | None = None


class CatalogueTrack(BaseModel):
    track_id: str
    filename: str
    num_segments: int
    duration: float
    title: str = ""
    artist: str = ""
    isrc: str = ""
    tags: list[str] = Field(default_factory=list)


class SegmentInfo(BaseModel):
    segment_id: str
    start_time: float
    end_time: float
    embedding_id: str


class TrackDetail(CatalogueTrack):
    segments: list[SegmentInfo]


class TrackMetadataUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=300)
    artist: str | None = Field(default=None, max_length=300)
    isrc: str | None = Field(default=None, max_length=32)
    tags: list[str] | None = None


class SegmentMatch(BaseModel):
    query_start: float
    query_end: float
    ref_start: float
    ref_end: float
    score: float


class QueryMatch(BaseModel):
    track_id: str
    filename: str
    score: float
    similarity: float
    coverage: float
    hits: int
    segments: list[SegmentMatch] = Field(default_factory=list)
    match_strength: str = "moderate"


class QueryResponse(BaseModel):
    query_embedding: list[float]
    matches: list[QueryMatch]
    confidence_gap: float
    confidence_label: str
    query_duration: float = 0.0
    similarity_gap: float = 0.0


class AuthVerifyRequest(BaseModel):
    api_key: str = Field(..., min_length=8, max_length=512)


class AuthVerifyResponse(BaseModel):
    authenticated: bool
    message: str


def _require_api_key_when_configured(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> None:
    """Require API key only when WAVEID_API_KEY is configured."""
    if not API_KEY_CONFIGURED:
        return
    from .security.auth import _extract_api_key

    provided = _extract_api_key(x_api_key, authorization)
    if not verify_api_key_value(provided):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def _match_strength(
    similarity: float,
    rank: int,
    top_similarity: float,
    similarity_gap: float,
) -> str:
    """Classify how trustworthy a match is for end users."""
    if similarity < 0.90:
        return "weak"
    if rank == 0 and similarity >= STRONG_MATCH_SIMILARITY and similarity_gap >= MIN_SIMILARITY_GAP:
        return "strong"
    if rank > 0 and (top_similarity - similarity) < MIN_SIMILARITY_GAP:
        return "weak"
    if similarity >= 0.93:
        return "moderate"
    return "weak"


_AUDIO_MEDIA_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".au": "audio/basic",
}


@app.get("/health")
@limiter.limit(RATE_LIMIT_DEFAULT)
async def health(request: Request) -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post("/auth/verify", response_model=AuthVerifyResponse)
@limiter.limit(RATE_LIMIT_AUTH)
async def auth_verify(request: Request, body: AuthVerifyRequest) -> AuthVerifyResponse:
    """
    Verify an API key. Limited to 5 attempts per 15 minutes per client IP.
    """
    if not API_KEY_CONFIGURED:
        raise HTTPException(
            status_code=503,
            detail="Authentication is not configured on this server.",
        )

    if verify_api_key_value(body.api_key.strip()):
        return AuthVerifyResponse(authenticated=True, message="API key is valid.")

    raise HTTPException(status_code=401, detail="Invalid API key.")


def _detect_duplicate(
    segment_embeddings: list[list[float]],
) -> DuplicateInfo | None:
    """Check sampled embeddings against the existing index for a near-duplicate.

    Returns the best existing track whose average similarity exceeds the
    configured threshold, or None. Runs before the new track is indexed.
    """
    if not segment_embeddings:
        return None

    # Sample evenly to bound the cost on long tracks.
    count = len(segment_embeddings)
    if count > DUPLICATE_SAMPLE_SEGMENTS:
        step = count / DUPLICATE_SAMPLE_SEGMENTS
        sampled = [
            segment_embeddings[int(i * step)] for i in range(DUPLICATE_SAMPLE_SEGMENTS)
        ]
    else:
        sampled = segment_embeddings

    scores: dict[str, dict[str, float | int | str]] = {}
    for emb in sampled:
        matches = query_similar(emb, top_k=1)
        if not matches:
            continue
        id_map = embedding_to_track_map([m["id"] for m in matches])
        best = matches[0]
        meta = id_map.get(best["id"])
        if meta is None:
            continue
        track_id = str(meta["track_id"])
        row = scores.setdefault(
            track_id,
            {"filename": str(meta["filename"]), "sum": 0.0, "n": 0},
        )
        row["sum"] = float(row["sum"]) + float(best["score"])
        row["n"] = int(row["n"]) + 1

    best_track: DuplicateInfo | None = None
    for track_id, row in scores.items():
        avg = float(row["sum"]) / max(int(row["n"]), 1)
        if avg >= DUPLICATE_THRESHOLD and (
            best_track is None or avg > best_track.similarity
        ):
            best_track = DuplicateInfo(
                track_id=track_id,
                filename=str(row["filename"]),
                similarity=avg,
            )
    return best_track


@app.post("/ingest-track", response_model=IngestResponse)
@limiter.limit(RATE_LIMIT_DEFAULT)
async def ingest_track(
    request: Request,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    artist: str | None = Form(default=None),
    isrc: str | None = Form(default=None),
    tags: str | None = Form(default=None),
) -> IngestResponse:
    """Ingest a reference track into the catalogue."""
    if REQUIRE_API_KEY and API_KEY_CONFIGURED:
        from .security.auth import _extract_api_key

        provided = _extract_api_key(
            request.headers.get("X-API-Key"),
            request.headers.get("Authorization"),
        )
        if not verify_api_key_value(provided):
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    contents = await read_bounded_upload(file, max_bytes)
    safe_filename = validate_upload(file.filename or "", contents)

    try:
        waveform, sr = load_audio_from_bytes(
            contents,
            filename=safe_filename,
            target_sr=SAMPLE_RATE,
            mono=MONO,
            normalize=NORMALIZE,
            max_duration_seconds=MAX_DURATION_SECONDS,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    duration_seconds = float(waveform.size / sr) if sr else 0.0
    segments = segment_audio(waveform, sr, SEGMENT_SECONDS, HOP_SECONDS)
    embeddings = [extract_embedding(segment.samples, sr) for segment in segments]

    # Duplicate check runs against the current index, before this track is added.
    duplicate_of = _detect_duplicate(embeddings)

    metadata = {
        "title": title or "",
        "artist": artist or "",
        "isrc": isrc or "",
        "tags": tags or "",
    }
    track_id = add_track(
        safe_filename, duration_seconds, sr, MODEL_VERSION, metadata=metadata
    )

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
    ext = Path(safe_filename).suffix.lower()
    reference_path = REFERENCE_DIR / f"{track_id}{ext}"
    reference_path.write_bytes(contents)

    message = f"Ingested {safe_filename}"
    if duplicate_of is not None:
        message += f" (possible duplicate of {duplicate_of.filename})"

    return IngestResponse(
        message=message,
        track_id=track_id,
        num_segments=len(segments),
        duration_seconds=duration_seconds,
        duplicate_of=duplicate_of,
    )


@app.post("/query", response_model=QueryResponse)
@limiter.limit(RATE_LIMIT_DEFAULT)
async def query_clip(
    request: Request,
    file: UploadFile = File(...),
) -> QueryResponse:
    """Identify an unknown audio clip."""
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    contents = await read_bounded_upload(file, max_bytes)
    safe_filename = validate_upload(file.filename or "", contents)

    try:
        waveform, sr = load_audio_from_bytes(
            contents,
            filename=safe_filename,
            target_sr=SAMPLE_RATE,
            mono=MONO,
            normalize=NORMALIZE,
            max_duration_seconds=MAX_DURATION_SECONDS,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    query_segments = segment_audio(waveform, sr, SEGMENT_SECONDS, HOP_SECONDS)
    if query_segments:
        segment_embeddings = [
            extract_embedding(segment.samples, sr) for segment in query_segments
        ]
        query_seg_times = [(s.start_time, s.end_time) for s in query_segments]
    else:
        segment_embeddings = [extract_embedding(waveform, sr)]
        query_seg_times = [(0.0, float(waveform.size / sr) if sr else 0.0)]
    query_embedding = np.mean(np.array(segment_embeddings, dtype=float), axis=0).tolist()
    query_duration = query_seg_times[-1][1] if query_seg_times else 0.0

    track_scores: dict[str, dict] = {}
    total_query_segments = max(len(segment_embeddings), 1)
    for segment_idx, segment_embedding in enumerate(segment_embeddings):
        segment_matches = query_similar(segment_embedding, top_k=QUERY_EMBEDDING_TOP_K)
        seg_map = embedding_to_segment_map([match["id"] for match in segment_matches])
        q_start, q_end = query_seg_times[segment_idx]
        for match in segment_matches:
            emb_id = match["id"]
            score = float(match["score"])
            meta = seg_map.get(emb_id)
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
                    "alignments": {},
                },
            )
            row["score_sum"] = float(row["score_sum"]) + score
            row["hits"] = int(row["hits"]) + 1
            row["matched_segments"].add(segment_idx)
            # Keep only the best reference segment per query segment for this track.
            existing = row["alignments"].get(segment_idx)
            if existing is None or score > existing["score"]:
                row["alignments"][segment_idx] = {
                    "query_start": float(q_start),
                    "query_end": float(q_end),
                    "ref_start": float(meta["ref_start"]),
                    "ref_end": float(meta["ref_end"]),
                    "score": score,
                }

    ranked_rows: list[dict] = []
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
        alignments = sorted(
            row.get("alignments", {}).values(),
            key=lambda a: a["query_start"],
        )
        ranked_rows.append(
            {
                "track_id": str(row["track_id"]),
                "filename": str(row["filename"]),
                "similarity": avg_score,
                "coverage": coverage,
                "hits": hits,
                "composite": composite_score,
                "alignments": alignments,
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

    top_similarity = float(top_rows[0]["similarity"]) if top_rows else 0.0
    similarity_gap = 0.0
    if len(top_rows) >= 2:
        similarity_gap = float(top_rows[0]["similarity"]) - float(top_rows[1]["similarity"])

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
            segments=[SegmentMatch(**a) for a in item.get("alignments", [])],
            match_strength=_match_strength(
                float(item["similarity"]), idx, top_similarity, similarity_gap
            ),
        )
        for idx, item in enumerate(top_rows)
    ]

    if len(confidences) >= 2:
        confidence_gap = float(confidences[0] - confidences[1])
    elif len(confidences) == 1:
        confidence_gap = float(confidences[0])
    else:
        confidence_gap = 0.0

    # Use raw acoustic similarity gap (not softmax) for the overall label.
    if top_similarity >= STRONG_MATCH_SIMILARITY and similarity_gap >= MIN_SIMILARITY_GAP:
        confidence_label = "high"
    elif top_similarity >= 0.90 and similarity_gap >= MIN_SIMILARITY_GAP / 2:
        confidence_label = "medium"
    else:
        confidence_label = "low"

    QUERY_DIR.mkdir(parents=True, exist_ok=True)
    query_id = uuid4().hex
    ext = Path(safe_filename).suffix.lower()
    query_path = QUERY_DIR / f"{query_id}{ext}"
    query_path.write_bytes(contents)

    return QueryResponse(
        query_embedding=query_embedding,
        matches=matches,
        confidence_gap=confidence_gap,
        confidence_label=confidence_label,
        query_duration=float(query_duration),
        similarity_gap=float(similarity_gap),
    )


@app.get("/catalogue", response_model=list[CatalogueTrack])
@limiter.limit(RATE_LIMIT_DEFAULT)
async def catalogue(request: Request) -> list[CatalogueTrack]:
    """List ingested tracks and segment counts."""
    return list_tracks()


@app.get("/catalogue/{track_id}", response_model=TrackDetail)
@limiter.limit(RATE_LIMIT_DEFAULT)
async def catalogue_track(request: Request, track_id: str) -> TrackDetail:
    """Get metadata and segment list for a track."""
    safe_id = validate_track_id(track_id)
    track = get_track(safe_id)
    if track is None:
        raise HTTPException(status_code=404, detail="Track not found.")
    return track


@app.get("/catalogue/{track_id}/audio")
@limiter.limit(RATE_LIMIT_DEFAULT)
async def stream_catalogue_audio(request: Request, track_id: str):
    """Stream the stored reference audio for a catalogue track."""
    from fastapi.responses import FileResponse

    safe_id = validate_track_id(track_id)
    if get_track(safe_id) is None:
        raise HTTPException(status_code=404, detail="Track not found.")

    if REFERENCE_DIR.exists():
        for ref in REFERENCE_DIR.glob(f"{safe_id}.*"):
            if ref.is_file():
                media_type = _AUDIO_MEDIA_TYPES.get(
                    ref.suffix.lower(), "application/octet-stream"
                )
                return FileResponse(ref, media_type=media_type, filename=ref.name)

    raise HTTPException(status_code=404, detail="Reference audio not found.")


@app.patch("/catalogue/{track_id}", response_model=TrackDetail)
@limiter.limit(RATE_LIMIT_DEFAULT)
async def update_catalogue_track(
    request: Request,
    track_id: str,
    body: TrackMetadataUpdate,
    _: None = Depends(_require_api_key_when_configured),
) -> TrackDetail:
    """Update editable metadata (title, artist, ISRC, tags) for a track."""
    safe_id = validate_track_id(track_id)
    provided = body.model_dump(exclude_none=True)
    updated = update_track_metadata(safe_id, provided)
    if updated is None:
        raise HTTPException(status_code=404, detail="Track not found.")
    return updated


@app.delete("/catalogue/{track_id}")
@limiter.limit(RATE_LIMIT_DEFAULT)
async def delete_catalogue_track(
    request: Request,
    track_id: str,
    _: None = Depends(_require_api_key_when_configured),
) -> dict[str, str | int]:
    """Delete a single track, its segments, embeddings, and stored audio."""
    safe_id = validate_track_id(track_id)
    summary = delete_track(safe_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Track not found.")

    embedding_ids = [str(e) for e in summary.get("embedding_ids", [])]
    removed = remove_embeddings(embedding_ids)

    if REFERENCE_DIR.exists():
        for ref in REFERENCE_DIR.glob(f"{safe_id}.*"):
            if ref.is_file():
                ref.unlink()

    return {
        "message": f"Deleted {summary.get('filename', safe_id)}",
        "track_id": safe_id,
        "removed_embeddings": removed,
    }


@app.post("/reset-catalogue")
@limiter.limit(RATE_LIMIT_DEFAULT)
async def reset_catalogue_endpoint(
    request: Request,
    _: None = Depends(_require_api_key_when_configured),
) -> dict[str, str]:
    """Wipe all ingested tracks and embeddings. Requires a valid API key."""
    reset_catalogue(persist=True)
    reset_search(persist=True)
    for folder in (REFERENCE_DIR, QUERY_DIR):
        if folder.exists():
            for f in folder.iterdir():
                if f.is_file():
                    f.unlink()
    return {"message": "Catalogue cleared."}
