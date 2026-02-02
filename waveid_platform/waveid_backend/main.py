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

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .services.embedding import extract_embedding
from .services.search import add_reference_embeddings, query_similar


app = FastAPI(title="WaveID Backend", version="0.1.0")


class IngestResponse(BaseModel):
    message: str
    num_segments: int


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
    # TODO: Implement actual preprocessing and segmentation.
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Placeholder: simulate creation of embeddings for 10 segments
    dummy_embedding = extract_embedding(b"dummy audio")
    add_reference_embeddings([dummy_embedding])

    return IngestResponse(message=f"Ingested {file.filename}", num_segments=1)


class QueryResponse(BaseModel):
    query_embedding: list[float]
    matches: list[dict[str, float]]


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
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    query_embedding = extract_embedding(b"dummy audio query")
    # TODO: Replace dummy audio with actual file data
    matches = query_similar(query_embedding, top_k=5)

    return QueryResponse(query_embedding=query_embedding, matches=matches)