"""
WaveID configuration
--------------------

Centralises project paths and tunable parameters so that
code does not rely on hardcoded file locations.

Sensitive values (API keys, secrets) are loaded from environment
variables only. See `.env.example` for required variables.
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root (waveid_platform)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"
QUERY_DIR = DATA_DIR / "query"
INDEX_DIR = DATA_DIR / "index"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Audio ingestion defaults
SAMPLE_RATE = int(os.getenv("WAVEID_SAMPLE_RATE", "16000"))
MONO = os.getenv("WAVEID_MONO", "true").lower() in {"1", "true", "yes"}
NORMALIZE = os.getenv("WAVEID_NORMALIZE", "true").lower() in {"1", "true", "yes"}
SEGMENT_SECONDS = float(os.getenv("WAVEID_SEGMENT_SECONDS", "2.0"))
HOP_SECONDS = float(os.getenv("WAVEID_HOP_SECONDS", "1.0"))
MAX_DURATION_SECONDS = float(os.getenv("WAVEID_MAX_DURATION_SECONDS", str(10 * 60)))
MAX_UPLOAD_MB = int(os.getenv("WAVEID_MAX_UPLOAD_MB", "50"))
MAX_FILENAME_LENGTH = int(os.getenv("WAVEID_MAX_FILENAME_LENGTH", "255"))
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".au"}

# Embedding defaults
EMBEDDING_DIM = int(os.getenv("WAVEID_EMBEDDING_DIM", "128"))
MFCC_COEFFS = int(os.getenv("WAVEID_MFCC_COEFFS", "20"))
MODEL_VERSION = os.getenv("WAVEID_MODEL_VERSION", "contrastive-v1")
CONTRASTIVE_MODEL_PATH = Path(
    os.getenv(
        "WAVEID_CONTRASTIVE_MODEL_PATH",
        str(DATA_DIR / "models" / "contrastive_encoder.pt"),
    )
)

# Query/search defaults
QUERY_TRACK_TOP_K = int(os.getenv("WAVEID_QUERY_TRACK_TOP_K", "5"))
QUERY_EMBEDDING_TOP_K = int(os.getenv("WAVEID_QUERY_EMBEDDING_TOP_K", "5"))
MIN_TRACK_SCORE = float(os.getenv("WAVEID_MIN_TRACK_SCORE", "0.15"))
# Minimum raw similarity gap between #1 and #2 for a "high confidence" identification.
MIN_SIMILARITY_GAP = float(os.getenv("WAVEID_MIN_SIMILARITY_GAP", "0.04"))
# Raw cosine similarity required for a "strong" same-song match label.
STRONG_MATCH_SIMILARITY = float(os.getenv("WAVEID_STRONG_MATCH_SIMILARITY", "0.97"))

# Duplicate detection: when ingesting, warn if an existing track exceeds this
# average similarity against the new track's sampled segments.
DUPLICATE_THRESHOLD = float(os.getenv("WAVEID_DUPLICATE_THRESHOLD", "0.9"))
# Max segments sampled from a new track when checking for duplicates.
DUPLICATE_SAMPLE_SEGMENTS = int(os.getenv("WAVEID_DUPLICATE_SAMPLE_SEGMENTS", "32"))

# Security / auth (never commit real values; use .env locally)
API_KEY: str = os.getenv("WAVEID_API_KEY", "").strip()
API_KEY_CONFIGURED: bool = bool(API_KEY)
# When true, ingest and reset require a valid API key.
REQUIRE_API_KEY: bool = os.getenv("WAVEID_REQUIRE_API_KEY", "false").lower() in {
    "1",
    "true",
    "yes",
}
# Wipe catalogue on startup (disable in production to preserve persisted index).
RESET_ON_STARTUP: bool = os.getenv("WAVEID_RESET_ON_STARTUP", "true").lower() in {
    "1",
    "true",
    "yes",
}

# Rate limiting (slowapi format strings)
RATE_LIMIT_DEFAULT: str = os.getenv("WAVEID_RATE_LIMIT_DEFAULT", "100/minute")
RATE_LIMIT_AUTH: str = os.getenv("WAVEID_RATE_LIMIT_AUTH", "5/15minutes")

# CORS: comma-separated origins, empty = same-origin only (no CORS middleware)
CORS_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv("WAVEID_CORS_ORIGINS", "").split(",")
    if origin.strip()
]

# External tool paths (optional; no secrets)
FPCALC_PATH: str = os.getenv("FPCALC", "fpcalc")
