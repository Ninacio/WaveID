"""
WaveID configuration
--------------------

Centralises project paths and tunable parameters so that
code does not rely on hardcoded file locations.
"""

from __future__ import annotations

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
SAMPLE_RATE = 16_000
MONO = True
NORMALIZE = True
SEGMENT_SECONDS = 2.0
HOP_SECONDS = 1.0
MAX_DURATION_SECONDS = 10 * 60
MAX_UPLOAD_MB = 50
ALLOWED_EXTENSIONS = {".wav", ".mp3"}

# Embedding defaults
EMBEDDING_DIM = 128
MFCC_COEFFS = 20
MODEL_VERSION = "baseline-v1"
