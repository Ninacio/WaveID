"""
Search Service
--------------

This module implements a minimal in-memory search service using
FAISS-like functionality. In the production system you would use
``faiss.IndexFlatL2`` or a more sophisticated index (e.g., HNSW) to
support efficient nearest neighbour search at scale. Here we fall
back to a naive list and compute cosine similarity by hand. This
implementation supports adding reference embeddings and querying
similar embeddings.

The search state is stored in the module-level lists ``_embeddings``
and ``_identifiers``. Each embedding corresponds to a track or
segment identifier. In this simplified version we assign
incrementing identifiers to each added embedding. When query_similar
is called, cosine similarity scores are computed against all stored
embeddings and the top_k results are returned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..config import INDEX_DIR


# Global in-memory store for embeddings and their identifiers
_embeddings: List[np.ndarray] = []
_identifiers: List[str] = []
_loaded = False
_EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
_IDS_PATH = INDEX_DIR / "embedding_ids.json"


def _load_state() -> None:
    """Load up thefingerprints from disk into memory if not already loaded."""
    global _embeddings, _identifiers, _loaded
    if _loaded:
        return  # already loaded — nothing to do
    _loaded = True
    if _EMBEDDINGS_PATH.exists() and _IDS_PATH.exists():
        array = np.load(_EMBEDDINGS_PATH)
        if array.ndim == 1:
            array = array.reshape(0, -1) if array.size else np.empty((0, 0))
        _embeddings = [row.astype(float) for row in array]
        _identifiers = json.loads(_IDS_PATH.read_text(encoding="utf-8"))


def _save_state() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if _embeddings:
        matrix = np.vstack(_embeddings)
        np.save(_EMBEDDINGS_PATH, matrix)
        _IDS_PATH.write_text(
            json.dumps(_identifiers, indent=2), encoding="utf-8"
        )
    else:
        if _EMBEDDINGS_PATH.exists():
            _EMBEDDINGS_PATH.unlink()
        if _IDS_PATH.exists():
            _IDS_PATH.unlink()


def reset_state(persist: bool = False) -> None:
    """Reset search state for isolated evaluation runs."""
    global _embeddings, _identifiers, _loaded
    _embeddings = []
    _identifiers = []
    _loaded = True
    if persist:
        _save_state()


def add_reference_embeddings(
    embeddings: List[list[float]], embedding_ids: List[str] | None = None
) -> List[str]:
    """Add a batch of reference embeddings to the store.

    Args:
        embeddings: List of embedding vectors (list of floats).
        embedding_ids: Optional list of identifiers for each embedding.

    Returns:
        List of embedding identifiers stored in the index.
    """
    _load_state()
    if embedding_ids is not None and len(embedding_ids) != len(embeddings):
        raise ValueError("embedding_ids must match embeddings length.")

    stored_ids: List[str] = []
    for idx, emb in enumerate(embeddings):
        vector = np.array(emb, dtype=float)
        _embeddings.append(vector)
        if embedding_ids is None:
            embedding_id = f"emb_{len(_identifiers) + 1}"
        else:
            embedding_id = embedding_ids[idx]
        _identifiers.append(embedding_id)
        stored_ids.append(embedding_id)
    _save_state()
    return stored_ids


def query_similar(query_embedding: list[float], top_k: int = 5) -> List[Dict[str, float]]:
    """
    Compares the query fingerprint against every stored reference fingerprint
    and returns the closest matches in ranked order.

    Args:
        query_embedding: The fingerprint of the query clip.
        top_k: Number of matches to return.

    Returns:
        List of matches with identifier and similarity score (highest first).
    """
    _load_state()
    if not _embeddings:
        return []  # no reference tracks have been ingested yet
    query_vec = np.array(query_embedding, dtype=float)
    # Normalise the query so comparisons are based on direction, not magnitude.
    # The small offset (1e-9) prevents division by zero if the vector is all zeros.
    q_norm = np.linalg.norm(query_vec) + 1e-9
    scores = []
    for vec, ident in zip(_embeddings, _identifiers):
        v_norm = np.linalg.norm(vec) + 1e-9
        # Cosine similarity: 1.0 = identical direction, 0.0 = completely unrelated
        score = float(np.dot(query_vec, vec) / (q_norm * v_norm))
        scores.append((ident, score))
    # Put the best match first
    scores.sort(key=lambda x: x[1], reverse=True)
    matches = [{"id": ident, "score": score} for ident, score in scores[:top_k]]
    return matches