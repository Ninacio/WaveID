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

import numpy as np
from typing import List, Dict


# Global in-memory store for embeddings and their identifiers
_embeddings: List[np.ndarray] = []
_identifiers: List[str] = []


def add_reference_embeddings(embeddings: List[list[float]]) -> None:
    """Add a batch of reference embeddings to the store.

    Args:
        embeddings: List of embedding vectors (list of floats).
    """
    for emb in embeddings:
        vector = np.array(emb, dtype=float)
        _embeddings.append(vector)
        _identifiers.append(f"track_{len(_identifiers) + 1}")


def query_similar(query_embedding: list[float], top_k: int = 5) -> List[Dict[str, float]]:
    """Return the top_k most similar reference embeddings.

    This function uses cosine similarity to compute distances between
    the query and all stored embeddings. It returns a list of
    dictionaries with ``id`` and ``score`` keys sorted by decreasing
    similarity.

    Args:
        query_embedding: The embedding vector for the query clip.
        top_k: Number of matches to return.

    Returns:
        List of matches with identifier and similarity score.
    """
    if not _embeddings:
        return []
    query_vec = np.array(query_embedding, dtype=float)
    # Normalise query for cosine similarity
    q_norm = np.linalg.norm(query_vec) + 1e-9
    scores = []
    for vec, ident in zip(_embeddings, _identifiers):
        v_norm = np.linalg.norm(vec) + 1e-9
        score = float(np.dot(query_vec, vec) / (q_norm * v_norm))
        scores.append((ident, score))
    # Sort by score descending and select top_k
    scores.sort(key=lambda x: x[1], reverse=True)
    matches = [ {"id": ident, "score": score} for ident, score in scores[:top_k] ]
    return matches