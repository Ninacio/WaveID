"""
Embedding Service
-----------------

This module provides a stub implementation of the embedding
extraction pipeline. It exposes a single function ``extract_embedding``
which accepts raw audio bytes and returns a list of floats
representing the embedding. In the final system this function would
load a pre-trained model (e.g., OpenL3, VGGish or a custom
contrastive network) and perform spectrogram extraction, batching,
inference, and any necessary normalisation.

For now we return a deterministic dummy vector of fixed length to
facilitate end-to-end testing of the API.
"""

from __future__ import annotations

import numpy as np


def extract_embedding(audio_bytes: bytes) -> list[float]:
    """Return a dummy embedding for the given audio data.

    Args:
        audio_bytes: Raw audio data (unused in placeholder implementation).

    Returns:
        list[float]: A 128-dimensional embedding vector with pseudo-random
            values determined by a fixed seed for reproducibility.
    """
    # We use a fixed seed to make the output reproducible across runs.
    rng = np.random.default_rng(seed=42)
    embedding = rng.random(128)
    return embedding.tolist()