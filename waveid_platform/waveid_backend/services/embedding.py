"""
Embedding Service
-----------------

This module provides a baseline implementation of the embedding
extraction pipeline. It exposes a single function ``extract_embedding``
which accepts a waveform and sample rate and returns a list of floats
representing the embedding. In the final system this function would
load a pre-trained model (e.g., OpenL3, VGGish or a custom
contrastive network) and perform spectrogram extraction, batching,
inference, and any necessary normalisation.

For now we return a compact MFCC-based vector padded or truncated to
the configured embedding size.
"""

from __future__ import annotations

import librosa
import numpy as np

from ..config import EMBEDDING_DIM, MFCC_COEFFS


def extract_embedding(waveform: np.ndarray, sr: int) -> list[float]:
    """Return a baseline embedding for the given waveform.

    Args:
        waveform: Audio samples (mono) as float32 array.
        sr: Sample rate of the waveform.

    Returns:
        list[float]: Fixed-length embedding vector.
    """
    if waveform.size == 0:
        return [0.0] * EMBEDDING_DIM

    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=MFCC_COEFFS)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    embedding = np.concatenate([mfcc_mean, mfcc_std])

    if embedding.size < EMBEDDING_DIM:
        pad_width = EMBEDDING_DIM - embedding.size
        embedding = np.pad(embedding, (0, pad_width))
    elif embedding.size > EMBEDDING_DIM:
        embedding = embedding[:EMBEDDING_DIM]

    return embedding.astype(float).tolist()