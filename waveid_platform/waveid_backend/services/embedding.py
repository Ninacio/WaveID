"""
Embedding Service
-----------------

This module provides a baseline implementation of the embedding
extraction pipeline. It exposes a single function ``extract_embedding``
which accepts a waveform and sample rate and returns a list of floats
representing the embedding. Supports MFCC baseline and contrastive CNN.
"""

from __future__ import annotations

from typing import Literal

import librosa
import numpy as np

from ..config import (
    CONTRASTIVE_MODEL_PATH,
    EMBEDDING_DIM,
    MFCC_COEFFS,
)


def _extract_mfcc(waveform: np.ndarray, sr: int) -> list[float]:
    """MFCC-based baseline embedding."""
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


_contrastive_model = None


def _extract_contrastive(waveform: np.ndarray, sr: int) -> list[float]:
    """Contrastive CNN embedding. Requires trained model at CONTRASTIVE_MODEL_PATH."""
    import torch

    from .contrastive_model import AudioEncoder

    global _contrastive_model
    if _contrastive_model is None:
        if not CONTRASTIVE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Contrastive model not found at {CONTRASTIVE_MODEL_PATH}. "
                "Run train_contrastive.py first."
            )
        ckpt = torch.load(CONTRASTIVE_MODEL_PATH, map_location="cpu", weights_only=True)
        _contrastive_model = AudioEncoder(embedding_dim=ckpt["embedding_dim"])
        _contrastive_model.load_state_dict(ckpt["state_dict"])
        _contrastive_model.eval()

    if waveform.size == 0:
        return [0.0] * _contrastive_model.embedding_dim

    x = waveform.astype(np.float32)
    if x.ndim == 1:
        x = x[np.newaxis, np.newaxis, :]
    else:
        x = np.expand_dims(x, axis=0)
    t = torch.from_numpy(x)
    with torch.no_grad():
        emb = _contrastive_model(t)
    return emb.squeeze(0).numpy().tolist()


def extract_embedding(
    waveform: np.ndarray,
    sr: int,
    model_version: str | Literal["baseline-v1", "contrastive-v1"] | None = None,
) -> list[float]:
    """Return embedding for the given waveform.

    Args:
        waveform: Audio samples (mono) as float32 array.
        sr: Sample rate of the waveform.
        model_version: "baseline-v1" (MFCC) or "contrastive-v1" (CNN). Default from config.

    Returns:
        list[float]: Fixed-length embedding vector.
    """
    from ..config import MODEL_VERSION

    version = model_version or MODEL_VERSION
    if version == "contrastive-v1":
        return _extract_contrastive(waveform, sr)
    return _extract_mfcc(waveform, sr)