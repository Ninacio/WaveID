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
    """Converts a short audio clip into a 128-number fingerprint using frequency statistics (MFCC)."""
    if waveform.size == 0:
        return [0.0] * EMBEDDING_DIM  # nothing to process — return a blank fingerprint

    # Summarise how energy is spread across frequency bands over time
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=MFCC_COEFFS)
    mfcc_mean = mfcc.mean(axis=1)  # average level of each band across the clip
    mfcc_std = mfcc.std(axis=1)    # how much each band varied across the clip
    embedding = np.concatenate([mfcc_mean, mfcc_std])  # combine into one fingerprint

    # Ensure the fingerprint is always exactly 128 numbers long
    if embedding.size < EMBEDDING_DIM:
        pad_width = EMBEDDING_DIM - embedding.size
        embedding = np.pad(embedding, (0, pad_width))  # fill any gap with zeros
    elif embedding.size > EMBEDDING_DIM:
        embedding = embedding[:EMBEDDING_DIM]           # trim if unexpectedly too long

    return embedding.astype(float).tolist()


_contrastive_model = None


def _extract_contrastive(waveform: np.ndarray, sr: int) -> list[float]:
    """Runs the audio clip through the trained CNN to produce a 128-number fingerprint."""
    import torch

    from .contrastive_model import AudioEncoder

    global _contrastive_model
    if _contrastive_model is None:
        # load up the trained network weights from disk the first time this is called
        if not CONTRASTIVE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Contrastive model not found at {CONTRASTIVE_MODEL_PATH}. "
                "Run train_contrastive.py first."
            )
        ckpt = torch.load(CONTRASTIVE_MODEL_PATH, map_location="cpu", weights_only=True)
        _contrastive_model = AudioEncoder(embedding_dim=ckpt["embedding_dim"])
        _contrastive_model.load_state_dict(ckpt["state_dict"])
        _contrastive_model.eval()  # switch to inference mode (disables dropout etc.)

    if waveform.size == 0:
        return [0.0] * _contrastive_model.embedding_dim  # blank fingerprint for empty audio

    # Reshape the audio into the format the network expects: (batch=1, channels=1, samples)
    x = waveform.astype(np.float32)
    if x.ndim == 1:
        x = x[np.newaxis, np.newaxis, :]
    else:
        x = np.expand_dims(x, axis=0)
    t = torch.from_numpy(x)
    with torch.no_grad():  # no gradient tracking needed — we are just predicting, not training
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