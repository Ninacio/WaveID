"""
Audio segmentation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segment:
    start_time: float
    end_time: float
    samples: np.ndarray


def segment_audio(
    waveform: np.ndarray,
    sr: int,
    segment_seconds: float,
    hop_seconds: float,
) -> list[Segment]:
    """Slice waveform into overlapping segments."""
    if sr <= 0:
        raise ValueError("Sample rate must be positive.")
    if segment_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("Segment and hop durations must be positive.")

    if waveform.size == 0:
        return []

    segment_len = int(round(segment_seconds * sr))
    hop_len = int(round(hop_seconds * sr))
    if segment_len <= 0 or hop_len <= 0:
        raise ValueError("Segment length and hop length must be positive.")

    total_len = waveform.shape[0]
    if total_len <= segment_len:
        duration = total_len / sr
        return [Segment(0.0, duration, waveform)]

    segments: list[Segment] = []
    for start in range(0, total_len - segment_len + 1, hop_len):
        end = start + segment_len
        segments.append(
            Segment(start / sr, end / sr, waveform[start:end].copy())
        )

    return segments
