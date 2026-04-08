"""
Contrastive audio encoder for WaveID.

Small 1D CNN that maps waveform segments to embeddings.
Designed for triplet/contrastive training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    """
    Neural network that converts a raw audio clip into a compact 128-number fingerprint.
    Uses three layers of pattern detection (1D convolution) on the waveform,
    each finding progressively more abstract sound characteristics.
    """

    def __init__(
        self,
        in_channels: int = 1,       # mono audio has a single channel
        embedding_dim: int = 128,   # size of the output fingerprint
        base_channels: int = 32,    # controls how wide the network is
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        # Three convolutional blocks applied in sequence.
        # Each block scans the audio for patterns, then compresses the signal length.
        self.conv = nn.Sequential(
            # Block 1: detect short, fine-grained sound patterns
            nn.Conv1d(in_channels, base_channels, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(base_channels),  # normalise outputs to keep training stable
            nn.ReLU(inplace=True),          # discard negative (irrelevant) responses
            nn.MaxPool1d(2),                # halve the signal length
            # Block 2: detect mid-level patterns using Block 1's output
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            # Block 3: detect high-level patterns and collapse to a single summary value
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # average across all remaining time positions
        )
        # Final step: map the summary to a fingerprint of the chosen size
        self.fc = nn.Linear(base_channels * 4, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Accepts (batch, 1, samples) and returns (batch, embedding_dim)."""
        h = self.conv(x)        # pass audio through all three convolutional blocks
        h = h.squeeze(-1)       # remove the now-redundant time dimension
        return self.fc(h)       # produce the final fingerprint


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,   # same track as anchor, but distorted
    negative: torch.Tensor,   # clip from a different track
    margin: float = 0.2,      # minimum gap required between correct and wrong match
) -> torch.Tensor:
    """
    Teaches the network to place same-track clips close together
    and different-track clips far apart in fingerprint space.
    """
    # Distance between the anchor and its correct match (should be small)
    d_pos = (anchor - positive).pow(2).sum(dim=1)
    # Distance between the anchor and the wrong track (should be large)
    d_neg = (anchor - negative).pow(2).sum(dim=1)
    # Only apply a penalty if the wrong track is not already far enough away.
    # clamp(min=0) ensures the loss never goes negative.
    return (d_pos - d_neg + margin).clamp(min=0).mean()
