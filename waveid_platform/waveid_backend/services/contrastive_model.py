"""
Contrastive audio encoder for WaveID.

Small 1D CNN that maps waveform segments to embeddings.
Designed for triplet/contrastive training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    """1D CNN encoder for audio segments."""

    def __init__(
        self,
        in_channels: int = 1,
        embedding_dim: int = 128,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(base_channels * 4, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1, samples) -> (batch, embedding_dim)"""
        h = self.conv(x)
        h = h.squeeze(-1)
        return self.fc(h)


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Triplet loss: L = max(0, d(a,p) - d(a,n) + margin)."""
    d_pos = (anchor - positive).pow(2).sum(dim=1)
    d_neg = (anchor - negative).pow(2).sum(dim=1)
    return (d_pos - d_neg + margin).clamp(min=0).mean()
