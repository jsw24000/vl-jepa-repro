"""Predictor that maps video tokens into the Y-encoder embedding space."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class EmbeddingPredictor(nn.Module):
    """Attention-pool video tokens, then predict a normalized text embedding."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 2048,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)
        self.attn = nn.MultiheadAttention(input_dim, num_heads=8, batch_first=True)
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        current_dim = input_dim
        for _ in range(max(depth - 1, 0)):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        query = self.query.expand(batch, -1, -1)
        pooled, _ = self.attn(query, tokens, tokens, need_weights=False)
        embedding = self.mlp(pooled.squeeze(1))
        return F.normalize(embedding, dim=-1)
