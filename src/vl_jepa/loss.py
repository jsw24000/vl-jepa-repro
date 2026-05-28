"""VL-JEPA embedding-space objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_embedding_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    predicted = F.normalize(predicted, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    return (1.0 - (predicted * target).sum(dim=-1)).mean()
