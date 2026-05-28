"""Frozen text/Y encoders for VL-JEPA targets."""

from __future__ import annotations

import hashlib

import torch
from torch import nn
import torch.nn.functional as F


class FrozenTextEncoder(nn.Module):
    """Wrapper around a Hugging Face text embedding model."""

    def __init__(
        self,
        model_name: str = "google/embeddinggemma-300m",
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        if device is not None:
            self.model.to(device)

        hidden = getattr(self.model.config, "hidden_size", None)
        if hidden is None:
            hidden = getattr(self.model.config, "text_config", self.model.config).hidden_size
        self.output_dim = int(hidden)

    @torch.no_grad()
    def forward(self, captions: list[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        encoded = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        outputs = self.model(**encoded)
        hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(pooled.detach(), dim=-1)


class DummyFrozenTextEncoder(nn.Module):
    """Deterministic offline text encoder for tests and plumbing checks."""

    def __init__(self, output_dim: int = 768) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.register_buffer("_anchor", torch.zeros(1), persistent=False)

    @torch.no_grad()
    def forward(self, captions: list[str]) -> torch.Tensor:
        vectors = []
        for caption in captions:
            digest = hashlib.sha256(caption.encode("utf-8")).digest()
            values = torch.tensor(list(digest), dtype=torch.float32)
            repeats = (self.output_dim + values.numel() - 1) // values.numel()
            vector = values.repeat(repeats)[: self.output_dim]
            vector = (vector / 127.5) - 1.0
            vectors.append(vector)
        out = torch.stack(vectors).to(self._anchor.device)
        return F.normalize(out, dim=-1)
