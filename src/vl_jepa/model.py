"""VL-JEPA model wrapper."""

from __future__ import annotations

import torch
from torch import nn

from .loss import cosine_embedding_loss


class VLJEPA(nn.Module):
    """Video-to-text-embedding JEPA model."""

    def __init__(
        self,
        video_encoder: nn.Module,
        predictor: nn.Module,
        y_encoder: nn.Module,
        freeze_vjepa: bool = True,
    ) -> None:
        super().__init__()
        self.video_encoder = video_encoder
        self.predictor = predictor
        self.y_encoder = y_encoder
        self.freeze_vjepa = freeze_vjepa
        if freeze_vjepa:
            self.video_encoder.eval()
            for param in self.video_encoder.parameters():
                param.requires_grad_(False)
        self.y_encoder.eval()
        for param in self.y_encoder.parameters():
            param.requires_grad_(False)

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        if self.freeze_vjepa:
            with torch.no_grad():
                return self.video_encoder(video).detach()
        return self.video_encoder(video)

    def forward(self, video: torch.Tensor, captions: list[str]) -> dict[str, torch.Tensor]:
        tokens = self.encode_video(video)
        predicted = self.predictor(tokens)
        with torch.no_grad():
            target = self.y_encoder(captions).detach()
        loss = cosine_embedding_loss(predicted, target)
        return {
            "loss": loss,
            "predicted_embedding": predicted,
            "target_embedding": target,
            "video_tokens": tokens,
        }
