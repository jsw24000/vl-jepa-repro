"""Video loading and preprocessing for V-JEPA-style inputs."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


VJEPA_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
VJEPA_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


def sample_indices(total_frames: int, num_frames: int) -> torch.Tensor:
    if total_frames <= 0:
        raise ValueError("Video must contain at least one frame")
    if total_frames >= num_frames:
        return torch.linspace(0, total_frames - 1, steps=num_frames).long()
    base = torch.arange(total_frames)
    pad = base.new_full((num_frames - total_frames,), total_frames - 1)
    return torch.cat([base, pad], dim=0)


def preprocess_video_tensor(
    video: torch.Tensor,
    num_frames: int = 16,
    image_size: int = 224,
) -> torch.Tensor:
    """Convert video to normalized `(C, T, H, W)` float tensor.

    Accepts `(T, H, W, C)`, `(T, C, H, W)`, or `(C, T, H, W)`.
    """

    if video.ndim != 4:
        raise ValueError(f"Expected 4D video tensor, got shape {tuple(video.shape)}")
    if video.shape[-1] == 3:
        video = video.permute(0, 3, 1, 2)
    elif video.shape[0] == 3:
        video = video.permute(1, 0, 2, 3)
    elif video.shape[1] != 3:
        raise ValueError(f"Cannot infer channel dimension from shape {tuple(video.shape)}")

    video = video.float()
    if video.max() > 2.0:
        video = video / 255.0
    indices = sample_indices(video.shape[0], num_frames).to(video.device)
    video = video.index_select(0, indices)
    video = F.interpolate(
        video,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    video = video.permute(1, 0, 2, 3).contiguous()
    mean = VJEPA_MEAN.to(video.device, video.dtype)
    std = VJEPA_STD.to(video.device, video.dtype)
    return (video - mean) / std


def load_video(
    path: str | Path,
    num_frames: int = 16,
    image_size: int = 224,
) -> torch.Tensor:
    """Read a video file and return normalized `(C, T, H, W)` tensor."""

    try:
        from torchvision.io import read_video
    except Exception as exc:  # pragma: no cover - depends on local torchvision build
        raise RuntimeError("torchvision.io.read_video is required for video loading") from exc

    frames, _, _ = read_video(str(path), pts_unit="sec", output_format="THWC")
    if frames.numel() == 0:
        raise ValueError(f"No frames decoded from {path}")
    return preprocess_video_tensor(frames, num_frames=num_frames, image_size=image_size)
