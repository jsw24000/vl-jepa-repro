"""CSV manifest dataset for video-text pairs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset

from src.vjepa_backbone.transforms import load_video


@dataclass(frozen=True)
class VideoTextSample:
    video_path: str
    caption: str
    split: str


class VideoTextManifestDataset(Dataset):
    """Read `video_path,caption,split` rows and decode videos on demand."""

    required_columns = {"video_path", "caption", "split"}

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        num_frames: int = 16,
        image_size: int = 224,
        video_loader: Callable[[str | Path, int, int], torch.Tensor] = load_video,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.num_frames = num_frames
        self.image_size = image_size
        self.video_loader = video_loader
        self.samples = self._load_rows()

    def _load_rows(self) -> list[VideoTextSample]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        with self.manifest_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Manifest has no header: {self.manifest_path}")
            missing = self.required_columns - set(reader.fieldnames)
            if missing:
                raise ValueError(f"Manifest missing columns: {sorted(missing)}")
            rows = [
                VideoTextSample(
                    video_path=row["video_path"],
                    caption=row["caption"],
                    split=row["split"],
                )
                for row in reader
                if row["split"] == self.split
            ]
        if not rows:
            raise ValueError(f"No rows for split '{self.split}' in {self.manifest_path}")
        return rows

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        video = self.video_loader(sample.video_path, self.num_frames, self.image_size)
        return {
            "video": video,
            "caption": sample.caption,
            "video_path": sample.video_path,
        }


def video_text_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    videos = torch.stack([item["video"] for item in batch])
    captions = [str(item["caption"]) for item in batch]
    paths = [str(item["video_path"]) for item in batch]
    return {"video": videos, "caption": captions, "video_path": paths}
