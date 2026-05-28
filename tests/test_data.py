import csv

import torch

from src.data import VideoTextManifestDataset, video_text_collate
from src.vjepa_backbone.transforms import preprocess_video_tensor


def test_manifest_dataset_and_collate(tmp_path):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_path", "caption", "split"])
        writer.writeheader()
        writer.writerow({"video_path": "one.mp4", "caption": "first", "split": "train"})
        writer.writerow({"video_path": "two.mp4", "caption": "second", "split": "train"})
        writer.writerow({"video_path": "three.mp4", "caption": "third", "split": "val"})

    def fake_loader(path, num_frames, image_size):
        return torch.zeros(3, num_frames, image_size, image_size)

    dataset = VideoTextManifestDataset(
        manifest,
        split="train",
        num_frames=4,
        image_size=32,
        video_loader=fake_loader,
    )
    batch = video_text_collate([dataset[0], dataset[1]])

    assert len(dataset) == 2
    assert batch["video"].shape == (2, 3, 4, 32, 32)
    assert batch["caption"] == ["first", "second"]


def test_preprocess_video_tensor_shape():
    video = torch.randint(0, 255, (3, 20, 24, 24), dtype=torch.uint8)
    out = preprocess_video_tensor(video, num_frames=4, image_size=32)
    assert out.shape == (3, 4, 32, 32)
    assert out.dtype == torch.float32
