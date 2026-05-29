"""Retrieve videos by natural-language query using a trained VL-JEPA predictor."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vjepa_backbone.transforms import load_video
from train_vl_jepa import build_model, load_config


@dataclass(frozen=True)
class VideoItem:
    video_id: str
    video_path: str


class UniqueVideoDataset(Dataset):
    """Load each video path from a manifest once for retrieval indexing."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        num_frames: int,
        image_size: int,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.num_frames = num_frames
        self.image_size = image_size
        self.items = self._load_items()

    def _load_items(self) -> list[VideoItem]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        items: list[VideoItem] = []
        seen: set[str] = set()
        with self.manifest_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if self.split != "all" and row["split"] != self.split:
                    continue
                video_path = row["video_path"]
                if video_path in seen:
                    continue
                seen.add(video_path)
                items.append(
                    VideoItem(
                        video_id=Path(video_path).stem,
                        video_path=video_path,
                    )
                )
        if not items:
            raise ValueError(f"No videos for split '{self.split}' in {self.manifest_path}")
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.items[index]
        video = load_video(item.video_path, self.num_frames, self.image_size)
        return {
            "video": video,
            "video_id": item.video_id,
            "video_path": item.video_path,
        }


def collate_videos(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "video": torch.stack([item["video"] for item in batch]),
        "video_id": [str(item["video_id"]) for item in batch],
        "video_path": [str(item["video_path"]) for item in batch],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text-to-video retrieval for VL-JEPA.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True, default="outputs/vl_jepa_epoch_1.pt", help="Path to vl_jepa_epoch_*.pt")
    parser.add_argument("--query", default="", help="Natural-language search query.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--manifest",
        default="",
        help="Manifest to index. Defaults to data.val_manifest from the config.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=("train", "val", "all"),
        help="Which split in the manifest to index.",
    )
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--cache",
        default="",
        help="Optional .pt file for cached video embeddings.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Recompute video embeddings even when --cache exists.",
    )
    return parser.parse_args()


def load_predictor_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["predictor"] if "predictor" in checkpoint else checkpoint
    model.predictor.load_state_dict(state_dict, strict=True)


@torch.no_grad()
def build_video_index(
    model: torch.nn.Module,
    config: dict,
    manifest_path: str | Path,
    split: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[str], list[str]]:
    model_cfg = config["model"]
    dataset = UniqueVideoDataset(
        manifest_path,
        split=split,
        num_frames=int(model_cfg["num_frames"]),
        image_size=int(model_cfg["image_size"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_videos,
        pin_memory=(device.type == "cuda"),
    )

    embeddings: list[torch.Tensor] = []
    video_ids: list[str] = []
    video_paths: list[str] = []
    for batch in tqdm(loader, desc="index videos"):
        video = batch["video"].to(device, non_blocking=True)
        tokens = model.encode_video(video)
        predicted = model.predictor(tokens)
        embeddings.append(predicted.cpu())
        video_ids.extend(batch["video_id"])
        video_paths.extend(batch["video_path"])

    return torch.cat(embeddings, dim=0), video_ids, video_paths


def load_or_build_index(
    model: torch.nn.Module,
    config: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, list[str], list[str]]:
    cache_path = Path(args.cache) if args.cache else None
    if cache_path and cache_path.exists() and not args.refresh_cache:
        cache = torch.load(cache_path, map_location="cpu")
        return cache["embeddings"], list(cache["video_ids"]), list(cache["video_paths"])

    data_cfg = config["data"]
    manifest_path = args.manifest or data_cfg["val_manifest"]
    batch_size = args.batch_size or int(data_cfg["batch_size"])
    embeddings, video_ids, video_paths = build_video_index(
        model=model,
        config=config,
        manifest_path=manifest_path,
        split=args.split,
        batch_size=batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "embeddings": embeddings,
                "video_ids": video_ids,
                "video_paths": video_paths,
                "manifest": str(manifest_path),
                "split": args.split,
            },
            cache_path,
        )
    return embeddings, video_ids, video_paths


@torch.no_grad()
def encode_query(model: torch.nn.Module, query: str) -> torch.Tensor:
    return model.y_encoder([query]).cpu().squeeze(0)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    requested_device = config["training"].get("device", "cuda")
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")

    query = args.query.strip() or input("Query: ").strip()
    if not query:
        raise ValueError("Query must not be empty")

    model = build_model(config, device)
    load_predictor_checkpoint(model, args.checkpoint)
    model.eval()

    video_embeddings, video_ids, video_paths = load_or_build_index(
        model,
        config,
        args,
        device,
    )
    query_embedding = encode_query(model, query)
    scores = video_embeddings @ query_embedding
    top_k = min(args.top_k, scores.numel())
    values, indices = torch.topk(scores, k=top_k)

    print(f"query: {query}")
    print(f"indexed_videos: {len(video_ids)}")
    for rank, (score, index) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        print(
            f"{rank}\t"
            f"score={score:.4f}\t"
            f"video_id={video_ids[index]}\t"
            f"path={video_paths[index]}"
        )


if __name__ == "__main__":
    main()
