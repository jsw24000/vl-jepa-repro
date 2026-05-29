# type: ignore[reportMissingImports]
"""Single-GPU VL-JEPA minimal training entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.data import VideoTextManifestDataset, video_text_collate
from src.vjepa_backbone import VIT_EMBED_DIMS, build_vjepa_encoder, load_vjepa_checkpoint
from src.vl_jepa import EmbeddingPredictor, FrozenTextEncoder, VLJEPA


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(config: dict, device: torch.device) -> VLJEPA:
    model_cfg = config["model"]
    backbone_name = model_cfg["backbone_name"]
    video_encoder = build_vjepa_encoder(
        backbone_name,
        image_size=int(model_cfg["image_size"]),
        num_frames=int(model_cfg["num_frames"]),
        patch_size=int(model_cfg["patch_size"]),
        tubelet_size=int(model_cfg["tubelet_size"]),
    )
    checkpoint_path = model_cfg.get("vjepa_checkpoint", "")
    if checkpoint_path:
        missing, unexpected = load_vjepa_checkpoint(video_encoder, checkpoint_path)
        print(f"Loaded V-JEPA checkpoint: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("Warning: no V-JEPA checkpoint configured; backbone is randomly initialized.")

    y_encoder = FrozenTextEncoder(model_cfg["y_encoder_name"], device=device)
    predictor = EmbeddingPredictor(
        input_dim=VIT_EMBED_DIMS[backbone_name],
        output_dim=y_encoder.output_dim,
        hidden_dim=int(model_cfg["predictor_hidden_dim"]),
        depth=int(model_cfg["predictor_depth"]),
    )
    model = VLJEPA(
        video_encoder=video_encoder,
        predictor=predictor,
        y_encoder=y_encoder,
        freeze_vjepa=bool(model_cfg.get("freeze_vjepa", True)),
    )
    return model.to(device)


def make_loader(config: dict, split: str) -> DataLoader:
    model_cfg = config["model"]
    data_cfg = config["data"]
    manifest = data_cfg["train_manifest"] if split == "train" else data_cfg["val_manifest"]
    dataset = VideoTextManifestDataset(
        manifest,
        split=split,
        num_frames=int(model_cfg["num_frames"]),
        image_size=int(model_cfg["image_size"]),
    )
    return DataLoader(
        dataset,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=(split == "train"),
        num_workers=int(data_cfg["num_workers"]),
        collate_fn=video_text_collate,
        pin_memory=True,
    )


def train_one_epoch(
    model: VLJEPA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    log_every: int,
) -> float:
    model.train()
    if model.freeze_vjepa:
        model.video_encoder.eval()
    model.y_encoder.eval()

    losses = []
    progress = tqdm(loader, desc="train", leave=False)
    for step, batch in enumerate(progress, start=1):
        video = batch["video"].to(device, non_blocking=True)
        captions = batch["caption"]
        optimizer.zero_grad(set_to_none=True)
        output = model(video, captions)
        loss = output["loss"]
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step % log_every == 0:
            progress.set_postfix(loss=sum(losses[-log_every:]) / len(losses[-log_every:]))
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def evaluate(model: VLJEPA, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    for batch in tqdm(loader, desc="val", leave=False):
        video = batch["video"].to(device, non_blocking=True)
        output = model(video, batch["caption"])
        losses.append(float(output["loss"].cpu()))
    return sum(losses) / max(len(losses), 1)


def save_checkpoint(model: VLJEPA, config: dict, output_dir: str | Path, epoch: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"vl_jepa_epoch_{epoch}.pt"
    torch.save(
        {
            "epoch": epoch,
            "predictor": model.predictor.state_dict(),
            "config": config,
        },
        path,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    requested_device = config["training"].get("device", "cuda")
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
    model = build_model(config, device)
    train_loader = make_loader(config, "train")
    val_loader = make_loader(config, "val")

    optimizer = torch.optim.AdamW(
        model.predictor.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            grad_clip=float(config["training"]["gradient_clip"]),
            log_every=int(config["training"]["log_every"]),
        )
        val_loss = evaluate(model, val_loader, device)
        path = save_checkpoint(model, config, config["training"]["output_dir"], epoch)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} checkpoint={path}"
        )


if __name__ == "__main__":
    main()
