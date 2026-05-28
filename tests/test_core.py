import torch

from src.vjepa_backbone import build_vjepa_encoder
from src.vl_jepa import DummyFrozenTextEncoder, EmbeddingPredictor, VLJEPA


def test_vljepa_forward_and_freezing():
    video_encoder = build_vjepa_encoder(
        "vit_tiny",
        image_size=32,
        num_frames=4,
        patch_size=16,
        tubelet_size=2,
    )
    y_encoder = DummyFrozenTextEncoder(output_dim=64)
    predictor = EmbeddingPredictor(input_dim=192, output_dim=64, hidden_dim=128, depth=2)
    model = VLJEPA(video_encoder, predictor, y_encoder, freeze_vjepa=True)

    video = torch.randn(2, 3, 4, 32, 32)
    out = model(video, ["a person runs", "a dog jumps"])

    assert out["predicted_embedding"].shape == (2, 64)
    assert out["target_embedding"].shape == (2, 64)
    assert out["video_tokens"].shape == (2, 8, 192)
    assert not out["target_embedding"].requires_grad
    assert not out["video_tokens"].requires_grad

    out["loss"].backward()
    assert any(p.grad is not None for p in predictor.parameters())
    assert all(p.grad is None for p in video_encoder.parameters())
    assert all(p.grad is None for p in y_encoder.parameters())


def test_vjepa_encoder_checkpoint_roundtrip(tmp_path):
    source = build_vjepa_encoder(
        "vit_tiny",
        image_size=32,
        num_frames=4,
        patch_size=16,
        tubelet_size=2,
    )
    path = tmp_path / "ckpt.pt"
    torch.save({"encoder": source.state_dict()}, path)

    target = build_vjepa_encoder(
        "vit_tiny",
        image_size=32,
        num_frames=4,
        patch_size=16,
        tubelet_size=2,
    )
    from src.vjepa_backbone import load_vjepa_checkpoint

    missing, unexpected = load_vjepa_checkpoint(target, path, strict=True)
    assert missing == []
    assert unexpected == []
