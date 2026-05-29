"""Checkpoint loading utilities for V-JEPA-style backbones."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


_STATE_KEYS = (
    "encoder",
    "target_encoder",
    "backbone",
    "model",
    "state_dict",
    "module",
)


def _looks_like_state_dict(value: Any) -> bool:
    # 这里只做轻量判断：checkpoint 可能直接就是 state_dict，也可能包在外层字典里。
    return isinstance(value, dict) and value and all(isinstance(k, str) for k in value)


def _select_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    # 兼容常见保存格式：{"encoder": ...}、{"state_dict": ...} 或裸 state_dict。
    if _looks_like_state_dict(checkpoint):
        tensor_values = [v for v in checkpoint.values() if torch.is_tensor(v)]
        if tensor_values:
            return checkpoint
        for key in _STATE_KEYS:
            if key in checkpoint and _looks_like_state_dict(checkpoint[key]):
                return checkpoint[key]
    raise ValueError("Could not find a model state_dict in checkpoint")


def _clean_key(key: str) -> str:
    # 官方/分布式训练 checkpoint 常带 module.backbone.encoder 等前缀；
    # 本地最小模型只保留 backbone 内部模块名，所以加载前要剥掉这些壳。
    prefixes = (
        "module.",
        "encoder.",
        "target_encoder.",
        "backbone.",
        "context_encoder.",
        "student.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def load_vjepa_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    strict: bool = False,
    map_location: str | torch.device = "cpu",
) -> tuple[list[str], list[str]]:
    """Load V-JEPA checkpoint weights into the local minimal backbone.

    Returns:
        A tuple of `(missing_keys, unexpected_keys)` from `load_state_dict`.
    """

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = _select_state_dict(checkpoint)
    # 本项目只加载视觉 encoder；checkpoint 里如果带 predictor 权重则忽略。
    cleaned = {
        _clean_key(key): value
        for key, value in state_dict.items()
        if not _clean_key(key).startswith(("predictor.", "module.predictor."))
    }
    incompatible = model.load_state_dict(cleaned, strict=strict)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)
