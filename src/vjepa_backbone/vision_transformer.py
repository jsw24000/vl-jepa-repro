"""Minimal V-JEPA-style video Vision Transformer.

This module intentionally keeps only the backbone features needed by VL-JEPA:
3D tubelet patch embedding, fixed sinusoidal positional embeddings, transformer
blocks, checkpoint-friendly module names, and token output.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable

import numpy as np
import torch
from torch import nn


class PatchEmbed3D(nn.Module):
    """Video to tubelet token embedding.

    输入形状是 `(B, C, T, H, W)`。3D 卷积会同时沿时间和空间切块：
    每个 tubelet 覆盖 `tubelet_size` 帧和一个 `patch_size x patch_size`
    空间块，输出是一串 token，形状为 `(B, num_tokens, embed_dim)`。
    """

    def __init__(
        self,
        patch_size: int = 16,
        tubelet_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv3d 输出 `(B, D, T/tubelet, H/patch, W/patch)`，再展平成 token 序列。
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch, tokens, dim)
        x = self.proj(x)
        return self.proj_drop(x)


class Block(nn.Module):
    """Pre-norm Transformer block used by the video encoder."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def _get_1d_sincos_pos_embed(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("Sin-cos embedding dimension must be even")
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)
    out = np.einsum("m,d->md", positions.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_3d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    grid_depth: int,
) -> np.ndarray:
    """Build temporal/spatial sin-cos embeddings for video tokens.

    V-JEPA 的视频 token 同时有时间位置和空间位置。这里把 embedding 维度
    分配给 `t/h/w` 三个轴，再拼成每个 tubelet token 的固定位置编码。
    """

    t_dim = embed_dim // 4
    h_dim = embed_dim // 4
    w_dim = embed_dim - t_dim - h_dim
    if t_dim % 2:
        t_dim += 1
        w_dim -= 1
    if h_dim % 2:
        h_dim += 1
        w_dim -= 1
    if w_dim % 2:
        w_dim -= 1
        h_dim += 1

    t = np.arange(grid_depth, dtype=np.float32)
    h = np.arange(grid_size, dtype=np.float32)
    w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(t, h, w, indexing="ij")
    emb_t = _get_1d_sincos_pos_embed(t_dim, grid[0])
    emb_h = _get_1d_sincos_pos_embed(h_dim, grid[1])
    emb_w = _get_1d_sincos_pos_embed(w_dim, grid[2])
    return np.concatenate([emb_t, emb_h, emb_w], axis=1).astype(np.float32)


class VisionTransformer(nn.Module):
    """V-JEPA-style video encoder that returns patch/tubelet tokens.

    这个 backbone 只负责把视频编码成 token 序列，不做分类头或文本对齐。
    在本项目中，它通常被冻结，然后由 `EmbeddingPredictor` 把这些视频 token
    映射到文本 embedding 空间。
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 16,
        tubelet_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.num_features = embed_dim
        self.embed_dim = embed_dim
        self.input_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size

        grid_size = img_size // patch_size
        grid_depth = num_frames // tubelet_size
        self.patch_embed = PatchEmbed3D(patch_size, tubelet_size, in_chans, embed_dim)
        self.num_patches = grid_depth * grid_size * grid_size

        # 固定 sin-cos 位置编码，不作为可训练参数；官方 V-JEPA checkpoint
        # 的位置编码形状必须和这里的 `(T/tubelet) * (H/patch) * (W/patch)` 匹配。
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim),
            requires_grad=False,
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        grid_size = self.input_size // self.patch_size
        grid_depth = self.num_frames // self.tubelet_size
        pos = get_3d_sincos_pos_embed(self.embed_dim, grid_size, grid_depth)
        self.pos_embed.data.copy_(torch.from_numpy(pos).unsqueeze(0))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
            elif isinstance(module, nn.Conv3d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def interpolate_pos_encoding(self, x: torch.Tensor) -> torch.Tensor:
        # 如果输入分辨率/帧数和初始化一致，直接复用 checkpoint 中的固定位置编码。
        _, _, frames, height, width = x.shape
        if (
            frames == self.num_frames
            and height == self.input_size
            and width == self.input_size
        ):
            return self.pos_embed

        dim = self.pos_embed.shape[-1]
        old_t = self.num_frames // self.tubelet_size
        old_h = old_w = self.input_size // self.patch_size
        new_t = frames // self.tubelet_size
        new_h = height // self.patch_size
        new_w = width // self.patch_size
        # 形状先还原成 3D 网格，再用三线性插值适配新的时间/空间尺寸。
        pos = self.pos_embed.reshape(1, old_t, old_h, old_w, dim)
        pos = pos.permute(0, 4, 1, 2, 3)
        pos = nn.functional.interpolate(
            pos,
            size=(new_t, new_h, new_w),
            mode="trilinear",
            align_corners=False,
        )
        return pos.permute(0, 2, 3, 4, 1).reshape(1, new_t * new_h * new_w, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 `(B, 3, T, H, W)` -> 输出 `(B, num_tokens, embed_dim)`。
        pos_embed = self.interpolate_pos_encoding(x)
        x = self.patch_embed(x)
        x = x + pos_embed.to(dtype=x.dtype, device=x.device)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def vit_tiny(patch_size: int = 16, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=2,
        num_heads=3,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_base(patch_size: int = 16, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_large(patch_size: int = 16, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_huge(patch_size: int = 16, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


VIT_EMBED_DIMS = {
    "vit_tiny": 192,
    "vit_base": 768,
    "vit_large": 1024,
    "vit_huge": 1280,
}


def build_vjepa_encoder(
    name: str,
    image_size: int = 224,
    num_frames: int = 16,
    patch_size: int = 16,
    tubelet_size: int = 2,
) -> VisionTransformer:
    factories = {
        "vit_tiny": vit_tiny,
        "vit_base": vit_base,
        "vit_large": vit_large,
        "vit_huge": vit_huge,
    }
    if name not in factories:
        valid = ", ".join(sorted(factories))
        raise ValueError(f"Unknown V-JEPA backbone '{name}'. Valid choices: {valid}")
    return factories[name](
        img_size=image_size,
        num_frames=num_frames,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
    )
