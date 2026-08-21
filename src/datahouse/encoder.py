"""Standalone frozen frame encoder owned by :mod:`contra_nes_data`.

The inference network deliberately has no dependency on ``contra_nes_policy``.
Its checkpoint embeds the architectural configuration; the accompanying datahouse
bundle records the checkpoint digest and image preprocessing contract.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


def _norm(channels: int) -> nn.Module:
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvEncoder(nn.Module):
    """Dreamer-compatible convolutional feature extractor without policy imports."""

    def __init__(self, *, size: int, depth: int, minres: int):
        super().__init__()
        layers: list[nn.Module] = []
        channels = 3
        n_layers = int(round(math.log2(size / minres)))
        if minres * 2 ** n_layers != size:
            raise ValueError("image size must be minres times a power of two")
        for index in range(n_layers):
            out = depth * 2 ** index
            layers += [nn.Conv2d(channels, out, 4, stride=2, padding=1),
                       _norm(out), nn.SiLU()]
            channels = out
        self.convs = nn.Sequential(*layers)
        self.conv_out_ch = channels

    def forward_features(self, image: torch.Tensor) -> torch.Tensor:
        return self.convs(image)


@dataclass(frozen=True)
class EncoderSpec:
    """The immutable token contract carried next to a datahouse encoder bundle."""

    checkpoint_sha256: str
    image_size: int
    interpolation: str
    token_dim: int
    token_dtype: str = "float16"
    input_layout: str = "uint8_rgb_hwc"

    @classmethod
    def from_checkpoint(cls, checkpoint: str) -> "EncoderSpec":
        digest = hashlib.sha256()
        with open(checkpoint, "rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)
        payload: dict[str, Any] = torch.load(checkpoint, map_location="cpu",
                                             weights_only=False)
        config = payload["config"]
        return cls(checkpoint_sha256=digest.hexdigest(), image_size=int(config["image_size"]),
                   interpolation="INTER_AREA", token_dim=int(config["hiddim"]))


class FrameEncoder(nn.Module):
    """The inference-only image → token portion of the published frame encoder."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = dict(config)
        size = int(config["image_size"])
        minres = int(config["minres"])
        proj_ch = int(config["proj_ch"])
        dim = int(config["hiddim"])
        self.view_backbone = ConvEncoder(size=size, depth=int(config["depth"]),
                                         minres=minres)
        self.reduce = nn.Sequential(nn.Conv2d(self.view_backbone.conv_out_ch, proj_ch, 1),
                                    _norm(proj_ch), nn.SiLU())
        self.proj = nn.Sequential(nn.Linear(proj_ch * minres * minres, dim),
                                  nn.LayerNorm(dim), nn.SiLU(), nn.Linear(dim, dim))
        self.token_ln = nn.LayerNorm(dim)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encode uint8 ``(B, H, W, 3)`` RGB images into frame tokens."""
        if image.dtype != torch.uint8 or image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError("encoder input must be uint8 BHWC RGB")
        x = image.permute(0, 3, 1, 2).float().div(255.0)
        z = self.reduce(self.view_backbone.forward_features(x))
        return self.token_ln(self.proj(z.flatten(1)))


def load_encoder(checkpoint: str, *, freeze: bool = True,
                 map_location: str = "cpu") -> FrameEncoder:
    """Load the published encoder's inference path, rejecting unexpected core keys."""
    payload: dict[str, Any] = torch.load(os.path.expanduser(checkpoint),
                                         map_location=map_location,
                                         weights_only=False)
    encoder = FrameEncoder(payload["config"])
    missing, unexpected = encoder.load_state_dict(payload["encoder"], strict=False)
    if missing or any(not key.startswith(("entity_head.", "recon_head."))
                      for key in unexpected):
        raise ValueError(f"encoder checkpoint does not match datahouse inference model: "
                         f"missing={missing}, unexpected={unexpected}")
    if freeze:
        encoder.requires_grad_(False).eval()
    return encoder
