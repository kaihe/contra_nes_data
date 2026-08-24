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

    def __init__(self, *, height: int, width: int, depth: int, n_layers: int):
        super().__init__()
        layers: list[nn.Module] = []
        channels = 3
        for index in range(n_layers):
            out = depth * 2 ** index
            layers += [nn.Conv2d(channels, out, 4, stride=2, padding=1),
                       _norm(out), nn.SiLU()]
            channels = out
        self.convs = nn.Sequential(*layers)
        self.conv_out_ch = channels
        self.output_hw = (height // 2 ** n_layers, width // 2 ** n_layers)
        if min(self.output_hw) < 1:
            raise ValueError("too many convolution stages for input dimensions")

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
        if "input_height" in config or "input_width" in config:
            height = int(config["input_height"])
            width = int(config["input_width"])
            n_layers = int(config["n_layers"])
        else:
            height = width = int(config["image_size"])
            minres = int(config["minres"])
            n_layers = int(round(math.log2(height / minres)))
            if minres * 2 ** n_layers != height:
                raise ValueError("image size must be minres times a power of two")
        self.input_hw = (height, width)
        proj_ch = int(config["proj_ch"])
        dim = int(config["hiddim"])
        self.view_backbone = ConvEncoder(height=height, width=width,
                                         depth=int(config["depth"]), n_layers=n_layers)
        self.reduce = nn.Sequential(nn.Conv2d(self.view_backbone.conv_out_ch, proj_ch, 1),
                                    _norm(proj_ch), nn.SiLU())
        feature_height, feature_width = self.view_backbone.output_hw
        self.proj = nn.Sequential(nn.Linear(proj_ch * feature_height * feature_width, dim),
                                  nn.LayerNorm(dim), nn.SiLU(), nn.Linear(dim, dim))
        self.token_ln = nn.LayerNorm(dim)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encode uint8 ``(B, H, W, 3)`` RGB images into frame tokens."""
        if image.dtype != torch.uint8 or image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError("encoder input must be uint8 BHWC RGB")
        if tuple(image.shape[1:3]) != self.input_hw:
            raise ValueError(f"encoder input must have spatial shape {self.input_hw}")
        x = image.permute(0, 3, 1, 2).float().div(255.0)
        z = self.reduce(self.view_backbone.forward_features(x))
        return self.token_ln(self.proj(z.flatten(1)))


class TemporalFrameEncoder(FrameEncoder):
    """0019 encoder over current RGB plus signed current-minus-previous RGB."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        old = self.view_backbone.convs[0]
        expanded = nn.Conv2d(6, old.out_channels, old.kernel_size,
                             stride=old.stride, padding=old.padding,
                             dilation=old.dilation, groups=old.groups,
                             bias=old.bias is not None,
                             padding_mode=old.padding_mode)
        self.view_backbone.convs[0] = expanded

    def encode_pair(self, current: torch.Tensor,
                    previous: torch.Tensor) -> torch.Tensor:
        """Encode equal-shape uint8 BHWC pairs; the delta remains signed float."""
        expected = (*self.input_hw, 3)
        if (current.dtype != torch.uint8 or previous.dtype != torch.uint8 or
                current.ndim != 4 or previous.shape != current.shape or
                tuple(current.shape[1:]) != expected):
            raise ValueError(f"temporal inputs must be equal uint8 B{expected}")
        current_float = current.permute(0, 3, 1, 2).float().div(255.0)
        previous_float = previous.permute(0, 3, 1, 2).float().div(255.0)
        x = torch.cat((current_float, current_float - previous_float), dim=1)
        z = self.reduce(self.view_backbone.forward_features(x))
        return self.token_ln(self.proj(z.flatten(1)))

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a standalone image with zero delta, used for visual goals."""
        return self.encode_pair(image, image)

    def encode_sequence(self, images: torch.Tensor) -> torch.Tensor:
        """Encode one episode's BHWC sequence without crossing its boundary."""
        if len(images) == 0:
            return torch.empty((0, int(self.config["hiddim"])), device=images.device)
        previous = torch.cat((images[:1], images[:-1]), dim=0)
        return self.encode_pair(images, previous)


class HeatmapHead(nn.Module):
    """Decode four occupancy heatmaps from the single frame token."""

    def __init__(self, dim: int, grid: int, n_classes: int, depth: int,
                 base: int = 4):
        super().__init__()
        n_up = int(round(math.log2(grid / base)))
        if base * 2 ** n_up != grid:
            raise ValueError("heatmap grid must be base times a power of two")
        channels = depth * 2 ** n_up
        self.base, self.seed_ch = base, channels
        self.seed = nn.Linear(dim, channels * base * base)
        layers: list[nn.Module] = []
        for _ in range(n_up):
            out = max(depth, channels // 2)
            layers += [nn.ConvTranspose2d(channels, out, 4, stride=2, padding=1),
                       _norm(out), nn.SiLU()]
            channels = out
        self.ups = nn.Sequential(*layers)
        self.out = nn.Conv2d(channels, n_classes, 3, padding=1)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        value = self.seed(token).view(len(token), self.seed_ch,
                                      self.base, self.base)
        return self.out(self.ups(value))


class EntityFrameEncoder(FrameEncoder):
    """Published frame encoder with its training-only entity probe attached."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.entity_head = HeatmapHead(
            dim=int(config["hiddim"]), grid=int(config["aux_size"]),
            n_classes=int(config["entity_classes"]),
            depth=int(config["head_depth"]))

    def entity_logits(self, image: torch.Tensor) -> torch.Tensor:
        return self.entity_head(self.encode(image))


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


def load_entity_encoder(checkpoint: str, *, map_location: str = "cpu"
                        ) -> EntityFrameEncoder:
    """Load the frozen encoder plus the entity head used for baseline measurement."""
    payload: dict[str, Any] = torch.load(os.path.expanduser(checkpoint),
                                         map_location=map_location,
                                         weights_only=False)
    encoder = EntityFrameEncoder(payload["config"])
    encoder.load_state_dict(payload["encoder"], strict=True)
    return encoder.requires_grad_(False).eval()


def load_temporal_encoder(checkpoint: str, *, freeze: bool = True,
                          map_location: str = "cpu") -> TemporalFrameEncoder:
    """Load the signed-frame-difference inference path from an exported bundle."""
    payload: dict[str, Any] = torch.load(os.path.expanduser(checkpoint),
                                         map_location=map_location,
                                         weights_only=False)
    if payload.get("config", {}).get("input_kind") != "rgb_signed_frame_difference":
        raise ValueError("checkpoint does not declare the temporal input contract")
    encoder = TemporalFrameEncoder(payload["config"])
    missing, unexpected = encoder.load_state_dict(payload["encoder"], strict=False)
    if missing or any(not key.startswith("entity_head.") for key in unexpected):
        raise ValueError(f"temporal checkpoint mismatch: missing={missing}, "
                         f"unexpected={unexpected}")
    if freeze:
        encoder.requires_grad_(False).eval()
    return encoder
