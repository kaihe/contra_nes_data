"""Compare boss frames with reconstructions from the legacy 1024-D autoencoder.

This does not evaluate the current datahouse encoder: its published 512-D checkpoint
has reconstruction disabled and therefore has no pixel decoder.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from datahouse.full_level import _download_archive, _extract_selected
from util.preview_player_bullet_heatmap import WEAPONS, replay_boss_samples, select_traces


def _legacy_classes(models_path: str):
    spec = importlib.util.spec_from_file_location("legacy_dreamer_models", models_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load legacy model definitions: {models_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ConvEncoder, module.ConvDecoder


def load_legacy_ae(checkpoint: str, models_path: str, device: str):
    ConvEncoder, ConvDecoder = _legacy_classes(models_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    encoder = ConvEncoder(config["size"], depth=config["depth"],
                          embed_dim=config["embed_dim"]).to(device).eval()
    decoder = ConvDecoder(config["size"], depth=config["depth"],
                          feat_dim=config["embed_dim"]).to(device).eval()
    encoder.load_state_dict(payload["encoder"])
    decoder.load_state_dict(payload["decoder"])
    return encoder, decoder, config


def reconstruct(encoder, decoder, frames: np.ndarray, *, size: int, device: str,
                chunk: int = 128) -> tuple[np.ndarray, float]:
    resized = np.asarray([cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
                          for frame in frames], dtype=np.uint8)
    outputs, squared_error = [], 0.0
    with torch.inference_mode():
        for start in range(0, len(resized), chunk):
            image = (torch.from_numpy(resized[start:start + chunk]).to(device)
                     .permute(0, 3, 1, 2).float().div(255))
            decoded = decoder(encoder(image)).clamp(0, 1)
            squared_error += float(F.mse_loss(decoded, image, reduction="sum"))
            outputs.append(decoded.permute(0, 2, 3, 1).cpu().numpy())
    reconstruction = np.concatenate(outputs)
    mse = squared_error / reconstruction.size
    return reconstruction, 10 * math.log10(1 / max(mse, 1e-10))


def save_sheet(weapon: str, fingerprint: str, steps: list[int], frames: np.ndarray,
               reconstruction: np.ndarray, psnr: float, destination: Path,
               *, columns: int = 4) -> None:
    rows = math.ceil(len(frames) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 2.5 * rows),
                                squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, step, frame, decoded in zip(axes.flat, steps, frames, reconstruction):
        size = decoded.shape[0]
        source = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA) / 255
        separator = np.ones((size, 4, 3), dtype=np.float32)
        axis.imshow(np.concatenate([source, separator, decoded], axis=1))
        axis.set_title(f"action {step}\ninput ←  |  → decoded", fontsize=8)
        axis.axis("off")
    figure.suptitle(f"{weapon} boss · legacy reconstruction AE · PSNR {psnr:.2f} dB · "
                    f"trace {fingerprint[:12]}", fontsize=12)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def render(snapshot_path: str, checkpoint: str, models_path: str, output_dir: str,
           *, every: int = 10, device: str = "cuda", client=None) -> list[Path]:
    if client is None:
        from google.cloud import storage
        client = storage.Client()
    snapshot = json.loads(Path(snapshot_path).read_text())
    selected = select_traces(snapshot)
    encoder, decoder, config = load_legacy_ae(checkpoint, models_path, device)
    outputs = []
    for weapon in WEAPONS:
        row = selected[weapon]
        batch = snapshot["batches"][row["batch_index"]]
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            archive = temporary / "traces.tar.zst"
            _download_archive(client, batch, archive)
            source = _extract_selected(archive, [row], temporary)[0]
            steps, frames, _ = replay_boss_samples(source, every=every)
            decoded, psnr = reconstruct(encoder, decoder, frames,
                                        size=int(config["size"]), device=device)
        destination = Path(output_dir) / f"{weapon.lower()}-legacy-reconstruction.png"
        save_sheet(weapon, row["fingerprint"], steps, frames, decoded, psnr, destination)
        outputs.append(destination)
        print(f"{weapon}: {len(frames)} samples, PSNR {psnr:.2f} dB -> {destination}",
              flush=True)
    return outputs


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--models-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    render(args.snapshot, args.checkpoint, args.models_path, args.output_dir,
           every=args.every, device=args.device)


if __name__ == "__main__":
    main()
