"""Render predicted player-bullet heatmaps during one boss fight per weapon."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from datahouse.encoder import EncoderSpec, load_entity_encoder
from datahouse.encoder_baseline import ENTITY_SIGMA_CELLS
from datahouse.full_level import _download_archive, _extract_selected
from env.entity import entity_heatmaps
from env.utility import boss_scene
from util.replay import make_env, rewind_state, step_env


WEAPONS = ("Regular", "Spread", "Laser", "Flamethrower")
PLAYER_BULLET_CHANNEL = 1


def select_traces(snapshot: dict) -> dict[str, dict]:
    """Choose the smallest selected fingerprint for every recorded weapon."""
    chosen = {}
    for row in snapshot["selected"]:
        weapon = row.get("boss_weapon")
        if weapon in WEAPONS and weapon not in chosen:
            chosen[weapon] = row
    missing = set(WEAPONS) - set(chosen)
    if missing:
        raise ValueError(f"collection has no selected trace for: {sorted(missing)}")
    return chosen


def replay_boss_samples(source: Path, *, every: int = 10
                        ) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Replay and capture the boss boundary plus every N subsequent decisions."""
    with np.load(source, allow_pickle=False) as trace:
        actions = np.asarray(trace["actions"], dtype=np.uint8)
        state = bytes(np.asarray(trace["initial_state"], dtype=np.uint8))
        skip = int(trace["skip"]) if "skip" in trace else 4
    env = make_env()
    rewind_state(env, state)
    steps, frames, targets = [], [], []
    boundary = None
    try:
        if boss_scene(env.unwrapped.get_ram()):
            boundary = 0
            steps.append(0)
            frames.append(env.em.get_screen().copy())
            targets.append(entity_heatmaps(env.unwrapped.get_ram(), grid=32,
                                           sigma=ENTITY_SIGMA_CELLS)[PLAYER_BULLET_CHANNEL])
        for observation_index, action in enumerate(actions, 1):
            step_env(env, action, skip)
            if boundary is None and boss_scene(env.unwrapped.get_ram()):
                boundary = observation_index
            if boundary is not None and (observation_index - boundary) % every == 0:
                steps.append(observation_index)
                frames.append(env.em.get_screen().copy())
                targets.append(entity_heatmaps(
                    env.unwrapped.get_ram(), grid=32,
                    sigma=ENTITY_SIGMA_CELLS)[PLAYER_BULLET_CHANNEL])
    finally:
        env.close()
    if boundary is None:
        raise ValueError(f"trace never entered the boss scene: {source}")
    return (steps, np.asarray(frames, dtype=np.uint8),
            np.asarray(targets, dtype=np.float32))


def predict(model, frames: np.ndarray, *, image_size: int, device: str,
            chunk: int = 256) -> np.ndarray:
    resized = np.asarray([cv2.resize(frame, (image_size, image_size),
                                     interpolation=cv2.INTER_AREA)
                          for frame in frames], dtype=np.uint8)
    output = []
    with torch.inference_mode():
        for start in range(0, len(resized), chunk):
            image = torch.from_numpy(resized[start:start + chunk]).to(device)
            logits = model.entity_logits(image)[:, PLAYER_BULLET_CHANNEL]
            output.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(output)


def heat_only(heatmap: np.ndarray, color: tuple[float, float, float], *,
              size: tuple[int, int] = (240, 224)) -> np.ndarray:
    """Render an absolute-probability map on black with a visibility gamma boost."""
    probability = cv2.resize(heatmap, size, interpolation=cv2.INTER_CUBIC)
    intensity = np.sqrt(np.clip(probability, 0, 1))[..., None]
    return intensity * np.asarray(color, dtype=np.float32)


def save_sheet(weapon: str, fingerprint: str, steps: list[int], frames: np.ndarray,
               heatmaps: np.ndarray, targets: np.ndarray, destination: Path, *,
               columns: int = 4) -> None:
    rows = math.ceil(len(frames) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 2.5 * rows),
                                squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, step, frame, heatmap, target in zip(
            axes.flat, steps, frames, heatmaps, targets):
        prediction = heat_only(heatmap, (1.0, 0.05, 0.75))
        truth = heat_only(target, (0.0, 1.0, 1.0))
        separator = np.ones((prediction.shape[0], 4, 3), dtype=np.float32)
        comparison = np.concatenate([prediction, separator, truth], axis=1)
        axis.imshow(comparison)
        peak = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        axis.set_title(f"action {step} · pred max {heatmap.max():.2f} · "
                       f"GT max {target.max():.0f}\n"
                       f"magenta prediction ←  |  → cyan ground truth · "
                       f"peak ({peak[1]},{peak[0]})",
                       fontsize=8)
        axis.axis("off")
    figure.suptitle(f"{weapon} boss · predicted player-bullet heatmap · "
                    f"trace {fingerprint[:12]}", fontsize=12)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def render(snapshot_path: str, checkpoint: str, output_dir: str, *, every: int = 10,
           device: str = "cuda", client=None) -> list[Path]:
    if client is None:
        from google.cloud import storage
        client = storage.Client()
    snapshot = json.loads(Path(snapshot_path).read_text())
    selected = select_traces(snapshot)
    spec = EncoderSpec.from_checkpoint(checkpoint)
    model = load_entity_encoder(checkpoint).to(device).eval()
    outputs = []
    for weapon in WEAPONS:
        row = selected[weapon]
        batch = snapshot["batches"][row["batch_index"]]
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            archive = temporary / "traces.tar.zst"
            _download_archive(client, batch, archive)
            source = _extract_selected(archive, [row], temporary)[0]
            steps, frames, targets = replay_boss_samples(source, every=every)
            heatmaps = predict(model, frames, image_size=spec.image_size, device=device)
        destination = Path(output_dir) / f"{weapon.lower()}-player-bullets.png"
        save_sheet(weapon, row["fingerprint"], steps, frames, heatmaps, targets,
                   destination)
        outputs.append(destination)
        print(f"{weapon}: {len(frames)} samples -> {destination}", flush=True)
    return outputs


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    render(args.snapshot, args.checkpoint, args.output_dir,
           every=args.every, device=args.device)


if __name__ == "__main__":
    main()
