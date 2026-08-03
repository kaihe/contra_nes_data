"""Materialize tasks into WebDataset shards for a Hugging Face dataset (ROCKET-2 style).

Each task ``.npz`` is *replayable* (save-state + actions) but needs the emulator +
ROM to render — not shippable. This exporter replays each task once and bakes a
self-contained, ROM-free sample into a WebDataset tar shard:

    <uid>.obs.mkv     observation frames, one per decision step (lossless FFV1, or mp4)
    <uid>.actions.npy (T, 9) uint8 button vectors, aligned to the frames
    <uid>.goal.png    the ROCKET goal frame (reference image + colored blob)
    <uid>.json        meta: goal_when/kind, goal points, per-frame centroids +
                      visibility (the ROCKET-2 aux targets), per-class entity
                      positions (``entities``), label, split, start/end x, …

One config per task type (kill / item / traverse / boss), and **one shard per split**
within each config — ``kill-train-00000.tar`` / ``kill-val-00000.tar``. The split is
owned by the dataset (:func:`task_maker.base.split_of`, keyed on the source trace),
so a consumer points at the shard it wants instead of reconstructing a permutation.
A WebDataset shard is just a tar of these grouped members, so no loading script is
needed on the Hub — declare the configs in the dataset card's ``README.md``.

    # 50-episode proof shard for one config, measure real bytes/episode
    python -m task_maker.export_hf --config kill --limit 50 --out tmp/hf_export

    # just the held-out half of one config
    python -m task_maker.export_hf --config kill --split val --out game_trace/hf

The ``.npz`` files stay the source of truth; these shards are a materialization.
"""

import argparse
import glob
import io
import json
import os
import tarfile
import tempfile
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*Gym.*")

import imageio.v2 as imageio

from env.entity import (ADDR_ENEMY_ROUTINE, ADDR_ENEMY_X, ADDR_ENEMY_Y,
                        ADDR_PLAYER_X, ADDR_PLAYER_Y, HEATMAP_CLASSES, on_screen,
                        overlay_pointers, scan)
from env.ground import left_edge
from task_maker.base import _boss_parts, load_task, split_of
from util.replay import make_env, rewind_state

CONFIG_GLOB = {
    "kill":     "game_trace/tasks/kill/*/*.npz",
    "item":     "game_trace/tasks/item/*/*.npz",
    "traverse": "game_trace/tasks/traverse/*/*.npz",
    "boss":     "game_trace/tasks/boss/*/*.npz",
}


def _xy_list(pts) -> list[list[int]]:
    """``(2,)`` / ``(n, 2)`` array or iterable of pairs → ``[[x, y], ...]``."""
    arr = np.asarray(pts)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        return [[int(arr[0]), int(arr[1])]]
    return [[int(x), int(y)] for x, y in arr.reshape(-1, 2)]


def entities_frame(ram) -> dict[str, list[list[int]]]:
    """One frame of per-class entity positions (PPU coords), same convention as
    ``centroids``. Always includes every :data:`HEATMAP_CLASSES` key; an empty
    list means none of that class is live this frame."""
    e = scan(ram)
    return {
        "player": _xy_list(e.player),
        "player_bullets": _xy_list(e.player_bullets),
        "enemies": _xy_list(e.enemies),
        "enemy_bullets": _xy_list(e.enemy_bullets),
    }


def _goal_points(seg, ram):
    """Goal entity position(s) in the *current* view + whether any is visible —
    the per-frame ROCKET-2 centroid / visibility signal."""
    when = seg.meta.get("goal_when", "first")
    if when == "last":                              # traverse: project the world-x goal
        sx = int(seg.meta["end_x"]) - left_edge(ram)
        sy = int(seg.meta.get("end_y", ADDR_PLAYER_Y and int(ram[ADDR_PLAYER_Y])))
        return ([(sx, sy)], True) if 0 <= sx < 256 else ([], False)
    if when == "boss":
        pts = _boss_parts(ram)
        return pts, len(pts) > 0
    slot = int(seg.meta["goal_slot"])               # kill / item: the target slot
    if int(ram[ADDR_ENEMY_ROUTINE + slot]) != 0:
        x, y = int(ram[ADDR_ENEMY_X + slot]), int(ram[ADDR_ENEMY_Y + slot])
        return [(x, y)], on_screen(x, y)
    return [], False


def materialize(seg, *, sigma=12.0):
    """Replay a task once; return obs frames, actions, goal image and aux targets."""
    when = seg.meta.get("goal_when", "first")
    kind = seg.meta.get("goal_kind", "target")
    env = make_env()
    rewind_state(env, seg.initial_state)
    frames, centroids, visibility = [], [], []
    # Per-class lists of length T; each entry is [[x,y], ...] for that frame.
    entities = {c: [] for c in HEATMAP_CLASSES}
    first_idx, boss_best = None, (-1, None)
    try:
        for i, act in enumerate(seg.actions):
            a = np.asarray(act, dtype=np.uint8)
            for _ in range(seg.skip):
                env.step(a)
            ram = env.unwrapped.get_ram()
            # Screen + labels share this post-action RAM snapshot (same as
            # centroids). Do not align entities to the pre-step action index.
            frames.append(np.ascontiguousarray(env.unwrapped.get_screen()))
            pts, vis = _goal_points(seg, ram)
            centroids.append([[int(x), int(y)] for x, y in pts])
            visibility.append(bool(vis))
            for cls, pts_cls in entities_frame(ram).items():
                entities[cls].append(pts_cls)
            if when == "first" and first_idx is None and pts:
                first_idx = i
            if when == "boss" and len(pts) > boss_best[0]:
                boss_best = (len(pts), i)
    finally:
        env.close()

    if when == "last":
        gidx = len(frames) - 1
    elif when == "boss":
        gidx = boss_best[1] if boss_best[1] is not None else 0
    else:
        gidx = first_idx if first_idx is not None else 0
    gpts = [tuple(p) for p in centroids[gidx]]
    goal_img, _hm = overlay_pointers(frames[gidx], gpts, kind=kind, sigma=sigma)
    return {
        "frames": frames,
        "actions": np.asarray(seg.actions, dtype=np.uint8),
        "goal_img": goal_img,
        "goal_points": [list(p) for p in gpts],
        "goal_frame_idx": gidx,
        "centroids": centroids,
        "visibility": visibility,
        "entities": entities,
    }


def verify_entities_vs_centroids(centroids, visibility, entities, *,
                                 goal_when=None) -> tuple[int, int]:
    """Check that each visible goal centroid appears in an enemy-slot class.

    Kill / item / boss goals read an enemy-array slot. ``scan`` splits that array
    into ``enemies`` vs ``enemy_bullets`` by ``ENEMY_TYPE``, and a few item frames
    briefly hold type ``0x01`` (bullet) during slot turnover — so the check
    accepts either class. Traverse goals (``goal_when == "last"``) are projected
    world landmarks, not live entities, and are skipped.

    Returns ``(checked, mismatches)``.
    """
    if goal_when == "last":
        return 0, 0

    checked = mismatches = 0
    for j, vis in enumerate(visibility):
        if not vis:
            continue
        # Enemy-array slots only (not player / player_bullets).
        live = {tuple(p) for p in entities["enemies"][j]}
        live.update(tuple(p) for p in entities["enemy_bullets"][j])
        for pt in centroids[j]:
            checked += 1
            if tuple(pt) not in live:
                mismatches += 1
    return checked, mismatches


# ── Encoding helpers ──────────────────────────────────────────────────────────

def _encode_video(frames, codec):
    """Encode frames to bytes. ``png`` = lossless (PNG-in-MKV, pixel-exact),
    ``h264`` = lossy but small (the ROCKET-2/VPT choice)."""
    ext = "mp4" if codec == "h264" else "mkv"
    fd, tmp = tempfile.mkstemp(suffix="." + ext)
    os.close(fd)
    try:
        kw = dict(fps=20, macro_block_size=1)
        # NOTE: set the pixel format via `pixelformat` (not a second -pix_fmt in
        # output_params, which ffmpeg warns about). Lossless codecs MUST use an RGB
        # format — the default yuv420p chroma-subsamples and is not pixel-exact.
        if codec == "png":
            kw.update(codec="png", pixelformat="rgb24")               # lossless RGB
        elif codec == "ffv1":
            kw.update(codec="ffv1", pixelformat="bgr0")               # lossless RGB
        else:
            kw.update(codec="libx264", pixelformat="yuv420p", output_params=["-crf", "18"])
        w = imageio.get_writer(tmp, **kw)
        for f in frames:
            w.append_data(f)
        w.close()
        with open(tmp, "rb") as fh:
            return fh.read(), ext
    finally:
        os.unlink(tmp)


def _png_bytes(img):
    b = io.BytesIO()
    imageio.imwrite(b, img, format="png")
    return b.getvalue()


def _npy_bytes(arr):
    b = io.BytesIO()
    np.save(b, arr)
    return b.getvalue()


def _add(tar, name, data):
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    tar.addfile(ti, io.BytesIO(data))


_META_PASSTHROUGH = ("target_entity", "source_entity", "item_weapon", "kind",
                     "start_seg", "end_seg", "weapon", "rapid",
                     "boss_hp_start", "offset_frac")


def write_shard(paths, out_tar, *, codec="ffv1", sigma=12.0, verify=True):
    os.makedirs(os.path.dirname(out_tar) or ".", exist_ok=True)
    n = total_frames = 0
    ver_checked = ver_bad = 0
    with tarfile.open(out_tar, "w") as tar:
        for p in paths:
            seg = load_task(p)
            m = materialize(seg, sigma=sigma)
            uid = seg.uid
            vid, ext = _encode_video(m["frames"], codec)
            meta = {
                "uid": uid, "label": seg.label, "level": seg.level,
                "split": seg.split, "src_trace": seg.src_trace,
                "goal_when": seg.meta.get("goal_when"),
                "goal_kind": seg.meta.get("goal_kind"),
                "goal_points": m["goal_points"], "goal_frame_idx": m["goal_frame_idx"],
                "window_len": len(m["frames"]),
                "start_x": seg.meta.get("start_x"), "end_x": seg.meta.get("end_x"),
                "centroids": m["centroids"], "visibility": m["visibility"],
                "entities": m["entities"],
                **{k: seg.meta[k] for k in _META_PASSTHROUGH if k in seg.meta},
            }
            if verify:
                c, bad = verify_entities_vs_centroids(
                    m["centroids"], m["visibility"], m["entities"],
                    goal_when=seg.meta.get("goal_when"),
                )
                ver_checked += c
                ver_bad += bad
            _add(tar, f"{uid}.obs.{ext}", vid)
            _add(tar, f"{uid}.actions.npy", _npy_bytes(m["actions"]))
            _add(tar, f"{uid}.goal.png", _png_bytes(m["goal_img"]))
            _add(tar, f"{uid}.json", json.dumps(meta).encode())
            n += 1
            total_frames += len(m["frames"])
            if n % 10 == 0:
                print(f"  {n}/{len(paths)} episodes…", flush=True)
    size = os.path.getsize(out_tar)
    print(f"\nshard {out_tar}")
    print(f"  {n} episodes, {total_frames} frames, {size/1e6:.1f} MB  "
          f"({size/n/1e3:.1f} KB/episode, {size/total_frames:.0f} B/frame)")
    if verify:
        print(f"  entities vs centroids: checked={ver_checked} mismatches={ver_bad}")
        if ver_bad:
            raise RuntimeError(
                f"entities/centroids disagree on {ver_bad}/{ver_checked} visible "
                f"goal points — export aborted"
            )
    return size, n, total_frames


def read_split(path: str) -> str:
    """The train/val split of a task ``.npz``, read without decompressing its arrays."""
    d = np.load(path, allow_pickle=True)
    if "split" in d.files:
        return str(d["split"])
    return split_of(str(d["src_trace"]))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", choices=list(CONFIG_GLOB), default="kill",
                   help="which task type to export")
    p.add_argument("--split", choices=["train", "val", "all"], default="all",
                   help="which split(s) to export; 'all' writes one shard per split")
    p.add_argument("--limit", type=int, default=None,
                   help="max episodes per shard (proof shards)")
    p.add_argument("--codec", choices=["png", "ffv1", "h264"], default="png",
                   help="obs video codec: png (lossless, default), ffv1 (lossless), "
                        "h264 (lossy, smallest — the ROCKET-2/VPT choice)")
    p.add_argument("--sigma", type=float, default=12.0, help="goal Gaussian std (px)")
    p.add_argument("--out", default="tmp/hf_export", help="output dir for shards")
    args = p.parse_args()

    paths = sorted(glob.glob(CONFIG_GLOB[args.config]))
    if not paths:
        raise SystemExit(f"no tasks for config {args.config!r}")

    # one shard per split: the split lives in the .npz, so grouping is a cheap
    # metadata read — no permutation to reconstruct downstream
    wanted = ["train", "val"] if args.split == "all" else [args.split]
    groups = {s: [] for s in wanted}
    for path in paths:
        s = read_split(path)
        if s in groups:
            groups[s].append(path)

    for split, group in groups.items():
        if args.limit:
            group = group[:args.limit]
        if not group:
            print(f"\n!! no {args.config} tasks in split {split!r} — no shard written")
            continue
        out_tar = os.path.join(args.out, f"{args.config}-{split}-00000.tar")
        print(f"\nexporting {len(group)} {args.config}/{split} episodes → {out_tar} "
              f"(codec={args.codec})")
        write_shard(group, out_tar, codec=args.codec, sigma=args.sigma)


if __name__ == "__main__":
    main()
