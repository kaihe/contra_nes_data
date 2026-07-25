"""Replay a Level-1 win trace and paint the engine's collision map on each frame.

For every frame we query :func:`env.entity.get_bg_collision` over the two loaded
screens and composite a translucent cell wherever the game says floor / water /
solid — aligned to the *world* 16-px grid so the overlay scrolls locked to the
terrain sprites. This is the visual sanity check that the RAM collision decode
matches what's actually drawn.

    # default: first Level-1 win trace -> tmp/level1_ground_overlay.mp4
    python -m util.ground_video

    # a specific trace, also box the player/enemies, quick 80-step preview
    python -m util.ground_video --trace game_trace/mc_trace/level1/win_level1_XXns.npz \
        --entities --max-steps 80
"""

import argparse
import glob
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*Gym.*")

from env.entity import annotate
from env.ground import (CELL, COLL_EMPTY, COLL_FLOOR, COLL_SOLID, COLL_WATER,
                        PPU_H, PPU_W, get_bg_collision, left_edge)
from util.replay import make_env, rewind_state
# high-contrast RGB per collision code (chosen to stand out from the natural
# green-grass / brown-rock / blue-water palette so alignment is easy to verify)
_COLOR = {
    COLL_FLOOR: (255, 60, 60),    # red   — walkable floor
    COLL_WATER: (0, 220, 255),    # cyan  — water / death plane
    COLL_SOLID: (255, 230, 0),    # yellow— solid block / wall
}
_FILL_A = 70                       # cell fill alpha; border drawn opaque


def overlay_ground(frame: np.ndarray, ram: np.ndarray):
    """Return a copy of `frame` with collision cells composited on top."""
    from PIL import Image, ImageDraw

    H, W = frame.shape[:2]
    xoff, yoff = (PPU_W - W) // 2, (PPU_H - H) // 2      # overscan crop (8, 8)
    base = Image.fromarray(frame).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    left = left_edge(ram)                             # world x of screen column 0
    first_cell = left // CELL
    last_cell = (left + PPU_W - 1) // CELL
    for wc in range(first_cell, last_cell + 1):
        screen_left = wc * CELL - left                  # cell's left edge in PPU x
        sx = screen_left + CELL // 2                     # sample the cell centre
        if not (0 <= sx < PPU_W):
            continue
        for row in range(0, PPU_H, CELL):                # y has no scroll on L1
            code = get_bg_collision(ram, sx, row + CELL // 2)
            if code == COLL_EMPTY:
                continue
            fx, fy = screen_left - xoff, row - yoff       # PPU -> frame coords
            rgb = _COLOR.get(code, (255, 0, 255))
            d.rectangle([fx, fy, fx + CELL - 1, fy + CELL - 1],
                        fill=rgb + (_FILL_A,), outline=rgb + (255,), width=1)
    out = Image.alpha_composite(base, ov).convert("RGB")
    return np.asarray(out)


def make_video(trace, out, *, entities=False, max_steps=None, fps=20):
    d = np.load(trace, allow_pickle=True)
    actions, skip = d["actions"], int(d["skip"])
    if max_steps:
        actions = actions[:max_steps]
    env = make_env()
    rewind_state(env, bytes(d["initial_state"]))

    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    writer = imageio.get_writer(out, fps=fps, macro_block_size=1) \
        if out.endswith(".mp4") else None
    frames = [] if writer is None else None

    sample = None
    for t, act in enumerate(actions):
        a = np.asarray(act, dtype=np.uint8)
        for _ in range(skip):
            env.step(a)
        ram = env.unwrapped.get_ram()
        img = env.unwrapped.get_screen()
        if entities:
            img = annotate(img, ram)
        img = overlay_ground(img, ram)
        if writer is not None:
            writer.append_data(img)
        else:
            frames.append(img)
        if t == len(actions) // 2:
            sample = img
    env.close()

    if writer is not None:
        writer.close()
    else:
        imageio.mimsave(out, frames, fps=fps)
    print(f"saved {len(actions)}-frame video -> {out}")
    if sample is not None:
        png = os.path.splitext(out)[0] + "_sample.png"
        imageio.imwrite(png, sample)
        print(f"saved mid-trace sample frame -> {png}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", default=None,
                   help="trace .npz (default: first game_trace/mc_trace/**/win_level1_*.npz)")
    p.add_argument("--out", default="tmp/level1_ground_overlay.mp4",
                   help="output video path (.mp4 or .gif); artifacts live under tmp/")
    p.add_argument("--entities", action="store_true",
                   help="also draw player/enemy/bullet boxes")
    p.add_argument("--max-steps", type=int, default=None, help="cap decision steps")
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args()

    trace = args.trace
    if trace is None:
        hits = sorted(glob.glob("game_trace/mc_trace/**/win_level1_*.npz", recursive=True))
        if not hits:
            raise SystemExit("no win_level1 traces found")
        trace = hits[0]
    print(f"replaying {trace}")
    make_video(trace, args.out, entities=args.entities,
               max_steps=args.max_steps, fps=args.fps)


if __name__ == "__main__":
    main()
