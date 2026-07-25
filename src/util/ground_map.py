"""Draw a level's ground outline straight from the game's collision map.

Thin visualiser over :func:`env.ground.build_ground`: it reconstructs the level's
terrain (floor / solid / water at 16-px cell resolution) from the engine's own
``BG_COLLISION_DATA`` and rasterises it to an equal-scale PNG under ``tmp/``. Unlike
``pos_heatmap`` (which infers the corridor statistically from where agents stood),
this is the level's *actual* terrain.

    # default: Level-1 ground -> tmp/level1_ground.png
    python -m util.ground_map

    # another level, and overlay the grounded player path as a cross-check
    python -m util.ground_map --level level1 --overlay-path
"""

import argparse
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*Gym.*")

from env.entity import ADDR_PLAYER_Y, is_grounded, player_x
from env.ground import (CELL, COLL_FLOOR, COLL_SOLID, COLL_WATER, GroundMap,
                        build_ground, find_win_trace)


def grounded_path(trace):
    """Grounded player (world_x, screen_y) points along a trace — for overlay."""
    from util.replay import make_env, rewind_state
    d = np.load(trace, allow_pickle=True)
    actions, skip = d["actions"], int(d["skip"])
    env = make_env()
    rewind_state(env, bytes(d["initial_state"]))
    xs, ys = [], []
    for act in actions:
        a = np.asarray(act, dtype=np.uint8)
        for _ in range(skip):
            env.step(a)
        r = env.unwrapped.get_ram()
        if is_grounded(r):
            xs.append(player_x(r))
            ys.append(int(r[ADDR_PLAYER_Y]))
    env.close()
    return np.asarray(xs), np.asarray(ys)


def render_png(g: GroundMap, out, title, *, path=None):
    """Rasterise a :class:`GroundMap` to an equal-scale PNG (y down, ground at bottom).
    Floor = green, solid = grey, water = blue."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # remap collision codes -> small contiguous palette indices
    grid = np.zeros_like(g.codes)
    grid[g.codes == COLL_FLOOR] = 1
    grid[g.codes == COLL_WATER] = 2
    grid[g.codes == COLL_SOLID] = 3

    ny, nx = grid.shape
    cmap = ListedColormap(["#101014", "#39d353", "#2b8cff", "#8a8a8a"])
    fig_w = max(nx / ny * 3.0 + 3.0, 6.0)
    plt.figure(figsize=(fig_w, 3.0))
    plt.imshow(grid, cmap=cmap, vmin=0, vmax=3, origin="upper", interpolation="nearest",
               extent=[g.x0, g.x_max, g.y_max, g.y0])   # world pixels; y down
    ax = plt.gca()
    ax.set_aspect("equal")
    handles = [Patch(color="#39d353", label="floor"),
               Patch(color="#8a8a8a", label="solid"),
               Patch(color="#2b8cff", label="water")]
    if path is not None and len(path[0]):
        ax.plot(path[0], path[1], ".", ms=0.6, color="#ffd000", alpha=0.25)
        handles.append(Line2D([], [], marker="o", ls="", color="#ffd000",
                              label="walked (grounded)"))
    ax.set_xlabel("world x  (level scroll + screen x)")
    ax.set_ylabel("screen y  (up = top)")
    ax.set_title(title)
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved ground map -> {out}  ({nx}×{ny} cells @ {g.cell}px, "
          f"x:[{g.x0},{g.x_max}] y:[{g.y0},{g.y_max}])")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", default="level1", help="level to map (e.g. level1)")
    p.add_argument("--out", default=None,
                   help="output PNG (default tmp/<level>_ground.png)")
    p.add_argument("--overlay-path", action="store_true",
                   help="also scatter the grounded player path for comparison")
    args = p.parse_args()

    out = args.out or f"tmp/{args.level}_ground.png"
    print(f"building ground for {args.level}…")
    g = build_ground(args.level)
    path = None
    if args.overlay_path:
        traces = find_win_trace(args.level)
        if traces:
            path = grounded_path(traces[0])
    render_png(g, out, f"{args.level} ground map  (BG_COLLISION_DATA)", path=path)


if __name__ == "__main__":
    main()
