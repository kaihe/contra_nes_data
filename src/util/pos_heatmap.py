"""Render a player-position density heatmap over many traces.

Replays every trace matching a glob, collects the player's world position each
decision step, and renders a 2-D density map — the level's traversal corridor as
flown by the agents. By default only **grounded** frames are kept (jump/fall
frames are filtered out), which sharpens the walked path and makes pits show up as
gaps in the ground band.

For a horizontal level (L1) the plane is ``world_x`` (level scroll + sprite x) vs.
the player's on-screen ``y`` — there is no vertical scroll to add.

    # all Level-1 traces -> PNG (slow: replays every trace once)
    python -m util.pos_heatmap --traces "game_trace/mc_trace/**/win_level1_*.npz"

    # quick preview on 20 traces, also print an ASCII map to the terminal
    python -m util.pos_heatmap --limit 20 --ascii

    # re-plot without re-replaying (reuses collected points)
    python -m util.pos_heatmap --cache tmp/level1_pos.npz --cell 4

Replaying is the expensive part; pass ``--cache PATH`` to save the collected
points and reuse them on later runs (different bins / ascii) for free.
"""

import argparse
import glob
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*Gym.*")

from env.entity import ADDR_PLAYER_Y, is_grounded, player_x
from util.replay import make_env, rewind_state


def collect_points(files, *, grounded_only=True):
    """Replay each trace; return (xs, ys) player positions and grounded fraction."""
    xs, ys = [], []
    total = kept = 0
    for i, f in enumerate(files, 1):
        d = np.load(f, allow_pickle=True)
        actions, skip = d["actions"], int(d["skip"])
        env = make_env()
        rewind_state(env, bytes(d["initial_state"]))
        for act in actions:
            a = np.asarray(act, dtype=np.uint8)
            for _ in range(skip):
                env.step(a)
            r = env.unwrapped.get_ram()
            total += 1
            if grounded_only and not is_grounded(r):
                continue
            kept += 1
            xs.append(player_x(r))
            ys.append(int(r[ADDR_PLAYER_Y]))
        env.close()
        if i % 20 == 0 or i == len(files):
            print(f"  replayed {i}/{len(files)} traces ({kept} points)", flush=True)
    return np.asarray(xs), np.asarray(ys), (kept / total if total else 0.0)


def print_ascii(xs, ys, w=90, h=16):
    """Terminal density map (x across, screen-y down = ground at the bottom)."""
    xr = (xs.max() - xs.min()) or 1
    yr = (ys.max() - ys.min()) or 1
    gx = np.clip(((xs - xs.min()) / xr * (w - 1)).astype(int), 0, w - 1)
    gy = np.clip(((ys - ys.min()) / yr * (h - 1)).astype(int), 0, h - 1)
    grid = np.zeros((h, w))
    np.add.at(grid, (gy, gx), 1)
    ramp = " .:-=+*#%@"
    mx = grid.max() or 1
    for row in grid:
        print("".join(ramp[min(int(v / mx * (len(ramp) - 1) * 3), len(ramp) - 1)]
                       for v in row))


def candidate_gaps(xs, bin_px=64, frac=0.03):
    """World-x bins with near-zero grounded occupancy — candidate pits / airborne terrain."""
    edges = np.arange(xs.min(), xs.max() + bin_px, bin_px)
    counts, _ = np.histogram(xs, bins=edges)
    thresh = counts.max() * frac
    return [int(edges[i]) for i, c in enumerate(counts) if c < thresh]


def render_png(xs, ys, out, title, *, cell=6, equal=True):
    """Render a log-density heatmap. With ``equal``, x and y share one scale
    (1 world-x px == 1 screen-y px), so the map shows the level's true proportions
    — a very wide strip. ``cell`` is the histogram bin size in world pixels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    xr = max(int(xs.max() - xs.min()), 1)
    yr = max(int(ys.max() - ys.min()), 1)
    nx, ny = max(xr // cell, 1), max(yr // cell, 1)   # square bins

    if equal:
        h = 3.0                                       # plot height in inches
        fig_w = h * xr / yr + 3.0                     # width follows the aspect
    else:
        fig_w, h = 16, 5
    plt.figure(figsize=(fig_w, h))
    plt.hist2d(xs, ys, bins=(nx, ny), norm=LogNorm(), cmap="inferno")
    plt.colorbar(label="frames (log)", fraction=0.01, pad=0.004)
    ax = plt.gca()
    if equal:
        ax.set_aspect("equal")                        # same scale on both axes
    ax.invert_yaxis()                                 # screen y grows downward → ground at bottom
    ax.set_xlabel("world x  (level scroll + sprite x)")
    ax.set_ylabel("screen y  (up = top)")
    ax.set_title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=100, bbox_inches="tight")
    print(f"saved heatmap -> {out}  ({nx}×{ny} bins, {'equal' if equal else 'stretched'} scale)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces", default="game_trace/mc_trace/**/win_level1_*.npz",
                   help="glob of trace .npz files (recursive)")
    p.add_argument("--out", default="tmp/level1_pos_heatmap.png",
                   help="output PNG path (research artifacts live under tmp/)")
    p.add_argument("--limit", type=int, default=None, help="max traces to replay")
    p.add_argument("--cell", type=int, default=6,
                   help="histogram bin size in world pixels (square cells)")
    p.add_argument("--stretch", action="store_true",
                   help="disable equal aspect (compact wide-but-short view)")
    p.add_argument("--include-airborne", action="store_true",
                   help="keep jump/fall frames too (default: grounded only)")
    p.add_argument("--ascii", action="store_true", help="also print an ASCII map")
    p.add_argument("--cache", default=None,
                   help="npz path to save/reuse collected points (skip replay if it exists)")
    args = p.parse_args()

    if args.cache and os.path.exists(args.cache):
        c = np.load(args.cache)
        xs, ys, frac = c["xs"], c["ys"], float(c["frac"])
        print(f"loaded {len(xs)} cached points ({frac*100:.0f}% grounded) from {args.cache}")
    else:
        files = sorted(glob.glob(args.traces, recursive=True))
        if args.limit:
            files = files[:args.limit]
        if not files:
            raise SystemExit(f"no traces matched: {args.traces}")
        print(f"replaying {len(files)} traces "
              f"({'all frames' if args.include_airborne else 'grounded frames only'})…")
        xs, ys, frac = collect_points(files, grounded_only=not args.include_airborne)
        if args.cache:
            np.savez_compressed(args.cache, xs=xs, ys=ys, frac=frac)
            print(f"cached points -> {args.cache}")

    print(f"{len(xs)} points  x:[{xs.min()},{xs.max()}]  y:[{ys.min()},{ys.max()}]  "
          f"({frac*100:.0f}% of frames grounded)")
    print("candidate ground gaps (pits / must-be-airborne x):", candidate_gaps(xs))

    kind = "all frames" if args.include_airborne else "grounded only"
    render_png(xs, ys, args.out, f"Level 1 player-position density  ({kind})",
               cell=args.cell, equal=not args.stretch)
    if args.ascii:
        print()
        print_ascii(xs, ys)


if __name__ == "__main__":
    main()
