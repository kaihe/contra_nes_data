"""Render ROCKET goal frames for sample tasks into a labelled montage (tmp/).

For each sampled task it calls :func:`task_maker.base.render_goal` — the reference
frame (first frame for kill/item, last frame for traverse) with a colored Gaussian
blob on the goal entity (enemy = red, item = cyan, player = green) — so we can eyeball
that the RAM-derived pointer lands on the right pixels before committing the format.

    # sample a few tasks from each dataset under game_trace/tasks -> tmp/goal_preview.png
    python -m util.goal_preview

    # specific task files
    python -m util.goal_preview --tasks game_trace/tasks/kill/*/*.npz --n 6
"""

import argparse
import glob
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*Gym.*")

from task_maker.base import load_task, render_goal

DEFAULT_GLOBS = [
    "game_trace/tasks/kill/*/*.npz",
    "game_trace/tasks/item/*/*.npz",
    "game_trace/tasks/traverse/*/*.npz",
]


def sample(globs, n, seed=0):
    rng = np.random.default_rng(seed)
    picks = []
    for g in globs:
        files = sorted(glob.glob(g))
        if files:
            idx = rng.choice(len(files), size=min(n, len(files)), replace=False)
            picks += [files[i] for i in sorted(idx)]
    return picks


def render_montage(paths, out, *, sigma=12.0, cols=4, heatmap=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(paths)
    if heatmap:                          # one row per task: overlay | raw heatmap channel
        fig, axes = plt.subplots(n, 2, figsize=(2 * 2.6, n * 2.7), squeeze=False)
        for (ax_img, ax_hm), path in zip(axes, paths):
            seg = load_task(path)
            img, hm, points = render_goal(seg, sigma=sigma)
            when, kind = seg.meta.get("goal_when", "?"), seg.meta.get("goal_kind", "?")
            ax_img.imshow(img); ax_img.axis("off")
            ax_img.set_title(f"{seg.label}  {kind}@{when} ×{len(points)}", fontsize=7)
            ax_hm.imshow(hm, cmap="inferno", vmin=0, vmax=1); ax_hm.axis("off")
            ax_hm.set_title("goal heatmap channel", fontsize=7)
    else:                                # compact grid of overlays
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.7))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        for ax, path in zip(axes, paths):
            seg = load_task(path)
            img, _hm, points = render_goal(seg, sigma=sigma)
            ax.imshow(img)
            when, kind = seg.meta.get("goal_when", "?"), seg.meta.get("goal_kind", "?")
            ax.set_title(f"{seg.label}\n{kind}@{when} ×{len(points)}", fontsize=7)
    fig.suptitle("ROCKET goal frames  (red=enemy, cyan=item, green=player, orange=boss)",
                 fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved goal-frame montage -> {out}  ({n} tasks)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=None,
                   help="task .npz glob(s); default samples kill/item/traverse")
    p.add_argument("--n", type=int, default=3, help="tasks to sample per dataset")
    p.add_argument("--sigma", type=float, default=12.0, help="Gaussian blob std (px)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--heatmap", action="store_true",
                   help="also show the raw Gaussian goal channel next to each overlay")
    p.add_argument("--out", default="tmp/goal_preview.png")
    args = p.parse_args()

    globs = args.tasks or DEFAULT_GLOBS
    paths = sample(globs, args.n, seed=args.seed)
    if not paths:
        raise SystemExit(f"no task files matched: {globs}")
    render_montage(paths, args.out, sigma=args.sigma, heatmap=args.heatmap)


if __name__ == "__main__":
    main()
