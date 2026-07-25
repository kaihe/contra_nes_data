"""Generate traversal (get-from-A-to-B) imitation-learning tasks.

A traversal task starts and ends with the player standing on **solid ground**
(``terrain_state == "ground"`` — water-wading and airborne frames are excluded as
anchors, though they may occur *within* a window). For each trace we build one
forward chain:

1. **start** at a random early on-ground step (action index in ``[0, start_max]``),
2. **hop** to the next on-ground step more than ``min_gap`` action-indices later,
3. repeat from each new end until the **boss** scene begins (we stop before it),
4. each **consecutive anchor pair** defines one task: start on ground A, drive to
   ground B (crossing whatever water / gaps lie between).

Tasks are deterministic slices replayed from a captured save-state, so no separate
verification pass is needed.

CLI
---
    python -m task_maker.traverse \
        --traces "game_trace/mc_trace/level1/win_level1_*.npz" \
        --out game_trace/tasks/traverse --min-gap 100

    # inspect one trace: print its anchor chain and render an overlay to tmp/
    python -m task_maker.traverse --inspect game_trace/mc_trace/level1/win_level1_XXns.npz
"""

import os
import re
from collections import namedtuple

import numpy as np

from env.entity import ADDR_PLAYER_Y, player_x, terrain_state

from .base import Segment, TaskMaker, iter_steps, load_trace
from env.utility import boss_scene

START_MAX = 100           # random start is an action index in [0, START_MAX]
MIN_GAP = 100             # minimum action-index gap between consecutive anchors
_LEVEL_RE = re.compile(r"(level\d+)")
_TRACE_ID_RE = re.compile(r"level\d+_(\d+)")   # the unique timestamp in the trace name

# one on-ground observation along a trace
GStep = namedtuple("GStep", "step snap x y")


def _level_of(stem: str) -> str:
    m = _LEVEL_RE.search(stem)
    return m.group(1) if m else "level1"


def on_ground_stream(ctx) -> list[GStep]:
    """Every step the player is on solid ground (not water, not airborne) before the
    boss scene, in order. Detection uses the pre-step RAM (== the captured save-state
    ``snap``), so a step's ``snap`` reproduces exactly the state at that footing."""
    stream: list[GStep] = []
    for step, prev, _cur, snap in iter_steps(ctx):
        if boss_scene(prev):                      # stop before the boss stage
            break
        if terrain_state(prev) == "ground":
            stream.append(GStep(step, snap, player_x(prev), int(prev[ADDR_PLAYER_Y])))
    return stream


def anchor_chain(stream: list[GStep], start_step: int, min_gap: int) -> list[GStep]:
    """Greedy forward chain: from the first on-ground step at/after ``start_step``,
    repeatedly jump to the next on-ground step more than ``min_gap`` actions later."""
    i0 = next((i for i, s in enumerate(stream) if s.step >= start_step), None)
    if i0 is None:
        return []
    chain = [stream[i0]]
    i = i0
    while True:
        cur = chain[-1]
        nxt = next((j for j in range(i + 1, len(stream))
                    if stream[j].step - cur.step > min_gap), None)
        if nxt is None:
            break
        chain.append(stream[nxt])
        i = nxt
    return chain


def extract_segments(trace_path: str, *, min_gap: int = MIN_GAP,
                     start_max: int = START_MAX,
                     rng: np.random.Generator | None = None) -> list[Segment]:
    """Replay a trace once and return the traversal tasks along one random chain."""
    ctx = load_trace(trace_path)
    stem = os.path.splitext(ctx.src)[0]
    level_str = _level_of(stem)
    level = int(level_str[len("level"):]) - 1
    m = _TRACE_ID_RE.search(stem)
    trace_id = m.group(1) if m else stem     # unique per-trace timestamp

    stream = on_ground_stream(ctx)
    if not stream:
        return []

    rng = rng or np.random.default_rng()
    start_step = int(rng.integers(0, start_max + 1))
    chain = anchor_chain(stream, start_step, min_gap)

    # bucket by level: game_trace/tasks/traverse/<level>/<trace_id>_<startx>_<endx>.npz
    segs: list[Segment] = []
    for a, b in zip(chain, chain[1:]):
        segs.append(Segment(
            initial_state=a.snap,
            actions=np.asarray(ctx.actions[a.step:b.step + 1], dtype=np.uint8),
            label=level_str,
            level=level,
            start_step=a.step,
            end_step=b.step,
            skip=ctx.skip,
            src_trace=ctx.src,
            uid=f"{trace_id}_{a.x}_{b.x}",
            meta={"start_x": a.x, "end_x": b.x, "dx": b.x - a.x,
                  "start_y": a.y, "end_y": b.y,
                  # ROCKET goal: mark the player's ending position on the last frame
                  "goal_when": "last", "goal_kind": "player"},
        ))
    return segs


# ── Maker ─────────────────────────────────────────────────────────────────────

class TraverseMaker(TaskMaker):
    kind = "traverse"
    task_noun = "traversal task"
    default_traces = "game_trace/mc_trace/level1/win_level1_*.npz"

    def __init__(self, *, min_gap: int = MIN_GAP, start_max: int = START_MAX,
                 seed: int = 0):
        self.min_gap = min_gap
        self.start_max = start_max
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def extract(self, trace_path: str) -> list[Segment]:
        return extract_segments(trace_path, min_gap=self.min_gap,
                                start_max=self.start_max, rng=self.rng)

    def goal_reached(self, seg: Segment, pre, cur) -> bool:
        """Goal predicate (state, not event): the player has reached the target
        world-x and is back on solid ground."""
        return player_x(cur) >= seg.meta["end_x"] and terrain_state(cur) == "ground"

    def add_arguments(self, p) -> None:
        p.add_argument("--min-gap", type=int, default=self.min_gap,
                       help="minimum action-index gap between consecutive anchors")
        p.add_argument("--start-max", type=int, default=self.start_max,
                       help="random start is an action index in [0, START_MAX]")
        p.add_argument("--seed", type=int, default=self.seed,
                       help="RNG seed for the random starts")
        p.add_argument("--inspect", metavar="TRACE", default=None,
                       help="print the anchor chain for one trace and render an overlay")

    def configure(self, args) -> None:
        self.min_gap, self.start_max, self.seed = args.min_gap, args.start_max, args.seed
        self.rng = np.random.default_rng(self.seed)

    def before_run(self, args) -> bool:
        if args.inspect:
            inspect(args.inspect, min_gap=args.min_gap, start_max=args.start_max,
                    seed=args.seed)
            return True
        return False


# ── Inspection (chain + overlay, no dataset written) ──────────────────────────

def inspect(trace_path: str, *, min_gap: int = MIN_GAP, start_max: int = START_MAX,
            seed: int = 0, out: str | None = None) -> None:
    """Print a trace's anchor chain and render an overlay PNG to tmp/."""
    ctx = load_trace(trace_path)
    stem = os.path.splitext(ctx.src)[0]
    stream = on_ground_stream(ctx)
    start_step = int(np.random.default_rng(seed).integers(0, start_max + 1))
    chain = anchor_chain(stream, start_step, min_gap)

    print(f"{_level_of(stem)}: {len(stream)} on-ground steps; start≈action {start_step}; "
          f"{len(chain)} anchors → {max(len(chain) - 1, 0)} traversal tasks "
          f"(min_gap={min_gap})")
    for k, a in enumerate(chain):
        tag = "start" if k == 0 else f"hop{k}"
        print(f"   {tag:5s} step {a.step:4d}  x={a.x:5d}")

    out = out or f"tmp/{stem}_chain.png"
    _render_overlay(_level_of(stem), stream, chain, out, title=f"{stem}: traversal chain")


def _render_overlay(level_str, stream, chain, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    from env.ground import (COLL_FLOOR, COLL_SOLID, COLL_WATER, get_ground)
    g = get_ground(level_str)
    grid = np.zeros_like(g.codes)
    grid[g.codes == COLL_FLOOR] = 1
    grid[g.codes == COLL_WATER] = 2
    grid[g.codes == COLL_SOLID] = 3
    cmap = ListedColormap(["#101014", "#2b3b2f", "#17324d", "#333333"])

    ny, nx = grid.shape
    plt.figure(figsize=(max(nx / ny * 3.0 + 3.0, 6.0), 3.0))
    plt.imshow(grid, cmap=cmap, vmin=0, vmax=3, origin="upper",
               interpolation="nearest", extent=[g.x0, g.x_max, g.y_max, g.y0])
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.plot([s.x for s in stream], [s.y for s in stream], ".", ms=1.2,
            color="#39d353", alpha=0.35)                  # on-ground footprint
    ax.plot([a.x for a in chain], [a.y for a in chain], "-o", color="#ff3b3b",
            ms=5, lw=1.2, zorder=5, label="anchor chain")
    for k, a in enumerate(chain):
        ax.annotate("S" if k == 0 else str(k), (a.x, a.y), color="#ffffff",
                    fontsize=6, ha="center", va="bottom")
    ax.set_xlabel("world x")
    ax.set_ylabel("screen y")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved traversal-chain overlay -> {out}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    TraverseMaker().main()


if __name__ == "__main__":
    main()
