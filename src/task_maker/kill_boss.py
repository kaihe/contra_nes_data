"""Extract per-level boss-fight tasks (``boss_level<N>``) by backward relabeling.

Anchored on the boss-stage event: for each trace we find where the boss fight
*begins* (the rising edge into the boss scene) and where the level is *cleared*
(level byte increments, or the post-boss transition routine starts), and emit one
self-contained episode (:class:`task_maker.base.Segment`) covering the whole
fight — the emulator save-state at the boss reveal + the actions that clear the
level.

CLI
---
    python -m task_maker.kill_boss \
        --traces "game_trace/mc_trace/level1/*.npz" \
        --out game_trace/tasks/boss --limit 20 --verify
"""

import os

import numpy as np

from env.constant import ADDR_LEVEL, ADDR_LEVEL_ROUTINE
from env.entity import player_x
from env.utility import boss_scene
from task_maker.base import (
    Segment,
    TaskMaker,
    iter_steps,
    load_trace,
)

_TRANSITION_ROUTINES = (0x08, 0x09)   # post-boss end-of-level sequence


def _routine(ram) -> int:
    return int(ram[ADDR_LEVEL_ROUTINE])


def _level(ram) -> int:
    return int(ram[ADDR_LEVEL])


def _boss_started(pre, cur) -> bool:
    """Rising edge into the boss scene (same signal as the in_boss_stage event)."""
    return boss_scene(cur) and not boss_scene(pre)


def _level_cleared(pre, cur) -> bool:
    """Level byte incremented, or the post-boss transition routine just started.

    Traces frequently stop at the transition before the level byte increments,
    so both are accepted as "boss beaten / level cleared".
    """
    if _level(cur) > _level(pre):
        return True
    return _routine(pre) not in _TRANSITION_ROUTINES and \
        _routine(cur) in _TRANSITION_ROUTINES


# ── Extraction ───────────────────────────────────────────────────────────────

def extract_boss(trace_path: str) -> Segment | None:
    """Replay a trace and return its boss fight, or None if it has none.

    Takes the first boss-stage rising edge and the first level-clear after it.
    """
    ctx = load_trace(trace_path)
    stem = os.path.splitext(ctx.src)[0]

    start_step = start_snap = start_level = start_x = None
    seg = None

    for step, prev, cur, snap in iter_steps(ctx):
        if start_step is None:
            if _boss_started(prev, cur):
                start_step, start_snap, start_level = step, snap, _level(prev)
                start_x = player_x(prev)
        elif seg is None and _level_cleared(prev, cur):
            seg = Segment(
                initial_state=start_snap,
                actions=np.asarray(ctx.actions[start_step:step + 1], dtype=np.uint8),
                label=f"boss_level{start_level + 1}",
                level=start_level,
                start_step=start_step,
                end_step=step,
                skip=ctx.skip,
                src_trace=ctx.src,
                uid=stem,
                # ROCKET goal: mark every boss component on the reveal frame
                meta={"start_x": start_x, "end_x": player_x(cur),
                      "goal_when": "boss", "goal_kind": "boss"},
            )
            # keep iterating so iter_steps runs to completion and closes the emu
    return seg


# ── Maker ─────────────────────────────────────────────────────────────────────

class KillBossMaker(TaskMaker):
    kind = "boss"
    task_noun = "boss task"

    def __init__(self, *, verify: bool = False):
        self.verify = verify

    def extract(self, trace_path: str) -> list[Segment]:
        seg = extract_boss(trace_path)
        return [seg] if seg is not None else []

    def goal_reached(self, seg: Segment, pre, cur) -> bool:
        """Goal predicate: the level clears (boss beaten) on this step."""
        return _level_cleared(pre, cur)

    def reject(self, seg: Segment, kept) -> str | None:
        if self.verify and not self.verify_segment(seg):
            return "no-clear"
        return None

    def add_arguments(self, p) -> None:
        p.add_argument("--verify", action="store_true",
                       help="replay each segment and drop any that doesn't clear the level")

    def configure(self, args) -> None:
        self.verify = args.verify


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    KillBossMaker().main()


if __name__ == "__main__":
    main()
