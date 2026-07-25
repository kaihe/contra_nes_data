"""Generate "kill this enemy" imitation-learning tasks by backward trajectory relabeling.

For every enemy destroyed in a source trace we trace *backward* to the frame the
enemy first appeared (its RAM slot became active) and emit a short, self-contained
episode (:class:`task_maker.base.Segment`): the emulator save-state at spawn + the
actions that lead to the kill. This is the ROCKET-1 "backward trajectory
relabeling" idea, anchored on the precise spawn frame the enemy slot arrays give
us (rather than a fixed-length backward window).

Every emitted segment passes two replay gates: it must reproduce its kill when
replayed as-is, and the enemy must *not* die when the same window is replayed
with the fire button muted (otherwise the killing bullet predates the window).

CLI
---
    python -m task_maker.kill_enemy \
        --traces "game_trace/mc_trace/level1/*.npz" \
        --out game_trace/tasks/kill --limit 20
"""

import os
from dataclasses import replace

import numpy as np

from env.constant import (
    ADDR_ENEMY_HP,
    ADDR_ENEMY_HP_COUNT,
    ADDR_ENEMY_TYPE,
    ADDR_LEVEL,
    ENEMY_TYPE_FALLING_ROCK,
    ENEMY_TYPE_SOLDIER,
    enemy_name,
)
from env.entity import ADDR_ENEMY_ROUTINE, ADDR_ENEMY_X, ADDR_ENEMY_Y, on_screen, player_x
from env.event import KillEvent
from env.utility import boss_scene
from task_maker.base import (
    Segment,
    TaskMaker,
    iter_segment,
    iter_steps,
    load_trace,
)

FIRE_BUTTON_IDX = 0  # index of the B (fire) button in the 9-button action vector

# Non-enemy / projectile / armored-unstable types that shouldn't anchor a kill
# task: weapon item, enemy bullets, unused slots, ting-reset wall plating.
_SKIP_TYPES = {0x00, 0x01, 0x09, 0x0A, 0x0B, 0x0D, 0x0F}

# 1-hit swarming enemies whose deaths are dominated by incidental fire in a
# bullet-saturated screen: with only 1 hp they die to whatever spray is already
# in the air, so a kill can't be cleanly attributed to a deliberate aimed action
# and the demonstration carries little targeting signal. Excluded from kill-task
# anchoring — this behaviour belongs to survival/progress tasks instead. (The
# indoor/jumping/group soldier variants at 0x15+ are candidates for the same
# treatment once we process those levels.)
_INCIDENTAL_TYPES = {ENEMY_TYPE_SOLDIER}


def _is_target(etype: int, level: int) -> bool:
    if etype in _SKIP_TYPES or etype in _INCIDENTAL_TYPES:
        return False
    if etype == ENEMY_TYPE_FALLING_ROCK and level == 2:   # L3 rock respawns
        return False
    return True


# ── Slot lifecycle tracker ───────────────────────────────────────────────────

class SlotTracker:
    """Track the 16 enemy slots across steps to pair each kill with its spawn.

    Liveness comes from ``ENEMY_ROUTINE`` (``!= 0``), *not* ``ENEMY_HP``: when an
    enemy walks off-screen the game clears its routine but leaves a stale ``hp``
    (and type), so an HP-based "alive" would treat the slot as continuously
    occupied and merge the departed enemy with the next one to reuse the slot.
    A **spawn** is a slot's routine turning non-zero (or being reused by a
    different type); a **kill** is an active slot whose hp crosses to 0. A slot
    going inactive (routine → 0) with hp still positive is an enemy that *left*,
    so its pending spawn is dropped — a kill is only ever paired with the spawn
    of that same enemy instance.
    """

    def __init__(self):
        n = ADDR_ENEMY_HP_COUNT
        self.active = [False] * n
        self.etype = [0] * n
        self.hp = [0] * n
        self.spawn_step: list[int | None] = [None] * n
        self.spawn_snap: list[bytes | None] = [None] * n
        self.spawn_x = [0] * n

    def update(self, cur: np.ndarray, step: int, snap: bytes):
        """Advance one step. Return kills as (slot, spawn_step, spawn_snap, spawn_x, etype)."""
        level = int(cur[ADDR_LEVEL])
        kills = []
        for s in range(ADDR_ENEMY_HP_COUNT):
            etype = int(cur[ADDR_ENEMY_TYPE + s])
            hp = int(cur[ADDR_ENEMY_HP + s])
            active = int(cur[ADDR_ENEMY_ROUTINE + s]) != 0
            was = self.active[s]

            if was and self.hp[s] > 0 and hp == 0:           # kill (hp crossed to 0)
                if self.spawn_step[s] is not None and _is_target(self.etype[s], level):
                    kills.append((s, self.spawn_step[s], self.spawn_snap[s],
                                  self.spawn_x[s], self.etype[s]))
                self.spawn_step[s] = self.spawn_snap[s] = None
            elif active and (not was or etype != self.etype[s]):  # spawn / slot reuse
                self.spawn_step[s], self.spawn_snap[s] = step, snap
                self.spawn_x[s] = player_x(cur)
            elif not active:                                 # left screen (no kill)
                self.spawn_step[s] = self.spawn_snap[s] = None

            self.active[s] = active
            self.etype[s] = etype
            self.hp[s] = hp
        return kills


# ── Extraction ───────────────────────────────────────────────────────────────

def extract_segments(trace_path: str, *, min_window: int = 2,
                     max_window: int = 240) -> list[Segment]:
    """Replay a trace once and return one Segment per cleanly-paired kill.

    Kills where the player never fires inside the window are dropped: the shot
    that killed the enemy predates its spawn (it walked into a bullet already in
    flight), so the demonstration would contain no aiming/shooting behaviour.
    """
    ctx = load_trace(trace_path)
    stem = os.path.splitext(ctx.src)[0]
    tracker = SlotTracker()
    segs: list[Segment] = []

    for step, _prev, cur, snap in iter_steps(ctx):
        if boss_scene(cur):
            break   # boss stage started — its kills belong to the boss task
        for slot, sp_step, sp_snap, sp_x, etype in tracker.update(cur, step, snap):
            if not (min_window <= step - sp_step + 1 <= max_window):
                continue
            if not on_screen(int(cur[ADDR_ENEMY_X + slot]),
                             int(cur[ADDR_ENEMY_Y + slot])):
                continue   # enemy killed off-screen → no visible target to learn from
            window_actions = np.asarray(ctx.actions[sp_step:step + 1], dtype=np.uint8)
            if not window_actions[:, FIRE_BUTTON_IDX].any():
                continue   # no shot in the window → enemy died to a stray bullet
            level = int(cur[ADDR_LEVEL])
            segs.append(Segment(
                initial_state=sp_snap,
                actions=window_actions,
                label=enemy_name(etype, level),
                level=level,
                start_step=sp_step,
                end_step=step,
                skip=ctx.skip,
                src_trace=ctx.src,
                uid=f"{stem}_slot{slot}_k{step}",
                meta={"enemy_type": etype, "slot": slot, "spawn_at_start": sp_step == 0,
                      "target_entity": enemy_name(etype, level),
                      "start_x": sp_x, "end_x": player_x(cur),
                      # ROCKET goal: mark the target enemy on the first frame
                      "goal_when": "first", "goal_kind": "target", "goal_slot": slot},
            ))
    return segs


def kill_reached(seg: Segment, pre, cur) -> bool:
    """Goal predicate: the target enemy dies on this step (its kill event fires)."""
    etype = seg.meta["enemy_type"]
    ev = KillEvent(etype, level=seg.level if etype > 0x0F else None)
    return bool(ev.trigger(pre, cur))


def kill_predates_window(seg: Segment) -> bool:
    """Replay with the fire button muted: True if the target slot still dies.

    If the enemy dies even though the player never fires inside the window, the
    killing bullet was already in flight at window start (fired before the
    enemy spawned), so the segment doesn't actually demonstrate the kill.
    The check pins the segment's own slot, so another same-type enemy dying in
    the muted replay can't trigger a false rejection.
    """
    slot = seg.meta["slot"]
    actions = seg.actions.copy()
    actions[:, FIRE_BUTTON_IDX] = 0
    muted = replace(seg, actions=actions)
    return any(int(pre[ADDR_ENEMY_HP + slot]) > 0
               and int(cur[ADDR_ENEMY_HP + slot]) == 0
               for pre, cur in iter_segment(muted))


# ── Runner ──────────────────────────────────────────────────────────────────

# ── Maker ─────────────────────────────────────────────────────────────────────

class KillEnemyMaker(TaskMaker):
    kind = "kill"
    task_noun = "kill task"

    def __init__(self, *, min_window: int = 2, max_window: int = 240,
                 per_type_cap: int | None = None):
        self.min_window = min_window
        self.max_window = max_window
        self.per_type_cap = per_type_cap

    def extract(self, trace_path: str) -> list[Segment]:
        return extract_segments(trace_path, min_window=self.min_window,
                                max_window=self.max_window)

    def goal_reached(self, seg: Segment, pre, cur) -> bool:
        return kill_reached(seg, pre, cur)

    def reject(self, seg: Segment, kept) -> str | None:
        if self.per_type_cap and kept[seg.label] >= self.per_type_cap:
            return "capped"
        if kill_predates_window(seg):        # kill by a bullet fired before the window
            return "stray-bullet"
        if not self.verify_segment(seg):     # kill doesn't reproduce from the state
            return "dropped"
        return None

    def add_arguments(self, p) -> None:
        p.add_argument("--min-window", type=int, default=self.min_window)
        p.add_argument("--max-window", type=int, default=self.max_window)
        p.add_argument("--per-type-cap", type=int, default=self.per_type_cap,
                       help="max segments to keep per enemy type (dataset balancing)")

    def configure(self, args) -> None:
        self.min_window = args.min_window
        self.max_window = args.max_window
        self.per_type_cap = args.per_type_cap


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    KillEnemyMaker().main()


if __name__ == "__main__":
    main()
