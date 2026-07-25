"""Generate dropped-item imitation-learning tasks by backward trajectory relabeling.

Every powerup in Contra is the *same* dropped item (enemy type 0x00, created when a
flying capsule / pill-box sensor / red soldier is destroyed); ``ENEMY_ATTRIBUTES &
0x07`` says which one it grants (0=R rapid, 1=M, 2=F, 3=S, 4=L guns, 5=B barrier,
6=Falcon). We watch each item's lifetime from creation to removal and emit:

* **pick_<item>** — the player collected it. The window runs from the *item's birth*
  (the moment the powerup spawns on screen, when the slot's type flips to 0x00) to
  the pickup, so the first frame already shows the item to take. Emitted for every
  item type. Task: *go take this powerup.* Whether a removal is a pickup is read
  from that item's own effect: the matching formal event (``WeaponChangeEvent`` for
  a gun, ``PowerUpEvent`` for rapid/barrier/falcon) firing on that step. The carrier
  that dropped it (flying capsule / sensor / soldier) is kept in ``source_entity``.

* **avoid_<gun>** — a *gun* item left the screen uncollected while the player kept a
  different gun. Emitted only for guns: skipping a barrier/falcon/rapid is never a
  real trade-off (they are pure benefits), so only weapon choice is a task. Because
  collecting a gun you don't hold must flip the weapon nibble, "item gone + pickup
  event didn't fire + you don't now hold that gun" is an unambiguous avoid.

CLI
---
    python -m task_maker.pick_item \
        --traces "game_trace/mc_trace/level1/*.npz" \
        --out game_trace/tasks/item --limit 20
"""

import os

import numpy as np

from env.constant import (
    ADDR_ENEMY_ATTR,
    ADDR_ENEMY_HP_COUNT,
    ADDR_ENEMY_TYPE,
    ADDR_LEVEL,
    ADDR_WEAPON,
    ITEM_GUN_ATTRS,
    ITEM_WEAPON_NAMES,
    enemy_name,
)
from env.entity import ADDR_ENEMY_ROUTINE, ADDR_ENEMY_X, ADDR_ENEMY_Y, on_screen, player_x
from env.event import WeaponChangeEvent, WeaponItemGoneEvent, make_powerup_events
from env.utility import boss_scene
from task_maker.base import (
    Segment,
    TaskMaker,
    iter_segment,
    iter_steps,
    load_trace,
)

ITEM_TYPE = 0x00   # ENEMY_TYPE of a dropped item

# Formal pickup event per item attr: guns fire a WeaponChangeEvent, the augments a
# PowerUpEvent. Reusing the events the rest of the pipeline detects keeps
# extraction and verification consistent.
_POWERUP_BY_TAG = {e.tag: e for e in make_powerup_events()}
_AUGMENT_PICKUP = {0: "pick_rapid_fire", 5: "pick_barrier", 6: "pick_falcon"}


def _item_name(attr: int) -> str:
    return ITEM_WEAPON_NAMES[attr].lower()


def _pickup_event(attr: int):
    """The event that fires the step this item is collected."""
    if attr in ITEM_GUN_ATTRS:
        return WeaponChangeEvent(curr_weapon=attr)
    return _POWERUP_BY_TAG[_AUGMENT_PICKUP[attr]]


# ── Item lifecycle tracker ────────────────────────────────────────────────────

class ItemTracker:
    """Track the 16 enemy slots and pair each item removal to its whole slot life.

    A dropped item is a later phase of the *same* slot the carrier (flying capsule /
    pill-box sensor / red soldier) occupied: when the carrier is destroyed the slot
    flips ``ENEMY_TYPE`` to 0x00 while ``ENEMY_ROUTINE`` stays non-zero. So we track
    two anchors per slot:

    * **origin** — when the slot activated (routine 0 → ≠0), i.e. the carrier's
      spawn. Pick tasks start here so the demo includes shooting the carrier.
    * **birth**  — when the slot became an item (type → 0x00). Avoid tasks start
      here (the item already exists; there is nothing to shoot).

    On removal (routine → 0, or the slot reused by a new enemy) we hand back both,
    plus the granted weapon and whether the item was on-screen when it appeared.
    """

    def __init__(self):
        n = ADDR_ENEMY_HP_COUNT
        self.active = [False] * n
        self.was_item = [False] * n
        self.attr = [0] * n
        self.carrier = [0] * n          # last non-item type in the slot (the dropping enemy)
        self.origin_step: list[int | None] = [None] * n
        self.origin_snap: list[bytes | None] = [None] * n
        self.origin_x = [0] * n
        self.birth_step: list[int | None] = [None] * n
        self.birth_snap: list[bytes | None] = [None] * n
        self.birth_x = [0] * n
        self.birth_onscreen = [False] * n

    def update(self, cur: np.ndarray, step: int, snap: bytes):
        """Advance one step. Return removals as ``(slot, origin_step, origin_snap,
        origin_x, birth_step, birth_snap, birth_x, attr, carrier, onscreen)``."""
        removed = []
        for s in range(ADDR_ENEMY_HP_COUNT):
            routine = int(cur[ADDR_ENEMY_ROUTINE + s])
            typ = int(cur[ADDR_ENEMY_TYPE + s])
            active = routine != 0
            is_item = active and typ == ITEM_TYPE

            if self.was_item[s] and not is_item:             # item removed (taken/gone/reused)
                if self.birth_step[s] is not None:
                    removed.append((s, self.origin_step[s], self.origin_snap[s], self.origin_x[s],
                                    self.birth_step[s], self.birth_snap[s], self.birth_x[s],
                                    self.attr[s], self.carrier[s], self.birth_onscreen[s]))
                self.origin_step[s] = self.birth_step[s] = None
                self.carrier[s] = 0

            if active and self.origin_step[s] is None:       # slot (re)activated → carrier spawn
                self.origin_step[s], self.origin_snap[s] = step, snap
                self.origin_x[s] = player_x(cur)
            elif not active:
                self.origin_step[s] = None
                self.carrier[s] = 0

            if active and not is_item:                       # carrier phase — remember its type
                self.carrier[s] = typ

            if is_item and not self.was_item[s]:             # slot became an item
                self.birth_step[s], self.birth_snap[s] = step, snap
                self.birth_x[s] = player_x(cur)
                self.attr[s] = int(cur[ADDR_ENEMY_ATTR + s]) & 0x07
                self.birth_onscreen[s] = on_screen(int(cur[ADDR_ENEMY_X + s]),
                                                   int(cur[ADDR_ENEMY_Y + s]))

            self.active[s] = active
            self.was_item[s] = is_item
        return removed


# ── Extraction ───────────────────────────────────────────────────────────────

def extract_segments(trace_path: str, *, min_window: int = 2, max_window: int = 240,
                     require_armed: bool = True) -> list[Segment]:
    """Replay a trace once and return one Segment per item pickup / gun-item avoid.

    ``require_armed`` keeps avoid tasks meaningful by requiring the player to
    already hold a real gun (nothing to protect otherwise).
    """
    ctx = load_trace(trace_path)
    stem = os.path.splitext(ctx.src)[0]
    tracker = ItemTracker()
    segs: list[Segment] = []

    for step, prev, cur, snap in iter_steps(ctx):
        if boss_scene(cur):
            break
        cur_gun = int(cur[ADDR_WEAPON]) & 0x0F
        prev_gun = int(prev[ADDR_WEAPON]) & 0x0F
        level = int(cur[ADDR_LEVEL])

        for (slot, o_step, o_snap, o_x, b_step, b_snap, b_x,
             attr, carrier, onscr) in tracker.update(cur, step, snap):
            if not onscr:                       # never visible → no real choice
                continue

            name = _item_name(attr)
            # Both pick and avoid anchor at the item's birth (b_step) — the moment the
            # powerup item spawns and is on screen — so the first frame already shows
            # the item (it can be pointed at) and the demo is "take / skip this item",
            # not "shoot the carrier that drops it". o_step (carrier spawn) is kept in
            # meta as the source, but is no longer the window start.
            if _pickup_event(attr).trigger(prev, cur):               # collected
                label, kind, kept = f"pick_{name}", "pick", prev_gun
                start, snap0, start_x = b_step, b_snap, b_x
            elif attr in ITEM_GUN_ATTRS and attr != cur_gun and \
                    (cur_gun in ITEM_GUN_ATTRS or not require_armed):  # gun left uncollected
                label, kind, kept = f"avoid_{name}", "avoid", cur_gun
                start, snap0, start_x = b_step, b_snap, b_x
            else:
                continue   # augment fly-off (no task) or ambiguous same-item collection

            if start is None or not (min_window <= step - start + 1 <= max_window):
                continue

            # the carrier is the item's source; blank if it activated straight as an item
            source = enemy_name(carrier, level) if carrier not in (0, ITEM_TYPE) else ""
            segs.append(Segment(
                initial_state=snap0,
                actions=np.asarray(ctx.actions[start:step + 1], dtype=np.uint8),
                label=label,
                level=level,
                start_step=start,
                end_step=step,
                skip=ctx.skip,
                src_trace=ctx.src,
                uid=f"{stem}_slot{slot}_i{step}",
                meta={"kind": kind, "item_weapon": attr, "kept_gun": kept, "slot": slot,
                      "item_born_step": b_step, "source_entity": source,
                      "start_x": start_x, "end_x": player_x(cur),
                      # ROCKET goal: mark the powerup item on the first frame
                      "goal_when": "first", "goal_kind": "item", "goal_slot": slot},
            ))
    return segs


def item_reached(seg: Segment, pre, cur) -> bool:
    """Goal predicate: the item's own formal event fires on this step — a pickup
    fires :class:`WeaponChangeEvent`/`PowerUpEvent` for its item, an avoid fires
    :class:`WeaponItemGoneEvent` (item left uncollected, gun unchanged)."""
    attr = seg.meta["item_weapon"]
    ev = _pickup_event(attr) if seg.meta["kind"] == "pick" else WeaponItemGoneEvent(weapon=attr)
    return bool(ev.trigger(pre, cur))


# ── Maker ─────────────────────────────────────────────────────────────────────

class PickItemMaker(TaskMaker):
    kind = "item"
    task_noun = "item task"

    def __init__(self, *, min_window: int = 2, max_window: int = 240,
                 min_pick_window: int = 10, require_armed: bool = True,
                 verify: bool = True):
        self.min_window = min_window
        self.max_window = max_window
        self.min_pick_window = min_pick_window   # pick-only floor (birth→pickup can be instant)
        self.require_armed = require_armed
        self.verify = verify

    def extract(self, trace_path: str) -> list[Segment]:
        return extract_segments(trace_path, min_window=self.min_window,
                                max_window=self.max_window,
                                require_armed=self.require_armed)

    def goal_reached(self, seg: Segment, pre, cur) -> bool:
        return item_reached(seg, pre, cur)

    def reject(self, seg: Segment, kept) -> str | None:
        # drop trivially-short pick tasks — items grabbed the instant they spawn
        if seg.meta.get("kind") == "pick" and seg.window_len < self.min_pick_window:
            return "too-short"
        if self.verify and not self.verify_segment(seg):
            return "dropped"
        return None

    def add_arguments(self, p) -> None:
        p.add_argument("--min-window", type=int, default=self.min_window)
        p.add_argument("--max-window", type=int, default=self.max_window)
        p.add_argument("--min-pick-window", type=int, default=self.min_pick_window,
                       help="drop pick tasks shorter than this many actions (birth→pickup)")
        p.add_argument("--keep-unarmed-avoid", action="store_true",
                       help="also emit avoid tasks when the player only holds the Regular gun")
        p.add_argument("--verify", action="store_true",
                       help="replay each segment and drop any that doesn't reproduce its outcome")

    def configure(self, args) -> None:
        self.min_window = args.min_window
        self.max_window = args.max_window
        self.min_pick_window = args.min_pick_window
        self.require_armed = not args.keep_unarmed_avoid
        self.verify = args.verify


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    PickItemMaker().main()


if __name__ == "__main__":
    main()
