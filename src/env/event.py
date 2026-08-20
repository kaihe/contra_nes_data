"""Contra NES events, as a small class hierarchy of concrete instances.

Each *family* of events is a class (``KillEvent``, ``HitEvent``,
``WeaponPickupEvent`` …) and each *concrete* event is an instance of that class
(kill a Soldier, kill a Rotating Gun, pick up the Laser, gain rapid fire …).

For now every event only implements ``trigger``:

    ev.trigger(pre_ram, cur_ram) -> float   # magnitude: count / HP delta / 0-or-1

Descriptions, narrative detail, reward weights and the per-level registry are
deliberately left out until later.

RAM addresses / lookup tables live in ``constant.py``; the RAM readers used
below live in ``utility.py``.
"""

from enum import Enum
import numpy as np

from .constant import (
    ADDR_ENEMY_ATTR,
    ADDR_ENEMY_HP_COUNT,
    ADDR_ENEMY_TYPE,
    ADDR_FALCON,
    ADDR_GUN_ANI,
    ADDR_INDOOR_CLEARED,
    ADDR_INVINCIBILITY,
    ADDR_LEVEL,
    ADDR_LIVES,
    ADDR_PLAYER_ADV,
    ADDR_PLAYER_DEATH,
    ADDR_PLAYER_STATE,
    ADDR_WEAPON,
    ADDR_XSCROLL_HI,
    ENEMY_TYPE_NAMES_BY_LEVEL,
    ENEMY_TYPE_NAMES_COMMON,
    ITEM_GUN_ATTRS,
    WEAPON_NAMES,
)
from .entity import ADDR_ENEMY_ROUTINE
from .utility import (
    boss_scene,
    live_slots,
    routine,
    xscroll,
    yscroll,
    yscroll_delta,
)


class EventType(Enum):

    KILL = "kill"
    POWERUP = "power_up"
    WEAPON_CHANGE = "weapon_change"
    WEAPON_ITEM_GONE = "weapon_item_gone"
    MARCH = 'march_through_level'
    TERMINAL = 'episode_result'
    STAGE = 'in_which_game_stage'

# ── Base event class ─────────────────────────────────────────────────────────

class BaseEvent:
    """Common interface for every event. Subclasses implement ``trigger``."""

    tag: str = ""

    def trigger(self, pre: np.ndarray, cur: np.ndarray) -> float:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.tag}>"


# ── Enemy events (scan the 16 enemy slots) ───────────────────────────────────

class KillEvent(BaseEvent):
    """A specific enemy type was destroyed this step (its HP crossed to 0).

    Armored / boss types that "ting"-reset their HP to 1 on every hit never
    read as killed here — use :class:`HitEvent` for those.

    ``level`` : 0-indexed level the enemy type belongs to. Enemy types 0x10+ are
                reused for different enemies across levels, so a level-specific
                kill event must pin the level; ``None`` matches any level (used
                for the common types 0x00-0x0F).
    """

    def __init__(self, enemy_type: int, level: int | None = None,
                 name: str | None = None):
        self.enemy_type = enemy_type
        self.level = level
        self.name = name or self._lookup_name(enemy_type, level)
        self.tag  = "kill_" + self.name.lower().replace(" ", "_")
        self.type = EventType.KILL

    @staticmethod
    def _lookup_name(enemy_type: int, level: int | None) -> str:
        table = (ENEMY_TYPE_NAMES_COMMON if level is None
                 else ENEMY_TYPE_NAMES_BY_LEVEL.get(level, {}))
        return table.get(enemy_type, f"0x{enemy_type:02x}")

    def trigger(self, pre: np.ndarray, cur: np.ndarray) -> float:
        if self.level is not None and int(pre[ADDR_LEVEL]) != self.level:
            return 0.0
        n = 0
        for etype, pre_hp, cur_hp in live_slots(pre, cur):
            if etype == self.enemy_type and pre_hp > 0 and cur_hp == 0:
                n += 1
        return float(n)

# ── Weapon / power-up events ─────────────────────────────────────────────────

class WeaponChangeEvent(BaseEvent):
    """The held weapon changed from ``pre_weapon`` to ``curr_weapon``.

    The gun-type nibble changing is the signal, so this covers both gains
    (picking up a new weapon) and losses (death resets the gun to Regular).

    ``curr_weapon`` : the new weapon; ``None`` matches any weapon.
    ``pre_weapon``  : the replaced weapon; ``None`` matches any weapon.

    Leaving one side ``None`` gives an open-ended event: ``curr_weapon=4`` is
    "picked up the Laser (from anything)"; ``pre_weapon=3`` is "lost the Spread
    (to anything)".
    """

    def __init__(self, curr_weapon: int | None = None,
                 pre_weapon: int | None = None):
        self.curr_weapon = curr_weapon
        self.pre_weapon  = pre_weapon
        self.tag = self._make_tag()
        self.type = EventType.WEAPON_CHANGE

    def _make_tag(self) -> str:
        if self.pre_weapon is None and self.curr_weapon is not None:
            return f"pick_{WEAPON_NAMES[self.curr_weapon].lower()}"
        if self.curr_weapon is None and self.pre_weapon is not None:
            return f"lose_{WEAPON_NAMES[self.pre_weapon].lower()}"
        if self.pre_weapon is not None and self.curr_weapon is not None:
            return (f"change_{WEAPON_NAMES[self.pre_weapon].lower()}"
                    f"_to_{WEAPON_NAMES[self.curr_weapon].lower()}")
        return "change_weapon"

    def trigger(self, pre: np.ndarray, cur: np.ndarray) -> float:
        prev = int(pre[ADDR_WEAPON]) & 0x0F
        curr = int(cur[ADDR_WEAPON]) & 0x0F
        if prev == curr:
            return 0.0
        if self.curr_weapon is not None and curr != self.curr_weapon:
            return 0.0
        if self.pre_weapon is not None and prev != self.pre_weapon:
            return 0.0
        return 1.0


class WeaponItemGoneEvent(BaseEvent):
    """A dropped gun item left the screen uncollected (the "un-spawn" of a pickup).

    A slot that held a weapon item (type 0x00) is no longer an item, the gun byte
    did **not** change this step, and the item granted a gun the player was *not*
    already holding — so collecting it would have flipped the weapon, and since it
    didn't, the item flew off / expired rather than being taken. This is the
    opposite of :class:`WeaponChangeEvent`'s ``pick_*`` and anchors the "keep your
    gun" (avoid) tasks.

    ``weapon`` : the gun the item would have granted (1=M … 4=L); ``None`` matches
    any gun item.
    """

    def __init__(self, weapon: int | None = None):
        self.weapon = weapon
        self.tag = (f"avoid_{WEAPON_NAMES[weapon].lower()}" if weapon is not None
                    else "avoid_weapon_item")
        self.type = EventType.WEAPON_ITEM_GONE

    def trigger(self, pre: np.ndarray, cur: np.ndarray) -> float:
        cur_gun = int(cur[ADDR_WEAPON]) & 0x0F
        if (int(pre[ADDR_WEAPON]) & 0x0F) != cur_gun:
            return 0.0   # the gun changed → a pickup / loss, not an uncollected item
        n = 0
        for s in range(ADDR_ENEMY_HP_COUNT):
            was_item = (int(pre[ADDR_ENEMY_TYPE + s]) == 0
                        and int(pre[ADDR_ENEMY_ROUTINE + s]) != 0)
            now_item = (int(cur[ADDR_ENEMY_TYPE + s]) == 0
                        and int(cur[ADDR_ENEMY_ROUTINE + s]) != 0)
            if not (was_item and not now_item):
                continue
            attr = int(pre[ADDR_ENEMY_ATTR + s]) & 0x07
            if (attr in ITEM_GUN_ATTRS and attr != cur_gun
                    and (self.weapon is None or attr == self.weapon)):
                n += 1
        return float(n)


class PowerUpEvent(BaseEvent):
    """A one-off beneficial pickup (rapid fire, extra life, …).

    Like :class:`TerminalEvent`, the condition is passed in as a callable, so
    each power-up is a one-line instance rather than its own subclass.
    """

    def __init__(self, tag: str, trigger_fn):
        self.tag = tag
        self._trigger_fn = trigger_fn
        self.type = EventType.POWERUP

    def trigger(self, pre: np.ndarray, cur: np.ndarray) -> float:
        return float(self._trigger_fn(pre, cur))



# ── Progress events ──────────────────────────────────────────────────────────
#
# How far the player advanced *this step*, in the unit the level advances in.
# Contra levels march in one of three styles (``env.utility.advance_style``):
#
#   "forward" — side-scroll levels: horizontal scroll pixels
#   "up"      — the waterfall climb: vertical scroll pixels
#   "inside"  — indoor bases, which barely scroll: the player instead breaks the
#               wall core, walks through the opened door, and enters the next
#               room, so progress is those three milestones rather than pixels
#
# Unlike the other families these fire on most steps and carry a magnitude, so
# they are a dense advancement signal (what a search or RL reward integrates)
# rather than a narrative event — which is why they stay out of
# ``get_event_instances`` / ``all_events``; ask for them via ``make_march_events``.

class MarchEvent(BaseEvent):
    """Advancement toward the end of the level, as a per-step magnitude.

    ``style`` is the advancement style the instance measures ("forward",
    "inside", "up"); a caller keeps the instances matching the level it is on.
    The magnitude is that style's own unit — scroll pixels for "forward"/"up",
    a 0-or-1 milestone for each "inside" event — so weights are per style, not
    comparable across them.

    Like :class:`TerminalEvent`, the condition is a callable, so each instance is
    one line in :func:`make_march_events`.
    """

    def __init__(self, tag: str, style: str, trigger_fn):
        self.tag = tag
        self.style = style
        self._trigger_fn = trigger_fn
        self.type = EventType.MARCH

    def trigger(self, pre: np.ndarray, cur: np.ndarray) -> float:
        return float(self._trigger_fn(pre, cur))


# ── Player / level-flow events ───────────────────────────────────────────────

class TerminalEvent(BaseEvent):
    """An event that ends the episode (death, level clear, game clear).

    The trigger condition is passed in as a callable, so each terminal
    condition is a one-line instance rather than its own subclass.
    """

    def __init__(self, tag: str, trigger_fn):
        self.tag = tag
        self._trigger_fn = trigger_fn
        self.type = EventType.TERMINAL

    def trigger(self, pre: np.ndarray, cur: np.ndarray) -> float:
        return float(self._trigger_fn(pre, cur))




class GameStageEvent(BaseEvent):
    """A game-flow phase signal (boss fight, level begin, transition, title).

    Not a reward event — it reports which phase the game is in so the caller can
    decide whether to send real actions or no-ops. Like :class:`TerminalEvent`,
    the condition is passed in as a callable rather than subclassed per phase.
    """

    def __init__(self, tag: str, trigger_fn):
        self.tag = tag
        self._trigger_fn = trigger_fn
        self.type = EventType.STAGE
        
    def trigger(self, pre: np.ndarray, cur: np.ndarray) -> float:
        return float(self._trigger_fn(pre, cur))


# ── Instance factories ───────────────────────────────────────────────────────
#
# One factory per event type builds that type's concrete instances.
# ``get_event_instances`` groups them by EventType value. March events are
# intentionally excluded: they fire on nearly every step, so they belong to a
# reward that integrates them, not to a narrative scan (see ``make_march_events``).

# Common enemy types that aren't real kill targets (items / projectiles /
# armored plating). Armored/ting-reset types are covered by damage, not kills.
_COMMON_KILL_SKIP = {0x00, 0x01, 0x0A, 0x0B, 0x0F}


def make_kill_events(level: int | None = None) -> list[KillEvent]:
    """Kill events: the common enemies (any level) plus level-specific enemies.

    ``level`` None includes every level's 0x10+ enemies; an int restricts to
    that one level's enemies.
    """
    events = [
        KillEvent(etype)
        for etype, name in ENEMY_TYPE_NAMES_COMMON.items()
        if etype not in _COMMON_KILL_SKIP and name != "Unused"
    ]
    by_level = (ENEMY_TYPE_NAMES_BY_LEVEL if level is None
                else {level: ENEMY_TYPE_NAMES_BY_LEVEL.get(level, {})})
    for lvl, types in by_level.items():
        events += [KillEvent(etype, level=lvl)
                   for etype, name in types.items() if name != "Unused"]
    return events


def make_weapon_change_events() -> list[WeaponChangeEvent]:
    """Weapon gains (pick_*) and the spread loss."""
    return [
        WeaponChangeEvent(curr_weapon=1),   # pick_machinegun
        WeaponChangeEvent(curr_weapon=2),   # pick_flamethrower
        WeaponChangeEvent(curr_weapon=3),   # pick_spread
        WeaponChangeEvent(curr_weapon=4),   # pick_laser
        WeaponChangeEvent(pre_weapon=3),    # lose_spread
    ]


def make_weapon_item_events() -> list[WeaponItemGoneEvent]:
    """Gun items (M/F/S/L) that left the screen uncollected (avoid_*)."""
    return [WeaponItemGoneEvent(weapon=w) for w in (1, 2, 3, 4)]


def make_powerup_events() -> list[PowerUpEvent]:
    """One-off beneficial pickups.

    Barrier ($b0) and Falcon ($7d) timers are set on pickup and only decrement
    afterwards, so a rising edge marks the pickup.
    """
    def _rapid_fire_gained(pre: np.ndarray, cur: np.ndarray) -> bool:
        """Rapid-fire flag turned on (same weapon type) via a gun pickup."""
        gun_ani_rising = int(cur[ADDR_GUN_ANI]) == 0x1F and int(pre[ADDR_GUN_ANI]) != 0x1F
        return bool(
            gun_ani_rising
            and (int(cur[ADDR_WEAPON]) & 0x0F) == (int(pre[ADDR_WEAPON]) & 0x0F)
            and (int(cur[ADDR_WEAPON]) & 0x10) and not (int(pre[ADDR_WEAPON]) & 0x10)
        )

    return [
        PowerUpEvent("pick_rapid_fire",
                     lambda pre, cur: 1.0 if _rapid_fire_gained(pre, cur) else 0.0),
        PowerUpEvent("pick_barrier",
                     lambda pre, cur: 1.0 if int(cur[ADDR_INVINCIBILITY]) > int(pre[ADDR_INVINCIBILITY]) else 0.0),
        PowerUpEvent("pick_falcon",
                     lambda pre, cur: 1.0 if int(cur[ADDR_FALCON]) > int(pre[ADDR_FALCON]) else 0.0),
        PowerUpEvent("gain_life",
                     lambda pre, cur: 1.0 if int(cur[ADDR_LIVES]) > int(pre[ADDR_LIVES]) else 0.0),
    ]


def make_march_events(style: str | None = None) -> list[MarchEvent]:
    """Per-step advancement events, optionally filtered to one level's style.

    ``push_right`` is the raw signed scroll delta, not clamped at 0: scrolling
    back is real lost ground, and the end-of-level xscroll reset shows up as one
    large negative step (which is why a searcher stops at the transition rather
    than stepping through it).

    Indoors, ``push_inside`` fires on every step the player is walking through
    the opened door (~44 steps), bridging the long gap between breaking the core
    and reaching the next room; the two milestones bracket it.
    """
    events = [
        MarchEvent("push_right", "forward",
                   lambda pre, cur: xscroll(cur) - xscroll(pre)),
        MarchEvent("push_up", "up", yscroll_delta),
        MarchEvent("core_broken", "inside",
                   lambda pre, cur: 1.0 if (int(pre[ADDR_INDOOR_CLEARED]) == 0
                                            and int(cur[ADDR_INDOOR_CLEARED]) != 0) else 0.0),
        MarchEvent("push_inside", "inside",
                   lambda _, cur: 1.0 if int(cur[ADDR_PLAYER_ADV]) != 0 else 0.0),
        MarchEvent("room_enter", "inside",
                   lambda pre, cur: 1.0 if int(cur[ADDR_XSCROLL_HI]) > int(pre[ADDR_XSCROLL_HI]) else 0.0),
    ]
    return [ev for ev in events if style is None or ev.style == style]


def make_terminal_events() -> list[TerminalEvent]:
    """Episode-ending events (death, level clear, game clear)."""
    return [
        TerminalEvent("die", lambda pre, cur: 1.0 if (
            int(pre[ADDR_PLAYER_STATE]) != 0x02
            and ((int(cur[ADDR_PLAYER_DEATH]) & 0x01) or int(cur[ADDR_PLAYER_STATE]) == 0x02)
        ) else 0.0),
        TerminalEvent("clear_level",
                      lambda pre, cur: 1.0 if int(cur[ADDR_LEVEL]) > int(pre[ADDR_LEVEL]) else 0.0),
        TerminalEvent("clear_game",
                      lambda pre, cur: 1.0 if int(pre[ADDR_LEVEL]) == 7 and int(cur[ADDR_LEVEL]) > 7 else 0.0),
    ]


def make_stage_events() -> list[GameStageEvent]:
    """Game-flow phase signals (boss fight, level begin, transition, title)."""
    return [
        GameStageEvent("in_boss_stage",
                       lambda pre, cur: 1.0 if (boss_scene(cur) and not boss_scene(pre)) else 0.0),
        GameStageEvent("in_begin",
                       lambda pre, cur: 1.0 if routine(pre) in (0, 1, 2, 3) and routine(cur) == 0x04 else 0.0),
        GameStageEvent("in_level_transition",
                       lambda pre, cur: 1.0 if routine(pre) not in (0x08, 0x09)
                       and routine(cur) in (0x08, 0x09) else 0.0),
        GameStageEvent("in_title",
                       lambda pre, cur: 1.0 if routine(cur) in (0, 1, 2, 3) else 0.0),
    ]


def get_event_instances(level: int | None = None) -> dict[str, list[BaseEvent]]:
    """All event instances grouped by EventType value (march excluded).

    ``level`` scopes the kill events to that level's enemies plus the common
    ones; None includes every level's enemies.
    """
    return {
        EventType.KILL.value:             make_kill_events(level),
        EventType.WEAPON_CHANGE.value:    make_weapon_change_events(),
        EventType.WEAPON_ITEM_GONE.value: make_weapon_item_events(),
        EventType.POWERUP.value:          make_powerup_events(),
        EventType.TERMINAL.value:      make_terminal_events(),
        EventType.STAGE.value:         make_stage_events(),
    }


def all_events(level: int | None = None) -> list[BaseEvent]:
    """Flat list of every event instance (march excluded)."""
    return [ev for group in get_event_instances(level).values() for ev in group]


def event_by_tag(tag: str, level: int | None = None) -> BaseEvent:
    """The single event instance carrying ``tag`` (e.g. "die", "pick_spread").

    Instances are built per call, so resolve the events you need once at import
    time rather than inside a per-step loop.
    """
    for ev in all_events(level):
        if ev.tag == tag:
            return ev
    raise KeyError(f"no event with tag {tag!r}")
