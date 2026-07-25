"""Low-level RAM readers and helpers shared by the event classes."""

import numpy as np

from .constant import (
    ADDR_ENEMY_HP,
    ADDR_ENEMY_HP_COUNT,
    ADDR_ENEMY_TYPE,
    ADDR_LEVEL,
    ADDR_LEVEL_ROUTINE,
    ADDR_LEVEL_STOP_SCROLL,
    ADDR_SCROLL_OFF,
    ADDR_XSCROLL,
    ADDR_XSCROLL_HI,
    BOSS_ENEMY_TYPES_BY_LEVEL,
    BOSS_ENEMY_TYPES_COMMON,
    ENEMY_TYPE_FALLING_ROCK,
    INSIDE_LEVELS,
)


def xscroll(ram: np.ndarray) -> int:
    return (int(ram[ADDR_XSCROLL]) << 8) | int(ram[ADDR_XSCROLL + 1])


def yscroll(ram: np.ndarray) -> int:
    """Absolute vertical climb height in pixels: screen index × 240 + offset.

    The offset byte (0x65) wraps at 0xF0 = 240 as each screen is climbed, and
    the screen index (0x64) increments, so this is monotonic up the level.
    """
    return int(ram[ADDR_XSCROLL_HI]) * 240 + int(ram[ADDR_SCROLL_OFF])


def yscroll_delta(pre: np.ndarray, cur: np.ndarray) -> float:
    """Vertical scroll progress in pixels, handling the wrap at 0xF0 = 240."""
    delta = int(cur[ADDR_SCROLL_OFF]) - int(pre[ADDR_SCROLL_OFF])
    if delta < -100:            # e.g. 238 → 2 means +4, not -236
        delta += 240
    return max(0.0, float(delta))

def routine(ram: np.ndarray) -> int:
    return int(ram[ADDR_LEVEL_ROUTINE])


def live_slots(pre: np.ndarray, cur: np.ndarray):
    """Yield (etype, pre_hp, cur_hp) for enemy slots active on both frames.

    HP >= 0xf0 marks an inactive / empty slot. Bullet-like projectile types
    carry no real HP and are skipped, as is the L3 Falling Rock, which respawns
    endlessly from the rock cave.
    """
    level = int(pre[ADDR_LEVEL])
    for slot in range(ADDR_ENEMY_HP_COUNT):
        etype  = int(pre[ADDR_ENEMY_TYPE + slot])
        pre_hp = int(pre[ADDR_ENEMY_HP + slot])
        cur_hp = int(cur[ADDR_ENEMY_HP + slot])
        if pre_hp >= 0xf0 or cur_hp >= 0xf0:
            continue
        if etype in (0x01, 0x0B, 0x0F):            # projectiles
            continue
        if etype == ENEMY_TYPE_FALLING_ROCK and level == 2:
            continue
        yield etype, pre_hp, cur_hp


def boss_enemy_present(ram: np.ndarray) -> bool:
    """True when a boss / boss-objective enemy type occupies an active slot."""
    level = int(ram[ADDR_LEVEL])
    types = BOSS_ENEMY_TYPES_COMMON | BOSS_ENEMY_TYPES_BY_LEVEL.get(level, set())
    for slot in range(ADDR_ENEMY_HP_COUNT):
        if int(ram[ADDR_ENEMY_HP + slot]) >= 0xf0:
            continue
        if int(ram[ADDR_ENEMY_TYPE + slot]) in types:
            return True
    return False


def boss_scene(ram: np.ndarray) -> bool:
    """True during an end-of-level boss fight.

    Scrolling levels flag LEVEL_STOP_SCROLL ($58) = 0xff when the boss-reveal
    auto-scroll begins. Indoor levels don't auto-scroll, so fall back to
    boss-enemy presence — there the boss types only spawn in the final room.
    """
    if int(ram[ADDR_LEVEL_STOP_SCROLL]) == 0xff:
        return True
    if int(ram[ADDR_LEVEL]) in INSIDE_LEVELS:
        return boss_enemy_present(ram)
    return False
