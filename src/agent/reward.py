"""Search reward for mc_search (trace-generation phase).

Scores one decision step (``skip`` NES frames) so the Monte-Carlo searcher can
rank rollouts. The objective here is *the cleanest winning trace, found
efficiently* — not a learnable RL shaping signal — so on top of the usual
advancement / combat / terminal components it charges a per-button hold penalty
that an RL reward must not have.

Components, all read from a (pre_ram, cur_ram) pair:

  * combat   — enemy HP decrements, split regular vs. boss (``env.utility.live_slots``)
  * items    — spread pickup (minus spread loss), rapid-fire pickup
  * terminal — level cleared, player died
  * advance  — chosen by the level's advancement style (``env.utility.advance_style``):
      "forward" xscroll pixels · "inside" core/door/room · "up" climb pixels
    Forward progress is switched off once the boss fight starts, where the screen
    auto-scrolls by itself (``env.utility.boss_scene``).
  * cleanliness — one weight per button (``F J U D L R``), charged on *every*
    step that button is held, not just the press edge.

Charging a held button drives the trace toward minimal button use: a step that
lands a hit still pays for itself via ``enemy_hp``/``boss_hp``, while a press
that achieves nothing goes net negative. Right (forward) is the canonical action
and is normally left free, so among reward-equivalent actions the search commits
the simplest one — a consistent state→action label for downstream BC instead of
an arbitrary R-vs-UR coin flip.

Per-level overrides of the button costs come from the level YAML's ``costs:``
block via :meth:`RewardConfig.with_costs` (see ``agent.sampler``).
"""

from dataclasses import dataclass

import numpy as np

from env.constant import (
    ADDR_LEVEL,
    BOSS_ENEMY_TYPES_BY_LEVEL,
    BOSS_ENEMY_TYPES_COMMON,
)
from env.event import event_by_tag, make_march_events
from env.utility import advance_style, boss_scene, live_slots

# Event instances are rebuilt on every ``event_by_tag`` call, so resolve the
# handful the reward needs once, here, not inside the per-step loop.
EV_DIE = event_by_tag("die")
EV_LEVELUP = event_by_tag("clear_level")
EV_SPREAD_PICK = event_by_tag("pick_spread")
EV_SPREAD_LOSE = event_by_tag("lose_spread")
EV_RAPID_FIRE = event_by_tag("pick_rapid_fire")
# The advancement events of each style, keyed the way advance_style names them.
MARCH_EVENTS = {style: make_march_events(style)
                for style in ("forward", "inside", "up")}

DEFAULT_REWARD_WEIGHTS = {
    # combat / items (level-agnostic)
    "enemy_hp": 1.0,
    "boss_hp": 1.0,
    "spread_pick": 20.0,
    "rapid_fire": 10.0,
    # terminal (level-agnostic)
    "levelup": 1.0,
    "player_die": -15.0,
    # advancement (env.event march events) — only the level's own style applies
    "push_right": 0.1,      # "forward": per xscroll pixel
    "push_inside": 1.0,     # "inside": dense progress through the door
    "room_enter": 1.0,      # "inside": per-room milestone
    "core_broken": 1.0,     # "inside": core-clear spike
    "push_up": 1.0,         # "up": per vertical-scroll pixel
    # generation-only per-button hold penalties, charged each step the bit is
    # held, keyed by the action-table nicknames (see BUTTON_BITS). fire/jump
    # default to a real penalty; the d-pad defaults to 0 (inert unless a level
    # opts in), and Right is the canonical forward action, normally left free.
    "F": -0.3,   # fire (B)
    "J": -0.3,   # jump (A)
    "U": 0.0,    # up
    "D": 0.0,    # down
    "L": 0.0,    # left
    "R": 0.0,    # right
}

# Button nickname → action-vector bit index.
# Bit order: [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A].
BUTTON_BITS = {
    "F": 0,   # fire (B)
    "U": 4,
    "D": 5,
    "L": 6,
    "R": 7,
    "J": 8,   # jump (A)
}


@dataclass(frozen=True)
class RewardConfig:
    """Named set of search reward weights."""

    name: str
    reward_weights: dict

    def with_costs(self, **costs: float | None) -> "RewardConfig":
        """Copy with per-button hold penalties overridden (from a level YAML).

        Each keyword is a button nickname (see :data:`BUTTON_BITS`); a ``None``
        value is ignored, so unspecified buttons keep their default.
        """
        w = dict(self.reward_weights)
        for key, val in costs.items():
            if val is None:
                continue
            if key not in BUTTON_BITS:
                raise ValueError(f"Unknown cost weight: {key!r}")
            w[key] = val
        return RewardConfig(name=self.name, reward_weights=w)


DEFAULT_CONFIG = RewardConfig(name="clean", reward_weights=DEFAULT_REWARD_WEIGHTS.copy())


def enemy_hp_deltas(pre: np.ndarray, cur: np.ndarray) -> tuple[float, float]:
    """(regular, boss) summed enemy-HP decrements this step.

    Armored / boss types "ting"-reset their HP on every hit, so their decrements
    are counted separately from regular enemies (which die at HP 0).
    """
    level = int(pre[ADDR_LEVEL])
    boss_types = BOSS_ENEMY_TYPES_COMMON | BOSS_ENEMY_TYPES_BY_LEVEL.get(level, set())
    regular = boss = 0.0
    for etype, pre_hp, cur_hp in live_slots(pre, cur):
        delta = float(pre_hp - cur_hp)
        if delta <= 0:
            continue
        if etype in boss_types:
            boss += delta
        else:
            regular += delta
    return regular, boss


def reward_components(pre: np.ndarray, cur: np.ndarray,
                      weights: dict[str, float]) -> dict[str, float]:
    """Level-aware reward components for one step (button costs excluded).

    Combat / item / terminal components are level-agnostic; the advancement
    component is picked from the level in RAM, so one config covers every level.
    """
    regular_hp, boss_hp = enemy_hp_deltas(pre, cur)
    spread = EV_SPREAD_PICK.trigger(pre, cur) - EV_SPREAD_LOSE.trigger(pre, cur)
    components = {
        "enemy_hp": weights["enemy_hp"] * regular_hp,
        "boss_hp": weights["boss_hp"] * boss_hp,
        "spread_pick": weights["spread_pick"] * spread,
        "rapid_fire": weights["rapid_fire"] * EV_RAPID_FIRE.trigger(pre, cur),
        "levelup": weights["levelup"] * EV_LEVELUP.trigger(pre, cur),
        "player_die": weights["player_die"] * EV_DIE.trigger(pre, cur),
    }

    for ev in MARCH_EVENTS[advance_style(int(pre[ADDR_LEVEL]))]:
        # The boss fight opens with a scripted auto-scroll that advances xscroll
        # on its own, so that progress is not the player's doing — paying for it
        # would reward idling through the reveal instead of killing the boss.
        marched = 0.0 if (ev.tag == "push_right" and boss_scene(cur)) else ev.trigger(pre, cur)
        components[ev.tag] = weights[ev.tag] * marched

    return components


def press_penalty(action: np.ndarray, weights: dict[str, float]) -> float:
    """Sum the hold cost of every button pressed in `action`.

    One search step is one decision (the action is held for ``skip`` frames), so
    a step pays for each button it holds regardless of the previous step.
    """
    return sum(weights[nick] for nick, bit in BUTTON_BITS.items() if action[bit])


def compute_reward(pre: np.ndarray, cur: np.ndarray,
                   config: RewardConfig = DEFAULT_CONFIG,
                   action: np.ndarray | None = None) -> float:
    """Single-step search reward: components + per-button hold penalties.

    ``action=None`` drops the cleanliness term, leaving the plain game reward for
    callers that don't track actions.
    """
    total = sum(reward_components(pre, cur, config.reward_weights).values())
    if action is not None:
        total += press_penalty(action, config.reward_weights)
    return total
