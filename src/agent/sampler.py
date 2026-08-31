"""ActionSampler — per-level action prior + state-masked random rollouts.

Owns everything ``agent.mc_search`` needs to *propose* actions during one search:

  * the level's action table (``agent/level<N>.yaml``, else ``baseline.yaml``),
  * the action **prior** for that level as a row-stochastic PMF, loaded from a
    compact versioned artifact (Levels 1, 2, and 4) or built from a trace glob
    for legacy level configs,
  * masked sampling from that prior against the stateful legal-action mask
    (``agent.action_mask.legal_mask``), and
  * the random **rollout** the Monte-Carlo lookahead scores.

It also carries the ``agent.reward.RewardConfig`` used to score a rollout.

A level YAML holds the action table (``skip`` + ``actions``, parsed by
:class:`agent.action.ActionSpace`) plus two optional blocks::

    costs:                 # per-button hold penalties (agent/reward.py)
      F: -0.02
      J: -0.02
    prior:
      artifact: "priors/level2.yaml"  # relative to the agent package

Levels without a YAML fall back to the baseline table with default costs and a
uniform prior.
"""

import glob
import hashlib
import os
from dataclasses import dataclass, field

import numpy as np
import yaml

from agent.action import CONFIG_DIR, ActionSpace, DEFAULT as BASELINE
from agent.action_mask import legal_mask
from agent.reward import BUTTON_BITS, DEFAULT_CONFIG, RewardConfig, compute_reward
from env.event import event_by_tag
from util.replay import SKIP as REPLAY_SKIP, rewind_state, step_env

# src/agent/sampler.py → repo root; prior globs in the YAML are relative to it so
# they read the same from any working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALLOWED_COST_KEYS = set(BUTTON_BITS)
ALLOWED_PRIOR_KEYS = {"artifact", "traces", "mode", "smooth"}

EV_PLAYER_DIE = event_by_tag("die")


# ── Level config ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LevelConfig:
    """Everything that tunes generation for one level, from a single YAML file."""

    action_space: ActionSpace
    costs: dict = field(default_factory=dict)
    prior: dict = field(default_factory=dict)


def _check_keys(block: dict, allowed: set, what: str, level: int) -> dict:
    unknown = sorted(set(block) - allowed)
    if unknown:
        raise ValueError(f"Unknown key(s) in {what}: of level{level}.yaml: "
                         f"{unknown}; allowed: {sorted(allowed)}")
    return block


def load_for_level(level: int) -> LevelConfig:
    """Load ``agent/level<N>.yaml``, falling back to the baseline action table.

    The fallback keeps every level runnable: a level without a tuned file just
    searches the full baseline action set with default penalties and a uniform
    prior.
    """
    path = os.path.join(CONFIG_DIR, f"level{level}.yaml")
    if not os.path.exists(path):
        return LevelConfig(action_space=BASELINE)
    with open(path) as f:
        raw = yaml.safe_load(f)
    return LevelConfig(
        action_space=ActionSpace.from_dict(raw),
        costs=_check_keys(raw.get("costs") or {}, ALLOWED_COST_KEYS, "costs", level),
        prior=_check_keys(raw.get("prior") or {}, ALLOWED_PRIOR_KEYS, "prior", level),
    )


# ── Prior ─────────────────────────────────────────────────────────────────────

def _combo_index(nes: np.ndarray) -> int:
    """Universal (d-pad, button) combo index for a 9-button NES vector.

    9 d-pad states × 4 button states = 36 combos. Used only to match a recorded
    action against a level's table, which spans the same combo space.
    """
    up, down, left, right = bool(nes[4]), bool(nes[5]), bool(nes[6]), bool(nes[7])
    fire, jump = bool(nes[0]), bool(nes[8])
    if   up and left:    dpad = 5
    elif up and right:   dpad = 6
    elif down and left:  dpad = 7
    elif down and right: dpad = 8
    elif left:           dpad = 1
    elif right:          dpad = 2
    elif up:             dpad = 3
    elif down:           dpad = 4
    else:                dpad = 0
    btn = (3 if fire and jump else 1 if jump else 2 if fire else 0)
    return dpad * 4 + btn


def _uniform_pmf(n: int) -> np.ndarray:
    return np.full((n, n), 1.0 / n, dtype=np.float32)


def _bigram_pmf(counts: np.ndarray) -> np.ndarray:
    """Normalize integer transition counts; unseen rows remain explorable."""
    n = len(counts)
    row_sums = counts.sum(axis=1, keepdims=True).astype(np.float64)
    return np.divide(counts, row_sums, out=np.full(counts.shape, 1.0 / n),
                     where=row_sums > 0).astype(np.float32)


def action_table_sha256(actions: np.ndarray) -> str:
    """Stable identity of the ordered NES button vectors behind a prior."""
    return hashlib.sha256(np.asarray(actions, dtype=np.uint8).tobytes()).hexdigest()


def load_prior_artifact(path: str, actions: np.ndarray,
                        names: tuple[str, ...]) -> tuple[np.ndarray, str]:
    """Load and strictly validate a versioned, optionally smoothed bigram."""
    raw_bytes = open(path, "rb").read()
    raw = yaml.safe_load(raw_bytes)
    if raw.get("format_version") != 1 or raw.get("mode") != "bigram":
        raise ValueError(f"unsupported prior artifact format: {path}")
    if tuple(raw.get("action_names", ())) != tuple(names):
        raise ValueError(f"prior action names do not match Level {raw.get('level')}: {path}")
    expected = action_table_sha256(actions)
    if raw.get("action_table_sha256") != expected:
        raise ValueError(f"prior action table digest does not match: {path}")
    counts = np.asarray(raw.get("transition_counts"), dtype=np.int64)
    shape = (len(actions), len(actions))
    if counts.shape != shape or np.any(counts < 0):
        raise ValueError(f"prior counts are {counts.shape}, expected nonnegative {shape}")
    if int(counts.sum()) != int(raw.get("included_pairs", -1)):
        raise ValueError(f"prior included_pairs does not match counts: {path}")
    smooth = float(raw.get("smooth", 0.0))
    if not 0.0 <= smooth <= 1.0:
        raise ValueError(f"prior smooth must be in [0, 1]: {path}")
    prior = _bigram_pmf(counts)
    if smooth > 0:
        prior = (1.0 - smooth) * prior + smooth / len(actions)
    return prior.astype(np.float32), hashlib.sha256(raw_bytes).hexdigest()


def build_prior(table: np.ndarray, files: list[str], mode: str = "bigram",
                smooth: float = 0.0, verbose: bool = False) -> np.ndarray:
    """Count `files` into an (N, N) row-stochastic prior over `table`'s actions.

      * ``"bigram"``  — pmf[i, :] = P(next | prev = i) from transition counts.
      * ``"unigram"`` — pmf[i, :] = P(action), the marginal frequency, identical
        for every row (so it slots into the same ``pmf[prev]`` machinery).

    Transitions touching an action absent from the table (a combo trimmed out of
    the level set) are skipped, so the prior only spans what the search can take.
    ``smooth`` ∈ [0, 1] blends each row toward uniform, keeping rare-but-legal
    actions explorable. Rows with no observed transitions fall back to uniform.
    """
    if mode not in ("bigram", "unigram"):
        raise ValueError(f"mode must be 'bigram' or 'unigram', got {mode!r}")
    n = len(table)
    combo_to_idx = {_combo_index(a): i for i, a in enumerate(table)}

    trans = np.zeros((n, n), dtype=np.int64)   # transition counts (bigram)
    visits = np.zeros(n, dtype=np.int64)       # action occurrences (unigram)
    nfiles = pairs = skipped = 0
    for fpath in files:
        try:
            rec = np.load(fpath, allow_pickle=True)["actions"]
        except Exception as e:
            print(f"  WARN: cannot load {fpath}: {e}")
            continue
        if rec.ndim != 2 or rec.shape[1] != 9:
            print(f"  WARN: unexpected action shape {rec.shape} in {fpath}, skipping")
            continue

        idxs = [combo_to_idx.get(_combo_index(a)) for a in rec]
        for k in idxs:
            if k is not None:
                visits[k] += 1
        for prev, curr in zip(idxs[:-1], idxs[1:]):
            if prev is None or curr is None:   # touches an out-of-table action
                skipped += 1
                continue
            trans[prev, curr] += 1
        pairs += max(0, len(idxs) - 1)
        nfiles += 1

    if verbose:
        print(f"  files={nfiles}  pairs={pairs:,}  skipped(out-of-table)={skipped:,}  "
              f"actions={n}  mode={mode}  smooth={smooth}")

    if mode == "bigram":
        probs = _bigram_pmf(trans)
    else:
        total = visits.sum()
        marginal = visits / total if total > 0 else np.full(n, 1.0 / n)
        probs = np.tile(marginal, (n, 1))
    if smooth > 0:
        probs = (1.0 - smooth) * probs + smooth / n
    return probs.astype(np.float32)


# ── Sampler ───────────────────────────────────────────────────────────────────

class ActionSampler:
    """Action prior + state-masked random rollout generator for one level."""

    def __init__(self, level: int, actions: np.ndarray, names: tuple,
                 reward_config: RewardConfig, prior_pmf: np.ndarray,
                 uniform_pmf: np.ndarray, prior_sha256: str = ""):
        self.level = level                  # the level this prior/action set is for
        self.actions = actions              # (N, 9) uint8 button vectors
        self.names = names                  # action labels, parallel to `actions`
        self.reward_config = reward_config  # scores each rollout step
        self.prior_pmf = prior_pmf          # (N, N) prior for `level` (uniform if none)
        self.uniform_pmf = uniform_pmf      # used for any other level (game_clear crossing)
        self.prior_sha256 = prior_sha256    # immutable artifact identity, empty for legacy
        self._index_by_bytes = {a.tobytes(): i for i, a in enumerate(actions)}

    @staticmethod
    def _level_config(level: int):
        """Return (LevelConfig, actions, names, reward_config) for `level`."""
        cfg = load_for_level(level)
        actions = cfg.action_space.actions_np()
        if cfg.action_space.skip != REPLAY_SKIP:
            raise ValueError(
                f"action-space skip ({cfg.action_space.skip}) != replay.SKIP "
                f"({REPLAY_SKIP}); search frame-skip must match replay/step_env or "
                "traces won't reproduce."
            )
        reward = DEFAULT_CONFIG.with_costs(**cfg.costs)  # absent buttons → defaults
        return cfg, actions, tuple(cfg.action_space.names), reward

    @classmethod
    def _from_files(cls, level, actions, names, reward, files, mode, smooth):
        """Construct a sampler whose prior is counted from `files` (uniform if empty)."""
        uniform = _uniform_pmf(len(actions))
        if not files:
            return cls(level, actions, names, reward, uniform, uniform)
        prior = build_prior(actions, files, mode=mode, smooth=smooth)
        return cls(level, actions, names, reward, prior, uniform)

    @classmethod
    def for_level(cls, level: int) -> "ActionSampler":
        """Build the sampler from the level YAML, prior included (uniform if none)."""
        cfg, actions, names, reward = cls._level_config(level)
        p = cfg.prior
        if p.get("artifact"):
            if p.get("traces"):
                raise ValueError(f"Level{level} prior cannot set artifact and traces")
            path = os.path.join(os.path.dirname(__file__), p["artifact"])
            prior, digest = load_prior_artifact(path, actions, names)
            uniform = _uniform_pmf(len(actions))
            return cls(level, actions, names, reward, prior, uniform, digest)
        pattern = os.path.join(REPO_ROOT, p["traces"]) if p.get("traces") else None
        files = sorted(glob.glob(pattern)) if pattern else []
        if pattern and not files:
            print(f"WARNING: Level{level} prior traces {p['traces']!r} matched no "
                  "files; using uniform.")
        return cls._from_files(level, actions, names, reward, files,
                               p.get("mode", "bigram"), float(p.get("smooth", 0.0)))

    @classmethod
    def from_traces(cls, level: int, trace_glob: str, *, mode: str = "bigram",
                    smooth: float = 0.0) -> "ActionSampler":
        """Build the prior from an explicit glob — ad-hoc A/B of source or mode.

        Bypasses the level YAML's ``prior:`` block; :meth:`for_level` is the
        configured path.
        """
        _, actions, names, reward = cls._level_config(level)
        files = sorted(glob.glob(trace_glob))
        if not files:
            raise ValueError(f"no traces match {trace_glob!r}")
        return cls._from_files(level, actions, names, reward, files, mode, smooth)

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    def pmf(self, level: int) -> np.ndarray:
        # The prior is in this level's own action ordering, so only this level has
        # a matching prior; a game_clear run that crosses into another level
        # samples uniformly there.
        return self.prior_pmf if level == self.level else self.uniform_pmf

    def row_for(self, action: np.ndarray) -> int:
        """Prior row index for a previously committed action (0 if unknown)."""
        return self._index_by_bytes.get(np.asarray(action, dtype=np.uint8).tobytes(), 0)

    @staticmethod
    def sample(pmf_row: np.ndarray, mask: np.ndarray, r: float) -> int:
        """Sample an action index from `pmf_row` restricted to `mask`, using r∈[0,1).

        The prior weight of illegal actions is zeroed and the remainder
        renormalised. If no legal action carries prior mass, fall back to uniform
        over the legal set.
        """
        w = pmf_row * mask
        s = w.sum()
        if s <= 0.0:
            legal = np.flatnonzero(mask)
            return int(legal[min(int(r * len(legal)), len(legal) - 1)])
        return min(int(np.searchsorted(np.cumsum(w), r * s)), len(pmf_row) - 1)

    def rollout(self, env, start_state: bytes, length: int, level: int,
                prev_action: np.ndarray) -> tuple[list, float, bool]:
        """Sample one prior-guided, state-masked random rollout from `start_state`.

        At each step the current RAM + previous action decide which presses are
        meaningful (``legal_mask``); the prior row is restricted to that legal set
        before sampling, so structurally inert fire/jump presses are never
        emitted. `prev_action` is the action committed just before this rollout,
        needed for the fire/jump press edge.

        Returns ``(actions, cumulative_reward, died)``. Stops early on death (that
        action is included but its reward is not).
        """
        rewind_state(env, start_state)
        pmf = self.pmf(level)
        actions = self.actions
        prev = self.row_for(prev_action)
        # Pre-sample all randoms at once (cheaper than per-step draws).
        rands = np.random.random(length).astype(np.float32)

        seq, total = [], 0.0
        for i in range(length):
            pre = env.unwrapped.get_ram().copy()
            mask = legal_mask(actions, pre, prev_action)
            prev = self.sample(pmf[prev], mask, rands[i])
            act = actions[prev].copy()
            step_env(env, act)
            cur = env.unwrapped.get_ram()
            seq.append(act)
            prev_action = act
            if EV_PLAYER_DIE.trigger(pre, cur):
                return seq, total, True
            total += compute_reward(pre, cur, self.reward_config, action=act)
        return seq, total, False
