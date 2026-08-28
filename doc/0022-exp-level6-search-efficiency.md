# Level 6 search efficiency

Status: Proposed

## 1. Goal

Find a high-throughput full-clear Monte Carlo configuration for Level 6. Start
from Level 5's `4/24/8/8` winner and select production parameters by
replay-valid Level 7 transitions per hour.

## 2. Setup

Search starts from the canonical Level 6 Spread + rapid-fire state. The action
table and costs match the trimmed outdoor Level 5 table. A frozen bigram prior
is counted from the old MC and human wins and blended 10% toward uniform. Both
sources replay from Level 6 into Level 7; together they contain 4,750 actions.
The trimmed table excludes 42 rare source actions.

Stage 1 runs five interleaved attempts per arm with eight workers, frame skip
3, a 600-second time limit, and a 6,000-action limit. The Level 5 winner runs
first in each round; remaining arms are deterministically shuffled. Working
results live under `tmp/level6-search-efficiency-screen/`.

| arm | rollouts | rollout length | settle margin | max rewind |
|---|---:|---:|---:|---:|
| l5_winner | 4 | 24 | 8 | 8 |
| rollouts_2 | 2 | 24 | 8 | 8 |
| rollouts_6 | 6 | 24 | 8 | 8 |
| rollouts_8 | 8 | 24 | 8 | 8 |
| length_20 | 4 | 20 | 8 | 8 |
| length_28 | 4 | 28 | 8 | 8 |

The low-compute screen was stopped after its first four completed attempts all
timed out at 600 seconds. Stage 2 moves to a high-compute scout grid around the
original Contra Agent default. Run one interleaved attempt per arm first under
`tmp/level6-search-efficiency-high-compute-scout/`; do not spend five timeouts
per arm before establishing which region can win.

| arm | rollouts | rollout length | settle margin | max rewind |
|---|---:|---:|---:|---:|
| old_contra_baseline | 64 | 48 | 16 | 60 |
| rollouts_16 | 16 | 48 | 16 | 60 |
| rollouts_32 | 32 | 48 | 16 | 60 |
| rollouts_96 | 96 | 48 | 16 | 60 |
| length_64 | 64 | 64 | 16 | 60 |
| rewind_30 | 64 | 48 | 16 | 30 |
| settle_8 | 64 | 48 | 8 | 60 |

After scouting, retain the three fastest replay-valid arms and run four more
attempts each, giving each finalist five total attempts. If fewer than three
arms win, retain every winning arm and add no speculative confirmation cells.

## 3. Evaluation metrics

Report attempts, search wins, replay-valid transitions, mean seconds per valid
win, wins/hour, and exact duplicates for every arm. Attempt rows in
`tmp/level6-search-efficiency-screen/results.jsonl` are the source of truth;
`summary.json` is derived after every attempt.

| source claim | provenance |
|---|---|
| Level 5 winner is `4/24/8/8` | `doc/0021-exp-level5-search-efficiency.md` |
| old MC win has 1,796 actions | `contra_agent/tmp/mc_trace_old/level6/win_level6_202606181024.npz` audit |
| old human win has 2,954 actions | `contra_agent/contra/human_recordings/Level6/03281909.npz` audit |
| both traces replay Level 6 to Level 7 | stable-retro replay audit on 2026-08-28 |

## 4. Conclusion

_Pending user conclusion after the measured sweep._
