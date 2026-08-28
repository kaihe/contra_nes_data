# Level 5 search efficiency

Status: Proposed

## 1. Goal

Find a high-throughput full-clear Monte Carlo configuration for Level 5. Start
from Level 1's winning `16/24/8/15` search shape, then vary one parameter at a
time. Select a production setup by replay-valid wins per hour, not successful
searches alone.

## 2. Setup

Search starts from the canonical Level 5 Spread + rapid-fire state and targets
the Level 5 to Level 6 transition. The Level 5 action table and button costs
match Level 1. A frozen bigram prior is counted from the two existing old-repo
wins and blended 10% toward uniform. Both sources replay from Level 5 into
Level 6; together they contain 5,360 actions. The trimmed table covers 5,339
actions, while 21 rare diagonal actions are excluded.

Stage 1 runs five interleaved attempts per arm, always running the Level 1
baseline first and deterministically shuffling the remaining arms. Hold eight
workers, frame skip 3, a 600-second time limit, and 6,000-action limit constant.

| arm | rollouts | rollout length | settle margin | max rewind |
|---|---:|---:|---:|---:|
| l1_fast | 16 | 24 | 8 | 15 |
| narrower | 8 | 24 | 8 | 15 |
| wider | 32 | 24 | 8 | 15 |
| shallower | 16 | 16 | 8 | 15 |
| deeper | 16 | 32 | 8 | 15 |

Working results, summaries, and replay-validation traces live under
`tmp/level5-search-efficiency-screen/`. Later stages must be added to this same
document before they run.

Stage 2 follows the 5/5 Stage 1 winner `8/24/8/15` and runs five attempts for
each breadth/lookahead arm under
`tmp/level5-search-efficiency-breadth-lookahead/`.

| arm | rollouts | rollout length | settle margin | max rewind |
|---|---:|---:|---:|---:|
| current_winner | 8 | 24 | 8 | 15 |
| rollouts_4 | 4 | 24 | 8 | 15 |
| rollouts_6 | 6 | 24 | 8 | 15 |
| rollouts_12 | 12 | 24 | 8 | 15 |
| length_20 | 8 | 20 | 8 | 15 |
| length_28 | 8 | 28 | 8 | 15 |

Stage 3 anchors rollouts and rollout length to the measured Stage 2 winner and
runs five attempts for each backtracking arm under
`tmp/level5-search-efficiency-settle-rewind/`.

Stage 2 selected `4/24`: it produced 5/5 replay-valid wins, zero duplicates,
and 152.48 wins/hour. The runner-up `8/28` produced 5/5 and 151.05 wins/hour.

| arm | settle margin | max rewind |
|---|---:|---:|
| stage2_winner | 8 | 15 |
| rewind_8 | 8 | 8 |
| rewind_12 | 8 | 12 |
| rewind_24 | 8 | 24 |
| settle_4 | 4 | 15 |
| settle_12 | 12 | 15 |

## 3. Evaluation metrics

For every arm report attempts, search wins, replay-valid Level 6 transitions,
wins/hour, mean wall seconds per valid win, and exact duplicate count. Attempt
rows are the source of truth in
`tmp/level5-search-efficiency-screen/results.jsonl`; the resumable driver writes
the derived `summary.json` after every attempt.

| source claim | provenance |
|---|---|
| Level 1 winner is `16/24/8/15` | `doc/0005-exp-l1-search-efficiency.md` |
| old MC win has 2,442 actions | `contra_agent/tmp/mc_trace_old/level5/win_level5_202606181019.npz` audit |
| old human win has 2,918 actions | `contra_agent/contra/human_recordings/Level5/03281906.npz` audit |
| both traces replay Level 5 to Level 6 | stable-retro replay audit on 2026-08-28 |

## 4. Conclusion

_Pending user conclusion after the measured sweep._
