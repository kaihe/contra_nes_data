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
rows are the source of truth; the resumable driver writes each derived
`summary.json` after every attempt. Every measured arm below produced 5/5
search wins, 5/5 replay-valid transitions, and zero exact duplicates.

| Stage 1 arm | mean seconds/win | wins/hour |
|---|---:|---:|
| narrower (`8/24/8/15`) | 23.07 | 156.05 |
| deeper (`16/32/8/15`) | 32.02 | 112.44 |
| l1_fast (`16/24/8/15`) | 34.92 | 103.09 |
| shallower (`16/16/8/15`) | 50.16 | 71.76 |
| wider (`32/24/8/15`) | 55.49 | 64.87 |

| Stage 2 arm | mean seconds/win | wins/hour |
|---|---:|---:|
| rollouts_4 (`4/24/8/15`) | 23.61 | 152.48 |
| length_28 (`8/28/8/15`) | 23.83 | 151.05 |
| current_winner (`8/24/8/15`) | 25.06 | 143.67 |
| rollouts_6 (`6/24/8/15`) | 29.21 | 123.26 |
| length_20 (`8/20/8/15`) | 30.35 | 118.62 |
| rollouts_12 (`12/24/8/15`) | 35.15 | 102.42 |

| Stage 3 arm | mean seconds/win | wins/hour |
|---|---:|---:|
| rewind_8 (`4/24/8/8`) | 17.45 | 206.35 |
| settle_4 (`4/24/4/15`) | 19.83 | 181.57 |
| rewind_12 (`4/24/8/12`) | 21.30 | 169.04 |
| stage2_winner (`4/24/8/15`) | 22.57 | 159.50 |
| rewind_24 (`4/24/8/24`) | 33.88 | 106.26 |
| settle_12 (`4/24/12/15`) | 55.58 | 64.77 |

| source claim | provenance |
|---|---|
| Level 1 winner is `16/24/8/15` | `doc/0005-exp-l1-search-efficiency.md` |
| old MC win has 2,442 actions | `contra_agent/tmp/mc_trace_old/level5/win_level5_202606181019.npz` audit |
| old human win has 2,918 actions | `contra_agent/contra/human_recordings/Level5/03281906.npz` audit |
| both traces replay Level 5 to Level 6 | stable-retro replay audit on 2026-08-28 |
| Stage 1 table | `tmp/level5-search-efficiency-screen/{results.jsonl,summary.json}` |
| Stage 2 table | `tmp/level5-search-efficiency-breadth-lookahead/{results.jsonl,summary.json}` |
| Stage 3 table | `tmp/level5-search-efficiency-settle-rewind/{results.jsonl,summary.json}` |

## 4. Conclusion

_Pending user conclusion after the measured sweep._
