# Level 4 search efficiency

Status: Proposed

## 1. Goal

Find a high-throughput Monte Carlo configuration for replay-valid full Level 4
wins on the eight-core cloud2 worker. First measure the production Level 2
setup as the Level 4 baseline; later stages can change rollouts, lookahead, and
rewind against that rate.

## 2. Setup

Cloud2 runs every stage with eight emulator workers, a 600-second per-attempt
limit, a 6,000-action limit, `level_up` goal, frame skip 3, and seed
`20260824`. Level 4 starts from `src/agent/states/spread_gun/Level4.state`.
The action table, fire/jump costs, and prior are `src/agent/level4.yaml`.
Search stages borrowed `src/agent/priors/level2.yaml`. After stage 5 the
adopted production search is 16 rollouts, rollout length 48, settle margin 8,
and max rewind 15. The Level 4 prior is `src/agent/priors/level4.yaml`, built
from 51 unique winning experiment NPZs on cloud2
(`source_set_sha256` `21de845e…ede44a5c`).

Stage 1 is the adopted Level 2 production search from
`doc/0011-exp-level2-search-efficiency.md`. It planned ten attempts and
stopped after five because every attempt was a replay-valid win and stage 2
started. Rows are `tmp/level4-search-efficiency-baseline/results.jsonl` and
`summary.json` on cloud2.

| arm | rollouts | rollout length | settle margin | max rewind |
|---|---:|---:|---:|---:|
| l2_production | 64 | 48 | 8 | 30 |

Stage 2 holds settle margin 8 and max rewind 30. It planned five interleaved
attempts per arm, baseline first each round, other arms shuffled with the
stage-1 seed. It stopped after six completed rows (one full round plus a
second baseline) because narrower led and shallower already failed, so
stage 3 moved toward the Level 1 thin shapes. Driver:
`python -m util.benchmark_l4_search --stage scale-up`. Rows:
`tmp/level4-search-efficiency-scale-up/results.jsonl` and `summary.json` on
cloud2.

| arm | rollouts | rollout length | settle margin | max rewind |
|---|---:|---:|---:|---:|
| baseline | 64 | 48 | 8 | 30 |
| wider | 96 | 48 | 8 | 30 |
| narrower | 32 | 48 | 8 | 30 |
| deeper | 64 | 64 | 8 | 30 |
| shallower | 64 | 32 | 8 | 30 |

Stage 3 is the opposite of wider/deeper: Level 1–style fewer rollouts and
shorter lookahead, five attempts per arm, `few_long` first each round. It
stopped after seven completed rows because 24-step arms were not clearing
and `few_long` was the only reliable win. Driver:
`python -m util.benchmark_l4_search --stage l1-shape`. Rows:
`tmp/level4-search-efficiency-l1-shape/results.jsonl` and `summary.json` on
cloud2.

| arm | rollouts | rollout length | settle margin | max rewind |
|---|---:|---:|---:|---:|
| few_long | 16 | 48 | 8 | 30 |
| l1_fast | 16 | 24 | 8 | 15 |
| l1_more | 32 | 24 | 8 | 15 |
| few_mid | 16 | 32 | 8 | 15 |

Stage 4 confirms the 48-step keepers only: `few_long` 16/48, `narrower`
32/48, and `baseline` 64/48, five interleaved attempts, `few_long` first.
Driver: `python -m util.benchmark_l4_search --stage confirm`. Rows:
`tmp/level4-search-efficiency-confirm/results.jsonl` and `summary.json` on
cloud2.

| arm | rollouts | rollout length | settle margin | max rewind |
|---|---:|---:|---:|---:|
| few_long | 16 | 48 | 8 | 30 |
| narrower | 32 | 48 | 8 | 30 |
| baseline | 64 | 48 | 8 | 30 |

Stage 5 holds `few_long` rollouts 16 and length 48 and changes one of settle
margin or max rewind. Five interleaved attempts, `few_long` (`8/30`) first.
Driver: `python -m util.benchmark_l4_search --stage settle-rewind`. Rows:
`tmp/level4-search-efficiency-settle-rewind/results.jsonl` and `summary.json`
on cloud2.

| arm | rollouts | rollout length | settle margin | max rewind |
|---|---:|---:|---:|---:|
| few_long | 16 | 48 | 8 | 30 |
| settle_4 | 16 | 48 | 4 | 30 |
| settle_16 | 16 | 48 | 16 | 30 |
| rewind_15 | 16 | 48 | 8 | 15 |
| rewind_45 | 16 | 48 | 8 | 45 |

## 3. Evaluation metrics

Stage 1 numbers come from cloud2
`tmp/level4-search-efficiency-baseline/summary.json` after five completed
attempts (420.80 s total wall).

| arm | valid wins / attempts | wins/hour | seconds/valid win | duplicates |
|---|---:|---:|---:|---:|
| l2_production | 5/5 | 42.78 | 84.16 | 0 |

Stage 2 numbers come from cloud2
`tmp/level4-search-efficiency-scale-up/summary.json` after nine completed
rows.

| arm | valid wins / attempts | wins/hour | seconds/valid win | duplicates |
|---|---:|---:|---:|---:|
| narrower | 2/2 | 57.80 | 62.28 | 0 |
| baseline | 2/2 | 42.49 | 84.73 | 0 |
| deeper | 2/2 | 39.79 | 90.47 | 0 |
| wider | 1/1 | 38.46 | 93.61 | 0 |
| shallower | 1/2 | 12.43 | 289.73 | 0 |

Stage 3 numbers come from
`tmp/level4-search-efficiency-l1-shape/summary.json` after seven completed
rows. 24-step arms have no valid win.

| arm | valid wins / attempts | wins/hour | seconds/valid win | duplicates |
|---|---:|---:|---:|---:|
| few_long | 2/2 | 68.57 | 52.50 | 0 |
| few_mid | 1/2 | 19.83 | 181.53 | 0 |
| l1_more | 0/2 | 0.00 | — | 0 |
| l1_fast | 0/1 | 0.00 | — | 0 |

Stage 4 numbers come from
`tmp/level4-search-efficiency-confirm/summary.json` after 15 completed rows.

| arm | valid wins / attempts | wins/hour | seconds/valid win | duplicates |
|---|---:|---:|---:|---:|
| few_long | 4/5 | 38.30 | 94.00 | 0 |
| baseline | 4/5 | 25.68 | 140.17 | 0 |
| narrower | 3/5 | 12.73 | 282.89 | 0 |

Stage 5 numbers come from
`tmp/level4-search-efficiency-settle-rewind/summary.json` after 25 completed
rows (five attempts per arm).

| arm | valid wins / attempts | wins/hour | seconds/valid win | duplicates |
|---|---:|---:|---:|---:|
| rewind_15 | 5/5 | 98.39 | 36.59 | 0 |
| few_long | 5/5 | 71.34 | 50.46 | 0 |
| settle_4 | 4/5 | 56.93 | 63.23 | 0 |
| settle_16 | 3/5 | 34.27 | 105.05 | 0 |
| rewind_45 | 2/5 | 23.87 | 150.84 | 0 |

## 4. Conclusion

_Pending — user drafts this section._
