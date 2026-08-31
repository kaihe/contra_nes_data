# Level 6 search efficiency

Status: Proposed

## 1. Goal

Find a reliable, high-throughput full-clear Monte Carlo configuration for
Level 6. Search must use the old Contra Agent 21-action vocabulary; select the
production setup by replay-valid Level 7 transitions per hour.

## 2. Setup

Search starts from the canonical Level 6 Spread + rapid-fire state. Use the
full ordered 21-action table from
`contra_agent/contra/action_configs/baseline.yaml`:
`_`, `J`, `F`, `L`, `LJ`, `LF`, `R`, `RJ`, `RF`, `U`, `UJ`, `UF`, `D`,
`DJ`, `DF`, `UL`, `ULJ`, `ULF`, `UR`, `URJ`, and `URF`. In particular,
do not use the 15-action Level 5 table: the old Level 6 wins contain `DJ` and
`LJ`.

Match the reliable late-level setup restored by Contra Agent commit `9bbf0e1`:
lookahead selects the highest-scoring surviving rollout whenever one exists and
rewinds only when every rollout dies. Use only `F` and `J` costs of `-0.02`.
Build a frozen bigram prior from the replay-verified human Level 6 win and blend
it 5% toward uniform. The source contains 2,954 actions and 2,944 in-table
adjacent pairs. Pin the action-table digest in the prior so an ordering or
vocabulary mismatch fails before search starts.

Pre-fix one-attempt scouting with the correct table found replay-valid wins
for `64/48/16/30` (182 seconds) and `96/48/16/60` (559 seconds). The old
`64/48/16/60` baseline and four lower-compute variants did not win in their
single attempts. A later targeted run also timed out at `64/48/16/30`. These
rows used the defective fatal-rollout selection and are historical diagnostics,
not comparable experiment evidence.

Run five interleaved attempts for each arm below with eight workers, frame skip
3, a 600-second time limit, and a 6,000-action limit. The grid isolates rollout
count and rewind depth around the repaired old setup. Run
`old_contra_baseline` first in every round and deterministically shuffle the
remaining arms.

| arm | rollouts | rollout length | settle margin | max rewind | purpose |
|---|---:|---:|---:|---:|---|
| old_contra_baseline | 64 | 48 | 16 | 60 | repaired historical setup |
| rewind_30 | 64 | 48 | 16 | 30 | shallower backtracking |
| rewind_15 | 64 | 48 | 16 | 15 | cheaper backtracking |
| rewind_45 | 64 | 48 | 16 | 45 | bracket rewind optimum |
| rollouts_32 | 32 | 48 | 16 | 60 | reduced breadth |
| rollouts_96 | 96 | 48 | 16 | 60 | increased breadth |

Working results, summaries, logs, and replay-validation traces belong under
`tmp/level6-search-efficiency-survivor-fix/`. Resume by `(arm, attempt)` and retain
failed and timed-out rows. Do not extend the grid until all six arms reach five
attempts. If no arm produces at least 3/5 replay-valid wins, plan a separate
second stage around the failure locations instead of selecting a weak winner.

## 3. Evaluation metrics

For every arm report attempts, search wins, replay-valid Level 7 transitions,
success rate, total wall time, mean seconds per valid win, wins/hour, median
sampled actions per valid win, and exact duplicate count. Attempt rows are the
source of truth; regenerate the summary after every attempt. Rank arms by
replay-valid wins/hour, with replay-valid success rate as the reliability gate
and mean wall time as the tie-breaker.

The repaired sweep completed all 30 attempts. Every search win replayed into
Level 7, and the 27 valid traces had distinct fingerprints.

| arm | valid / attempts | success | total wall (s) | mean s/win | wins/hour | median sampled/win | duplicates |
|---|---:|---:|---:|---:|---:|---:|---:|
| rewind_15 | 5/5 | 100% | 314.61 | 62.92 | 57.21 | 219,393 | 0 |
| old_contra_baseline | 5/5 | 100% | 369.73 | 73.95 | 48.68 | 222,743 | 0 |
| rollouts_32 | 4/5 | 80% | 310.97 | 77.74 | 46.31 | 195,889 | 0 |
| rewind_45 | 5/5 | 100% | 433.33 | 86.67 | 41.54 | 275,357 | 0 |
| rollouts_96 | 5/5 | 100% | 541.69 | 108.34 | 33.23 | 396,761 | 0 |
| rewind_30 | 3/5 | 60% | 580.03 | 193.34 | 18.62 | 343,305 | 0 |

| source claim | provenance |
|---|---|
| 21-action vocabulary and order | `contra_agent/contra/action_configs/baseline.yaml` |
| old human win has 2,954 actions | `contra_agent/contra/human_recordings/Level6/03281909.npz` audit |
| human source replays into Level 7 | stable-retro replay audit on 2026-08-28 |
| survivor-first late-level repair | `contra_agent` commit `9bbf0e1` |
| pre-fix diagnostic scout | `tmp/level6-search-efficiency-full-actions-scout/results.jsonl` |
| pre-fix targeted timeout | `tmp/level6-search-efficiency-targeted/results.jsonl` |
| repaired-grid metrics | `tmp/level6-search-efficiency-survivor-fix/results.jsonl` |

## 4. Conclusion

_Pending user conclusion after the measured sweep._
