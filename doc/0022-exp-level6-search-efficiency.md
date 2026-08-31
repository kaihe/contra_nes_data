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

Build a frozen bigram prior from the replay-verified old MC and human Level 6
wins, then blend it 10% toward uniform. The two sources contain 4,750 actions
and 4,739 in-table adjacent pairs. Pin the action-table digest in the prior so
an ordering or vocabulary mismatch fails before search starts.

Earlier one-attempt scouting with the correct table found replay-valid wins
for `64/48/16/30` (182 seconds) and `96/48/16/60` (559 seconds). The old
`64/48/16/60` baseline and four lower-compute variants did not win in their
single attempts. Treat these as routing evidence, not final performance.

Run five interleaved attempts for each arm below with eight workers, frame skip
3, a 600-second time limit, and a 6,000-action limit. The grid isolates rollout
count and rewind depth, then tests their interaction. Run `scout_winner` first
in every round and deterministically shuffle the remaining arms.

| arm | rollouts | rollout length | settle margin | max rewind | purpose |
|---|---:|---:|---:|---:|---|
| scout_winner | 64 | 48 | 16 | 30 | measured anchor |
| rewind_15 | 64 | 48 | 16 | 15 | cheaper backtracking |
| rewind_45 | 64 | 48 | 16 | 45 | bracket rewind optimum |
| rollouts_32 | 32 | 48 | 16 | 30 | reduce compute around winner |
| rollouts_96 | 96 | 48 | 16 | 30 | breadth/rewind interaction |
| old_contra_baseline | 64 | 48 | 16 | 60 | historical control |

Working results, summaries, logs, and replay-validation traces belong under
`tmp/level6-search-efficiency-targeted/`. Resume by `(arm, attempt)` and retain
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

| source claim | provenance |
|---|---|
| 21-action vocabulary and order | `contra_agent/contra/action_configs/baseline.yaml` |
| old MC win has 1,796 actions | `contra_agent/tmp/mc_trace_old/level6/win_level6_202606181024.npz` audit |
| old human win has 2,954 actions | `contra_agent/contra/human_recordings/Level6/03281909.npz` audit |
| both sources replay into Level 7 | stable-retro replay audit on 2026-08-28 |
| correct-table scout results | `tmp/level6-search-efficiency-full-actions-scout/results.jsonl` |
| targeted-grid metrics | `tmp/level6-search-efficiency-targeted/results.jsonl` |

## 4. Conclusion

_Pending user conclusion after the measured sweep._
