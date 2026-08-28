# Level 3 search starts from a Spread checkpoint near the first ladder

Status: Implemented

**Question.** How should distributed Level 3 search avoid the unrewarded opening
approach while preserving the carried weapon and recording reproducible lineage?

**Answer.** Level 3 search uses the manually captured `Level3-spread-right`
checkpoint: player x=161 near the first ladder, Spread gun with rapid fire. A
frozen bigram prior from 14 old wins uses 0.1 uniform smoothing. Production runs
the measured winner `32/48/16/60` (rollouts/length/settle/rewind). Other levels
and canonical replay states remain unchanged.

---

## The search-only checkpoint preserves Spread and canonical replay

`src/agent/states/search_start/Level3-spread-right.state` is a gzip-compressed
stable-retro state captured interactively from the canonical Level 3 Spread
state. It replaces the regular-gun frame-40 checkpoint as the Level 3 search
default, but does not replace `src/agent/states/spread_gun/Level3.state` used by
replay, task extraction, and policy environments.

The adjacent manifest pins the raw emulator-state checksum, capture facts, and
`trace_scope: checkpoint_suffix`. `mc_search` rejects missing, unlisted, or
checksum-mismatched state files. Generated traces retain the checkpoint filename,
checksum, and manifest metadata.

| field | value |
|---|---:|
| raw state SHA-256 | `bbe37b1359f2fafa7f35b71a013a41f8c901c01f18f30477c120c06cbde0b9c7` |
| level / lives | 3 / 3 |
| player position | x=161, y=180 |
| weapon byte | 19 (Spread + rapid fire) |
| trace scope | checkpoint suffix |

## A frozen smoothed prior makes the checkpoint searchable

`src/agent/priors/level3.yaml` stores integer transition counts from the 14 old
Level 3 MC wins under `contra_agent/tmp/mc_trace_old/level3/`: 64,412 actions,
64,398 candidate pairs, and no out-of-table transitions. Its action table is the
21-action baseline table. The artifact records `smooth: 0.1`; workers normalize
counts and blend each row with 10% uniform probability so unseen checkpoint
states retain exploration.

The old trace files are build inputs, not deployment inputs. Workers load only
the committed YAML artifact and verify its action-table digest and prior digest.
With the new checkpoint but a uniform prior, search timed out at 600 seconds;
the smoothed win prior produced a win in 190.4 seconds with otherwise matching
settings.

## The measured production winner is 32/48/16/60

Three local parameter sweeps used the same Spread checkpoint, prior, 32 workers,
600-second limit, and 6,000-action limit. The selected setup uses 32 rollouts,
48-action rollout length, settle margin 16, and maximum rewind 60.

| setup | wins | mean wall time | median |
|---|---:|---:|---:|
| 64/48/16/60 | 5/5 | 153.1 s | 165.6 s |
| **32/48/16/60** | **5/5** | **90.2 s** | **83.7 s** |
| 64/64/16/60 | 5/5 | 107.1 s | 97.2 s |
| 32/56/16/60 | 5/5 | 97.1 s | 90.6 s |
| 32/64/16/60 | 5/5 | 109.7 s | 114.4 s |

Reducing breadth below 32 caused one timeout at both 16 and 24 rollouts in the
length-64 sweep. Increasing length did not beat 32×48. Production therefore
keeps `32/48/16/60` and targets 25,000 committed unique Level 3 traces.

## Provenance remains auditable across repositories and workers

| claim | source |
|---|---|
| checkpoint capture and raw checksum | `contra_agent/tmp/state_capture/Level3-20260825_142219.state` |
| prior has 14 wins and 64,412 actions | `contra_agent/tmp/mc_trace_old/level3/*.npz` audit |
| uniform-prior timeout and 190.4-second prior win | `contra_agent/tmp/mc_baseline/` logs |
| 32×48 winner is 5/5 at 90.2 seconds | `contra_agent/tmp/mc_baseline/level3_rewind60_sweep.jsonl` |
| interaction sweep does not beat winner | `contra_agent/tmp/mc_baseline/level3_stage3_sweep.jsonl` |

Before fleet launch, one worker must verify the checkpoint checksum, non-uniform
smoothed prior, exact `32/48/16/60` command, and one successful canary upload.
