# Add a train-only boss-search curriculum without changing validation

Status: Proposed

**Question.** How should level-1 boss demonstrations be diversified without
changing the frozen validation set or losing the source-trace split boundary?

**Answer.** Search from save-states replayed from the existing 466 train boss
tasks. Select a source task first, use its reveal state 30% of the time and a
uniform decision offset within its fight 70% of the time, and optimize boss HP
damage with forward-scroll reward disabled during the boss scene. Save each win
as an additive raw trace and replayable boss task whose `src_trace` and `split`
remain those of the original train example. Never sample or rewrite validation.

---

## 1. Why

The dataset has 523 level-1 boss tasks from 523 distinct source traces: 466
train and 57 validation. Different traces currently provide execution diversity
but only one search objective and therefore one broad strategy. The learned
policy completes the approach, then usually dies near the approach-to-engage
transition. Re-searching full fights explores strategies on the measured start
distribution; partial starts create a reverse curriculum with less HP and less
survival time remaining.

The validation set is a published comparison contract. Its 57 boss tasks are
part of the fixed 846-task evaluation suite and must remain byte-for-byte
unchanged.

## 2. The design

### Sampling and provenance

The input unit is an existing `boss_level1` task, not an arbitrary frame pooled
across all tasks. Inputs with `split != "train"` are rejected before sampling.
For every requested search:

1. sample a source task uniformly;
2. choose offset zero with probability `full_fraction` (default 0.3), otherwise
   choose a decision offset uniformly after full reveal, excluding the final
   eight-decision transition tail;
3. replay the task to the start of that decision and capture the emulator state;
4. search from the captured state to the unchanged level-clear predicate;
5. save a raw trace under `game_trace/mc_trace/boss_level1/` and a task under a
   separate staging output, then merge the verified task additively.

`src_trace` remains the original root trace name and `split` is copied as
`train`; generated filenames never participate in split hashing. The generated
UID contains the source UID, offset and search instance so independent searches
cannot overwrite each other.

### Search objective

`env.utility.boss_hp(ram)` sums HP in active boss-objective slots. The ordinary
search reward pays boss HP decrements and suppresses `push_right` once
`boss_scene` begins. The boss driver uses the same action sampler, frame skip,
backtracking and level-clear goal as normal `mc_search`, but supplies an
arbitrary initial emulator state.

### Task and export contract

Every generated boss task preserves the existing keys and adds:

| key | meaning |
|---|---|
| `weapon` | interpreted weapon name at the sampled start |
| `rapid` | whether weapon RAM bit 4 is set |
| `boss_hp_start` | total active boss-objective HP at the sampled start |
| `offset_frac` | source decision offset divided by source fight length |

The same four fields pass through to HF JSON. `KillBossMaker.boss_hp(ram)` is a
public accessor delegating to the RAM helper so policy reward shaping needs no
`ADDR_*` knowledge.

### Acceptance contract

- Existing 523 boss task files remain byte-identical.
- The boss validation set remains exactly 57 tasks and the full validation set
  remains exactly 846 tasks.
- Every new task is `split=train`, names an original train `src_trace`, retains
  the source `skip`, and replays to the existing success predicate.
- A pilot reports search win rate and the action/weapon/offset distribution
  before a large generation run is accepted.

## 3. What was rejected, and why

**Sampling frames globally.** This weights a source in proportion to its fight
length; current median lengths differ substantially by weapon. Trace-first
sampling makes the weighting explicit and preserves the existing source mix.

**Hashing generated trace filenames for the split.** A new filename can hash to
validation even when its state was derived from training. The root source trace
is the data lineage and therefore remains the split key.

**Weakening `extract_boss` to accept any trace that starts in a boss scene.** A
mid-fight trace has no reveal rising edge, and silently changing the general
extractor would blur raw-trace and task provenance. The boss-search driver emits
an explicit task alongside its raw trace instead.

**Replacing the existing boss tasks.** Existing tasks anchor all published
evaluation comparisons. New tasks are additive and train-only.

## 4. Risks and gates

| risk | why it is plausible | gate |
|---|---|---|
| apparent diversity is only action noise | all searches share a sampler | pilot action profiles and lengths differ across independent searches |
| partial starts contain no active boss HP | reveal timing and component slots vary | reject starts with `boss_hp_start <= 0` |
| generated data leaks validation | filenames and derived states can obscure lineage | all inputs and outputs assert original `split=train`; frozen hashes unchanged |
| search wins do not replay | skip/state/action alignment is fragile | every accepted task passes `KillBossMaker.verify_segment` |
| easier partial tasks inflate metrics | offset changes difficulty | export `boss_hp_start` and `offset_frac` for stratified reporting |

## 5. Sequencing

1. Add RAM accessors, metadata export and tests.
2. Generalize `mc_search` to accept a supplied initial state without changing
   its default level-start behavior.
3. Add the train-only boss sampling/generation driver and provenance tests.
4. Run a small replay-verified pilot; inspect win rate and diversity.
5. Generate and export the accepted batch, then hand the new shard/API contract
   to policy through a GitHub issue.

## Appendix — provenance

| claim | source |
|---|---|
| 523 boss tasks, 466 train and 57 val, all distinct sources | scan of `game_trace/tasks/boss/boss_level1/*.npz` on 2026-08-03 |
| validation total is fixed at 846 | `contra_nes_evaluation/doc/0008-grpo-with-boss.md` |
| policy fails around the approach/engage transition | `contra_nes_data` issue #2 and `contra_nes_policy/doc/0004-grpo-experiment-plan.md` |
| existing median boss length is about 140 decisions | scan of the same 523 boss task files on 2026-08-03 |
