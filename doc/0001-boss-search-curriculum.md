# Add a train-only boss-search curriculum without changing validation

Status: Proposed

**Question.** How should level-1 boss demonstrations be diversified without
changing the frozen validation set or losing the source-trace split boundary?

**Answer.** Materialize a small, reproducible bank of train-derived full-fight
boss savestates, then run ordinary `mc_search` directly from those files. The
bank contains one reveal start for each observed weapon class and records the
root task, split, HP, loadout and checksum in a manifest. Generated traces enter
training only after replay verification and diversity filtering, then ship in a
versioned, frame-balanced train release. Never sample or rewrite validation.

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

The first-stage bank lives in `src/agent/states/boss_level1/`, matching the
gzip-compressed format used by the per-level Spread states. For each observed
weapon class, the median-length train source contributes its reveal state.
`mc_search --initial-state FILE` loads the state, verifies its manifest
checksum, and copies its lineage metadata into the raw winning trace under
`game_trace/mc_trace/boss_level1/`.

This fixed full-fight bank is the experiment surface. Partial starts are not
part of this release: they change the task difficulty and can make apparent
diversity come from the start state instead of the strategy.

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

### Candidate acceptance and shard releases

Raw `mc_search` files are candidates, not training samples. A release builder:

1. matches each trace to a checksummed full-fight state-bank entry;
2. replays it through the unchanged boss-clear predicate and writes a train-only
   replayable task into a release staging directory;
3. computes action and replay-state diversity against both the candidate batch
   and the existing train set, rejecting exact duplicates and reporting nearest
   neighbours rather than hiding the threshold;
4. assigns accepted tasks deterministically to shards by total decision frames,
   with weapon/start strata distributed across shards;
5. materializes self-contained WebDataset tar files and writes a JSON manifest
   containing task IDs, hashes, counts, frame totals and the frozen validation
   shard hash.

Candidate releases live outside the production shard directory. Promotion
creates a versioned directory rather than overwriting the existing shard. The
57-example validation tar is copied byte-for-byte; it is never re-exported.
Train shard targets are expressed in frames (default 60,000), not episodes,
because boss trace lengths vary substantially. Very small delta shards are not
mixed with baseline shards implicitly: shard-uniform consumers could otherwise
oversample the generated data.

### Acceptance contract

- Existing 523 boss task files remain byte-identical.
- The boss validation set remains exactly 57 tasks and the full validation set
  remains exactly 846 tasks.
- Every new task is `split=train`, names an original train `src_trace`, retains
  the source `skip`, and replays to the existing success predicate.
- A release manifest proves the validation tar hash and lists the exact accepted
  task and shard hashes.
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

**Partial starts in the full-fight release.** They provide useful curriculum
data but confound strategy diversity with reduced HP and elapsed fight state.
This release accepts only `stage=full`; partial starts may be evaluated later as
a separately named dataset configuration.

## 4. Risks and gates

| risk | why it is plausible | gate |
|---|---|---|
| apparent diversity is only action noise | all searches share a sampler | pilot action profiles and lengths differ across independent searches |
| generated data leaks validation | filenames and derived states can obscure lineage | all inputs and outputs assert original `split=train`; frozen hashes unchanged |
| search wins do not replay | skip/state/action alignment is fragile | every accepted task passes `KillBossMaker.verify_segment` |
| variable lengths make episode-balanced shards uneven | long traces dominate bytes and tokens | deterministic frame-balanced assignment |
| small generated shard is oversampled | some loaders sample shards uniformly | versioned release + explicit manifest; no implicit tiny delta |

## 5. Sequencing

1. Add RAM accessors, metadata export and tests.
2. Generalize `mc_search` to accept a supplied initial state without changing
   its default level-start behavior.
3. Add the train-only boss sampling/generation driver and provenance tests.
4. Build the fixed full-fight state bank and run `mc_search` from every entry;
   inspect win rate, latency and diversity.
5. Import replay-verified candidates, measure nearest-neighbour diversity, and
   select the accepted full-fight task set.
6. Export a versioned, frame-balanced release and hand its manifest/API contract
   to policy through a GitHub issue.

## 6. Execution update (2026-08-03)

The initial `k=1` batch was stopped after 59 completed full-fight outputs. Those
trace/task pairs remain preserved, but no further bulk requests are running.
The experiment pivoted to the fixed state bank above because it gives direct,
repeatable `mc_search` starting points and makes search behavior measurable
before committing compute to 1,553 requests.

The first bank contained eight states: full and partial starts for
Flamethrower, Laser, Regular and Spread. That mixed-start design was retired
before production generation. The active bank keeps only the four full-fight
states so every candidate solves the same task horizon.

## Appendix — provenance

| claim | source |
|---|---|
| 523 boss tasks, 466 train and 57 val, all distinct sources | scan of `game_trace/tasks/boss/boss_level1/*.npz` on 2026-08-03 |
| validation total is fixed at 846 | `contra_nes_evaluation/doc/0008-grpo-with-boss.md` |
| policy fails around the approach/engage transition | `contra_nes_data` issue #2 and `contra_nes_policy/doc/0004-grpo-experiment-plan.md` |
| existing median boss length is about 140 decisions | scan of the same 523 boss task files on 2026-08-03 |
