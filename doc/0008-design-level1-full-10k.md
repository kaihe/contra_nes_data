# Encode 10k full Level 1 traces once and expose three sequence views

Status: Proposed

**Question.** How should the data repository turn 10,000 full Level 1 traces in
GCS into policy-ready shards with precomputed image tokens, while allowing an
experiment to load only the approach to the boss, only the boss fight, or the
entire level without storing the episode three times?

**Answer.** Freeze a deterministic 10k source collection from committed
`level1/full` GCS batches, replay and encode every full trace once, and publish
immutable token shards under `datahouse/level1/full/mixed`. Each episode stores
one observation-token array, one action array, and a verified boss-entry
boundary. The reader exposes `start_to_boss`, `boss_fight`, and `full` as virtual
slices over those arrays. A versioned collection manifest and `catalog.sqlite`
record source identity, shard membership, encoder identity, boundaries, and
counts. Policy chooses a view and train/validation membership but creates no
token cache or shard copy.

---

## Frozen 10k source collection

The builder lists only GCS batches with a valid `COMMITTED.json` beneath
`schema-v1/level1/full`. At the start of the build it freezes their object
generations in `l1-full-10k-v1.json`, deduplicates trace fingerprints, and selects
the 10,000 smallest fingerprints. Hash ordering makes selection independent of
worker, filename, upload order, and later arrivals. The manifest records the
eligible snapshot, selected fingerprint, source object/member, NPZ SHA-256, boss
weapon, rapid-fire flag, and initial-state identity.

This repository does not create a train/validation split. Weapon proportions are
measured and recorded, not balanced by duplicating or resampling episodes.

## One encoding with three aligned sequence views

Replay produces environment observations `o[0:T+1]` and actions `a[0:T]`.
The builder detects the first boss-scene observation during replay and stores its
index as `boss_observation_index = b`; it does not assume the legacy
`boss_entry_step` has observation-index semantics. The three views are:

| view | observations | target actions | instruction label |
|---|---|---|---|
| `start_to_boss` | `o[0:b+1]` | `a[0:b]` | reach the Level 1 boss |
| `boss_fight` | `o[b:T+1]` | `a[b:T]` | defeat the Level 1 boss |
| `full` | `o[0:T+1]` | `a[0:T]` | complete Level 1 |

The boundary observation belongs to both partial views: it is the terminal
context for approaching the boss and the initial context for fighting it. Every
view must satisfy `observation_count = action_count + 1`. Traces that never enter
the boss scene or fail alignment are quarantined and replaced from the frozen
eligible snapshot before the collection manifest is finalized.

`view` is a scalar projection chosen once for a dataset reader, not a request to
expand each episode into three examples. The reader rejects multiple views in one
request. In particular, `full` is mutually exclusive with either partial view,
so a normal training dataset cannot contain the whole episode and its component
segments at the same time. Separate experiments may intentionally choose
different views, but their manifests have distinct `(collection, view)` identities.
If the two partial views are trained separately, their target-action ranges are
disjoint; only the boundary observation is shared as necessary context.

## Immutable token-shard members

Physical shards live at:

```text
game_trace/datahouse/
  encoder/<encoder_sha256>/{encoder.pt,spec.json}
  level1/full/mixed/token-<ordinal>.tar
  collections/l1-full-10k-v1.json
  catalog.sqlite
```

Each episode contributes `<uid>.tokens.npy`, `<uid>.actions.npy`, and
`<uid>.json`. Tokens contain only the encoded environment observations; actions
use the data-owned baseline vocabulary. Episode JSON carries both lengths,
`boss_observation_index`, the three instruction labels, source provenance,
weapon metadata, and encoder digest. Policy-specific BOS/EOS and text-token IDs
remain policy-owned; the data reader returns the semantic instruction label,
image-token slice, and aligned action targets.

Shards are frame-balanced to bound replay, staging, and GPU memory. An episode is
never split across shards. The 10k collection manifest lists exact shard hashes
and episode members, so later 20k/40k collections can reuse existing shards
instead of copying their first 10k episodes.

## Streaming GCS-to-GPU construction

The CPU producer downloads one required GCS archive at a time, verifies the
committed object generation, archive hash, member hash, and frozen selection,
then replays selected NPZs into a bounded temporary RGB stage. It deletes the raw
archive after its selected members are staged. The emulator closes before a
CUDA-only consumer loads the datahouse-owned encoder, batch-encodes frames, and
atomically publishes the token tar.

After the tar hash and catalog transaction succeed, staged RGB files are deleted.
A journal records downloaded members, replay boundaries, encoded episodes, and
published shards, making every phase restartable without exposing partial data.

## Catalog-backed no-copy loading

The catalog adds collection membership and per-episode
`boss_observation_index`, observation count, action count, and source GCS
identity. A data-owned reader accepts:

```text
collection=l1-full-10k-v1
view=start_to_boss | boss_fight | full
```

It requires exactly one `view`, resolves immutable shard paths, verifies hashes,
and slices members according to the table above. The resulting dataset identity
is `(collection, view, episode subset)`, and the reader never materializes view
files. The policy repository may choose any episode subset for training or
validation, but it reads the same shards in place and does not write image-token
caches or shard-index JSON files.

## Publication gates and recovery

Publication requires exactly 10,000 unique fingerprints; zero missing boss
boundaries; zero token/action alignment failures; matching encoder and shard
hashes; and successful reconstruction of all three views for every episode.
`catalog.sqlite` and the collection manifest become visible only after all shards
pass. Failed or interrupted builds retain their journal and bounded staging data
but never replace an existing ordinal or collection version.

## Provenance and auditability

| claim | source |
|---|---|
| committed GCS batches are the raw visibility boundary | `doc/0007-design-distributed-trace-ingest.md` |
| full traces record boss entry and weapon metadata | `src/worker/search_loop.py` |
