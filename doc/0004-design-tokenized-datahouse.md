# Token-shard datahouse

Status: Implemented

## Decision

Store concrete encoded token shards by game taxonomy, not release name:

```
game_trace/datahouse/
  catalog.sqlite
  encoder/<sha256>/{encoder.pt,spec.json}
  level1/{boss,kill,full}/
  level2/... through level8/...
  level1/boss/{spread,laser,regular}/token-00000.tar
```

The datahouse owns the encoder and every token shard. A shard contains only
episode tokens, action indices, and episode metadata. `catalog.sqlite` records
each immutable shard's taxonomy, encoder digest, hash, episode/frame counts, and
the source trace and episode fingerprint of every member.

Production is two-phase per shard. CPU replay first writes resized RGB frames to
a resumable temporary staging directory. The emulator then closes; a CUDA-only
consumer batches those staged images, atomically publishes and catalogs the token
tar, and deletes that shard's staging directory. A crash exposes neither a partial
tar nor a catalog row, and temporary disk use is bounded to one shard.

Policy queries the catalog for a taxonomy and episode budget. For example,
`level1/boss/{spread,laser,regular}`, `episodes=40000` returns deterministic
whole shards in catalog order, with their paths and hashes. Policy reads them in place;
it owns any train/validation selection and creates no token cache or shard copy.

## Catalog contract

`shards` has one row per tar: path, SHA-256, taxonomy, encoder digest,
ordinal, episodes, and frames. `episodes` maps a stable fingerprint and UID to
one source trace. `shard_episodes` maps each member to its shard. Transactions
make a tar and its catalog entries visible together only after hash validation.

Shard ordinals are append-only within `(level, task, weapon, encoder)`.
The policy may request an exact whole-shard prefix or a minimum budget; the
catalog returns the selected count so experiments cannot silently consume more
or fewer episodes than intended.

## Provenance

| claim | source |
|---|---|
| action/state identity defines an episode | `task_maker.boss_release.task_fingerprint` |
| data must own ROM-derived task/shard formats | repository `CLAUDE.md` |
