# Publish matched boss pixels beside token shards

Status: Proposed

**Question.** How does policy train an unfrozen image encoder on the exact Laser
D10k episodes used by the frozen L/D10k/C20k baseline without changing or
invalidating the token datahouse?

**Answer.** Publish native RGB and actions as a separate `png-mkv-v1` frame
release. Membership is the fingerprint set of a declared token-shard prefix:
Laser ordinals 0–17 for D10k. Frame shards have separate catalog tables, preserve
the token release unchanged, and record the post-action frame convention so the
policy loader must apply the causal shift explicitly.

---

## Separate frame catalog preserves token identity

Frame artifacts use `frame_shards` and `frame_shard_episodes`, not `shards`.
Existing token consumers require one encoder fingerprint per `(level, task,
weapon)` slice; inserting a sentinel encoder for pixels would make valid token
slices look stale. The frame tables instead discriminate representations by
`format`, record native geometry, and reference the existing episode fingerprint.

The initial Laser release selects token shards 0–17 before publication. This is
the same D10k membership consumed by the frozen Laser experiment. Identity is
therefore a catalog join rather than a filesystem ordering convention. The token
tars and their catalog rows are never rewritten.

## Native RGB episodes retain auditable action alignment

Each episode contributes:

```
{uid}.obs.mkv       native 224x240 RGB, all-intra PNG
{uid}.actions.npy   int64 indices in the 21-action baseline vocabulary
{uid}.json          fingerprint, source, counts, and per-frame hashes
```

The tar also contains `manifest.json` with member offsets, sizes, and hashes for
direct seeking. Frames are lossless, unresized, and contain no synthetic goal row.
The policy remains responsible for input resizing.

The trace materializer records the screen after applying the corresponding action.
Thus `frames[i]` is post-action state for `actions[i]`; a behavior-cloning reader
must shift the target rather than predict an action from its result. Publication
asserts equal frame/action counts, validates the action vocabulary, and decodes
every video to check frame count and boundary-frame hashes.

## Atomic shards make replay resumable

The producer resolves wanted fingerprints from the token prefix, subtracts already
cataloged frame fingerprints, and replays only the remainder. It writes each tar to
a process-specific temporary path, atomically renames it, then registers the shard
and episode rows in one transaction. A failed shard is absent from the catalog and
can be retried without rewriting completed shards.

Fixed episode-count shards are appropriate because consumers stream video rather
than memory-map tokens. Publication may run with multiple emulator workers, while
the parent alone writes tar files and catalog rows.

## Provenance and auditability

| claim | source |
|---|---|
| Laser D10k is the first 18 of 70 token shards | `game_trace/datahouse/catalog.sqlite`; policy `config_bc_laser.yaml` |
| frozen comparison is L/D10k/C20k | evaluation `doc/0029-result-laser-encoder-comparison.md` |
| token readers reject mixed encoder ids | policy `src/contra_policy/datahouse.py` |
| source task membership and frame counts | `episodes`, `shards`, and `shard_episodes` catalog tables |
