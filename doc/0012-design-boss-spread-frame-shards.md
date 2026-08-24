# 0012 — Boss Spread frame shards

Status: Proposed

The datahouse publishes Level-1 boss episodes as 512-D token shards. The ViT policy
direction needs the same episodes as pixels. This adds a raw-frame release beside the
token release, addressed by the same catalog, so a pixel run and a token run can be
proven to have seen the same episodes.

## A separate `frame_shards` table, not another encoder id

The obvious move — register MKV shards in `shards` under a sentinel
`encoder_sha256` — breaks every existing policy run. `contra_nes_policy`'s
`DatahouseTokens.__init__` selects on `(level, task, weapon)` **without** filtering by
encoder, then asserts a single encoder across the result:

```python
shas = {r[1] for r in rows}
if len(shas) != 1:
    raise StaleCache(f"shards for {self.slice} mix encoders: {sorted(shas)}")
```

A second `encoder_sha256` under `(1, 'boss', 'spread')` therefore turns every Spread
run into a load-time `StaleCache`. The check is correct and should not be relaxed:
mixing representations in one training set is exactly the error it exists to catch.

So frame shards get their own tables:

```sql
CREATE TABLE frame_shards (
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL UNIQUE,
  sha256 TEXT NOT NULL UNIQUE,
  level INTEGER NOT NULL CHECK(level BETWEEN 1 AND 8),
  task TEXT NOT NULL CHECK(task IN ('boss', 'kill', 'full')),
  weapon TEXT NOT NULL,
  format TEXT NOT NULL,          -- 'png-mkv-v1'
  frame_height INTEGER NOT NULL,
  frame_width INTEGER NOT NULL,
  ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
  episodes INTEGER NOT NULL CHECK(episodes > 0),
  frames INTEGER NOT NULL CHECK(frames >= 0),
  UNIQUE(level, task, weapon, format, ordinal)
);
CREATE TABLE frame_shard_episodes (
  shard_id INTEGER NOT NULL REFERENCES frame_shards(id) ON DELETE RESTRICT,
  fingerprint TEXT NOT NULL REFERENCES episodes(fingerprint) ON DELETE RESTRICT,
  ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
  PRIMARY KEY(shard_id, fingerprint),
  UNIQUE(shard_id, ordinal)
);
```

`format` replaces `encoder_sha256` as the representation discriminator, carrying the
same "do not mix these" role. Frame geometry is recorded per shard so a consumer can
reject a resized release without decoding one.

Existing queries touch only `shards`, so this is additive: no migration of existing
rows, and no change required in `contra_nes_policy` before it chooses to read frames.

## Episode set pinned to the D10k token prefix by fingerprint

`frame_shard_episodes.fingerprint` references the **existing** `episodes` table rather
than a private copy. Episode-set identity between the token and frame releases is then
a join, not a convention:

```sql
SELECT COUNT(*) FROM frame_shard_episodes f
JOIN shard_episodes s USING (fingerprint)
JOIN shards ON shards.id = s.shard_id
WHERE shards.weapon = 'spread' AND shards.ordinal < 13;
```

The initial release targets the set that `config_bc_mixed_d10.yaml` already trains on.
That config sets `shard_counts.spread: 13`, and the policy orders shards by
`(weapon, ordinal)`, so D10k Spread is the first 13 token shards:

```
episodes         9,815      (frame-balanced shards land under a round 10,000)
frames         768,976      78.3 frames/episode
```

Selecting by fingerprint rather than by re-globbing traces is what makes the release
reproducible. Trace glob order is filesystem-dependent; the catalog is not.

The token shards are **not** rebuilt. Re-replaying to emit both representations in one
pass would give the same guarantee, but it would rewrite the 13 shards that current
baselines were measured on, and the fingerprint join delivers the guarantee without
touching them.

## Native 224x240 frames with no goal row

`materialize()` returns `frames` from `env.unwrapped.get_screen()`, which is already
`(224, 240, 3)` — the overscan crop lives in the retro configuration, so no cropping
happens here. Frames are stored exactly as returned.

Two things the token producer does are deliberately **not** done:

- **No resize.** `boss_spread._stage_episode` applies `_resize(frames, spec.image_size)`
  to a square because that is the frozen encoder's input geometry. A frame release that
  bakes in one model's input size is a frame release that the next model cannot use.
  Downstream resizing is a consumer decision.
- **No goal image.** The token producer prepends `goal_img`, making token arrays
  `length + 1` and forcing every reader to remember `arr[1 + start]`. The frame release
  stores frames only, so index `i` is decision `i` with no offset to get wrong.

Dropping the goal row is what makes the next feature exact.

## Actions as 21-way baseline indices, one per frame

`materialize()` captures one screen per action — the post-action RAM snapshot and the
screen come from the same emulator state — so frames and actions are the same length.
The catalog agrees: `SUM(episodes.action_steps)` over the D10k prefix is 768,976,
equal to the shard frame total.

Actions are stored as `int64` indices into `src/agent/baseline.yaml` (21 entries), via
the same bit-packed lookup `boss_spread._action_indices` uses, so an index means the
same button combination in both releases. A trace containing an action outside the
vocabulary is an error, not a silent drop.

`frames[i]` is the screen **after** action `i` was applied. A behaviour-cloning
consumer predicting `actions[i]` from `frames[i]` is therefore predicting an action
from its own result and must shift; this is a property of the existing trace format,
recorded here so the shift is a decision rather than a bug.

## All-intra PNG in MKV with tar-offset member access

Each episode contributes three tar members, following `compressed_episodes.py`:

```
{uid}.obs.mkv       all-intra PNG-in-MKV, one entry per frame
{uid}.actions.npy   int64, shape (frames,)
{uid}.json          uid, fingerprint, frames, source_trace, per-frame sha256
```

A `manifest.json` member records every member's offset, size and sha256, so a reader
seeks to a member instead of scanning the tar. Episodes are decoded sequentially in
512-frame windows, which is what `CompressedEpisodeDataset` already does.

Size, measured on a published 4-episode shard rather than projected:

```
7,499 B/frame   ->  768,976 frames  ->  5.37 GiB
```

That is 29% denser than the 5,807 B/frame of the Level-1 corpus in `doc/0010`. Boss
frames are busier — more sprites, more distinct tiles, less flat background — and
`doc/PNG.md` shows this codec's output tracks tile variety closely. Projecting boss
storage from a traversal corpus would have under-budgeted by a gigabyte.

`doc/PNG.md` measures 4,397 B/frame for truecolor PNG with filtering disabled against
6,197 B for the adaptive default. If that 0.71 ratio carries, roughly 3.8 GiB is
available for identical pixels. It is not taken here: ffmpeg's PNG-in-MKV path does
not expose per-scanline filter choice, and changing encoders would break
byte-comparability with the existing corpus. Recorded as headroom, gated on storage
actually becoming a constraint.

Decode throughput measured on this hardware is **5,330 frames/s**, flat past four
workers. Per-frame training reaches maybe 2% of that, but a sequence policy consumes
its whole context per sample: at batch 32 with a 64-frame context the loader caps at
2.6 steps/s before the GPU does anything. Consumers that need more must shorten
context, share decodes across a batch, or accept the ceiling — caching decoded pixels
is not an option at 193 GB for the full corpus, which is why `doc/0010` retired the
indexed-chunk cache.

## Resumable atomic publication through the existing catalog

The producer follows `boss_spread.build_house`'s structure, with one phase removed:

1. Resolve target fingerprints from the catalog; skip any already in
   `frame_shard_episodes` for this `(level, task, weapon, format)`.
2. Replay each episode in a process pool. Each worker returns that episode's finished
   members — video, actions, metadata — and the parent streams them into the tar.
3. Write to `frames-NNNNN.tar.tmp-<pid>`, then `os.replace` into place.
4. Register the shard and its episode fingerprints in one transaction.

`boss_spread` stages pixels to disk between replay and encoding because its emulator
pool and its CUDA phase must not hold memory at the same time. This release loads no
encoder, so that constraint is absent and staging would only add ~3 GB of write churn
for data consumed once. An episode's members are ~450 KiB, small enough to return
through the pool.

An interrupted run resumes at step 1 with no manual cleanup, and a shard is either
absent or complete and cataloged. Shards hold a fixed episode count rather than being
frame-balanced: balancing exists to equalize token-shard memory maps, and frame shards
are streamed, not mapped.

Verification mirrors `compressed_episodes._verify_video`: after encoding, the first and
last frame of every episode are decoded and compared against the sha256 recorded at
replay time. A mismatch fails the shard before publication.

## What `contra_nes_policy` reads

The consumer contract is the two tables plus the member layout above. A reader selects
`(level=1, task='boss', weapon='spread', format='png-mkv-v1')` ordered by `ordinal`,
takes the first N shards exactly as `shard_counts` does today, and gets frames and
actions with no goal-row offset.

This repo publishes; the policy repo consumes. The contract change is filed as a
GitHub issue on `contra_nes_policy` when the first shard is cataloged, per the
cross-repo convention.
