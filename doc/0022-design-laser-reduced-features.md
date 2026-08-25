# Publish Laser reduced encoder features

Status: Implemented

**Question.** How can policy tune the static encoder projection on the exact Laser
D10k corpus without rerunning the much more expensive convolutional trunk?

**Answer.** Publish the frozen checkpoint output immediately after
`view_backbone + reduce` as a versioned float16 `T×256×4×4` representation.  The
release mirrors all 42 `png-mkv-v1` Laser frame shards, retains unshifted actions,
and is selected by representation name plus encoder checkpoint digest.

## Representation boundary

The producer losslessly decodes each native 224×240 RGB frame, resizes it to
256×256 with OpenCV `INTER_AREA`, converts to float32 RGB divided by 255, and runs
the checkpoint through:

```
view_backbone: RGB -> 1024×4×4
reduce:        1×1 convolution -> GroupNorm -> SiLU -> 256×4×4
```

It does not run or serialize `proj` or `token_ln`. Policy therefore owns the
trainable `4096 -> 512 -> 512` projection and final token normalization while the
published convolutional work remains frozen.

Each episode contains:

```
{uid}.features.npy  float16 (T,256,4,4), TCHW
{uid}.actions.npy   original int64 21-way action indices
{uid}.json          membership, shape, boundary, and alignment
```

`frames[i]` is the post-action state produced by `actions[i]`; behavior cloning
must predict `actions[i+1]`. The producer copies the action payload byte for byte
from the source frame shard.

## Policy-facing selector and measured release

```json
{
  "level": 1,
  "task": "boss",
  "weapon": "laser",
  "representation": "reduced-view-v1",
  "encoder_sha256": "f36041bc69f1ce20781d5200bc89970b1b305e12bff5ae826b23581ca0f1923c"
}
```

The release is under
`game_trace/datahouse/level1/boss/laser/features/reduced-view-v1/<encoder_sha256>/`.
Its `spec.json` contains every ordered fingerprint and shard digest.

| property | value |
|---|---:|
| feature shards | 42 |
| episodes | 10,293 |
| frames | 1,075,404 |
| stored bytes | 8,856,750,080 (8.86 GB / 8.25 GiB) |
| ordered membership SHA-256 | `616126422ac17bcfe3d07fec9cd329533d616d08e80c8bd65565932b492184e3` |
| checkpoint SHA-256 | `f36041bc69f1ce20781d5200bc89970b1b305e12bff5ae826b23581ca0f1923c` |

The catalog join reports 10,293 frame members, 10,293 feature members, and zero
missing fingerprints. Nine episodes sampled across the ordered release were
decoded again and checked at the first, middle, and final frame. The largest
absolute float16 difference from a fresh CUDA forward was `0.00390625`, within
the `0.0078125` verification tolerance required for batch-shape-dependent CUDA
rounding.

## Implementation and invariants

`datahouse.reduced_features` writes one output tar per source frame tar, hashes
the source before use, atomically renames each completed artifact, then registers
it in the separate `feature_shards` and `feature_shard_episodes` tables. Resuming
skips only whole source shards. A partial membership overlap is an error.

The existing `shards`, `frame_shards`, and their artifact identities are not
changed. Representation name and encoder SHA-256 jointly prevent consumers from
mixing incompatible frozen producers.

