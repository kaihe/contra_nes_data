# Feed four discrete image codes directly to the policy

Status: Proposed

**Question.** Can the datahouse eliminate images and continuous image embeddings from
training shards while preserving raw-frame detail and avoiding image encoding during
policy training?

**Answer.** Train a datahouse-owned VQ image autoencoder offline and initially encode
each frame as a 2×2 grid of four `uint16` codes. Shards store only these codes, actions,
boundaries, and metadata. The policy learns code and slot embeddings, combines the four
codes through its normal attention layers, and uses four consecutive sequence positions
per frame. Four codes are accepted only if reconstruction, entity visibility, and
closed-loop policy gates match raw RGB; otherwise increase the code count to 8, 16, or
32 without changing the shard protocol.

---

## Four offline codes replace continuous frame tensors

The datahouse owns the VQ encoder, reconstruction decoder, codebook, preprocessing,
and weights. Dataset construction performs image encoding once. A shard contains no
RGB frames, image packs, videos, or 512-D floating-point image embeddings:

```text
codes:       uint16 [frames, 4]
actions:     uint8  [actions]
boundaries:  episode and view offsets
metadata:    tokenizer identity, shapes, hashes, and source fingerprints
```

Four `uint16` values cost 8 bytes per frame and allow codebooks larger than 256
entries without packed-bit decoding. The current Level 1 full release uses about
1,035 bytes per frame, so its 11,950,720 frames would shrink from 12.37 GB to about
95.6 MB of image codes before actions, tar headers, indexes, and metadata. Encoder and
decoder weights are collection artifacts referenced by hash, not copied into shards.

Every shard records a `tokenizer_id` derived from the preprocessing specification,
encoder weights, codebook, spatial layout, and code dtype. Loaders reject mixed
identities. Re-encoding publishes a new collection rather than mutating existing
shards.

## Four policy positions preserve spatial detail

The four codes describe fixed positions in a 2×2 spatial grid. The policy performs
four trainable embedding lookups and adds a learned slot embedding to distinguish the
positions:

```text
frame t -> [top-left, top-right, bottom-left, bottom-right] -> four GPT positions
```

There is no frame combiner before the temporal transformer. The action prediction
after the fourth position can attend to all four regions and merge them differently
for each game context. This avoids introducing a second learned bottleneck after the
VQ encoder.

Actions are predicted only after all four codes for a frame are available. Goal images
use the same four-token order. Rolling windows are specified in frames, aligned to
frame boundaries, and expanded to four image positions per included frame. The design
accepts four times as many image positions and up to roughly sixteen times the
full-attention work for the same frame horizon. Policy checkpoints own the learned code
and slot embeddings; the datahouse decoder retains a separate immutable embedding table
for reconstruction. Policy learning therefore cannot change what stored code indices
mean or break dataset reproducibility.

## Detail-preserving training targets native game structure

The VQ model reconstructs the native NES frame rather than a blurred natural-image
proxy. Training combines palette-aware pixel reconstruction with edge/high-frequency
loss and the existing RAM-derived entity heatmaps. Entity supervision includes the
player, enemies, player projectiles, and enemy projectiles so a low average pixel loss
cannot hide missing one-pixel gameplay details.

The four-code bottleneck carries only 64 stored bits per frame and cannot be declared
equivalent to arbitrary raw RGB by construction. Contra frames occupy a much smaller
manifold, so equivalence is an empirical gate. Evaluate 4 codes first, then 8, 16, and
32 using the same architecture and training data. Select the smallest rate that passes
all gates; do not compensate for a failed visual representation by weakening the
metrics.

| gate | acceptance measurement |
|---|---|
| reconstruction | exact palette-pixel accuracy, PSNR, and edge error against native frames |
| entity detail | fresh probes from codes versus RGB for all entity heatmaps, including positive and empty projectile frames |
| storage | total shard bytes and bytes/frame including actions, indexes, and metadata |
| loader | frames/s and CPU use with code lookup only; no encoder invocation |
| policy | matched action loss and Spread/Laser closed-loop win-rate intervals versus raw RGB |

Raw-frame equivalence requires the code probe to be statistically indistinguishable
from the RGB probe and the policy result to be non-inferior. Reconstruction quality
alone cannot accept the encoder.

## Offline training removes only the training-time encoder

Policy training reads integer codes and performs embedding lookup, so experiments do
not decode pixels or run the VQ encoder. This makes shard loading small and makes code
embeddings trainable like text-token embeddings.

Closed-loop play still requires `frame -> VQ encoder -> four codes` for each live
frame. The evaluation runtime must use the exact datahouse encoder and tokenizer
identity used to build the training collection. Encoder latency and code stability
under emulator capture differences are therefore release gates even though they are
absent from offline policy training.

---

## Provenance and auditability

| claim | source |
|---|---|
| Current image representation is 512 `float16` values per frame | `src/datahouse/encoder.py`; shard `tokens.npy` shape |
| Level 1 full contains 11,950,720 frames | `catalog.sqlite`: sum of `shards.frames` for `level=1, task='full'` |
| Level 1 full token tar files occupy 12,368,629,760 bytes | sum of `game_trace/datahouse/level1/full/mixed/token-*.tar` |
| Four `uint16` codes occupy 8 bytes/frame | proposed `[frames, 4]` dtype and shape |
| Projectile presence must include empty frames | experiment 0010 visual review and presence measurements |
