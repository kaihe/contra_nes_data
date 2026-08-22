# One-token image encoder

Status: Implemented

**Question.** What exact image representation does the current datahouse publish for
each game observation?

**Answer.** Consume the native 224×240 RGB frame without interpolation, compress it
with a six-stage convolutional network, and emit one normalized 512-D continuous
vector. That vector occupies one transformer position. Entity heatmaps are training
supervision only; neither their decoder nor a pixel decoder is part of the published
inference path.

## One frame becomes one continuous transformer token

The encoder accepts native `uint8` RGB images in `(B, 224, 240, 3)` layout. It validates
the dtype and exact spatial layout, converts to `(B, 3, 224, 240)`, and scales values
to `[0,1]`. It does not resize, crop, pad, or interpolate NES pixels.

The network is:

| stage | operation | output shape per frame |
|---|---|---|
| input | native `uint8` RGB | `224×240×3` |
| backbone 1–6 | 4×4 stride-2 convolution, GroupNorm, SiLU | `112×120×32`, `56×60×64`, `28×30×128`, `14×15×256`, `7×7×512`, `3×3×1024` |
| reduction | 1×1 convolution, GroupNorm, SiLU | `3×3×256` |
| projection | flatten, `2304→512`, LayerNorm, SiLU, `512→512` | `512` |
| token normalization | LayerNorm | `512` |

There is no temporal operation inside the encoder: every observation is encoded
independently. The output is one continuous vector, not one scalar, one VQ code, or a
grid of patch tokens. Datahouse shards store it as 512 `float16` values (1,024 bytes)
per frame; the policy treats the whole vector as one sequence position after its input
projection.

## Auxiliary entity decoding shapes the bottleneck during training

The published training model attaches a four-channel 32×32 heatmap decoder to the
512-D token. A linear layer produces `256×4×4`; three 4×4 stride-2 transposed
convolutions with GroupNorm and SiLU expand it through 128, 64, and 32 channels; a
3×3 convolution emits the heatmap logits. These RAM-derived targets force the single
token to retain spatially useful entity information despite collapsing the frame to
one vector.

The auxiliary head is not required to encode frames and is not stored in token
shards. `load_encoder` loads only the backbone, reduction, projection, and final token
normalization; it accepts checkpoint keys belonging to training-only entity or
reconstruction heads but does not execute them. `load_entity_encoder` exists for
offline measurement.

Experiment 0010 retrains this same inference architecture from scratch with a native
RGB decoder and a three-channel player/enemy/merged-projectile head. Those decoders
measure what the bottleneck retains; they do not change the one-token contract.

## Compressed all-intra episodes are the prepared training dataset

Experiment 0010 uses the 1,000 traces frozen by 0012: 800 train, 100 validation, and
100 untouched test episodes, with all 1,196,977 observations retained. Trace replay
has already produced a lossless 16,119,787,694-byte tar corpus containing native PNG
frames and JSON entity coordinates. Dataset preparation repacks this corpus without
replaying the emulator, resizing pixels, or changing split membership.

Each prepared episode contains:

```text
<uid>.obs.mkv          # 224×240 RGB, PNG codec, all frames are keyframes
<uid>.entities.npz     # compact int16 coordinates and per-frame int32 offsets
<uid>.json             # count, split, source fingerprint and source hashes
```

Episodes are grouped into tar shards. A shard manifest records member offsets and
sizes so a worker can read one episode without scanning the tar. Preparation writes a
temporary shard, verifies that decoded frame counts, first/last frames, coordinates,
splits, and source hashes match the frozen corpus, then publishes it by atomic rename.
The command is resumable by manifest identity. PNG is lossless and all-intra MKV gives
every frame its own seek point; the container adds indexing without introducing
inter-frame dependencies.

The existing 186.719 GiB indexed cache is evidence against raw materialization, not
the final format. Its 179.790 GiB of frames cannot fit in 19 GiB host RAM, and shuffled
memory-map access eventually becomes random NVMe page faults. The same frozen PNG
corpus is 15.013 GiB and can remain substantially resident in the page cache. Allowing
for MKV indexes, compact coordinates, manifests, and tar headers, reserve 20 GiB for
the prepared dataset; record its exact component sizes after construction. The raw
cache becomes deletable after the prepared dataset passes equivalence and throughput
checks.

Training converts decoded frames to floating point and divides by 255 on the device.
It expands compact entity coordinates into three 32×32 heatmaps and the weighted RGB
mask per batch. Coordinates, rather than 6.849 GiB of precomputed `float16` heatmaps,
keep supervision small without changing its declared sigmas `(6,6,4)`.

## Five-hundred-twelve-frame decode windows amortize storage work

The loader shuffles shard and episode order, seeks to an episode's all-intra video,
and decodes contiguous 512-frame windows. It then shuffles those decoded frames and
their coordinates in RAM and emits GPU micro-batches of 32. The I/O window is not a
model context: frames remain independent, and effective batch 128 is unchanged.
Validation decodes episodes and frames in fixed sequential order.

The frozen episodes range from 1,094 to 1,580 frames (median 1,171), so 512 reduces
container/seek work by 16× relative to the policy loader's 32-frame window without
usually decoding unused episode tails. One window holds about 81.8 MiB after decode.
With two persistent workers, prefetch factor two, and an allowance for pinned copies,
approximately 654 MiB is in flight—well below this machine's host-memory limit.

The policy repo already measured all-intra PyAV decode at 0.192 ms/frame and supplied
72 32-frame windows/s against a demand of 42; its 0.097 ms/frame resize also disappears
for this native encoder. Re-measure rather than transplanting those numbers: benchmark
128, 256, and the selected 512 only if a canary shows a regression, using a cold page
cache and reporting samples/second, GPU duty cycle, physical bytes read, host-memory
peak, and checkpoint pauses. The prepared loader is accepted when 512 keeps the GPU
fed for at least one complete cold-cache shard traversal and reproduces sampled pixels
and entity targets exactly.

## Checkpoint identity fixes preprocessing and tensor semantics

The previous implemented bundle is
`game_trace/datahouse/encoder/f36041bc69f1ce20781d5200bc89970b1b305e12bff5ae826b23581ca0f1923c/`.
It remains the immutable 256×256 compatibility reference; its `spec.json` records
`INTER_AREA` preprocessing. The native encoder is a new identity with height 224,
width 240, and preprocessing `none`. `EncoderSpec` recomputes the checkpoint digest
and reads architectural dimensions from the checkpoint. Consumers must reject a
different digest rather than assuming that equal output shapes imply equal token
meanings.

The previous checkpoint completed 20,000 steps with seed 0, AdamW at `3e-4`, 500
warmup steps followed by cosine decay, bf16 mixed precision, weight decay `0.01`, and
gradient clipping at `1.0`. Its backbone was trained rather than frozen. These values
describe that artifact. The native controlled scratch run and its measurements belong
to experiment 0010. The indexed corpus already stores native frames, so changing this
contract requires neither replay nor a second frame cache.

## A replacement must preserve or explicitly version the token contract

Changing preprocessing, weights, width, dtype, or representation type creates a new
encoder identity and requires rebuilding dependent token shards. A multi-position VQ
encoder therefore does not silently replace this model: it changes both stored data
and transformer sequence length and is specified separately in 0011.

The one-token encoder remains the control representation until a candidate passes the
0010 reconstruction/entity comparison and downstream policy evaluation. Its main
advantage is minimal sequence length; its architectural risk is forcing an entire
frame, including small projectiles, through one global 512-D bottleneck.

| recorded fact | provenance |
|---|---|
| layer dimensions and inference behavior | `src/datahouse/encoder.py`; published checkpoint `config` and state-dict shapes |
| checkpoint step and original optimization recipe | published checkpoint `step` and `train_config` |
| legacy digest, preprocessing, output width and dtype | published `spec.json` |
| 1,196,977 rows and split membership | indexed chunk manifests generated from experiment 0012 |
| cache component sizes | file-size sum under `tmp/0012-vq-codebook/indexed-1k-all/` on 2026-08-22 |
| 1,094–1,580 frame range and 1,171 median | episode-prefix counts from the frozen indexed keys |
| policy decode rate and per-frame costs | `contra_nes_policy/README.md` and `src/contra_policy/token_cache.py` |
| reconstruction and entity-retention baseline | experiment 0010 |
| four-position discrete replacement | design 0011 |
