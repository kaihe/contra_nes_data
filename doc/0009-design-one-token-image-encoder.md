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

## Native indexed chunks preprocess the training population once

Experiment 0010 uses the 1,000-trace corpus frozen by 0012: 800 train, 100 validation,
and 100 untouched test episodes, with every observation retained. Trace replay first
produces a lossless 16,119,787,694-byte tar corpus containing PNG frames and JSON
entity coordinates. The disposable 0014 cache decodes that corpus once into 100
memory-mappable chunk directories. It performs no resize or image augmentation:
`frames.npy` stores native RGB `uint8`, while `targets.npy` stores the RAM-derived
player, enemy, and merged-projectile 32×32 heatmaps as `float16`. Keys and manifests
preserve frame identity, split, source hash, shape, and dtype.

The measured cache contains 1,196,977 frames and occupies 200,487,925,488 bytes
(186.719 GiB):

| member | bytes/frame | measured size |
|---|---:|---:|
| native frame, `224×240×3 uint8` | 161,280 | 179.790 GiB |
| three `32×32 float16` targets | 6,144 | 6.849 GiB |
| keys and manifests | variable | 0.079 GiB |

One uncompressed row therefore costs 167,424 bytes before its key. The 958,192-row
train split accounts for about 149.407 GiB; a 20,000-step run at effective batch 128
consumes 2.56 million samples, or 2.67 nominal passes and about 399 GiB of frame/target
payload reads. Scaling the same materialization to all 10,000 traces would approach
1.8 TiB, so the indexed cache is deliberately limited to the fixed 1,000-trace
encoder experiment and remains reproducible from the smaller tar corpus.

Training converts each selected frame to floating point and divides by 255 on the
device. It converts stored target heatmaps to `float32` and derives the weighted RGB
loss mask there. Precomputing heatmaps removes JSON parsing and Gaussian construction
from the hot path. Compact coordinate targets could save at most the measured 6.849
GiB plus keys, only about 3.7% of this cache; frames dominate the storage decision.

## Block-local shuffling should keep storage reads sequential

Memory mapping eliminates PNG decoding but does not make arbitrary reads cheap. The
current iterator shuffles every row index in a roughly 1.9 GiB chunk and then follows
that random order. Once the operating-system page cache is cold, this turns an NVMe
scan into small page faults. The observed symptom is low GPU clocks and utilization
while loader workers accumulate physical reads; the smaller native encoder cannot
improve wall time while it waits for batches.

The next loader revision should shuffle at two levels. Shuffle chunk order each epoch,
then shuffle contiguous blocks of 256–512 rows. Read one block sequentially into RAM,
randomize its rows in memory, and yield its batches before moving to another block.
Two persistent workers may prefetch pinned blocks. This preserves stochastic ordering
at chunk, block, and row scales while replacing disk-wide random access with large
sequential reads. Evaluation remains strictly sequential and deterministic.

Benchmark throughput from a cold page cache and report samples/second, GPU duty cycle,
physical bytes read, and checkpoint pauses. A warm-cache first minute is not a valid
run estimate. Keep uncompressed `uint8` frames in the training cache: returning to PNG
would trade the random-I/O problem for CPU decode work. If the block loader still
cannot feed the GPU, the next experiment should compare a train-only cache on faster
local storage or modest block compression with asynchronous decode; neither changes
the frozen frame population.

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
| 2.56 million samples and 2.67 passes | declared 20,000 steps × effective batch 128, divided by 958,192 train rows |
| reconstruction and entity-retention baseline | experiment 0010 |
| four-position discrete replacement | design 0011 |
