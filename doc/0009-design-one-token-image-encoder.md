# One-token image encoder

Status: Implemented

**Question.** What exact image representation does the current datahouse publish for
each game observation?

**Answer.** Resize one RGB frame to 256×256, compress it with a six-stage convolutional
network, and emit one normalized 512-D continuous vector. That vector occupies one
transformer position. Entity heatmaps are training supervision only; neither their
decoder nor a pixel decoder is part of the published inference path.

## One frame becomes one continuous transformer token

The encoder accepts a batch of `uint8` RGB images in `(B, 256, 256, 3)` layout. Frame
rendering code resizes native 224×240 observations with OpenCV `INTER_AREA`; the
standalone encoder validates the dtype and layout, converts to `(B, 3, 256, 256)`, and
scales values to `[0,1]`.

The network is:

| stage | operation | output shape per frame |
|---|---|---|
| input | `uint8` RGB, `INTER_AREA` resize | `256×256×3` |
| backbone 1–6 | 4×4 stride-2 convolution, GroupNorm, SiLU | `128²×32`, `64²×64`, `32²×128`, `16²×256`, `8²×512`, `4²×1024` |
| reduction | 1×1 convolution, GroupNorm, SiLU | `4×4×256` |
| projection | flatten, `4096→512`, LayerNorm, SiLU, `512→512` | `512` |
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

## Checkpoint identity fixes preprocessing and tensor semantics

The implemented bundle is
`game_trace/datahouse/encoder/f36041bc69f1ce20781d5200bc89970b1b305e12bff5ae826b23581ca0f1923c/`.
Its `spec.json` records the checkpoint SHA-256, `INTER_AREA` preprocessing, input
layout, token width, token dtype, and sequence alignment. `EncoderSpec` recomputes the
checkpoint digest and reads architectural dimensions from the checkpoint. Consumers
must reject a different digest rather than assuming that equal tensor shapes imply
equal token meanings.

The published checkpoint completed 20,000 steps with seed 0, AdamW at `3e-4`, 500
warmup steps followed by cosine decay, bf16 mixed precision, weight decay `0.01`, and
gradient clipping at `1.0`. Its backbone was trained rather than frozen. These values
describe the existing artifact; the controlled scratch baseline and its measurements
belong to experiment 0010.

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
| digest, preprocessing, output width and dtype | published `spec.json` |
| reconstruction and entity-retention baseline | experiment 0010 |
| four-position discrete replacement | design 0011 |
