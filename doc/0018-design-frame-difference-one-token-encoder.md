# Frame-difference one-token image encoder

Status: Proposed
Supersedes: 0016

**Question.** Can the encoder retain temporal changes and small moving assets without
depending on the unreliable global registration rejected by 0017?

**Answer.** Concatenate the current RGB frame with its signed difference from the
preceding frame and feed the resulting six channels through the existing 0010
backbone. Keep current RGB because a difference alone cannot represent stationary
entities or level geometry. Keep the output contract at one 512-D token. Compare this
early-fusion candidate with the completed 0010 baseline at the same 20,000-step budget
before considering a larger two-stream model.

## Current RGB and signed difference form a six-channel input

For observations within one episode, normalize RGB to `[0,1]` and calculate
`delta_t = rgb_t - rgb_(t-1)` in floating point, preserving its `[-1,1]` sign. The
encoder input is channel-wise `[rgb_t, delta_t]`, shaped `B×6×224×240`. Never subtract
`uint8` arrays because negative changes wrap around. At the first observation of an
episode, set all delta channels to zero. Pairs never cross episode or split boundaries.

The current RGB channels preserve absolute player and entity positions, terrain,
weapon appearance, and other static state. The delta channels expose temporal changes
without claiming to be optical flow. Camera scrolling therefore produces a coherent
full-background difference as well as local object motion. Unlike 0016, this design
does not pre-align frames: 0017 showed that erroneous alignment can create stronger
false motion than raw differences. The convolutional network must learn whether a
change is global or local from the joint six-channel input.

## Early fusion preserves the 512-D architecture

Only the first convolution changes from `3→32` to `6→32`; its 4×4 kernel, stride 2,
padding 1, GroupNorm, and SiLU remain unchanged. All later layers are identical to
0010. The exact encoder path is:

| stage | operation | output per observation |
|---|---|---|
| input | concatenate current RGB and signed delta | `6×224×240` |
| conv 1 | 4×4, stride 2, `6→32`; GroupNorm; SiLU | `32×112×120` |
| conv 2 | 4×4, stride 2, `32→64`; GroupNorm; SiLU | `64×56×60` |
| conv 3 | 4×4, stride 2, `64→128`; GroupNorm; SiLU | `128×28×30` |
| conv 4 | 4×4, stride 2, `128→256`; GroupNorm; SiLU | `256×14×15` |
| conv 5 | 4×4, stride 2, `256→512`; GroupNorm; SiLU | `512×7×7` |
| conv 6 | 4×4, stride 2, `512→1024`; GroupNorm; SiLU | `1024×3×3` |
| reduction | 1×1, `1024→256`; GroupNorm; SiLU | `256×3×3` |
| projection | flatten 2,304; Linear `2304→512`; LayerNorm; SiLU; Linear `512→512`; token LayerNorm | `512` |

The training-only heads remain unchanged. The entity head maps the token through a
linear 4×4 seed and three transposed-convolution upsampling blocks to player, enemy,
and projectile 32×32 heatmaps. The reconstruction decoder maps the token through a
`256×2×2` seed, five convolution blocks with interpolation, and a final 3-channel
224×240 sigmoid output. Neither head becomes part of the published encoder.

Early fusion is the minimum useful test: it lets every spatial layer combine
appearance and change while adding almost no parameters. A separate motion stream is
considered only if early fusion improves motion-sensitive validation metrics but shows
insufficient capacity.

## Parameter growth is confined to the first convolution

The first convolution has `out_channels × in_channels × 4 × 4 + out_channels`
parameters. It grows from `32×3×4×4+32 = 1,568` to
`32×6×4×4+32 = 3,104`, an increase of exactly 1,536 weights. Everything else is
unchanged.

| component | 0010 RGB baseline | six-channel candidate | delta |
|---|---:|---:|---:|
| convolutional backbone | 11,181,472 | 11,183,008 | +1,536 |
| reduction + projection + token norm | 1,707,776 | 1,707,776 | 0 |
| published inference encoder | 12,889,248 | 12,890,784 | +1,536 |
| training-only entity head | 2,790,915 | 2,790,915 | 0 |
| training-only reconstruction decoder | 1,985,379 | 1,985,379 | 0 |
| complete training model | 17,665,542 | 17,667,078 | +1,536 |

The candidate is 0.0119% larger at inference and 0.0087% larger during training. These
counts are exact for `depth=32`, `proj_ch=256`, `hiddim=512`, six convolution stages,
and three entity classes; they were obtained by instantiating the 0010 PyTorch modules
and applying the first-layer formula above. Activation and input bandwidth increase
more than parameter count because the first stage reads twice as many channels, but
later-stage compute is unchanged.

## Paired initialization and validation isolate the temporal input

Instantiate a fresh three-channel 0010 reference with seed 0, copy every shared
parameter into the candidate, copy its first-convolution RGB weights into input
channels 0–2, and initialize delta-channel weights 3–5 to zero. The candidate therefore
produces exactly the reference function before training and must learn whether to use
delta. Train from step 0 for the same 20,000-step WSD schedule, effective batch 128,
AdamW `3e-4`, bf16, gradient clipping, corpus, and 800/100/100 episode split as 0010.
The loader constructs consecutive pairs before shuffling them.

Evaluate the complete validation split against the fixed 0010 result. Retain RGB
MSE/PSNR, player/enemy/projectile Dice, presence AP, positive/empty counts, and
empty-frame FPR at 0.5. Add projectile peak localization error on positive frames and
report metrics separately for RAM-stationary and scrolling pairs for diagnosis only;
RAM is never an encoder input. The design is useful only if projectile Dice and
localization improve without worsening projectile AP, empty-frame FPR, player/enemy
Dice, or weighted MSE. A lower training loss alone does not pass.

| recorded value or decision | source |
|---|---|
| baseline architecture, parameter modules, and validation protocol | `datahouse.encoder`, `datahouse.one_token_baseline`, experiment 0010 |
| exact baseline component counts | instantiated 0010 `OneTokenAutoencoder`, recorded above |
| alignment rejection and whole-trace failure rates | experiment 0017 |
| frozen corpus and episode split | experiments 0010 and 0012 |
