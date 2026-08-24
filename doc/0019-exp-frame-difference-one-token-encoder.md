# Does signed frame difference improve the one-token encoder?

Status: Proposed

## 1. Goal

Test the 0018 early-fusion design against the completed 0010 baseline. Determine
whether adding signed `RGB(t)-RGB(t-1)` to current RGB helps one 512-D token retain
projectiles and other moving entities without damaging its static image representation.
Use the existing ordered episode corpus; do not collect or materialize another dataset.

## 2. Setup

Use the frozen 0010 compressed corpus at `tmp/0010-one-token-compressed-1k/`: 800 train,
100 validation, and 100 untouched test episodes. Decode adjacent frames inside each
episode, calculate the signed difference in floating point, use zero difference for an
episode's first observation, and construct pairs before shuffling. Never pair across an
episode or split boundary.

Compare exactly these runs:

| run | input | token | budget | output |
|---|---|---:|---:|---|
| baseline | current RGB | 512-D | 20,000 completed steps | `runs/encoder-baseline/one-token-wsd-20000/` |
| candidate | `[RGB(t), RGB(t)-RGB(t-1)]` | 512-D | 20,000 steps | `runs/encoder-motion/frame-difference-wsd-20000/` |

The candidate follows 0018 exactly. Its six-channel 224×240 input enters six 4×4
stride-2 convolutions with channels `32,64,128,256,512,1024` and spatial outputs
`112×120,56×60,28×30,14×15,7×7,3×3`. A 1×1 `1024→256` reduction produces a
`256×3×3` map; flattening 2,304 values followed by `2304→512→512` linear projection
and LayerNorm emits one 512-D token. The existing 32×32 three-class entity head and
224×240 RGB decoder remain unchanged.

Only the first convolution changes from `3→32` to `6→32`, adding exactly 1,536
parameters. The full candidate has 17,667,078 trainable parameters versus 17,665,542
for the baseline. Instantiate the three-channel reference with seed 0, retain all its
initial parameters, expand its first convolution, copy RGB weights into channels 0–2,
and zero channels 3–5. Before optimization, zero-delta candidate output must match the
reference output exactly.

Train with 200 warmup steps, stable learning rate through step 16,000, cosine decay to
zero at step 20,000, effective batch 128 (micro-batch 32), AdamW `3e-4`, bf16 mixed
precision, gradient clipping at 1.0, weight decay 0.01, and seed 0. Train reconstruction
and player/enemy/projectile heads jointly with the unchanged weighted MSE, BCE, and soft
Dice objective. The candidate is trained from step 0; it does not initialize from the
completed baseline checkpoint.

```sh
python -m datahouse.frame_difference train \
  --corpus tmp/0010-one-token-compressed-1k \
  --output runs/encoder-motion/frame-difference-wsd-20000 \
  --steps 20000 --warmup 200 --decay-start 16000 \
  --micro-batch 32 --effective-batch 128 --workers 2
```

## 3. Evaluation metrics

Evaluate the final candidate checkpoint over the same complete 119,410-frame validation
split used by 0010 and 0015. Do not inspect the test split.

| metric | decision use | source |
|---|---|---|
| exact RGB channel/pixel accuracy, MSE, weighted MSE, PSNR | static representation guardrail | current frame and reconstruction |
| player/enemy/projectile soft Dice | spatial entity retention | RAM-derived heatmaps |
| per-class presence AP and empty-frame FPR at 0.5 | detection and hallucination | heatmap maxima and entity presence |
| positive/empty frame counts | metric population audit | validation metadata |
| projectile peak localization error | small moving-asset precision | predicted peak versus RAM target peak on positive frames |
| train loss and wall time | optimization diagnosis only | candidate metrics log |

The candidate passes only if projectile Dice improves over 0.389318, projectile AP does
not fall below 0.987193, projectile empty-frame FPR falls below 0.851081, player and
enemy Dice do not fall by more than 0.01 absolute, and weighted MSE does not worsen by
more than 2% from 0.013778. Projectile localization error is reported in pixels and
must improve in any later comparison; the 0010 evaluator did not record it, so it is
diagnostic in this first run rather than a numeric gate. A lower training loss alone
cannot pass.

| recorded fact | source |
|---|---|
| baseline validation values and 119,410-frame population | `doc/0010-exp-level1-encoder-baseline.md`; baseline `validation.json` |
| candidate architecture and parameter derivation | `doc/0018-design-frame-difference-one-token-encoder.md` |
| corpus identity and split | experiments 0010 and 0012 |
| candidate measurements | candidate `config.json`, `metrics.jsonl`, and `validation.json` |

## 4. Conclusion

_Pending user conclusion after the candidate completes full validation._
