# Does a 1024-D one-token bottleneck retain small entities better?

Status: Implemented

## 1. Goal

Test whether widening the native one-token image encoder from 512 to 1024 dimensions
reduces false projectile detections and improves retained image/entity information.
Use the completed 20,000-step 512-D run from experiment 0010 as the fixed baseline.
Change token width only, compare at equal optimization steps, and do not inspect the
test split. The candidate is exploratory: it does not replace the production encoder
or trigger token-shard rebuilding without a later downstream policy decision.

## 2. Setup

Use the same frozen corpus and episode split as 0010:
`tmp/0010-one-token-compressed-1k/`, with 800 train, 100 validation, and 100 untouched
test episodes. Inputs remain native 224×240 RGB frames. The backbone remains six
4×4 stride-2 convolution stages followed by the 3×3 reduction map. Reconstruction
and player/enemy/projectile heads, targets, sigmas, loss weights, seed, and data order
remain unchanged.

Run exactly two configurations:

| run | token width | initialization | schedule | budget | output |
|---|---:|---|---|---:|---|
| baseline | 512 | random | 200 warmup, stable to 16k, cosine decay to 20k | 20,000 | existing `runs/encoder-baseline/one-token-wsd-20000/` |
| candidate | 1024 | random | 200 warmup, stable to 16k, cosine decay to 20k | 20,000 | `runs/encoder-enhanced/one-token-1024-wsd-20000/` |

The instantiated models contain 17,665,542 and 22,256,134 trainable parameters,
respectively. The candidate is therefore 26.0% larger; equal steps do not mean equal
FLOPs.

Launch the candidate as one resumable run; an existing `latest.pt` in the output
directory resumes automatically:

```sh
python -m datahouse.one_token_baseline train \
  --corpus tmp/0010-one-token-compressed-1k \
  --output runs/encoder-enhanced/one-token-1024-wsd-20000 \
  --steps 20000 --schedule wsd --warmup 200 --decay-start 16000 \
  --token-dim 1024 --micro-batch 32 --effective-batch 128 \
  --workers 2 --save-every 1000
```

Both use effective batch 128, micro-batch 32, AdamW at `3e-4`, bf16 mixed precision,
gradient clipping at `1.0`, weight decay `0.01`, and seed 0. The 1024-D run is trained
from scratch; neither its weights nor optimizer state come from the 512-D run.

Widening increases both bottleneck capacity and width-dependent projection/head
parameters. This experiment therefore answers whether the complete 1024-D one-token
model performs better at equal steps, not whether token width alone is causally
responsible. Record total/trainable parameter counts and wall time for both models so
that the compute difference is explicit.

## 3. Evaluation metrics

Evaluate both final checkpoints over the same complete 119,410-frame validation split.
Retain 0010's exact RGB channel/pixel accuracy, unweighted and weighted MSE, PSNR, and
per-class soft Dice. Add presence AP and empty-frame FPR at a fixed maximum-heatmap
probability threshold of 0.5 separately for player, enemy, and projectile.

Every class reports positive- and empty-frame counts beside AP and FPR. A class with
zero positive frames reports AP as `null`; a class with zero empty frames reports FPR
as `null`, not zero. Player FPR is expected to be weak or undefined because valid
gameplay frames almost always contain the player; enemy and especially projectile
empty frames provide the useful hallucination tests. Per-class reporting prevents the
strong player/enemy localization result from hiding projectile failure.

| gate | requirement |
|---|---|
| primary | projectile empty-frame FPR at 0.5 is lower than the 512-D baseline |
| guardrail | projectile presence AP and projectile Dice do not decrease |
| representation | weighted MSE does not worsen by more than 2% relative |
| interpretation | report player/enemy/projectile counts, AP, FPR, and Dice even when a gate fails |

A lower training loss is diagnostic only and cannot pass the experiment. The candidate
must improve validation FPR without trading away projectile detection. Because the
baseline artifact currently contains projectile presence metrics only, re-evaluate
its unchanged checkpoint with the new per-class evaluator before comparing the two
runs.

```sh
python -m datahouse.one_token_baseline evaluate \
  --corpus tmp/0010-one-token-compressed-1k \
  --decoder runs/encoder-baseline/one-token-wsd-20000/latest.pt \
  --output runs/encoder-baseline/one-token-wsd-20000/validation-v2.json

python -m datahouse.one_token_baseline evaluate \
  --corpus tmp/0010-one-token-compressed-1k \
  --decoder runs/encoder-enhanced/one-token-1024-wsd-20000/latest.pt \
  --output runs/encoder-enhanced/one-token-1024-wsd-20000/validation.json
```

Both checkpoints completed the full 119,410-frame validation pass:

| metric | 512-D baseline | 1024-D candidate | candidate delta |
|---|---:|---:|---:|
| exact RGB channel accuracy | 0.273368 | 0.240194 | -0.033174 |
| exact RGB pixel accuracy | 0.107388 | 0.100170 | -0.007218 |
| unweighted MSE | 0.010438 | 0.010716 | +2.66% |
| weighted MSE | 0.013778 | 0.014107 | +2.38% |
| PSNR (dB) | 19.8138 | 19.6996 | -0.1142 |
| player Dice | 0.617711 | 0.617725 | +0.000014 |
| enemy Dice | 0.643461 | 0.643054 | -0.000407 |
| projectile Dice | 0.389318 | 0.388444 | -0.000873 |
| player presence AP | 1.000000 | 1.000000 | 0 |
| enemy presence AP | 0.999754 | 0.999714 | -0.000040 |
| projectile presence AP | 0.987193 | 0.988096 | +0.000904 |
| enemy empty-frame FPR @ 0.5 | 0.018583 | 0.019744 | +0.001161 |
| projectile empty-frame FPR @ 0.5 | 0.851081 | 0.840980 | -0.010101 (-1.19%) |

Player FPR is `null` for both runs because all 119,410 frames contain the player.
Enemy counts are 113,383 positive and 6,027 empty frames; projectile counts are
91,294 positive and 28,116 empty frames in both evaluations.

| gate | result | evidence |
|---|---|---|
| lower projectile FPR | pass | 0.851081 → 0.840980 |
| projectile AP and Dice do not decrease | fail | AP rises 0.000904, but Dice falls 0.000873 |
| weighted MSE worsens by no more than 2% | fail | weighted MSE worsens 2.38% |
| report every class | pass | counts, AP, FPR, and Dice recorded; player FPR correctly `null` |

| recorded fact | source |
|---|---|
| 512-D baseline recipe and result | experiment 0010; `runs/encoder-baseline/one-token-wsd-20000/validation.json` |
| 119,410 validation frames | the same validation artifact |
| corpus membership and preprocessing | experiment 0010 and compressed shard manifests |
| 17,665,542 and 22,256,134 parameter counts | direct model instantiation from the published architecture config with widths 512 and 1024 |
| expanded 512-D measurements | `runs/encoder-baseline/one-token-wsd-20000/validation-v2.json` |
| candidate configuration and measurements | `runs/encoder-enhanced/one-token-1024-wsd-20000/config.json`, `checkpoint-020000.pt`, `metrics.jsonl`, and `validation.json` |

## 4. Conclusion

Widening the one-token representation from 512 to 1024 dimensions does not improve
projectile accuracy or hallucination enough to justify the larger model. Projectile
Dice decreases from 0.389318 to 0.388444, while the empty-frame FPR falls only from
0.851081 to 0.840980. Weighted MSE also worsens by 2.38%. Keep the 512-D token as the
baseline; final token width is not the effective bottleneck for small-projectile
information.
