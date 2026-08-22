# How much image and entity information does the one-token encoder retain?

Status: Proposed

## 1. Goal

Establish the validation baseline against which four-position continuous and VQ image
encoders are compared. Retrain the current one-token architecture from scratch and
measure the native-frame and entity information carried by its 512-D continuous token.
Do not initialize from published weights or inspect the test split.

## 2. Setup

Use the 1,000 complete Level 1 traces selected by experiment 0012 from snapshot
`l1-full-10k-v1` (`14cf8463…bf85be8a`): 800 train, 100 validation, and 100 untouched
test episodes, with every observation retained. Native targets are 224×240 RGB frames
and RAM-derived 32×32 player, enemy, and merged-projectile heatmaps.
Generate a disposable indexed cache from the frozen lossless corpus using the 0014
layout. Training and validation read its memory-mapped frames and precomputed targets;
the split and frame population are unchanged.

Instantiate the native one-token architecture in 0009 from scratch: preserve the
224×240 frame, apply six stride-2 convolutional stages and a 3×3-to-512 projection,
and emit one 512-D continuous token. Attach a native 224×240 reconstruction decoder
and a fresh three-channel 32×32 entity head. Train every component jointly for 20,000 steps with
effective batch 128, AdamW at `3e-4`, 2,000 warmup steps, cosine decay, mixed precision,
and seed 0. Use the same weighted pixel MSE plus player/enemy/projectile BCE and soft
Dice objective as the four-continuous-token warmup. Generate targets with native-pixel
sigmas `(6,6,4)`. Store the run under
`runs/encoder-baseline/one-token-reconstruction/`.

The superseded 256×256 attempt was stopped at step 3,740 after its step-3,000
checkpoint. It is not evaluated or included in the baseline because the input
contract changed before completion.

## 3. Evaluation metrics

Report the complete 100-episode validation split. Do not inspect test results. Use the
same definitions for every later continuous/VQ candidate.

| metric | purpose | source |
|---|---|---|
| exact RGB-pixel accuracy, unweighted/weighted MSE, PSNR | reconstruction fidelity | native frame and trained decoder output |
| player, enemy, projectile soft Dice | entity information retained | jointly trained auxiliary head and RAM heatmaps |
| projectile presence AP and empty-frame FPR at 0.5 | detect hallucination hidden by positive-only Dice | merged projectile heatmap maximum |

Record both exact RGB-pixel accuracy (all three channels match) and the diagnostic
per-channel accuracy. Weighted MSE uses the 0011 Gaussian mask and is normalized by
total pixel weight. Presence AP is threshold-free; FPR uses only ground-truth empty
frames and the fixed maximum-probability threshold 0.5.

| recorded number | provenance |
|---|---|
| 1,000 episodes and 1,196,977 frames | 0012 corpus markers under `tmp/0012-vq-codebook/corpus-1k-all/` |
| native one-token architecture and 512-D width | design 0009; weights initialized from scratch |
| split, decoder recipe, objectives, and metrics | predeclared setup above |

## 4. Conclusion

_Pending user conclusion after validation measurements._
