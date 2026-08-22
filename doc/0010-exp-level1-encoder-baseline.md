# How much image and entity information does the one-token encoder retain?

Status: Proposed

## 1. Goal

Establish the validation baseline against which four-position continuous and VQ image
encoders are compared. Measure native-frame information recoverable from the published
512-D continuous token and entity information exposed by its published auxiliary head.
This experiment does not retrain the encoder or inspect the test split.

## 2. Setup

Use the 1,000 complete Level 1 traces selected by experiment 0012 from snapshot
`l1-full-10k-v1` (`14cf8463…bf85be8a`): 800 train, 100 validation, and 100 untouched
test episodes, with every observation retained. Native targets are 224×240 RGB frames
and RAM-derived 32×32 player, enemy, and merged-projectile heatmaps.

Freeze published encoder `f36041bc…1923c`, which maps a frame resized to 256×256 into
one 512-D continuous token. Its checkpoint has no reconstruction decoder. Train a new
decoder from only the frozen token to the native 224×240 frame for 10,000 steps with
effective batch 128, AdamW at `3e-4`, 2,000 warmup steps, cosine decay, mixed precision,
and seed 0. Train with the entity-weighted pixel MSE from 0011. The decoder measures
recoverability from the token; it cannot add image information absent from the token.

Evaluate entity metrics without fitting a new probe. Use the checkpoint's published
four-channel head and map it to the comparison taxonomy: player is the player channel,
enemy is the enemy channel, and projectile is the maximum of player-bullet and
enemy-bullet probabilities. Generate comparison targets with native-pixel sigmas
`(6,6,4)`. Store the run under
`runs/encoder-baseline/one-token-reconstruction/`.

## 3. Evaluation metrics

Report the complete 100-episode validation split. Do not inspect test results. Use the
same definitions for every later continuous/VQ candidate.

| metric | purpose | source |
|---|---|---|
| exact RGB-pixel accuracy, unweighted/weighted MSE, PSNR | reconstruction fidelity | native frame and trained decoder output |
| player, enemy, projectile soft Dice | entity information retained | published auxiliary head and RAM heatmaps |
| projectile presence AP and empty-frame FPR at 0.5 | detect hallucination hidden by positive-only Dice | merged projectile heatmap maximum |

Record both exact RGB-pixel accuracy (all three channels match) and the diagnostic
per-channel accuracy. Weighted MSE uses the 0011 Gaussian mask and is normalized by
total pixel weight. Presence AP is threshold-free; FPR uses only ground-truth empty
frames and the fixed maximum-probability threshold 0.5.

| recorded number | provenance |
|---|---|
| 1,000 episodes and 1,196,977 frames | 0012 corpus markers under `tmp/0012-vq-codebook/corpus-1k-all/` |
| encoder identity, 512-D float16 token | published encoder `spec.json` and checkpoint SHA-256 |
| split, decoder recipe, objectives, and metrics | predeclared setup above |

## 4. Conclusion

_Pending user conclusion after validation measurements._
