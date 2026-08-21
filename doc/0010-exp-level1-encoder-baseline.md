# Can image tokens preserve Spread and Laser projectiles?

Status: Implemented

## 1. Goal

Test whether the frozen 512-D datahouse token preserves the two player-projectile
patterns used by the policy experiments: multi-pellet Spread shots and elongated Laser
shots. Compare a fresh token probe with a direct-image CNN on balanced boss-fight data.
Do not pool the weapons or include Regular, Flamethrower, or enemy bullets.

## 2. Setup

Freeze 1,000 unique canonical Spread and 1,000 unique canonical Laser traces from the
committed canonical GCS boss prefixes. Select the smallest fingerprints per weapon,
then assign the first 800 to training and last 200 to validation. Record every source
object generation and fingerprint in
`runs/encoder-baseline/l1-boss-projectile-probe-v1/snapshot.json`. Boss-only
traces make the manifest weapon valid for every frame. Both weapons receive identical
episode counts, frame sampling, optimizer steps, seeds, and validation exposure.
Canonical Spread uses `full_spread.state` with rapid fire; canonical Laser uses
`full_laser.state` without rapid fire. Therefore only the paired RGB-minus-token gap
within a weapon is causal; raw Spread-versus-Laser accuracy is descriptive.

The target is the RAM-derived 32×32 player-projectile occupancy map with sigma 6 screen
pixels. It represents every simultaneously live Spread pellet or Laser segment rather
than reducing a shot to one point. Train with weighted BCE (`pos_weight=10`), AdamW, learning rate `3e-4`, cosine
decay, 500 warmup steps, 20,000 total steps, and seed 0. Sample weapons
equally and use the same positive/empty-frame policy in both learned arms.

| run | input and trainable path | purpose |
|---|---|---|
| published control | frozen encoder `f36041bc…1923c` plus its published player-bullet channel | measure current Spread and Laser behavior without fitting on the snapshot |
| fresh token probe | frozen 512-D token plus a new player-projectile heatmap head | measure each weapon pattern recoverable from the production token |
| direct-image CNN | RGB image through a fully convolutional encoder-decoder with no vector bottleneck | measure each weapon pattern available directly from pixels |

The completed full-trace control remains a distribution check: on 1,000 held-out
episodes it measured player-bullet Dice 0.6734, MSE skill 0.2889, and peak hit 0.7522
over 726,799 positive observations. It is not compared numerically with the balanced
boss snapshot because its episode distribution differs.

## 3. Evaluation metrics

Report every metric separately for the 200 Spread and 200 Laser validation episodes;
there is no pooled headline score.
Aggregate by frame for continuity with the encoder baseline and bootstrap whole
episodes for 95% confidence intervals.

| metric | role | source |
|---|---|---|
| player-bullet soft Dice | primary localization score | predicted and RAM-derived heatmaps |
| direct-CNN Dice minus fresh-token-probe Dice | primary representation gap | paired validation episodes |
| bullet-presence average precision and F1 | detect whether any player projectile exists | per-frame predicted heatmap maximum and RAM presence |
| empty-frame false-positive rate | expose hallucinated projectiles hidden by positive-only Dice | fraction of empty frames with predicted maximum ≥ 0.5 |
| MSE skill | secondary map-quality diagnostic against an all-zero predictor | same heatmaps |
| peak hit | secondary strongest-location diagnostic | same heatmaps |
| positive/empty frames, parameters, frames/s | coverage, capacity, and throughput checks | frozen manifest and run logs |

A weapon has evidence of token information loss when the direct-image CNN improves
Dice by at least 0.05 and the episode-bootstrap 95% interval for the paired difference
stays above zero. A larger Laser gap than Spread localizes the concern to Laser visual
detail. If the published head trails the fresh token probe, head training—not token
capacity—explains that portion of the deficit.

The frozen snapshot contains 1,000 episodes per weapon and 186,524 observations in
total. Validation contains 200 episodes per weapon: 15,698 Spread observations
(13,641 positive) and 21,997 Laser observations (17,823 positive). The matched learned
comparison uses seed 0. Two additional completed token-probe seeds are retained as
supplemental artifacts but excluded because no matching direct-image seeds were run.

| weapon | published Dice | fresh-token Dice | direct-image Dice | paired RGB − token Dice (95% CI) |
|---|---:|---:|---:|---:|
| Spread | 0.7685 | 0.8533 | 0.7861 | −0.0673 [−0.0708, −0.0639] |
| Laser | 0.8278 | 0.9277 | 0.8961 | −0.0331 [−0.0353, −0.0308] |

The paired intervals use 10,000 whole-episode bootstrap samples. Both gaps are
negative, so neither weapon satisfies the predeclared token-information-loss rule.
However, localization Dice is computed only on frames containing a ground-truth
projectile. Visual inspection of the published Laser preview found substantial
predictions on empty frames—for example maxima 0.45 at action 1099 and 0.63 at action
1129—so the Dice result alone does not establish reliable bullet presence detection.

| weapon | arm | presence AP | precision | recall | F1 | empty-frame FPR |
|---|---|---:|---:|---:|---:|---:|
| Spread | published control | 0.9761 | 0.9108 | 0.9735 | 0.9411 | 0.6325 |
| Spread | fresh token probe | 0.9986 | 0.9907 | 0.9738 | 0.9821 | 0.0608 |
| Spread | direct-image CNN | 0.9977 | 0.9839 | 0.9733 | 0.9786 | 0.1055 |
| Laser | published control | 0.9593 | 0.9148 | 0.9454 | 0.9299 | 0.3759 |
| Laser | fresh token probe | 0.9896 | 0.9212 | 0.9652 | 0.9427 | 0.3527 |
| Laser | direct-image CNN | 0.9906 | 0.9463 | 0.9264 | 0.9362 | 0.2245 |

Presence uses each frame's predicted heatmap maximum; precision, recall, F1, and
empty-frame FPR use the fixed 0.5 threshold, while AP is threshold-free. The fresh
head sharply reduces Spread hallucinations but barely changes the Laser empty-frame
FPR. Its high Laser AP shows useful ranking information remains, so the fixed-threshold
failure may include calibration rather than proving token information loss.

| weapon | arm | MSE skill | peak hit |
|---|---|---:|---:|
| Spread | published control | 0.5213 | 0.8911 |
| Spread | fresh token probe | 0.6733 | 0.9342 |
| Spread | direct-image CNN | 0.5433 | 0.9484 |
| Laser | published control | 0.6544 | 0.8409 |
| Laser | fresh token probe | 0.8723 | 0.9589 |
| Laser | direct-image CNN | 0.8320 | 0.9670 |

The token head has 2,790,337 trainable parameters and ran at 13,768 sampled frames/s;
the direct-image CNN has 169,217 parameters and ran at 1,449 sampled frames/s. These
throughput figures divide 1.28 million sampled training frames by total train-plus-
validation wall time, so they are conservative and intended only as run diagnostics.
Machine-readable metrics are in
`runs/encoder-baseline/l1-boss-projectile-probe-v1/results/summary.json`.

## 4. Conclusion

Awaiting user conclusion.
