# Can image tokens preserve Spread and Laser projectiles?

Status: Proposed

## 1. Goal

Test whether the frozen 512-D datahouse token preserves the two player-projectile
patterns used by the policy experiments: multi-pellet Spread shots and elongated Laser
shots. Compare a fresh token probe with a direct-image CNN on balanced boss-fight data.
Do not pool the weapons or include Regular, Flamethrower, or enemy bullets.

## 2. Setup

Freeze 1,000 unique canonical Spread and 1,000 unique canonical Laser traces from the
committed canonical GCS boss prefixes. Select the smallest fingerprints per weapon,
then assign the first 800 to training and last 200 to validation. Record every source
object generation and fingerprint in `l1-boss-projectile-probe-v1.json`. Boss-only
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
| MSE skill | secondary map-quality diagnostic against an all-zero predictor | same heatmaps |
| peak hit | secondary strongest-location diagnostic | same heatmaps |
| positive/empty frames, parameters, frames/s | coverage, capacity, and throughput checks | frozen manifest and run logs |

A weapon has evidence of token information loss when the direct-image CNN improves
Dice by at least 0.05 and the episode-bootstrap 95% interval for the paired difference
stays above zero. A larger Laser gap than Spread localizes the concern to Laser visual
detail. If the published head trails the fresh token probe, head training—not token
capacity—explains that portion of the deficit.

## 4. Conclusion

Pending measurements and user conclusion.
