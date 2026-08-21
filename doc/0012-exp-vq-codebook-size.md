# How large should the four-token VQ codebook be?

Status: Proposed

## 1. Goal

Select the smallest shared VQ codebook that preserves native Level 1 frame detail when
every frame is represented by the four spatial codes defined in 0011. Compare 256,
1,024, 4,096, and 16,384 entries while holding the frame corpus, model, objective,
optimizer, seed, and four-token rate fixed. This experiment selects an encoder
candidate; it does not approve policy adoption without a later closed-loop comparison.

## 2. Setup

Freeze the existing `l1-full-10k-v1` snapshot (10,000 complete Level 1 episodes;
snapshot SHA-256 `14cf8463…bf85be8a`). Sort a salted hash of each trace fingerprint,
retain the first 1,000 episodes, and assign 800 episodes to train, 100 to validation,
and 100 to test. Frames from one episode never cross splits. The smaller episode set
removes near-duplicate complete playthroughs while retaining every temporal transition.

Replay each selected trace at the emulator's native 224×240 RGB resolution without
resize or interpolation and retain every observation, including the initial frame.
Record trace fingerprint, action index, frame hash, split, and RAM-derived player,
enemy, and merged-projectile centers in a frozen sample manifest.

The exact corpus is 1,196,977 frames; split totals are recorded after replay.
Store it only as a disposable cache under `tmp/0012-vq-codebook/`: lossless native PNGs
in sequential ten-episode tar files plus compact center/offset arrays. Generate
Gaussian masks in the loader. A 200-frame native replay sample averaged 8.6 KB per PNG,
so the image payload is expected to be about 8.6 GB. All four runs consume the same
cache; final policy shards contain no PNGs.

Use one convolutional encoder/decoder with a 2×2 quantized grid, 256-dimensional
latents, one codebook shared across the four positions, and the 0011 objective. The
three 32×32 heatmap channels are player, enemy, and all projectiles. Weighted L2 uses center
Gaussians with `(sigma_player, sigma_enemy, sigma_projectile)=(6,6,4)` native pixels,
class weights `(3,3,15)`, and maximum pixel weight 16. The auxiliary head reads only
the quantized grid.

Warm the continuous autoencoder for 20,000 steps, then reuse that checkpoint for every
cell. Initialize each codebook by k-means over four-position latents from 100,000 frozen
train frames. Train each VQ cell for 100,000 steps with batch 128, AdamW at `3e-4`,
2,000 warmup steps, cosine decay, mixed precision, and seed 0. Use standard
straight-through VQ codebook and commitment losses with commitment weight 0.25.

| run | codes/frame | shared entries | stored index width |
|---|---:|---:|---:|
| vq-k256 | 4 | 256 | 1 byte |
| vq-k1024 | 4 | 1,024 | 2 bytes |
| vq-k4096 | 4 | 4,096 | 2 bytes |
| vq-k16384 | 4 | 16,384 | 2 bytes |

## 3. Evaluation metrics

Report metrics separately for natural validation frames and frames containing a
projectile. Aggregate frames within episodes, then bootstrap episodes for 95%
confidence intervals. Do not inspect test results until selecting a codebook from
validation.

| metric | purpose | source |
|---|---|---|
| exact RGB-pixel accuracy, unweighted/weighted MSE, PSNR | reconstruction fidelity | native frame and decoder output |
| player, enemy, projectile soft Dice | entity information retained after quantization | quantized auxiliary head and RAM heatmaps |
| projectile presence AP and empty-frame FPR at 0.5 | detect hallucination hidden by positive-only Dice | projectile heatmap maximum |

Choose the smallest codebook whose paired validation interval is no worse than
`vq-k16384` by more than 0.005 exact-pixel accuracy or 0.01 Dice on any entity channel.
Evaluate only that selection and `vq-k16384` on test.

| recorded number | provenance |
|---|---|
| 1,000 episodes, 1,196,977 frames, and snapshot hash | salted selection from `game_trace/datahouse/collections/l1-full-10k-v1.json` |
| native 224×240 shape and 8.6 KB PNG mean | 200 frames replayed from `win_level1_20260819231221_i512.npz` |
| four spatial codes and Gaussian objective | `doc/0011-design-vq-image-encoder.md` |
| split, sample, optimizer, and codebook values | predeclared 0012 setup above |

## 4. Conclusion

_Pending user conclusion after measurements._
