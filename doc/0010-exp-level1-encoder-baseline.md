# How much image and entity information does the one-token encoder retain?

Status: Implemented

## 1. Goal

Establish a fixed 20,000-step validation baseline for the native one-token encoder.
Retrain the current architecture from scratch and measure the native-frame and entity
information carried by its 512-D continuous token. Do not initialize from published
weights or inspect the test split.

## 2. Setup

Use the 1,000 complete Level 1 traces selected by experiment 0012 from snapshot
`l1-full-10k-v1` (`14cf8463…bf85be8a`): 800 train, 100 validation, and 100 untouched
test episodes, with every observation retained. Native targets are 224×240 RGB frames
and RAM-derived 32×32 player, enemy, and merged-projectile heatmaps.

Read the compressed all-intra episode shards specified in 0009, published at
`tmp/0010-one-token-compressed-1k/` — 100 shards, 6.470 GiB, one PNG-in-MKV video and
one compact coordinate archive per episode. They were repacked from the frozen corpus
without replaying the emulator: decoded frames and coordinates are byte-identical to
`corpus-1k-all`, verified for one whole shard and for sampled episodes of three others.
The loader decodes contiguous 512-frame windows, shuffles each window in RAM, seeks an
episode by tar offset, and partitions shards between data workers; training waits on it
for 0.4% of wall time at two workers, so every run below is GPU-bound. Shards are
published by atomic rename after their members verify, and a rebuild skips a shard only
when its manifest identity and source digest match, so an interrupted build resumes
without accepting partial data.

The data path is an optimization beneath this experiment. Model inputs, objectives,
splits, frame population, and evaluation definitions do not change with it, and the
lossless tar corpus remains the reproducible source from which any cache is
regenerated. Supervision is likewise unchanged in definition: the same three player,
enemy, and merged-projectile Gaussian maps on the same 32×32 grid, with native-pixel
sigmas `(6,6,4)`, maximum composition, and coordinate convention. Only where the work
happens moved — coordinates are expanded into maps per batch on the training device
rather than materialized once to disk, and pixel weights are still derived from those
maps in one vectorized device operation.

The earlier indexed-chunk cache is retired. It stored `frames.npy` (`uint8`,
N×224×240×3) and precomputed `targets.npy` (`float16`, N×3×32×32) per source shard, so
that memory-mapped integer indices replaced PNG decoding, JSON parsing, and Gaussian
construction on every epoch. That trade needed about 200 GB — 186.719 GiB at this
corpus size, of which 179.790 GiB was frames — which cannot stay resident in 19 GiB of
host RAM, so shuffled access degraded into random NVMe page faults. The compressed
shards hold the same frames in 6.470 GiB and can remain substantially in page cache.
The cache has been deleted; the chunk reader remains only for corpora already in that
layout.

Instantiate the native one-token architecture in 0009 from scratch: preserve the
224×240 frame, apply six stride-2 convolutional stages and a 3×3-to-512 projection,
and emit one 512-D continuous token. Attach a native 224×240 reconstruction decoder
and a fresh three-channel 32×32 entity head. Train every component jointly with
effective batch 128 (micro-batch 32), AdamW at `3e-4`, bf16 mixed precision, gradient
clipping at `1.0`, and seed 0. Use the same weighted pixel MSE plus
player/enemy/projectile BCE and soft Dice objective as the four-continuous-token
warmup. Generate targets with native-pixel sigmas `(6,6,4)`.

Schedule the learning rate as warmup–stable–decay: 200 linear warmup steps, a constant
`3e-4` phase through step 16,000, then cosine decay to zero at step 20,000. The measured
run branches from the scratch-trained stable checkpoint at step 16,000, so its weights
are one continuous 20,000-step trajectory rather than a second initialization.

The train split holds 958,192 frames, so at effective batch 128 a budget buys far
less data coverage than its step count suggests: one epoch is 7,486 steps and the
20,000-step result covers 2.67 epochs. NES frames are strongly redundant, so epochs
overstate effective diversity. The stable trunk later continued to step 26,400 only
for curve inspection; its slowly falling training loss is not a second validation
result. The planned larger-budget rungs were not run. Experiment 0015 instead compares
a 1024-D candidate against this exact compute-matched 20,000-step baseline.

Two earlier attempts are excluded from the baseline. The 256×256 attempt was stopped
at step 3,740 after its step-3,000 checkpoint, because the input contract changed
before it completed. The native cosine attempt under
`runs/encoder-baseline/one-token-reconstruction/` was stopped at step 2,460; it read
the superseded per-frame tar path and its cosine schedule is not comparable to the
completed WSD baseline. Neither is evaluated.

## 3. Evaluation metrics

Report every metric below on the complete 100-episode, 119,410-frame validation split.
Do not inspect test results. Use the same definitions for every later one-token width
candidate measured at the same 20,000-step budget.

The completed run writes `validation.json` beside its checkpoints, and its summarized
row in `runs/encoder-baseline/ladder.json` carries steps, frames seen, epochs, and every
metric below.

| metric | purpose | source |
|---|---|---|
| exact RGB-pixel accuracy, unweighted/weighted MSE, PSNR | reconstruction fidelity | native frame and trained decoder output |
| player, enemy, projectile soft Dice | entity information retained | jointly trained auxiliary head and RAM heatmaps |
| projectile presence AP and empty-frame FPR at 0.5 | detect hallucination hidden by positive-only Dice | merged projectile heatmap maximum |

The 20,000-step rung completed evaluation on the full validation split. The abandoned
40,000- and 80,000-step rows are omitted because they produced no measurements.

| budget | frames seen | epochs | exact RGB channel | exact RGB pixel | unweighted MSE | weighted MSE | PSNR (dB) | player Dice | enemy Dice | projectile Dice | projectile AP | empty-frame FPR @ 0.5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20,000 | 2,560,000 | 2.672 | 0.273368 | 0.107388 | 0.010438 | 0.013778 | 19.8138 | 0.617711 | 0.643461 | 0.389318 | 0.987193 | 0.851081 |

| recorded values | source |
|---|---|
| 20,000-step validation metrics and 119,410-frame population | `runs/encoder-baseline/one-token-wsd-20000/validation.json` |
| frames seen and epoch count | `runs/encoder-baseline/ladder.json` |


## 4. Conclusion

Training loss continued to improve after the 20,000-step budget, but only slowly.
Frame-by-frame inspection found that player and enemy positions were represented well,
while the projectile heatmap had a very high empty-frame false-positive rate. The
512-D one-token encoder therefore serves as the baseline, but its encoding of small
assets is not working well enough; experiment 0015 tests whether widening the same
one-token bottleneck to 1024 dimensions reduces those false positives.
