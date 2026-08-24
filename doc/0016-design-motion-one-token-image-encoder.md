# Motion-aware one-token image encoder

Status: Superseded by 0018

**Question.** How should one 512-D image token retain motion and small moving objects
without spending most of its capacity repeatedly encoding a static or globally
scrolling background?

**Answer.** Encode the current RGB frame together with a camera-aligned signed residual
from the preceding frame. Estimate one global translation from pixels only, never from
Contra RAM, and mask newly exposed borders. Fuse a full appearance pathway and a
lightweight residual pathway into the existing single 512-D contract. First validate
registration; then compare the motion-aware model against 0010 at the same 20,000-step
budget and unchanged validation split.

## Pixel-only translation removes global camera motion

For consecutive frames `(previous, current)`, estimate one global integer translation
`(dx,dy)` by robust image registration. Alignment operates on luminance, searches a
bounded translation window, and minimizes trimmed absolute error over overlapping
pixels so independently moving sprites cannot dominate the background match. The
search bound and trimming fraction must be fixed by a registration canary before
encoder training, then recorded in the checkpoint configuration.

Shift the previous RGB frame into current-frame coordinates and calculate a signed
floating-point residual in `[-1,1]`. A one-channel validity mask marks pixels with a
real correspondence; newly exposed borders are zeroed rather than reported as motion.
Scene cuts, episode starts, low-confidence matches, and displacements outside the
declared bound produce an all-invalid residual. The estimated normalized `(dx,dy)` is
retained as input so alignment does not erase useful global progress information.

This is deliberately global registration, not dense optical flow. It estimates two
translation values per frame pair instead of a vector per pixel. Contra scroll RAM may
audit estimator accuracy during development, but it is never an encoder input or
published contract, keeping the method applicable to ordinary video.

## Appearance and residual pathways fuse into one 512-D token

The appearance pathway consumes the current native 224×240 RGB frame and preserves
the 0010 spatial backbone. A lower-capacity motion pathway consumes three signed
residual channels plus the validity mask. Their reduced feature maps, along with
normalized `(dx,dy)`, are concatenated and projected to one normalized 512-D token.
The exact motion-path width and resulting parameter count are fixed in the experiment
setup before training; the output width does not change.

This follows the established two-stream separation of appearance and motion while
using residuals instead of optical flow. The appearance path protects stationary
player, enemy, terrain, and weapon identity. The residual path spends its capacity on
independent changes such as projectile displacement. One fused token preserves the
policy sequence-length advantage of 0009.

The first frame of every episode has no predecessor and uses an invalid zero residual.
Pairs never cross episode or split boundaries. Training windows may be shuffled only
after consecutive pairs and their alignment metadata have been constructed.

## Registration and representation gates precede publication

Registration is audited before model training. Report estimated-versus-RAM horizontal
scroll error on Contra as a development oracle, the fraction of low-confidence pairs,
and residual energy inside known static background regions before and after alignment.
Also inspect scrolling, stationary boss, episode-start, and transition examples. A
failed registration audit blocks training rather than teaching the motion branch from
dense false residuals.

The representation experiment uses the same frozen corpus, 800/100/100 episode split,
seed, objectives, batch sizes, and 20,000-step WSD schedule as 0010. It compares exactly
the implemented 512-D static baseline and one 512-D motion-aware candidate. Required
validation metrics remain RGB MSE/PSNR, per-class Dice and AP, positive/empty counts,
and per-class empty-frame FPR. Add projectile localization error on positive frames.

The candidate must materially improve projectile Dice and localization without
worsening projectile AP, enemy/player Dice, or weighted MSE beyond the 0015 guardrails.
A threshold-only FPR improvement cannot pass. Passing creates an offline candidate;
publishing new token shards and downstream policy evaluation remain separate decisions.

| claim | source |
|---|---|
| static 512-D baseline and validation population | experiments 0010 and 0015 |
| appearance/motion pathway separation | Simonyan and Zisserman, *Two-Stream Convolutional Networks*, NeurIPS 2014 |
| lightweight fast motion pathway | Feichtenhofer et al., *SlowFast Networks*, ICCV 2019 |
| residual video signals reduce temporal redundancy | Wu et al., *Compressed Video Action Recognition*, CVPR 2018 |
| Contra horizontal scroll oracle | `env.utility.xscroll`; `ADDR_XSCROLL` in `env.constant` |
