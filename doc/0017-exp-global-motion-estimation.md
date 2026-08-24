# Can pixels recover global camera motion over a complete trace?

Status: Proposed

## 1. Goal

Determine whether pixel-only global translation can align every consecutive frame pair
in one complete Level-1 trace accurately enough to construct a motion residual. Contra
RAM supplies audit ground truth only. The selected estimator must not read RAM, entity
coordinates, actions, or trace metadata. This experiment gates the motion pathway in
0016: do not train that encoder if global registration is unreliable.

## 2. Setup

Replay the first validation episode selected by the frozen `l1-full-10k-v1` split:
fingerprint `9c15be3fc41e7febaafd1dcc8ca77a468b0d1283a14b13dfec571355aa8fc6ce`,
batch 26. Its source is the generation-pinned archive recorded in the collection
manifest. Retain the initial observation and every post-action observation, with native
224×240 RGB and the corresponding RAM snapshot. Never cross the episode boundary.

For pair `(previous,current)`, define audit truth as
`dx = -(xscroll(current)-xscroll(previous))` and `dy = 0`: when the viewport advances
right, the previous background must shift left into current-frame coordinates. A RAM
delta outside the estimator's ±16-pixel range is a discontinuity, reported separately
rather than counted as an ordinary alignment error.

Run exactly two estimators:

| run | estimator | inputs | output |
|---|---|---|---|
| zero | always `(0,0)` | none | control for stationary frames |
| robust | bounded coarse-to-fine translation | consecutive RGB only | integer `(dx,dy)` plus confidence |

The robust estimator converts RGB to deterministic integer luminance
`(77R+150G+29B)>>8`. Its coarse pass area-averages 4×4 pixels and exhaustively scores
translations at four-pixel increments over `[-16,16]²`. Its fine pass exhaustively
scores every full-resolution integer translation within ±3 pixels of the coarse
winner, clipped to `[-16,16]²`. This is bounded exhaustive search, not greedy search.

Each candidate is scored on its valid overlap by sorting absolute luminance errors,
discarding the largest 20%, and averaging the remainder. Trimming prevents independently
moving sprites from controlling a background-motion estimate. The winning score and
gap to the second-best distinct translation are recorded; no confidence threshold is
tuned or applied in this first measurement. Align RGB with the winner, zero newly
exposed borders, and emit a validity mask for later residual-energy analysis.

Write per-pair JSONL and a summary JSON under `tmp/0017-global-motion/`. Also render a
small audit sheet containing the worst exact-ground-truth misses, largest residuals on
RAM-stationary pairs, scroll onset, boss-entry auto-scroll, and the final transition.

## 3. Evaluation metrics

Report all metrics over the entire trace and separately for RAM-stationary, in-range
scrolling, and discontinuity pairs. The primary population is every non-discontinuity
pair; do not silently discard low-confidence or visually difficult frames.

| metric | purpose | source |
|---|---|---|
| exact `(dx,dy)` accuracy | strict registration correctness | estimate versus RAM scroll delta |
| within-one-pixel accuracy | tolerance relevant to small sprites | per-axis absolute error ≤1 |
| `dx`/`dy` MAE and maximum error | magnitude and direction of failures | estimate versus RAM truth |
| nonzero estimate on stationary pairs | false camera-motion rate | pairs with RAM delta `(0,0)` |
| aligned versus unaligned trimmed residual | whether registration removes static change | RGB overlap before/after estimated shift |
| confidence-gap distribution | whether failure rejection is feasible later | best and second-best robust scores |
| runtime per pair | feasibility inside corpus preparation | wall time excluding replay |

The estimator passes this one-trace gate only if exact accuracy is at least 99%, every
non-discontinuity pair is within one pixel, and median aligned residual is lower than
the zero-shift control on scrolling pairs without increasing it on stationary pairs.
The audit sheet remains required even when these aggregate gates pass.

| recorded fact | source |
|---|---|
| frozen trace identity, split, batch, and archive generation | `game_trace/datahouse/collections/l1-full-10k-v1.json`; deterministic split in `datahouse.vq_codebook` |
| RGB/RAM observation alignment | replay loop using `rewind_state`, `step_env`, and post-step capture |
| horizontal ground truth | `env.utility.xscroll` and `env.constant.ADDR_XSCROLL` |
| estimator outputs and measurements | `tmp/0017-global-motion/pairs.jsonl` and `summary.json` |

## 4. Conclusion

_Pending user conclusion after the whole-trace registration audit._
