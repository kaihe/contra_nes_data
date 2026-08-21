# Level 1 full-trace encoder baseline

Status: Proposed

## 1. Goal

Measure how well the frozen datahouse encoder preserves Level 1 entity locations on
the new full-trace distribution before deciding whether to retrain it on that data.
The checkpoint passes only if its enemy and enemy-bullet soft Dice meet the existing
0.96 and 0.91 gates.

## 2. Setup

Evaluate checkpoint `f36041bc…1923c` without training. Use the 1,000 largest
fingerprints in the frozen `l1-full-10k-v1` collection as a deterministic validation
set; reserve the other 9,000 episodes for a later training arm. Replay the immutable
GCS NPZ sources, resize RGB observations to 256×256, derive four 32×32 occupancy
heatmaps from emulator RAM with sigma 6 px, and evaluate every aligned observation.
Save the resolved collection, checkpoint, code revision, and aggregate counts under
`runs/encoder-baseline/l1-full-10k-v1/`.

## 3. Evaluation metrics

| metric | role | source |
|---|---|---|
| soft Dice per entity class | primary; enemy ≥0.96 and enemy-bullet ≥0.91 are gates | replayed RGB and RAM-derived heatmaps |
| MSE skill per entity class | secondary diagnostic against the all-zero predictor | same fixed validation observations |
| peak hit per entity class | secondary diagnostic for strongest-location accuracy | same fixed validation observations |
| evaluated observations and episodes | coverage and alignment check | frozen collection manifest and evaluator output |

## 4. Conclusion

Pending measurements and user conclusion.
