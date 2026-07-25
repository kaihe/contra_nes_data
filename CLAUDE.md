# contra_nes_data — project instructions

## Where things go
- **`tmp/`** — research & design artifacts: heatmaps, analysis plots, exploratory
  scans, scratch data and one-off scripts used to *inform* model/task design.
  This directory is gitignored and disposable. Default any such output here (e.g.
  `util.pos_heatmap` writes to `tmp/`). Do **not** put research artifacts in
  `game_trace/` or `src/`.
- **`game_trace/`** — real datasets only: source traces (`mc_trace/`) and generated
  task datasets (`tasks/<kind>/`). Not scratch output.
- **`src/`** — library code (`env/`, `agent/`, `util/`, `task_maker/`), installed
  editable so imports drop the `src.` prefix.

## Rule of thumb
If an output exists to help us *understand or design* (a chart, a probe, a stat
dump), it belongs in `tmp/`. If it's *product* (a task dataset, reusable code), it
belongs in `game_trace/` or `src/`.
