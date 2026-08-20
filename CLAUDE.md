# contra_nes_data — project instructions

## Document names

Numbered documents use `doc/NNNN-design-<topic>.md` for design decisions and
`doc/NNNN-exp-<topic>.md` for experiments. Both share one global sequence. `pytest`
enforces this for numbered files directly under `doc/`; `README.md`, unnumbered
references, and `doc/archive/` are exempt.

Every `doc/NNNN-exp-*.md` uses exactly four level-two sections, in order: Goal,
Setup, Evaluation metrics, and Conclusion. List every run in Setup and source every
number in Evaluation metrics. The conclusion is drafted by the user, never an
assistant; leave the appropriate `_Pending_` line and ask for it.

Every new `doc/NNNN-design-*.md` organizes level-two sections by concrete design
feature. Do not use generic headings such as Decision, Why, Evidence, The design,
Rejected alternatives, Risks, Sequencing, or Appendix provenance. Put evidence,
tradeoffs, and gates inside the feature they affect. Historical design docs are
allowlisted by `tests/test_doc_names.py`; the rule applies to all new design docs.

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

## Cross-repo work
Coordinate with `contra_nes_policy` and `contra_nes_evaluation` via **GitHub
issues** on the target repo (skill: `contra-nes-handoff`). Do not leave long
handoffs only in chat. File work where it must be done; open a consumer issue
when shards/API are ready for the other side.
