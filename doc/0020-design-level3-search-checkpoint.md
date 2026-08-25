# Level 3 search starts from a replay-derived checkpoint

Status: Implemented

**Question.** How should Monte Carlo search avoid Level 3's unrewarded opening
approach without replacing the canonical level state or losing the origin of
generated traces?

**Answer.** Level 3 search defaults to a dedicated checkpoint captured at frame
40 of a replayable human win. The checkpoint lives in a manifested search-state
bank, is checksum-verified when loaded, and records its source trace and exact
subframe offset in every generated trace. Other consumers and levels continue
to use the canonical Spread states.

---

## A search-only state bank preserves the canonical start

`src/agent/states/search_start/Level3-frame40.state` is a gzip-compressed
stable-retro state. It does not replace
`src/agent/states/spread_gun/Level3.state`: replay, task extraction, and policy
environments therefore retain the full-level start unless they explicitly use
the search bank.

The adjacent `manifest.yaml` owns the state checksum and capture lineage.
`mc_search` verifies that manifest before using the checkpoint. A missing,
unlisted, or checksum-mismatched state is an error rather than a silent fallback.

## Level 3 alone receives the checkpoint override

`make_search_env(3, ...)` loads the search checkpoint by default. Levels 2 and
4–8 continue loading their Spread states, and Level 1 continues using the
integration state. An explicit `--initial-state` remains authoritative for
one-off experiments.

The selected inspector frame occurs after 39 complete decisions and one NES
subframe of decision 40. The environment resets before rewinding to the captured
bytes; this avoids stable-retro initialization shifting the selected frame.
Search then begins from the exact state, so its three-frame decision clock is
deterministic even though the source capture is not on a prior decision boundary.

## Generated traces carry checkpoint identity

Every default Level 3 search trace records `initial_state_file`,
`initial_state_sha256`, source trace SHA-256, inspector frame, completed decision
count, subframe offset, and source skip. This makes checkpoint-produced traces
distinguishable from canonical-start traces and supports later prefix stitching.

The first production gate is replay validity from the checkpoint. These are
checkpoint-scope traces until a separately validated prefix-stitching path
replays them from the canonical Level 3 state; they must not be described as
canonical full-level traces before that gate exists.

---

## Provenance and auditability

| claim | source |
|---|---|
| selected frame is 40 | manual inspection of `Level3_win_03220002.npz` |
| state SHA-256 is `36d2c9e8c27269f36cc7a6c54ee484414e9dcf27f8a2808dff77a239c8562212` | `tmp/level3-start-state/frame40.json` |
| capture is 39 decisions plus one subframe | old `contra/gui_state_inspect.py` replay loop |
| source trace SHA-256 is `8eae2e84129c7d1a0351c5e80e3282d92c2d421756273350c773f821ea165e38` | `tmp/level3-start-state/frame40.json` |
