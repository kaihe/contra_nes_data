# Spread-only boss validation release

Status: Proposed

## Decision

Publish a new immutable `boss-spread-v2` release with a disjoint 1,317/57
train/validation split drawn exclusively from the 1,374 replay-verified,
full-fight Spread traces. Leave `boss-spread-v1` intact: it remains useful for
training-only scale experiments but is invalid for held-out evaluation because
it contains every available Spread trace in train.

## Split contract

The release builder will calculate every candidate's state/action fingerprint
after replay conversion, order the distinct fingerprints lexicographically, and
reserve the first 57 for validation. The remaining 1,317 candidates form the
generated-only training set. This makes the split reproducible from the raw
trace bank and ensures no task fingerprint can appear in both partitions.

The validation shard is newly encoded from the held-out Spread tasks; it is not
a copy of the former mixed-weapon validation tar. Train shards contain only the
remaining Spread tasks. The existing 466-task boss train shard is used only as
a diversity reference and is not included in either v2 split.

## Verification

The manifest must record both partition task fingerprints and SHA-256 hashes,
the validation episode count, and train/validation frame totals. The builder
must reject duplicate fingerprints before partitioning and prove the two UID
sets are disjoint. `contra_nes_policy` and `contra_nes_evaluation` need a
handoff because their previous 57-example comparison metric is no longer the
appropriate metric for this Spread-only experiment.

## Provenance

| claim | source |
|---|---|
| 1,374 full-fight Spread traces are available | state-bank provenance scan on 2026-08-07 |
| `boss-spread-v1` has all 1,374 in train and old mixed 57 in validation | `game_trace/releases/boss-spread-v1/manifest.json` |
