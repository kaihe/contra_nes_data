# Incremental Spread scaling releases

Status: Proposed

## Decision

Build immutable, shard-only Spread releases at fixed trace-count snapshots so
policy experiments can start before the complete trace bank is encoded. The
first snapshots contain 10,000 and 20,000 traces from the 2026-08-09 Spread
session. Each has a fresh 100-task deterministically fingerprint-ranked
validation holdout; the remainder forms generated-only training shards.

Raw traces remain in the production trace bank. Converted task files are build
intermediates, not release artifacts. Each completed release contains only
training-ready WebDataset shards and a manifest with source/task/shard hashes.

## Rationale

The previous all-scale builder computes nearest-neighbour diversity for every
candidate. At 50k candidates this is quadratic work yet, with a zero threshold,
does not reject near-similar trajectories. Keep exact state/action deduplication
and manifest counts, but omit pairwise diversity scores for scaling releases.

## Scaling contract

Subsequent releases use independently immutable candidate snapshots and the
same 100-task validation size. Shard prefixes are deterministic, enabling
reproducible 1, 2, 4, ... shard training-scale experiments.
