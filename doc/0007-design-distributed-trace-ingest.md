# Commit immutable trace batches to Google Drive

Status: Proposed

**Question.** How should many CPU workers publish MC wins to shared cloud
storage without losing traces, creating duplicates, or exposing partial uploads?

**Answer.** Google Drive is the primary durable store. Each worker saves wins
atomically on local disk and closes a batch at 100 traces, five minutes, or
graceful shutdown. It uploads a compressed archive and manifest through
resumable Drive API sessions, then creates `COMMITTED.json` as the visibility
boundary. One ingester validates committed batches, deduplicates trace
fingerprints, and owns canonical metadata. Workers never write shared SQLite or
derive their action prior from live output.

---

## Worker identity and ownership

| role | responsibility |
|---|---|
| worker | search, atomic local save, batch, resumable upload, retain until acknowledged |
| Google Drive | primary durable batch, manifest, marker, and acknowledgement storage |
| ingester | validate, deduplicate, publish canonical raw traces and metadata |
| datahouse builder | consume canonical traces and produce token shards later |

Every launch has a random `run_id`, each machine a stable `worker_id`, and every
upload a random `batch_id`. A trace's identity is the SHA-256 fingerprint of
normalized initial state, actions, frame skip, level, and goal—not its filename.

Workers authenticate with an OAuth refresh token kept outside Git and search
configuration. The first implementation may distribute one encrypted
`rclone`/Drive credential to trusted workers. A later upload gateway may keep
the credential on one stable host without changing the batch format.

## Google Drive layout and file identity

The Google AI Pro account owns one dedicated root folder:

```text
Contra MC Tracehouse/
  schema-v1/
    level1/full/
      priors/<prior_sha256>/level1.yaml
      runs/<run_id>/run.json
      batches/<worker_id>/<batch_id>/
        traces.tar.zst
        manifest.json
        COMMITTED.json
      acknowledgements/<worker_id>/<batch_id>.json
      quarantine/<batch_id>.json
```

`run.json` records code commit, search configuration, prior digest, worker
identity, machine shape, start time, and software versions. `manifest.json` is
stored both beside and inside the archive. For every trace it records
fingerprint, member name, SHA-256, byte size, outcome, action/search steps,
sampled actions, deaths, wall-clock time, and initial-state/config provenance.
Full traces also carry `boss_weapon`, `boss_rapid`, and zero-based
`boss_entry_step`, captured from RAM at the boss-scene edge. This makes Spread
and rapid-fire filtering a metadata query rather than an ingestion-time replay.

Google Drive permits duplicate names, so paths are not identity. Bootstrap
records the Drive folder ID for `schema-v1`; journals and markers record all
parent/file IDs. Files carry `run_id`, `worker_id`, and `batch_id` as Drive
`appProperties`. A retry queries those properties before creating a file. Two
files with one batch ID are duplicates and conflicting copies are quarantined.

## Resumable worker commit protocol

1. Write each NPZ to a temporary local name, `fsync`, then rename atomically.
2. Close a batch at 100 wins, five minutes, or graceful shutdown.
3. Build `manifest.json` and `traces.tar.zst` atomically on local disk.
4. Create or find the Drive batch folder by `batch_id`. Upload archive and
   manifest with resumable sessions; retry 403, 429, and 5xx responses with
   exponential backoff.
5. Verify Drive-reported size/MD5 and locally recorded SHA-256. Create
   `COMMITTED.json` last with Drive file IDs, hashes, counts, and creation time.
6. Keep local NPZ/archive files until the ingester uploads an acknowledgement.
   Retried batches retain their original IDs.

An archive without `COMMITTED.json` is invisible and eligible for trash after
24 hours. The protocol does not depend on Drive move or rename being atomic.
Matching repeated markers are idempotent; conflicting markers are quarantined.

## Canonical ingestion and deduplication

One logical ingester lists commit markers beneath the configured Drive root ID,
groups them by `batch_id`, verifies file IDs, hashes, and manifest invariants,
and recomputes trace fingerprints. It accepts unseen fingerprints and records
duplicates as provenance without creating a second canonical trace. Only after
its database transaction commits does it upload the immutable acknowledgement.

Workers never open `game_trace/datahouse/catalog.sqlite`; that catalog describes
token shards, not raw searches. Raw search metadata uses a separate ingester-owned
database. This keeps retries and schema migration out of the search hot path.

## Recovery, monitoring, and fixed priors

Workers journal batch state as `open`, `uploaded`, `committed`, or
`acknowledged`. Startup resumes the oldest incomplete batch before new search.
Before joining production, a worker performs an authenticated Drive
upload/download/delete canary and forces OAuth refresh. Sustained Drive failure
pauses new search while durable local batches continue retrying.

Metrics cover wins/hour, pending batches, upload age, retries, Drive API errors,
duplicate traces, bytes, and save-to-ack latency. Alerts fire when a committed
batch lacks acknowledgement for 15 minutes or a worker misses twice its expected
commit interval.

Every run uses one compact prior identified by SHA-256 in `run.json`. It is never
rebuilt from live output. Operational gates are: fewer than 0.5% wins lost in a
10-worker interruption test; median archive size 1–64 MiB; duplicate rate below
1%; p95 commit-to-ack below 15 minutes at twice planned scale; and zero accepted
checksum mismatches during fault injection.

## Staged Google Drive scale-up

1. Archive 100 current NPZs and accept the default only if the size gate passes.
2. Define fingerprint and JSON schemas with golden tests.
3. Implement a local-filesystem adapter and fault every worker state transition.
4. Implement Google Drive resumable upload, OAuth refresh, `appProperties`,
   file-ID journals, checksum verification, and the authenticated canary.
5. Run two workers for 1,000 wins, interrupt one, and pass all integrity gates.
6. Scale through 2, 4, 8, then the target worker count; stop at a failed gate.

Policy consumes token shards after datahouse construction, not Drive raw batches,
so no policy-repository change is required.

## Provenance and auditability

| claim | source |
|---|---|
| Level 1 has a committed fixed prior | `src/agent/priors/level1.yaml`, `ActionSampler.for_level` |
| token-shard catalog has a single transactional owner | `doc/0004-design-tokenized-datahouse.md` |
| bootstrap keeps ROM and data outside Git | `deploy/README.md`, `deploy/setup_cloud_worker.sh` |
| configured China worker reached Drive API and OAuth endpoints | remote canary, 2026-08-20; authenticated upload remains a gate |
