# Commit immutable trace batches to Google Cloud Storage

Status: Proposed

**Question.** How should many CPU workers publish MC wins to shared cloud
storage without losing traces, creating duplicates, or exposing partial uploads?

**Answer.** Google Cloud Storage (GCS) is the primary durable store. Each live worker saves wins
atomically on local disk and closes a batch at exactly 100 traces; finite canonical
archives close at 1,000 traces. Each producer uploads a
compressed archive and manifest with create-only object preconditions, then
creates `COMMITTED.json` as the visibility
boundary. One ingester validates committed batches, deduplicates trace
fingerprints, and owns canonical metadata. Workers never write shared SQLite or
derive their action prior from live output.

---

## Worker identity and ownership

| role | responsibility |
|---|---|
| worker | search, atomic local save, batch, resumable upload, retain until acknowledged |
| GCS | primary durable batch, manifest, marker, and acknowledgement storage |
| ingester | validate, deduplicate, publish canonical raw traces and metadata |
| datahouse builder | consume canonical traces and produce token shards later |

Every launch has a random `run_id`, each machine a stable `worker_id`, and every
upload a random `batch_id`. A trace's identity is the SHA-256 fingerprint of
normalized initial state, actions, frame skip, level, and goal—not its filename.

Workers authenticate through Application Default Credentials. A GCP worker uses
an attached service account; a worker on another cloud prefers Workload Identity
Federation and may use a narrowly scoped service-account key only during initial
deployment. Workers receive bucket-level Object Creator and Object Viewer roles,
not bucket administration or object deletion.

## GCS layout and object identity

One dedicated bucket contains the tracehouse prefix:

```text
gs://<bucket>/contra-mc-tracehouse/
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
    level1/boss/
      batches/<collection_id>/<batch_id>/
        traces.tar.zst
        manifest.json
        COMMITTED.json
```

`run.json` records code commit, search configuration, prior digest, worker
identity, machine shape, start time, and software versions. `manifest.json` is
stored both beside and inside the archive. For every trace it records
fingerprint, member name, SHA-256, byte size, outcome, action/search steps,
sampled actions, deaths, wall-clock time, and initial-state/config provenance.
Full traces also carry `boss_weapon`, `boss_rapid`, and zero-based
`boss_entry_step`, captured from RAM at the boss-scene edge. This makes Spread
and rapid-fire filtering a metadata query rather than an ingestion-time replay.
Every trace records `trace_scope`: `full_level` beneath `level1/full` and
`boss_fight` beneath `level1/boss`. The physical prefix prevents accidental
mixed loading; the manifest field lets catalogs classify traces without relying
on object paths or initial-state hashes.
Finite archives also record a stable `collection_id` in every manifest row.
The canonical 40k-per-weapon set uses `level1-boss-canonical-80k-v1`; valid wins
outside that selection use `level1-boss-extra-v1`. Consumers therefore opt into
extra candidates explicitly without treating them as failures or inferring
membership from producer IDs.

## Legacy trace replay and source preservation

The worker also accepts existing NPZ globs as a finite pseudo-search source. It
hard-links each source into the durable spool when possible (copying only across
filesystems), records its SHA-256 in a restart journal, and never edits or deletes
the source file. Legacy traces missing boss loadout fields are replayed once with
one persistent emulator, except boss-start traces whose equivalent legacy
`weapon` and `rapid` fields can be mapped directly; recovered `boss_weapon`, `boss_rapid`, and
`boss_entry_step` are written into the batch manifest while the archived NPZ
remains byte-identical. Imports accept an explicit source list for catalog-selected
collections. Canonical bulk archives close at 1,000 traces and explicitly flush
their final partial batch; live worker batches remain 100 traces for bounded loss
and upload latency.

Every GCS object name is unique within a bucket. Uploads use
`if_generation_match=0`, so retries cannot overwrite an existing object. Object
metadata carries `run_id`, `worker_id`, `batch_id`, and schema version. A retry
accepts an existing object only when its size and checksum match; a mismatch is
quarantined.

## Resumable worker commit protocol

1. Write each NPZ to a temporary local name, `fsync`, then rename atomically.
2. Close a batch at 100 wins. Graceful shutdown leaves a partial batch in the
   local journal for the next launch; an explicit final flush may close it early
   when permanently retiring a worker.
3. Build `manifest.json` and `traces.tar.zst` atomically on local disk.
4. Upload archive and manifest with resumable GCS sessions and
   `if_generation_match=0`; retry transient failures with exponential backoff.
5. Verify GCS-reported size and checksum against local files. Create
   `COMMITTED.json` last with object names, generations, hashes, counts, and creation time.
6. Keep local NPZ/archive files until the ingester uploads an acknowledgement.
   Retried batches retain their original IDs.

An archive without `COMMITTED.json` is invisible and eligible for trash after
24 hours. The protocol does not depend on object move or rename being atomic.
Matching repeated markers are idempotent; conflicting markers are quarantined.

## Canonical ingestion and deduplication

One logical ingester lists commit markers beneath the configured GCS prefix,
groups them by `batch_id`, verifies object generations, hashes, and manifest invariants,
and recomputes trace fingerprints. It accepts unseen fingerprints and records
duplicates as provenance without creating a second canonical trace. Only after
its database transaction commits does it upload the immutable acknowledgement object.

Workers never open `game_trace/datahouse/catalog.sqlite`; that catalog describes
token shards, not raw searches. Raw search metadata uses a separate ingester-owned
database. This keeps retries and schema migration out of the search hot path.

## Recovery, monitoring, and fixed priors

Workers journal batch state as `open`, `uploaded`, `committed`, or
`acknowledged`. Startup resumes the oldest incomplete batch before new search.
Before joining production, a worker performs an authenticated GCS
upload/download/delete canary. Sustained GCS failure
pauses new search while durable local batches continue retrying.

Metrics cover wins/hour, pending batches, upload age, retries, GCS API errors,
duplicate traces, bytes, and save-to-ack latency. Alerts fire when a committed
batch lacks acknowledgement for 15 minutes or a worker misses twice its expected
commit interval.

Every run uses one compact prior identified by SHA-256 in `run.json`. It is never
rebuilt from live output. Operational gates are: fewer than 0.5% wins lost in a
10-worker interruption test; median archive size 1–64 MiB; duplicate rate below
1%; p95 commit-to-ack below 15 minutes at twice planned scale; and zero accepted
checksum mismatches during fault injection.

## Staged GCS scale-up

1. Archive 100 current NPZs and accept the default only if the size gate passes.
2. Define fingerprint and JSON schemas with golden tests.
3. Implement a local-filesystem adapter and fault every worker state transition.
4. Implement GCS resumable upload, generation preconditions, object metadata,
   generation journals, checksum verification, and the authenticated canary.
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
| GCS supports Application Default Credentials and generation preconditions | Google Cloud Storage client documentation; authenticated upload remains a gate |
