# Commit immutable trace batches from stateless cloud workers

Status: Proposed

**Question.** How should many CPU workers publish MC wins to shared cloud
storage without losing traces, creating duplicates, or turning the object store
into millions of tiny mutable files?

**Answer.** Each worker writes every win atomically to local disk, then publishes
an immutable batch of 100 traces or every five minutes, whichever comes first.
It uploads a compressed archive followed by a small commit marker. A single
ingester accepts only committed, checksum-valid batches, deduplicates trace
fingerprints, and records canonical traces and search metadata. Workers never
write the shared SQLite catalog or derive their action prior from live output.

---

## Worker identity and ownership

| role | responsibility |
|---|---|
| worker | search, local atomic save, batch, upload, retain until acknowledged |
| object store | durable immutable batch and marker storage |
| ingester | validate, deduplicate, publish canonical raw traces and metadata |
| datahouse builder | consume canonical traces and produce token shards later |

Every launch receives a random `run_id`; each machine receives a stable
`worker_id`; every upload receives a random `batch_id`. Trace filenames may stay
human-readable, but identity is the SHA-256 fingerprint of normalized initial
state, actions, frame skip, level, and goal—not a timestamp or filename.

## Immutable object layout

One-object-per-trace makes discovery, retry, and lifecycle operations scale with
episode count. Copying whole worker directories provides no atomic boundary:
consumers can observe partial uploads and retries can create duplicates. Batches
provide one immutable visibility unit:

```text
mc-traces/v1/level1/full/
  runs/<run_id>/worker.json
  batches/<worker_id>/<batch_id>.tar.zst
  commits/<worker_id>/<batch_id>.json
  acknowledgements/<worker_id>/<batch_id>.json
```

`worker.json` records code commit, search configuration, prior digest, machine
shape, start time, and software versions. A batch archive contains NPZ traces
and `manifest.json`; the manifest records each trace's fingerprint, member name,
SHA-256, byte size, outcome, action/search steps, sampled actions, deaths,
wall-clock time, and initial-state/config provenance.

## Worker commit protocol

1. Write each NPZ to a temporary local name, `fsync`, then rename atomically.
2. Close a batch at 100 wins, five minutes, or graceful shutdown.
3. Build `manifest.json`, then create `<batch_id>.tar.zst` atomically locally.
4. Upload the archive under its final immutable key and verify size/checksum.
5. Upload the commit marker last with archive key, hash, manifest hash, counts,
   and creation time. Creating an existing marker is an idempotent success only
   when its bytes match.
6. Retain local NPZ/archive files until the ingester writes an acknowledgement.
   Retry uploads with the same IDs; never mint a new ID for the same batch.

An archive without a commit marker is invisible and eligible for garbage
collection after 24 hours. A marker is the visibility boundary; there is no
rename assumption because common object stores do not provide one.

## Canonical ingestion and deduplication

One logical ingester lists commit markers, claims each marker through its own
durable queue/lease, verifies all hashes and manifest invariants, and computes
fingerprints independently. It accepts unseen fingerprints and records duplicate
ones as provenance without copying them into canonical storage. Only after its
database transaction commits does it write the immutable acknowledgement.

The ingester owns the searchable metadata database. Cloud workers never open
`game_trace/datahouse/catalog.sqlite`; that catalog continues to describe token
shards, not partially ingested raw searches. Raw-search metadata gets a separate
database whose schema can later be designed around the fields above.

## Recovery, monitoring, and fixed priors

Workers keep a local journal with batch state: `open`, `uploaded`, `committed`,
or `acknowledged`. Startup resumes the oldest non-acknowledged batch before new
search. Spot-instance shutdown flushes the open batch and attempts upload, but
local persistence is not assumed; the five-minute limit bounds expected loss.

Metrics are wins/hour, attempted searches, pending local batches, upload age,
retries, rejected batches, duplicate traces, bytes, and time from local save to
acknowledgement. Alerts fire when the oldest committed batch is unacknowledged
for 15 minutes or a worker has not committed for twice its expected batch time.

The search prior is a compact immutable artifact identified by SHA-256 in
`worker.json`. Every worker in a run uses the same prior for its lifetime. It is
never rebuilt from live output: that would make workers incomparable and make
startup cost grow with the dataset.

Operational gates live with this feature: fewer than 0.5% wins lost in a
10-worker interruption test; median archive size 1–64 MiB; duplicate rate below
1%; p95 commit-to-ack below 15 minutes at twice the planned worker count; and
zero accepted checksum mismatches during fault injection.

## Staged scale-up

1. Measure 1,000 current NPZ sizes and archive 100; accept the default only if
   the archive-size gate passes.
2. Define normalized trace fingerprint and JSON schemas with golden tests.
3. Implement a local-filesystem object-store adapter and fault tests for every
   worker state transition.
4. Choose the cloud object store and add one backend; require checksum and
   create-if-absent semantics before using paid workers.
5. Run two workers for 1,000 wins, interrupt one, and pass all loss, duplicate,
   corruption, and latency gates.
6. Scale geometrically: 2, 4, 8, then the target worker count. Stop at the first
   gate failure.

No policy-repository handoff is required: policy consumes token shards after the
datahouse builder, not raw cloud batches.

---

## Provenance and auditability

| claim | source |
|---|---|
| Level 1 currently globs every NPZ for its prior | `src/agent/level1.yaml`, `ActionSampler.for_level` |
| parent and each pool worker rebuild the sampler | `agent.mc_search._run_one_search`, `_worker_init` |
| token-shard catalog has a single transactional owner | `doc/0004-design-tokenized-datahouse.md` |
| remote bootstrap separates ROM and generated data from Git | `deploy/README.md`, `deploy/setup_cloud_worker.sh` |
