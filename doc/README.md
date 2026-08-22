# Design documents

- [0001 Boss-search curriculum](0001-design-boss-search-curriculum.md) — Implemented — verified full-fight boss candidates and frame-balanced shard releases
- [0002 Spread-only boss validation](0002-design-spread-only-validation.md) — Proposed — disjoint Spread train/validation release
- [0003 Incremental Spread scaling](0003-design-incremental-spread-scaling.md) — Proposed — shard-only fixed-size scaling releases
- [0004 Tokenized trace datahouse](0004-design-tokenized-datahouse.md) — Implemented — data-owned encoder and deduplicated token releases
- [0005 L1 search efficiency](0005-exp-l1-search-efficiency.md) — Implemented — fast search reaches the Level 1 boss 2.89x faster
- [0006 Cloud trace worker bootstrap](0006-design-cloud-trace-worker.md) — Implemented — safe minimal Ubuntu provisioning for MC search
- [0007 Distributed trace ingest](0007-design-distributed-trace-ingest.md) — Accepted — immutable cloud batches with single-writer ingestion
- [0008 Level 1 full 10k](0008-design-level1-full-10k.md) — Accepted — one full encoding with three virtual sequence views
- [0009 One-token image encoder](0009-design-one-token-image-encoder.md) — Implemented — native 224×240 frame to one 512-D continuous position
- [0010 One-token image encoder baseline](0010-exp-level1-encoder-baseline.md) — Proposed — reconstruction and entity baseline for continuous/VQ comparisons; carries the retired 0014 indexed-chunk record
- [0011 Level 2 search efficiency](0011-exp-level2-search-efficiency.md) — Implemented — `64/48/8/30`; keep Platinum workers only
- [0012 Boss Spread frame shards](0012-design-boss-spread-frame-shards.md) — Implemented — native 224×240 MKV + actions for the D10k Spread episode set, cataloged beside the token shards
- [PNG](PNG.md) — Reference — how PNG turns a 161,280 B frame into ~4,400 B losslessly, and why its bytes are a bad model input
- [Entities](ENTITIES.md) — Legacy — RAM entity taxonomy and extraction notes
- [Events](EVENTS.md) — Legacy — event semantics and trace statistics
