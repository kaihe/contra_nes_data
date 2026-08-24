# Design documents

- [0001 Boss-search curriculum](0001-design-boss-search-curriculum.md) — Implemented — verified full-fight boss candidates and frame-balanced shard releases
- [0002 Spread-only boss validation](0002-design-spread-only-validation.md) — Proposed — disjoint Spread train/validation release
- [0006 Cloud trace worker bootstrap](0006-design-cloud-trace-worker.md) — Implemented — safe minimal Ubuntu provisioning for MC search
- [0007 Distributed trace ingest](0007-design-distributed-trace-ingest.md) — Accepted — immutable cloud batches with single-writer ingestion
- [0008 Level 1 full 10k](0008-design-level1-full-10k.md) — Accepted — one full encoding with three virtual sequence views
- [0009 One-token image encoder](0009-design-one-token-image-encoder.md) — Implemented — native 224×240 frame to one 512-D continuous position
- [0010 One-token image encoder baseline](0010-exp-level1-encoder-baseline.md) — Implemented — fixed 20k/512-D reconstruction and entity baseline
- [0011 Four-code VQ image encoder](0011-design-vq-image-encoder.md) — Proposed — four offline spatial codes enter GPT as four positions per frame
- [0012 Four-token VQ codebook size](0012-exp-vq-codebook-size.md) — Proposed — fixed frame corpus and four codebook capacities
- [0015 Enhanced one-token image encoder](0015-exp-enhanced-one-token-image-encoder.md) — Implemented — 1024-D does not improve projectile localization or hallucination enough to replace 512-D
- [Entities](ENTITIES.md) — Legacy — RAM entity taxonomy and extraction notes
- [Events](EVENTS.md) — Legacy — event semantics and trace statistics
