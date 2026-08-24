# Level 2 search efficiency

Status: Implemented

## 1. Goal

Find the highest-throughput Monte Carlo configuration for replay-valid full
Level 2 wins on the eight-core cloud1 worker. First compare the fast Level 1
shape and the proven default from the old `contra_agent` repository; then scale
up breadth and lookahead around the winning default, then isolate commit and
rewind controls. Select a production configuration only after the control arms
complete 10 attempts. After adoption, measure live throughput on the cloud
workers that run that configuration, because their host CPUs differ. Screen
new hosts from `lscpu` before bootstrap and keep only Platinum-class CPUs.

## 2. Setup

Stage 1 planned 10 attempts per arm in interleaved rounds, with the Level 1
setup first and the other arms shuffled with seed `20260822`. It stopped after
23 relevant attempts because the 48-step default clearly led the 24-step arms.
It held Level 2, the `level_up` goal, eight emulator workers, frame skip 3, a
180-second per-attempt limit, 6,000-action limit, action table, reward, and
four-recording bigram prior constant.

| arm | rollouts | rollout length | settle margin | max rewind |
|---|---:|---:|---:|---:|
| Level 1 fast | 16 | 24 | 8 | 15 |
| fewer rollouts | 8 | 24 | 8 | 15 |
| more rollouts | 32 | 24 | 8 | 15 |
| old Level 2 default | 64 | 48 | 16 | 30 |

Stage 2 runs 10 attempts per arm in interleaved rounds, with the baseline first
and the other arms shuffled with the same seed. It raises the limit to 240
seconds and holds every other runtime and data setting from stage 1 constant.
It stopped after 13 attempts because all arms were reliable and the unscaled
baseline led throughput.

| arm | rollouts | rollout length | settle margin | max rewind |
|---|---:|---:|---:|---:|
| baseline | 64 | 48 | 16 | 30 |
| wider | 96 | 48 | 16 | 30 |
| deeper | 64 | 64 | 20 | 30 |
| wider + deeper | 96 | 64 | 20 | 30 |

Stage 3 holds rollouts 64 and rollout length 48 constant, interleaves the
baseline first, and changes one control at a time.
Settle margin 8 permits commit lengths 20–40; margin 16 permits 16–32; margin
24 permits 12–24. It retains the stage 2 seed, eight workers, and 240-second
limit.

| arm | settle margin | max rewind |
|---|---:|---:|
| baseline | 16 | 30 |
| longer commits | 8 | 30 |
| shorter commits | 24 | 30 |
| shallow rewind | 16 | 15 |
| deep rewind | 16 | 45 |

Stage 3 stopped its broad schedule after 11 attempts because longer commits led
at 61.69 wins/hour. A narrowed confirmation resumes the same append-only log
until longer commits reach five total attempts. Adoption requires all five to
replay successfully and their throughput to remain above the three-attempt
baseline.

Cloud1 uses commit `99047377fbd1a9277285cf054da91663e0e5c206` on
`feat/gcs-legacy-import`, Python 3.12.3, and eight logical CPUs. The Level 2
configuration SHA-256 is `28488bc5...3857`; source-prior trace hashes are
recorded by `sha256sum game_trace/human_recordings/Level2/*.npz` on cloud1.
Stage 1 attempt rows and its final aggregate are respectively
`tmp/level2-search-efficiency-focused/results.jsonl` and `summary.json` on
cloud1. Stage 2 writes the same filenames under
`tmp/level2-search-efficiency-scale-up/`; stage 3 uses
`tmp/level2-search-efficiency-control-sweep/`. Three superseded pilot attempts remain in
`tmp/level2-search-efficiency/results.jsonl`; the removed `16/16/4/15` arm is
excluded from this comparison.

Production search on cloud1–cloud8 uses the adopted `64/48/8/30` configuration,
eight emulator workers, a 600-second per-attempt limit, 6,000-action limit,
`level_up` goal, and commit `99047377fbd1a9277285cf054da91663e0e5c206`. Each
worker writes `game_trace/worker_spool/cloudN-level2` and uploads to
`gs://contra_nes_trace/contra-mc-tracehouse/schema-v1/level2/full`. Cloud1
started at 12:31 CST; cloud2–cloud4 started together at 12:32 CST; cloud5
started at 13:26 CST on a host that had been up for 28 minutes. Cloud6 is a
later screen of another Platinum 8468 host; it started at 13:56 CST after
bootstrap from the same git bundle, ROM, GCS credential, and Level 2 human
priors. Cloud7 is a later screen of an unseen Broadwell SKU (Xeon E5-2698 v4);
it started at 14:33 CST. Cloud8 is a later Platinum 8468 keeper; it started at
14:48 CST. The first fleet snapshot is 2026-08-22 13:45 CST; cloud6 is 14:08
CST; cloud7 is 14:43 CST; cloud8 is 14:53 CST.

| worker | class | screen | CPU | start (CST) | parent elapsed |
|---|---|---|---|---|---:|
| cloud1 | Gold 6133 | drop | Xeon Gold 6133 @ 2.50 GHz | 12:31 | 74.46 min |
| cloud2 | Platinum 8352V | keep | Xeon Platinum 8352V @ 2.10 GHz | 12:32 | 73.06 min |
| cloud3 | Platinum 8468 | keep | Xeon Platinum 8468 @ 2.10 GHz | 12:32 | 73.06 min |
| cloud4 | Gold 6133 | drop | Xeon Gold 6133 @ 2.50 GHz | 12:32 | 73.07 min |
| cloud5 | Gold 6133 | drop | Xeon Gold 6133 @ 2.50 GHz | 13:26 | 18.86 min |
| cloud6 | Platinum 8468 | keep | Xeon Platinum 8468 @ 2.10 GHz | 13:56 | 11.53 min |
| cloud7 | E5-2698 v4 | drop | Xeon E5-2698 v4 @ 2.20 GHz | 14:33 | 9.81 min |
| cloud8 | Platinum 8468 | keep | Xeon Platinum 8468 @ 2.10 GHz | 14:48 | 4.33 min |

## 3. Evaluation metrics

Stage 1 stopped at 2026-08-22 11:40:36 CST after 23 relevant attempts.

| arm | valid wins / attempts | wins/hour | seconds/valid win | duplicates |
|---|---:|---:|---:|---:|
| old Level 2 default | 6/6 | 47.21 | 76.25 | 0 |
| Level 1 fast | 4/6 | 22.62 | 159.17 | 0 |
| more rollouts | 4/6 | 19.39 | 185.69 | 0 |
| fewer rollouts | 1/5 | 4.73 | 761.13 | 0 |

All stage 1 numbers come from cloud1
`tmp/level2-search-efficiency-focused/summary.json`; that aggregate counts only
replay-valid wins and charges failed attempts' full wall time to throughput.
Stage 2 stopped at 2026-08-22 12:00:24 CST after 13 attempts; every result was a
unique replay-valid win.

| arm | valid wins / attempts | wins/hour | seconds/valid win | duplicates |
|---|---:|---:|---:|---:|
| baseline | 4/4 | 53.06 | 67.85 | 0 |
| deeper | 3/3 | 47.42 | 75.91 | 0 |
| wider + deeper | 3/3 | 42.97 | 83.77 | 0 |
| wider | 3/3 | 41.14 | 87.51 | 0 |

These stage 2 numbers come from cloud1
`tmp/level2-search-efficiency-scale-up/summary.json`. They do not replace the
planned 10-attempt comparison because the stage stopped early, but they identify
`64/48/16/30` as the highest-throughput arm.

Stage 3 broad-sweep snapshot at 2026-08-22 12:19:02 CST; all 11 attempts were
unique replay-valid wins.

| arm | valid wins / attempts | wins/hour | seconds/valid win | duplicates |
|---|---:|---:|---:|---:|
| longer commits | 2/2 | 61.69 | 58.36 | 0 |
| baseline | 3/3 | 40.21 | 89.53 | 0 |
| shallow rewind | 2/2 | 35.07 | 102.64 | 0 |
| shorter commits | 2/2 | 33.63 | 107.06 | 0 |
| deep rewind | 2/2 | 32.13 | 112.05 | 0 |

These numbers come from cloud1
`tmp/level2-search-efficiency-control-sweep/summary.json`. The final stage 3
table replaces this snapshot after the three longer-commit confirmations.

The narrowed confirmation completed at 2026-08-22 12:28 CST.

| arm | valid wins / attempts | wins/hour | seconds/valid win | duplicates |
|---|---:|---:|---:|---:|
| longer commits | 5/5 | 62.53 | 57.57 | 0 |

The five attempt times were 61.45, 55.26, 53.48, 68.46, and 49.20 seconds.
Numbers come from cloud1
`tmp/level2-search-efficiency-control-sweep/results.jsonl` and `summary.json`.
Against the three-attempt settle-16 baseline at 40.21 wins/hour, settle 8
improved observed throughput by 55.5% while retaining perfect replay validity.

Production snapshot at 2026-08-22 13:45 CST. Wins/hour uses parent-process
elapsed time, so failed attempts and startup are charged. Search-wall seconds
are the successful-win `search_wall_s` values stored in each NPZ.

| worker | class | wins | wins/hour | seconds/valid win | search-wall mean / median | p10 / p90 wall s |
|---|---|---:|---:|---:|---|---|
| cloud1 | Gold 6133 | 59 | 47.54 | 75.72 | 74.70 / 68.47 | 55.10 / 94.54 |
| cloud2 | Platinum 8352V | 97 | 79.66 | 45.19 | 44.21 / 42.22 | 33.74 / 55.12 |
| cloud3 | Platinum 8468 | 174 | 142.90 | 25.19 | 24.48 / 22.97 | 19.85 / 30.40 |
| cloud4 | Gold 6133 | 65 | 53.37 | 67.45 | 66.79 / 64.06 | 49.33 / 84.71 |
| cloud5 | Gold 6133 | 11 | 35.00 | 102.87 | 74.51 / 76.28 | 58.31 / 87.23 |
| cloud6 | Platinum 8468 | 19 | 98.85 | 36.42 | 35.98 / 32.30 | 25.79 / 47.83 |
| cloud7 | E5-2698 v4 | 9 | 55.07 | 65.37 | 63.76 / 58.60 | 55.83 / 82.81 |
| cloud8 | Platinum 8468 | 8 | 110.74 | 32.51 | 31.18 / 30.67 | 27.42 / 35.45 |

Cloud1–cloud4 overlap for 73 minutes; their combined rate is 323.47 wins/hour.
Cloud3 is 3.01× cloud1 and 2.68× cloud4. The three Gold 6133 hosts cluster
well below the two Platinum hosts. Cloud5’s 18.86-minute sample includes one
362.85-second gap between wins (search-wall max 93.59 s), which pulls
parent-elapsed throughput below its successful-win wall times. Only cloud3 had
sealed and committed a 100-trace GCS batch by the 13:45 snapshot
(`e8394e6216224cf19fca112289eaa8b9` at 13:13 CST). Steal time was 0 on every
host.

Cloud6’s 11.53-minute screen is 2.08× cloud1 and 1.24× cloud2, but only 0.69×
cloud3 despite the same Platinum 8468 model name. Median successful-win wall
is 32.30 s versus cloud3’s 22.97 s; that gap is consistent across the 19 wins,
not one long failure. Cloud1, cloud3, and cloud6 all run GNOME, and steal time
is 0, so the 8468 split is not explained by desktop processes or visible steal.
Zero uniform-prior warnings after the Level 2 human recordings were copied.

CPU model from `lscpu` is the immediate screen. All eight hosts had 8 logical
CPUs, ~15 GiB RAM, and 0 steal; those fields did not separate them.

Cloud7’s 9.81-minute screen is 55.07 wins/hour: 1.16× cloud1 and 1.03× cloud4,
but 0.69× cloud2. Median successful-win wall is 58.60 s. Zero uniform-prior
warnings. Steal is 0. That puts E5-2698 v4 with the Gold 6133 drop class for
new hosts. Cloud7 itself stays in the live production fleet until the operator
stops it.

Cloud8’s 4.33-minute screen is 110.74 wins/hour: 1.12× cloud6 and 0.77× cloud3
on the same Platinum 8468 class. Median successful-win wall is 30.67 s. Zero
uniform-prior warnings.

| CPU class | workers | wins/hour | screen |
|---|---|---|---|
| Platinum 8468 | cloud3, cloud6, cloud8 | 142.90, 98.85, 110.74 | keep |
| Platinum 8352V | cloud2 | 79.66 | keep |
| E5-2698 v4 | cloud7 | 55.07 | drop |
| Gold 6133 | cloud1, cloud4, cloud5 | 47.54, 53.37, 35.00 | drop |

The 8468 hosts still differ by up to 1.45×, so keep is not a throughput forecast.
`python -m util.probe_cloud_host CLOUDN` reads the same fields over SSH.
Numbers come from each worker’s
`game_trace/worker_spool/cloudN-level2/search.log` and NPZ `search_wall_s`
fields; the local copies are
`tmp/level2-search-efficiency-production/snapshot-20260822-1345.json`,
`tmp/level2-search-efficiency-production/cloud6-snapshot-20260822-1408.json`,
and
`tmp/level2-search-efficiency-production/cloud7-snapshot-20260822-1443.json`,
and
`tmp/level2-search-efficiency-production/cloud8-snapshot-20260822-1453.json`.

## 4. Conclusion

Use `64/48/8/30` for Level 2 production search: 64 rollouts, rollout length 48,
settle margin 8, and max rewind 30.

Keep Platinum hosts only (8468 and 8352V). Drop Gold 6133 and E5-2698 v4 for
new workers. Probe with `lscpu` before bootstrap; same-SKU Platinum boxes still
vary, but they are the class worth running.
