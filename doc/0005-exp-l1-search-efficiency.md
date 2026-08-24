# L1 search efficiency

Status: Implemented

## 1. Goal

Test whether the fast Spread-derived setup reaches the Level 1 boss more
efficiently than the classic search setup. Stop at boss entry; weapon does not
gate success.

## 2. Setup

The planned screen was 20 interleaved attempts per arm. It was stopped early at
six paired attempts per arm after the user accepted any weapon. One seventh fast
attempt completed before shutdown and is excluded from the comparison. Use a
`boss_entry` goal that succeeds on the first `boss_scene: false → true` edge.
Hold the `clean` reward, Level 1 action table, 28 workers, 300-second limit,
3,000-action limit, and frame skip 3 constant.

| arm | rollouts | rollout length | settle margin | max rewind |
|---|---:|---:|---:|---:|
| classic | 64 | 48 | 16 | 30 |
| fast Spread | 16 | 24 | 8 | 15 |

Every successful route was independently replayed to the boss-entry edge. Raw
attempt records are in `tmp/level1-spread-efficiency/results.jsonl`.

## 3. Evaluation metrics

| metric | classic | fast Spread |
|---|---:|---:|
| replay-valid entries | 6/6 | 6/6 |
| entries/hour | 60.5 | 175.1 |
| mean / p90 wall seconds | 59.5 / 71.7 | 20.6 / 23.8 |
| mean sampled actions | 218,998 | 54,638 |
| mean trace steps | 1,053 | 1,057 |
| exact duplicates | 0 | 0 |

Numbers come from the first six attempts per arm in
`tmp/level1-spread-efficiency/results.jsonl`.

## 4. Conclusion

Use the fast `16/24/8/15` setup for Level 1 search to boss entry. It is 2.89x
faster than classic with equal observed success and comparable trace length.
Weapon at boss entry is not a selection requirement.
