# Indexed frame chunks for encoder training

Status: Proposed

## Memory-mapped chunk layout

Encoder training consumes one directory per source shard. Each directory contains
`frames.npy` (`uint8`, N×224×240×3), `targets.npy` (`float16`, N×3×32×32),
`keys.json`, and `manifest.json`. Standard NPY headers make both arrays directly
memory-mappable without adding a storage dependency. Chunks preserve source order;
loaders shuffle integer indices while the split remains an episode-level property.

The uncompressed arrays trade disk capacity for predictable random access. At the
0012 corpus size they require about 200 GB, which fits the training host and avoids
repeated PNG decoding, JSON parsing, and Gaussian construction on every epoch.
The lossless tar corpus remains the reproducible source and may regenerate this
disposable cache.

## Precomputed supervision

Conversion computes the three player, enemy, and merged-projectile Gaussian maps once
from source JSON. It uses the same 32×32 grid, native-pixel sigmas `(6,6,4)`, maximum
composition, and coordinate convention as the existing online target builder. Pixel
weights remain derived in batches from these maps, so their interpolation can run as
one vectorized operation on the training device.

## Resumable conversion

Each chunk is written to a temporary directory, flushed, and published by atomic
rename only after its arrays, keys, and manifest are complete. The manifest records
the source tar name and SHA-256, split, count, shapes, and dtypes. A subsequent build
skips a published chunk only when that identity and layout match, making interrupted
conversions resumable without accepting partial arrays.

## Shared loader contract

The indexed loader yields `(frame, target, key)` and partitions chunks between data
workers. It is an optimization beneath the experiment: model inputs, objectives,
splits, frame population, and evaluation definitions do not change. The tar loader
stays available for corpus verification and cache regeneration.
