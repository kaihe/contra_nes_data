# How PNG compresses a Contra frame

A reference for reading the numbers in `doc/0010-exp-level1-encoder-baseline.md` and
for deciding what a model should be fed. Every figure here was measured on real
frames from `tmp/0010-one-token-compressed-1k` with `tmp/png-anatomy.py`,
`tmp/png-filters.py`, and `tmp/png-lz77.py`.

The short version: a frame is 161,280 raw bytes and PNG stores it in about 4,400
losslessly, roughly 37x. Almost all of that comes from one stage — LZ77 matching —
and the reason it works so well is that NES art is a mosaic of repeated 8x8 tiles
drawn from a 14-colour palette.

## The container is not the compressor

A PNG file is an 8-byte signature followed by length-tagged chunks, each carrying a
4-byte CRC. For our frames there are only three:

```
IHDR        13 B payload + 12 B framing     width, height, bit depth, colour type
IDAT     ~4,400 B payload + 12 B framing     the entire compressed image
IEND         0 B payload + 12 B framing     terminator
```

Framing costs 45 bytes total, under 1%. Everything interesting happens inside the
single `IDAT` payload, which is one continuous zlib stream. That single-stream
design matters later: there is no place in a PNG where high-entropy content is
stored separately from low-entropy content. The entire image is one blob.

## Stage 0 — what the raw bytes look like

The image is serialized as scanlines, top to bottom, each scanline being
`width x 3` bytes of interleaved R, G, B. For 240x224 that is 720 bytes per
scanline and 161,280 bytes total.

These bytes are extraordinarily redundant. A representative frame uses **14 distinct
colours**, and the top one covers 43.7% of the screen:

```
#000000  43.7%    #201888  16.3%    #009400  10.6%
#402C00   9.9%    #B8BCB8   7.5%    #80D010   4.7%
```

Measured as a plain symbol distribution, the byte stream carries only **2.37 bits per
byte** of zeroth-order entropy. If you did nothing but Huffman-code the raw bytes you
would already reach 47,791 B, a 3.4x saving, purely because most byte values never
occur.

## Stage 1 — scanline filtering

Before compression, PNG lets each scanline choose one of five *filters*. Each
predicts a byte from its already-decoded neighbours and stores only the residual,
modulo 256. With `bpp = 3`, `a` is the byte three positions left (same colour
channel, previous pixel), `b` is the byte directly above, `c` is above-left:

| type | name | stored value |
|---|---|---|
| 0 | None | `x` |
| 1 | Sub | `x - a` |
| 2 | Up | `x - b` |
| 3 | Average | `x - floor((a + b) / 2)` |
| 4 | Paeth | `x - p`, where `p` is whichever of `a`, `b`, `c` is closest to `a + b - c` |

One filter-type byte is prepended to each scanline, so filtering costs 224 bytes per
frame before it saves anything. Each scanline picks independently, which is why the
filter byte is per-row rather than per-image.

The intent is photographic: in a smooth gradient, neighbouring bytes are numerically
close, so residuals cluster near zero and the alphabet collapses. **On NES art this
backfires.** Here are 24 real bytes from scanline 120, and the same columns of
scanline 119:

```
row 120 raw    64  44   0  64  44   0   0   0   0   0   0   0   0   0   0  64  44   0   0   0   0  64  44   0
row 119 raw     0   0   0   0   0   0  64  44   0   0   0   0   0   0   0   0   0   0  64  44   0   0   0   0
Up filtered    64  44   0  64  44   0 192 212   0   0   0   0   0   0   0  64  44   0 192 212   0  64  44   0
```

The raw row is nine consecutive zeros and three exact copies of `64 44 0` — the kind
of structure a match-finder devours. The filtered row introduces `192` and `212`,
byte values that appear nowhere in the palette, and breaks the alignment between the
repeated groups. Filtering traded long-range exact repetition for local numerical
smallness, and on tiled art the repetition was worth far more.

The measurement, over 256 frames, building genuine PNGs with each filter forced and
counting every byte of framing:

| filter | truecolor | ratio | 4-bit palette | ratio |
|---|---:|---:|---:|---:|
| **None** | **4,397 B** | **36.7x** | **3,161 B** | **51.0x** |
| Sub | 5,392 B | 29.9x | 3,882 B | 41.5x |
| Up | 5,943 B | 27.1x | 3,687 B | 43.7x |
| Average | 7,790 B | 20.7x | 4,657 B | 34.6x |
| Paeth | 6,034 B | 26.7x | 3,867 B | 41.7x |

Filter `None` wins every column. Standard encoders do not discover this, because the
usual adaptive heuristic scores each scanline by the sum of absolute residuals — a
proxy for local entropy that is blind to what LZ77 will do later. Pillow's default
truecolor output is 6,197 B against 4,397 B for forced `None`: **29.1% larger than
necessary**, on this corpus, from a heuristic tuned for photographs.

## Stage 2 — DEFLATE, where the compression actually happens

The filtered scanlines are concatenated and passed through zlib, which is LZ77
followed by Huffman coding.

**LZ77** slides a 32 KiB window and replaces any sequence of 3-258 bytes that
appeared earlier with a `(length, distance)` back-reference. **Huffman** then
entropy-codes the resulting stream of literals, lengths, and distances, using either
a fixed table or a per-block table transmitted in the block header.

Running a greedy LZ77 over one real frame's unfiltered bytes shows how lopsided the
split is:

```
raw bytes            161,280
emitted literals          49   (0.03% of bytes)
emitted matches        3,796   covering 161,231 B (99.97%)
tokens total           3,845   -> 41.9 B per token before Huffman
match length      mean 42.5   median 16   max 258
```

Essentially **the entire frame is back-references**. Only 49 bytes in the whole image
had to be sent literally. That is why Huffman-alone gets 3.4x while DEFLATE gets 37x:
the redundancy in this data is repetition, not symbol skew, and only LZ77 can see it.

The distances make the reason explicit. Sorted by frequency, with the geometry
annotated:

```
     1  x397           run of a flat colour region
   720  x300   = 1 scanline
    96  x246   = 32 px = 4 tile widths
 1,440  x161   = 2 scanlines
    48  x130   = 16 px = 2 tile widths
     3  x92    = 1 px
    24  x44    = 8 px = 1 tile width
```

Grouped by band, as a share of matched bytes:

```
< 8 px, within a tile row                19.7%
8-80 px                                  28.5%
80 px to 1 scanline                       7.7%
1-8 scanlines, i.e. one tile row         40.3%
> 8 scanlines, earlier tile rows          3.9%
```

**LZ77 is rediscovering the tile grid.** The peaks sit at exact multiples of 8 pixels
horizontally and at whole scanlines vertically, because the PPU composed the screen
from 8x8 tiles in the first place. PNG has no concept of a tile; it finds the
structure by brute-force string matching, which is why it needs a 32 KiB window to
capture what a tile-index representation states directly in two bytes.

That is also the ceiling. Encoding the frame as indices into a learned 8x8 tile
dictionary costs about **787 B/frame** at the measured tile entropy of 7.49 bits per
tile — roughly 5.6x smaller than PNG, because it represents the repetition natively
instead of rediscovering it through a sliding window.

## Why it is lossless

Every stage is exactly invertible, in reverse order:

1. Huffman decoding is a prefix code — unique decode by construction.
2. LZ77 expansion copies from already-reconstructed output, so the decoder rebuilds
   the identical byte stream. (Overlapping copies, e.g. distance 1 length 20, are
   defined to copy byte-by-byte, which is how runs work.)
3. Filter reversal reads the filter byte and adds back the predictor, using
   already-decoded neighbours — the decoder always has `a`, `b`, `c` available before
   it needs them. All arithmetic is mod 256, so it wraps identically.
4. Chunk CRCs verify nothing was corrupted.

No stage discards anything. PNG is not throwing away detail you cannot see; it is
describing the same bytes in a shorter language. That is the whole answer to "how
does it shrink 161,280 bytes without losing information": the frame never contained
161,280 bytes of *information*. It contained roughly 4,400 bytes of information
padded out by a fixed-width encoding, and DEFLATE removes the padding.

## Summary for this corpus

| representation | bytes/frame | vs raw |
|---|---:|---:|
| raw RGB uint8 | 161,280 | 1x |
| Huffman-coded raw bytes | 47,791 | 3.4x |
| PNG, Pillow default (truecolor) | 6,197 | 26.0x |
| PNG, truecolor, filter None | 4,397 | 36.7x |
| PNG, 4-bit palette, filter None | 3,161 | 51.0x |
| learned 8x8 tile dictionary | ~787 | ~205x |

Two practical consequences. First, if PNG size ever matters for this corpus, write
palettized PNGs with filtering disabled — 51x against Pillow's 26x, for identical
pixels. Second, the gap between PNG and the tile dictionary is the measure of how
much structure PNG leaves on the table by being general-purpose.

## Why these bytes are a bad model input

The pipeline above explains the result in `doc/0012-*` and is worth stating plainly.

- **Entropy coding targets exactly what learning needs.** A good coder succeeds when
  its output has no exploitable local structure left. DEFLATE's output is engineered
  to look like uniform random bits.
- **Byte position carries no spatial meaning.** In an RGB array, index maps to pixel
  by a fixed bijection. In an `IDAT` stream, byte *k*'s meaning depends on the entire
  preceding stream and on the dynamic Huffman table in its block header.
- **The representation is violently unstable.** Two consecutive frames of the same
  level share the majority of their content, yet their PNG streams are identical for
  only the first 0.61% of their length, and just 1.46% of bytes match positionally —
  against 0.39% for independent random bytes. One moved sprite shifts every
  subsequent back-reference distance.
- **Length varies per frame**, 4,412 to 8,147 B over 200 sampled frames, so a fixed
  input shape requires padding. Truncating is not an option: it deletes the bottom
  scanlines of the image.

The working rule is that transform domains are fine as model inputs and entropy-coded
bitstreams are not. Training CNNs on dequantized JPEG DCT coefficients works because
those coefficients still sit in a spatial grid; training over arithmetic-coded text
famously does not, and needs the coder state reset on fixed windows to become
learnable at all.

For this project the useful conclusion is not about PNG. It is that the low-entropy
and high-entropy parts of a Contra frame *are* separable — just not by a general
codec. The background is a level id plus a scroll offset; the foreground is a short
list of `(type, x, y)`. That factorization is already in our labels, and it is exact.
