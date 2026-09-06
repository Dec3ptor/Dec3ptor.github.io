# nvc — a tiny neural video codec

Feed it a video. It trains a small neural network to memorise that one clip,
then throws the video away. Playback runs entirely out of the network's weights.

```
encode:  video ──▶ overfit  f(t) ≈ frame_t  ──▶ quantise ──▶ file.nvc
decode:  file.nvc ──▶ rebuild network ──▶ forward pass ──▶ video
```

The weights **are** the file. Nothing else is stored — no frames, no residuals,
no motion vectors.

There are two pieces here:

| | |
|---|---|
| `nvc.py` | command-line codec (PyTorch). Real quantisation, real entropy coding, and an `x264` benchmark. |
| `index.html` | the same idea in the browser (TensorFlow.js). Drop in a video, watch it learn, play it back from the weights. Quantises to 8 bits after training only — the QAT and progressive work below is CLI-side. |

---

## Quick start

```bash
pip install torch numpy imageio imageio-ffmpeg

# no video handy? make one
python nvc.py demo-video -o sample.mp4 --frames 48

# learn it, and check the result against x264 on the same frames
python nvc.py encode sample.mp4 -o sample.nvc \
    --size 128 --frames 48 --params 200k --epochs 600 \
    --preview compare.mp4 --compare

# play it back out of the weights alone
python nvc.py decode sample.nvc -o replay.mp4
python nvc.py info sample.nvc
```

`--preview` writes a side-by-side original | reconstruction video, which is the
fastest way to see what the network did and did not manage to learn.

---

## How it works

### `--arch nerv` (default) — one timestamp in, one frame out

```
t ──▶ Fourier features ──▶ MLP ──▶ [C × H/32 × W/32] feature map
                                        │
                          conv + PixelShuffle ×3  (4×, 4×, 2× upsample)
                                        │
                                        ▼
                                  [3 × H × W] RGB
```

A bare scalar `t` is useless to an MLP — it cannot express the difference
between adjacent frames. Projecting it onto `sin`/`cos` at doubling frequencies
gives the network a basis with enough high-frequency content to build sharp
detail from. This is the NeRV architecture (Chen et al., NeurIPS 2021), shrunk.

### `--arch siren` — one pixel coordinate in, one pixel out

`f(x, y, t) → RGB`, a sine-activated coordinate MLP (SIREN, Sitzmann et al.,
NeurIPS 2020). This is the literal reading of "learn where the pixels are
supposed to be", and it has a party trick: the video becomes a *continuous*
function, so you can decode at frame times that were never in the source.

```bash
python nvc.py decode sample.nvc --frames 240   # 4× slow motion, no interpolation code
```

It is also much worse per parameter than `nerv`, and far slower to decode —
every pixel is its own forward pass. Included because it makes the idea obvious.

### From weights to a file

After training, every weight tensor is quantised to 8 bits (per-tensor
asymmetric min/scale; biases keep 16 bits since they are a rounding error of the
total but matter for quality), then the whole blob goes through LZMA. The
container is a 4-byte magic, a JSON header describing the architecture and the
per-tensor quantisation parameters, and the compressed payload.

The reported quality is measured **after** the round trip through quantisation,
because that is what a decoder actually sees.

---

## So — is this a method for video compression?

**Yes, genuinely.** It is a real, published branch of lossy compression called
*implicit neural representation* (INR) coding. The lineage is COIN (Dupont et
al., 2021) for images and NeRV for video, with a stack of follow-ups — E-NeRV,
HNeRV, FFNeRV, HiNeRV. The framing is the interesting part: compression becomes
*model fitting*, the bitstream is a quantised network, and the whole
model-compression toolbox (pruning, quantisation, weight entropy coding) becomes
codec engineering.

It is compression for a concrete reason. The network has a fixed parameter
budget that is much smaller than the pixel count, so it physically cannot store
frames independently. To drive the loss down it has to discover what the frames
share — a background that persists, an object that translates, a colour ramp
that sweeps — and encode that structure once in the weights. That is the same
redundancy H.264 attacks with motion vectors, just found by gradient descent
instead of block search.

**But it is not competitive compression, and you should know why before you get
attached to it:**

- **Rate–distortion.** Here is a real measurement from `--compare`, on a
  deliberately hard 32-frame 96×96 clip (panning texture plus a hard-edged
  moving square), 147k parameters, 500 epochs:

  | | size | bpp | PSNR |
  |---|---|---|---|
  | this codec | 72.4 KB | 2.012 | **29.33 dB** |
  | x264 crf 18 | 21.1 KB | 0.587 | 29.50 dB |
  | x264 crf 23 | 12.9 KB | 0.357 | 28.73 dB |
  | x264 crf 28 | 8.5 KB | 0.236 | 27.50 dB |

  At matched quality that is **3.4× the bytes x264 needs** (it was 5.9× before
  quantisation-aware training and the 5-bit default; see *Where the bits went*). The serious research
  versions do far better — HiNeRV and friends land in roughly HEVC/x265 territory
  on standard test sets — but they are much larger, trained far longer, and still
  generally trail the best conventional codecs (VVC) and the best learned
  autoencoder codecs (the DCVC line). Do not take the table above as the ceiling
  of the idea; do take it as the ceiling of *this* 700-line version.

- **Short clips are the worst case.** The weights are a fixed cost, and bits per
  pixel is `file_bytes × 8 / (frames × H × W)`. Spreading one 125 KB model over
  32 frames is expensive; the same model over 300 frames of similar content costs
  a tenth the bpp. This is exactly why the NeRV papers train on long sequences,
  and why a short test clip flatters conventional codecs. If you want this to
  look its best, give it a long clip that keeps returning to the same content.
- **Encoding cost is the real problem.** Encoding *is* training. Minutes here,
  hours on a GPU for research configurations, versus x264 running faster than
  real time. Nothing about the approach fixes this; the fit is per-video by
  construction, so there is no pretrained model to amortise it against.
- **One model per video.** A `.nvc` decodes exactly one clip. There is no
  shared codebook to amortise across a library.
- **No hardware decode.** Every phone has an H.264 block in silicon. None of
  them have this.
- **Quality degrades in a specific way.** No blocking artifacts — instead the
  network spends its capacity on the low frequencies and loses fine texture and
  sharp edges first. Whether that looks better or worse than blocking is a
  matter of taste; PSNR does not capture the difference well.

Where it is genuinely attractive: **decoding is trivially parallel and
random-access**. There is no GOP, no reference frames, no decode order. Frame
9,000 costs exactly one forward pass, same as frame 1. That is a real structural
advantage over every block-based codec, and it is the reason people keep
publishing on this.

The honest summary: it is a legitimate and active research direction, it is
excellent for understanding what compression *is* (find the shortest description
that reproduces the data), and it is not something to replace H.264 with today.

---

## Where the bits went

Rather than guess at improvements, measure. Three experiments on the clip above,
all reproducible from this repo.

**1. A better entropy coder is not worth building.** LZMA lands within 2.5% of
the zeroth-order entropy of the quantised weights (6.965 vs 6.787 bits/weight at
8 bits). There is almost nothing left for an arithmetic coder to take, which
kills the most obvious "improvement" before any of it gets written.

**2. Eight bits was wasteful, and quantisation-aware training is what makes low
bit depths usable.** Rounding weights only *after* training means the network
never sees the rounding error and cannot compensate. Rounding them in the
forward pass for the second half of training (with a straight-through estimator
for the backward pass) changes the picture completely:

| bits | post-training only | with QAT | gain |
|---|---|---|---|
| 6 | 29.43 dB / 89.5 KB | 29.54 dB / 90.4 KB | +0.11 dB |
| 5 | 28.70 dB / 71.6 KB | **29.35 dB / 72.7 KB** | +0.65 dB |
| 4 | 26.59 dB / 56.5 KB | **28.84 dB / 55.8 KB** | +2.25 dB |
| 3 | 22.23 dB / 37.4 KB | **27.65 dB / 38.9 KB** | +5.42 dB |

QAT costs nothing at 6 bits and rescues the codec entirely at 3. Together with
dropping the default from 8 bits to 5, this is a 42% smaller file at 0.3 dB.
Both are on by default (`--qat-start`, `--bits`).

**3. Per-tensor sensitivity varies ~40×, but mixed precision barely pays.**
Quantising one tensor at a time to 3 bits and leaving the rest alone:

| tensor | share of weights | dB lost | per % of weights |
|---|---|---|---|
| `head.weight` | 0.3% | 0.87 | 2.90 |
| `fc1.weight` | 0.5% | 1.20 | 2.40 |
| `fc2.weight` | 7.5% | 2.18 | 0.29 |
| `blocks.1.weight` | 26.8% | 3.72 | 0.14 |
| `blocks.0.weight` | 58.6% | 4.28 | 0.07 |

The two smallest tensors are by far the most sensitive per parameter, which
looks like free money: spend more bits there, they cost almost nothing. In
practice it is worth about **+0.2 dB at matched rate** — the single-tensor
measurement overstates it, because once everything is quantised the errors
compound and no one tensor dominates. Filed under "true but not useful".

---

## Knobs that matter

| flag | effect |
|---|---|
| `--params` | the bitrate dial. Parameter budget → layer width → file size. Try `60k`, `200k`, `500k`. A budget below what `--min-channels` allows is reported as a warning, not silently exceeded. |
| `--epochs` | quality dial. Undertrained is the most common reason results look bad — this is overfitting, so there is no such thing as too much. Start at 600. |
| `--size` | long side of the frame, snapped to a multiple of the stride product (32 by default). Cost scales with pixels. |
| `--bits` | weight quantisation, default 5. With QAT even 3 bits stays usable; without it, 4 falls apart. |
| `--qat-start` | fraction of training after which weights are rounded in the forward pass. Default 0.5. Set to 1.0 to disable and quantise only at the end. |
| `--progressive` / `--qat-jitter` | write a truncatable file, and train it to survive truncation. See below. |
| `--strides` | nerv upsample factors, e.g. `2,2,2,2,2`. Their product sets the base feature map size and must divide the frame dimensions. |
| `--min-channels` | floor on block widths, and therefore the smallest model the architecture can express: a floor of 16 cannot go below ~66k parameters, 8 reaches ~22k, 4 reaches ~8k. Chosen automatically from `--params` (largest floor that fits, since a high floor also measures better — 30.00 dB vs 29.55 dB at a 150k budget); set it by hand to override. |
| `--compare` | benchmark against libx264 at matched quality. Use it before believing any compression claim, including this README's. |

Rough guide: a 48-frame 128×128 clip at `--params 200k --epochs 600` takes a
couple of minutes on a CPU. Push resolution or frame count and use a GPU
(`--device cuda`) — it is picked up automatically when present.

---

## A file you can cut in half

`--progressive` writes the weights as bit planes, most significant first,
instead of one entropy-coded blob. Truncating the file then lowers the bitrate:
every weight simply loses precision. One encode serves every rate.

```bash
python nvc.py encode clip.mp4 -o v.nvc --bits 8 --progressive --qat-jitter 4
python nvc.py truncate v.nvc -o small.nvc --bits 5   # 142.6 KB -> 88.8 KB
head -c 70000 v.nvc > chopped.nvc                    # even this still decodes
python nvc.py decode chopped.nvc -o out.mp4
```

That matters because for an implicit codec *encoding is the expensive part*.
Serving five bitrates normally means five training runs; here it means five
calls to `truncate`.

`--qat-jitter N` trains across a range of bit depths rather than one, so the
weights are reasonable at every truncation point instead of only the trained
one. What that actually buys, measured:

| planes kept | jitter-trained | plain, truncated | trained at that depth |
|---|---|---|---|
| 8 | 29.04 dB | 29.67 dB | — |
| 6 | 28.82 dB | 28.89 dB | 29.54 dB |
| 5 | **28.56 dB** | 28.25 dB | 29.35 dB |
| 4 | **27.46 dB** | 26.38 dB | 28.84 dB |

Read it honestly. Jitter flattens the curve — worth +1.08 dB three planes down —
but it costs 0.63 dB at full precision, and a purpose-trained model still beats
the truncated one by more than a dB. The bit-plane layout itself costs a further
12%, because splitting values into planes throws away exactly the
value-distribution redundancy that LZMA was living on.

So this is a capability, not a free win: pay ~12% in size and ~1 dB against
per-rate training, and in exchange never re-encode. Off by default.

---

## File format

```
"NVC1"            4 bytes
header length     uint32 little-endian
header            JSON — architecture, resolution, frame count, fps,
                  and per-tensor {shape, bits, min, scale}
payload           LZMA( concatenated quantised weight tensors )
```

Small enough to reimplement a decoder anywhere. `index.html` ships a simpler
uncompressed variant (`.nvcw`) so the browser can write one without an LZMA
dependency.

---

## References

- Sitzmann et al., *Implicit Neural Representations with Periodic Activation Functions* (SIREN), NeurIPS 2020
- Dupont et al., *COIN: COmpression with Implicit Neural representations*, 2021
- Chen et al., *NeRV: Neural Representations for Videos*, NeurIPS 2021
- Chen et al., *HNeRV: A Hybrid Neural Representation for Videos*, CVPR 2023
- Kwan et al., *HiNeRV: Video Compression with Hierarchical Encoding-based Neural Representation*, NeurIPS 2023
