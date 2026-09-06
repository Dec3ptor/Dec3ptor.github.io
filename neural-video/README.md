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
| `index.html` | the same idea in the browser (TensorFlow.js). Drop in a video, watch it learn, play it back from the weights. |

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

- **Rate–distortion.** Run `--compare` and read the table. On a short clip this
  toy typically needs several times the bits x264 needs for the same PSNR. The
  serious research versions do far better — HiNeRV and friends land in roughly
  HEVC/x265 territory on standard test sets — but they are much larger, trained
  far longer, and still generally trail the best conventional codecs (VVC) and
  the best learned autoencoder codecs (the DCVC line).
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

## Knobs that matter

| flag | effect |
|---|---|
| `--params` | the bitrate dial. Parameter budget → layer width → file size. Try `60k`, `200k`, `500k`. |
| `--epochs` | quality dial. Undertrained is the most common reason results look bad — this is overfitting, so there is no such thing as too much. Start at 600. |
| `--size` | long side of the frame, snapped to a multiple of the stride product (32 by default). Cost scales with pixels. |
| `--bits` | weight quantisation. 8 is nearly free; 6 costs a little quality and shrinks the file; 4 usually falls apart. |
| `--strides` | nerv upsample factors, e.g. `2,2,2,2,2`. Their product sets the base feature map size and must divide the frame dimensions. |
| `--compare` | benchmark against libx264 at matched quality. Use it before believing any compression claim, including this README's. |

Rough guide: a 48-frame 128×128 clip at `--params 200k --epochs 600` takes a
couple of minutes on a CPU. Push resolution or frame count and use a GPU
(`--device cuda`) — it is picked up automatically when present.

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
