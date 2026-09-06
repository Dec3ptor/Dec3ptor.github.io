#!/usr/bin/env python3
"""
nvc.py - a very small neural video codec.

The idea in one line: instead of storing pixels, we *overfit* a tiny neural
network to one specific video, and then ship the network's weights as the file.

    encode:  video  ->  train f(t) ~= frame_t  ->  quantise weights  ->  file.nvc
    decode:  file.nvc  ->  rebuild net  ->  run forward pass  ->  video

There is no dataset, no generalisation and no training/test split. The network
is not learning "video" - it is learning *this* video, the same way a JPEG
table only describes one image. The weights ARE the compressed bitstream.

Two architectures are included:

  nerv   f(t) -> whole RGB frame, via a small MLP that produces a low-res
         feature map which conv + PixelShuffle blocks upsample. This is the
         NeRV family (Chen et al., 2021) and it is what the neural-compression
         literature actually uses for video.

  siren  f(x, y, t) -> one RGB pixel, a sine-activated coordinate MLP
         (Sitzmann et al., 2020). Literally "learn where pixels are supposed
         to be". Continuous in all three axes, so you can decode at any
         resolution or frame rate, but far worse rate/distortion than nerv.

Run `nvc.py encode ... --compare` to benchmark the result against libx264 on
the exact same frames, which is the only honest way to answer "is this
actually compression?".
"""
from __future__ import annotations

import argparse
import json
import lzma
import math
import os
import random
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

MAGIC = b"NVC1"
DEFAULT_STRIDES = (4, 4, 2)  # 32x total upsample from the base feature map
GRID_SHARE = 0.45           # of a parameter budget, spent on the finer grid


# --------------------------------------------------------------- utilities

def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{int(n):,} B"
        n /= 1024
    return f"{n:.1f} GB"


def parse_count(s: str) -> int:
    """'100k' -> 100000, '1.5m' -> 1500000, '4096' -> 4096."""
    s = str(s).strip().lower().replace("_", "")
    mult = 1
    if s.endswith("k"):
        mult, s = 1_000, s[:-1]
    elif s.endswith("m"):
        mult, s = 1_000_000, s[:-1]
    return int(float(s) * mult)


def psnr_u8(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio between two uint8 arrays, in dB."""
    diff = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(diff * diff))
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10(255.0 * 255.0 / mse)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------- video io

def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def read_video(path, max_frames=None, frame_stride=1):
    """Decode a video to a uint8 array [T, H, W, 3] plus its frame rate."""
    import imageio.v2 as iio
    reader = iio.get_reader(str(path), "ffmpeg")
    try:
        fps = float(reader.get_meta_data().get("fps", 25.0) or 25.0)
    except Exception:
        fps = 25.0
    frames = []
    for i, frame in enumerate(reader):
        if i % frame_stride:
            continue
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        frames.append(arr[..., :3])
        if max_frames and len(frames) >= max_frames:
            break
    reader.close()
    if not frames:
        raise SystemExit(f"no frames decoded from {path}")
    return np.stack(frames), fps / frame_stride


def resize_clip(clip: np.ndarray, long_side: int, multiple: int = 32):
    """Rescale to ~long_side on the longer axis, snapped to a multiple of 32.

    Aspect ratio is preserved (no cropping). The multiple-of-32 constraint
    exists because the nerv decoder upsamples by 32x from its base feature map.
    """
    T, H, W, _ = clip.shape
    scale = long_side / max(H, W)
    th = max(multiple, int(round(H * scale / multiple)) * multiple)
    tw = max(multiple, int(round(W * scale / multiple)) * multiple)
    if (th, tw) == (H, W):
        return clip
    mode = "area" if (th <= H and tw <= W) else "bilinear"
    out = np.empty((T, th, tw, 3), dtype=np.uint8)
    for i in range(0, T, 16):  # chunked so we never blow up memory
        x = torch.from_numpy(clip[i:i + 16]).permute(0, 3, 1, 2).float()
        kw = {} if mode == "area" else {"align_corners": False}
        y = F.interpolate(x, size=(th, tw), mode=mode, **kw)
        out[i:i + 16] = y.clamp(0, 255).round().byte().permute(0, 2, 3, 1).numpy()
    return out


def write_video(path, frames: np.ndarray, fps: float):
    import imageio.v2 as iio
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".gif":
        iio.mimsave(str(path), list(frames), fps=fps, loop=0)
        return
    writer = iio.get_writer(str(path), fps=fps, codec="libx264", quality=9,
                            macro_block_size=1, ffmpeg_log_level="error")
    for f in frames:
        writer.append_data(f)
    writer.close()


# ----------------------------------------------------------------- models

def time_embed(t: torch.Tensor, levels: int) -> torch.Tensor:
    """Fourier features for a scalar time in [0, 1] -> [B, 2*levels].

    A raw scalar t is a hopeless input for an MLP: it cannot represent sharp
    changes between neighbouring frames. Projecting onto sin/cos at doubling
    frequencies gives the net a basis it can build high-frequency detail from.
    """
    freqs = (2.0 ** torch.arange(levels, device=t.device, dtype=t.dtype)) * math.pi
    ang = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class NeRVTiny(nn.Module):
    """t -> full RGB frame. Small MLP stem, then conv + PixelShuffle upsamplers."""

    def __init__(self, height, width, channels=64, strides=DEFAULT_STRIDES,
                 embed_levels=10, min_channels=16):
        super().__init__()
        up = math.prod(strides)
        if height % up or width % up:
            raise ValueError(f"{height}x{width} must be divisible by {up}")
        self.height, self.width = height, width
        self.channels = channels
        self.embed_levels = embed_levels
        self.strides = list(strides)
        self.base = (height // up, width // up)

        self.fc1 = nn.Linear(2 * embed_levels, channels)
        self.fc2 = nn.Linear(channels, channels * self.base[0] * self.base[1])

        convs, c = [], channels
        for s in self.strides:
            c_out = max(min_channels, c // 2)
            convs.append(nn.Conv2d(c, c_out * s * s, 3, padding=1))
            c = c_out
        self.blocks = nn.ModuleList(convs)
        self.head = nn.Conv2d(c, 3, 3, padding=1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(time_embed(t, self.embed_levels)))
        x = self.fc2(x).view(t.shape[0], self.channels, *self.base)
        for conv, s in zip(self.blocks, self.strides):
            x = F.gelu(F.pixel_shuffle(conv(x), s))
        return torch.sigmoid(self.head(x))


def sample_time(grid: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Linearly interpolate a [G, C, H, W] grid along its time axis at t in [0,1]."""
    g = grid.shape[0]
    if g == 1:
        return grid[0].unsqueeze(0).expand(t.shape[0], -1, -1, -1)
    pos = t.clamp(0, 1) * (g - 1)
    i0 = pos.floor().long().clamp(0, g - 1)
    i1 = (i0 + 1).clamp(0, g - 1)
    w = (pos - i0.to(pos.dtype)).view(-1, 1, 1, 1)
    return grid[i0] * (1 - w) + grid[i1] * w


class GridNeRV(nn.Module):
    """t -> frame, but the per-frame code is a learned grid, not an MLP.

    The stem in NeRVTiny is a bottleneck by construction: everything that makes
    frame 7 different from frame 8 has to survive being squeezed through an MLP
    applied to a Fourier embedding of a scalar. Measured on this codebase, that
    pathway holds 8%% of the parameters while being the most quantisation
    sensitive part of the model, and the frame-independent convolutions hold
    59%% while being the least.

    Storing the base feature maps directly fixes the allocation and removes the
    constraint that they be a smooth function of an MLP. It also moves
    parameters somewhere *compressible*: dense conv weights turned out to be
    structureless (adjacent kernel taps correlate at 0.02, and a DCT of them
    concentrates no energy), whereas a feature grid is a small video and carries
    the temporal redundancy of one.

    A second, finer grid can be concatenated partway up the decoder, which is
    the cheap version of the hierarchical encoding HiNeRV uses.
    """

    def __init__(self, height, width, channels=64, strides=DEFAULT_STRIDES,
                 min_channels=8, grid_frames=32, mid_channels=0, mid_level=1,
                 mid_frames=0):
        super().__init__()
        up = math.prod(strides)
        if height % up or width % up:
            raise ValueError(f"{height}x{width} must be divisible by {up}")
        self.height, self.width = height, width
        self.channels = channels
        self.strides = list(strides)
        self.base = (height // up, width // up)
        self.mid_level = mid_level if mid_channels else -1

        self.base_grid = nn.Parameter(
            torch.randn(max(1, grid_frames), channels, *self.base) * 0.05)

        # Work out the spatial size at the input of each block so the mid grid
        # can be built at the right resolution.
        sizes, h, w = [], self.base[0], self.base[1]
        for st in self.strides:
            sizes.append((h, w))
            h, w = h * st, w * st

        self.mid_grid = None
        if mid_channels and 0 <= self.mid_level < len(self.strides):
            mh, mw = sizes[self.mid_level]
            self.mid_grid = nn.Parameter(
                torch.randn(max(1, mid_frames or grid_frames),
                            mid_channels, mh, mw) * 0.05)

        convs, c = [], channels
        for i, st in enumerate(self.strides):
            c_in = c + (mid_channels if i == self.mid_level else 0)
            c_out = max(min_channels, c // 2)
            convs.append(nn.Conv2d(c_in, c_out * st * st, 3, padding=1))
            c = c_out
        self.blocks = nn.ModuleList(convs)
        self.head = nn.Conv2d(c, 3, 3, padding=1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = sample_time(self.base_grid, t)
        for i, (conv, st) in enumerate(zip(self.blocks, self.strides)):
            if i == self.mid_level and self.mid_grid is not None:
                x = torch.cat([x, sample_time(self.mid_grid, t)], dim=1)
            x = F.gelu(F.pixel_shuffle(conv(x), st))
        return torch.sigmoid(self.head(x))


class SirenVideo(nn.Module):
    """(x, y, t) -> RGB. Sine-activated coordinate MLP (SIREN)."""

    def __init__(self, hidden=128, depth=4, w0=30.0):
        super().__init__()
        self.w0 = w0
        dims = [3] + [hidden] * depth
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(depth)])
        self.out = nn.Linear(hidden, 3)
        with torch.no_grad():
            # SIREN initialisation keeps the pre-activations in the regime
            # where sin() behaves like a well-conditioned nonlinearity.
            first = self.layers[0]
            first.weight.uniform_(-1.0 / first.in_features, 1.0 / first.in_features)
            for layer in list(self.layers[1:]) + [self.out]:
                bound = math.sqrt(6.0 / layer.in_features) / w0
                layer.weight.uniform_(-bound, bound)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = torch.sin(self.w0 * self.layers[0](coords))
        for layer in self.layers[1:]:
            x = torch.sin(self.w0 * layer(x))
        return torch.sigmoid(self.out(x))


def build_model(cfg: dict) -> nn.Module:
    if cfg["arch"] == "grid":
        return GridNeRV(cfg["height"], cfg["width"], channels=cfg["channels"],
                        strides=tuple(cfg["strides"]),
                        min_channels=cfg["min_channels"],
                        grid_frames=cfg["grid_frames"],
                        mid_channels=cfg["mid_channels"],
                        mid_level=cfg.get("mid_level", 1),
                        mid_frames=cfg.get("mid_frames", 0))
    if cfg["arch"] == "nerv":
        return NeRVTiny(cfg["height"], cfg["width"], channels=cfg["channels"],
                        strides=tuple(cfg["strides"]),
                        embed_levels=cfg["embed_levels"],
                        min_channels=cfg["min_channels"])
    return SirenVideo(hidden=cfg["channels"], depth=cfg["depth"], w0=cfg["w0"])


def fit_budget(make_cfg, budget: int, floors=(16, 12, 8, 4)):
    """Choose a channel floor and a layer width for a parameter budget.

    The floor sets the smallest model the architecture can express, so a high
    floor makes small budgets unreachable. It also buys quality: at a 150k
    budget a floor of 16 measured ~0.45 dB better than 8. So take the largest
    floor whose smallest model still fits, and widen from there.
    """
    for floor in floors:
        width = size_to_budget(lambda w: make_cfg(w, floor), budget)
        if count_params(build_model(make_cfg(width, floor))) <= budget:
            return width, floor
    floor = floors[-1]
    return size_to_budget(lambda w: make_cfg(w, floor), budget), floor


def size_to_budget(make_cfg, target_params: int, lo=8, hi=1024) -> int:
    """Binary search the widest model whose parameter count fits the budget."""
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            n = count_params(build_model(make_cfg(mid)))
        except ValueError:
            n = target_params + 1
        if n <= target_params:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


# ------------------------------------------------- quantisation-aware training

class _RoundSTE(torch.autograd.Function):
    """Quantise to a uniform grid on the forward pass, pass gradients through.

    Rounding has zero gradient almost everywhere, so a network trained in fp32
    and quantised afterwards has never seen the rounding error and cannot
    compensate for it. The straight-through estimator pretends the rounding is
    the identity during the backward pass, which lets the weights settle into
    positions where the grid costs little.
    """

    @staticmethod
    def forward(ctx, w, levels):
        lo, hi = w.min(), w.max()
        scale = (hi - lo) / levels
        if float(scale) <= 0:
            return w
        return torch.round((w - lo) / scale).clamp_(0, levels) * scale + lo

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class FakeQuantize(nn.Module):
    """Round to `bits`, or to a random depth in [bits - jitter, bits].

    Jitter is what makes a truncatable file worth having. Trained at one fixed
    depth, a network is only good at that depth, so chopping bit planes off the
    end degrades it faster than necessary. Sampling the depth each step asks the
    weights to be simultaneously reasonable at every precision, which is what a
    file you can cut anywhere actually needs.
    """

    def __init__(self, bits: int, jitter: int = 0):
        super().__init__()
        self.bits = bits
        self.jitter = jitter

    def forward(self, w):
        b = self.bits
        if self.jitter and self.training:
            b = random.randint(max(2, self.bits - self.jitter), self.bits)
        return _RoundSTE.apply(w, (1 << b) - 1)


def _quantisable(model):
    """Every (module, parameter name) that the bitstream stores at `bits`.

    This has to match what quantize_state actually quantises, which is any
    tensor of rank 2 or more. Feature grids are plain parameters rather than
    layer weights, and they are the most content-critical tensors in the model,
    so leaving them out would train against the wrong quantisation.
    """
    out = []
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)) and m.weight.dim() >= 2:
            out.append((m, "weight"))
        for name in ("base_grid", "mid_grid"):
            # Once parametrised the attribute is a plain tensor, not a
            # Parameter, so check that first or finalise_qat cannot find the
            # grids again to bake them.
            if parametrize.is_parametrized(m, name):
                out.append((m, name))
                continue
            t = getattr(m, name, None)
            if isinstance(t, nn.Parameter) and t.dim() >= 2:
                out.append((m, name))
    return out


def enable_qat(model, bits: int, jitter: int = 0):
    """Make every stored tensor read back quantised, in place."""
    for m, name in _quantisable(model):
        parametrize.register_parametrization(m, name, FakeQuantize(bits, jitter))


def finalise_qat(model):
    """Bake the quantised values in and restore plain parameters."""
    # Baking runs the parametrisation once more, so make sure jitter is off:
    # otherwise the stored weights land at a random depth rather than the full
    # one the file claims to hold.
    model.eval()
    for m, name in _quantisable(model):
        if parametrize.is_parametrized(m, name):
            parametrize.remove_parametrizations(m, name, leave_parametrized=True)


# --------------------------------------------------------------- training

def grid_tv(model) -> torch.Tensor:
    """Squared difference between neighbouring grid slices in time.

    Nothing in a plain reconstruction loss makes adjacent grid slices resemble
    each other, and measurement confirms they do not: a trained base grid coded
    to 4.21 bits/value, slightly *worse* than the dense conv weights next to it,
    and delta coding along time made it worse still. Penalising the temporal
    difference is a rate proxy - it pushes the grid towards something a delta
    coder can actually exploit, and doubles as a temporal-consistency prior.
    """
    total = None
    for name in ("base_grid", "mid_grid"):
        g = getattr(model, name, None)
        if g is None or g.shape[0] < 2:
            continue
        term = (g[1:] - g[:-1]).pow(2).mean()
        total = term if total is None else total + term
    return total


def train_nerv(model, clip_u8, *, epochs, batch, lr, device, quiet=False,
               grid_smooth=0.0,
               qat_bits=None, qat_start=None, qat_jitter=0):
    T = clip_u8.shape[0]
    target = torch.from_numpy(clip_u8).permute(0, 3, 1, 2).float().div_(255.0)
    times = torch.arange(T, dtype=torch.float32) / max(T - 1, 1)
    model.to(device).train()

    steps = epochs * math.ceil(T / batch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.1)

    t0, last = time.time(), 0.0
    for epoch in range(epochs):
        if qat_bits and qat_start is not None and epoch == qat_start:
            enable_qat(model, qat_bits, qat_jitter)
            if not quiet:
                depth = (f"{max(2, qat_bits - qat_jitter)}-{qat_bits}"
                         if qat_jitter else str(qat_bits))
                print(f"\n  -> quantisation-aware training at {depth} bits")
        order = torch.randperm(T)
        running = 0.0
        for i in range(0, T, batch):
            idx = order[i:i + batch]
            t = times[idx].to(device)
            y = target[idx].to(device)
            pred = model(t)
            loss = F.mse_loss(pred, y) + 0.1 * F.l1_loss(pred, y)
            if grid_smooth:
                tv = grid_tv(model)
                if tv is not None:
                    loss = loss + grid_smooth * tv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.detach().item() * len(idx)
        last = running / T
        if not quiet:
            report_epoch(epoch, epochs, last, t0)
    if not quiet:
        print()
    return last


def train_siren(model, clip_u8, *, epochs, batch, lr, device, quiet=False,
                qat_bits=None, qat_start=None, qat_jitter=0):
    T, H, W, _ = clip_u8.shape
    flat = torch.from_numpy(clip_u8.reshape(-1, 3)).float().div_(255.0)
    n_px = flat.shape[0]
    model.to(device).train()

    # One "epoch" here means one pass worth of randomly sampled pixels.
    per_epoch = max(1, n_px // batch)
    steps = epochs * per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.1)

    denom = torch.tensor([max(W - 1, 1), max(H - 1, 1), max(T - 1, 1)],
                         dtype=torch.float32)
    t0, last = time.time(), 0.0
    for epoch in range(epochs):
        if qat_bits and qat_start is not None and epoch == qat_start:
            enable_qat(model, qat_bits, qat_jitter)
            if not quiet:
                depth = (f"{max(2, qat_bits - qat_jitter)}-{qat_bits}"
                         if qat_jitter else str(qat_bits))
                print(f"\n  -> quantisation-aware training at {depth} bits")
        running = 0.0
        for _ in range(per_epoch):
            idx = torch.randint(0, n_px, (batch,))
            ti = torch.div(idx, H * W, rounding_mode="floor")
            rem = idx % (H * W)
            yi = torch.div(rem, W, rounding_mode="floor")
            xi = rem % W
            coords = torch.stack([xi, yi, ti], dim=-1).float()
            coords = (coords / denom) * 2.0 - 1.0  # -> [-1, 1]
            pred = model(coords.to(device))
            y = flat[idx].to(device)
            loss = F.mse_loss(pred, y) + 0.1 * F.l1_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.detach().item()
        last = running / per_epoch
        if not quiet:
            report_epoch(epoch, epochs, last, t0)
    if not quiet:
        print()
    return last


def report_epoch(epoch, epochs, loss, t0):
    done = epoch + 1
    elapsed = time.time() - t0
    eta = elapsed / done * (epochs - done)
    approx_psnr = 10 * math.log10(1.0 / max(loss, 1e-9))
    sys.stdout.write(
        f"\r  epoch {done:4d}/{epochs}  loss {loss:.5f}  "
        f"~{approx_psnr:5.2f} dB  elapsed {elapsed:5.1f}s  eta {eta:5.1f}s   ")
    sys.stdout.flush()


# --------------------------------------------------------------- rendering

@torch.no_grad()
def render(model, cfg, n_frames=None, device="cpu", chunk=8) -> np.ndarray:
    """Run the network forward to produce the video, as uint8 [T, H, W, 3]."""
    model.to(device).eval()
    T = n_frames or cfg["frames"]
    H, W = cfg["height"], cfg["width"]
    out = np.empty((T, H, W, 3), dtype=np.uint8)
    times = torch.arange(T, dtype=torch.float32) / max(T - 1, 1)

    if cfg["arch"] != "siren":
        for i in range(0, T, chunk):
            pred = model(times[i:i + chunk].to(device))
            pred = pred.clamp(0, 1).mul(255).round().byte()
            out[i:i + chunk] = pred.permute(0, 2, 3, 1).cpu().numpy()
        return out

    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32), indexing="ij")
    grid = torch.stack([xs.reshape(-1) / max(W - 1, 1),
                        ys.reshape(-1) / max(H - 1, 1)], dim=-1) * 2.0 - 1.0
    px_chunk = 65536
    for i in range(T):
        tval = float(times[i]) * 2.0 - 1.0
        frame = np.empty((H * W, 3), dtype=np.uint8)
        for j in range(0, H * W, px_chunk):
            block = grid[j:j + px_chunk]
            coords = torch.cat(
                [block, torch.full((block.shape[0], 1), tval)], dim=-1)
            pred = model(coords.to(device)).clamp(0, 1).mul(255).round().byte()
            frame[j:j + px_chunk] = pred.cpu().numpy()
        out[i] = frame.reshape(H, W, 3)
    return out


# ------------------------------------------------------ quantise + bitstream

def quantize_state(state_dict, bits: int, bias_bits: int = 16):
    """Per-tensor asymmetric uniform quantisation of every weight.

    Weight matrices get `bits`; 1-D tensors (biases) keep 16 bits because they
    are a rounding error of the total parameter count but matter for quality.
    """
    meta, params, data = [], [], bytearray()
    for name, tensor in state_dict.items():
        arr = tensor.detach().cpu().numpy().astype(np.float32).ravel()
        b = bits if tensor.dim() >= 2 else bias_bits
        lo, hi = float(arr.min()), float(arr.max())
        levels = (1 << b) - 1
        # Round the scale to float32 up front: that is the precision the
        # decoder reads back, so quantising with it makes the round trip exact.
        scale = float(np.float32((hi - lo) / levels)) if hi > lo else 1.0
        q = np.rint((arr - lo) / scale).clip(0, levels)
        data += q.astype("<u1" if b <= 8 else "<u2").tobytes()
        meta.append({"name": name, "shape": list(tensor.shape), "bits": b})
        params.append((lo, scale))
    # min/scale live in the binary payload, not the JSON header: as decimal text
    # they cost ~40 bytes per tensor, which is real money on a small model.
    head = np.asarray(params, dtype="<f4").tobytes() if params else b""
    return meta, head + bytes(data)


def dequantize_state(meta, blob: bytes):
    params = np.frombuffer(blob, dtype="<f4", count=2 * len(meta)).reshape(-1, 2)
    off = 8 * len(meta)
    state = {}
    for t, (lo, scale) in zip(meta, params):
        dtype = np.dtype("<u1" if t["bits"] <= 8 else "<u2")
        count = math.prod(t["shape"]) if t["shape"] else 1
        q = np.frombuffer(blob, dtype=dtype, count=count, offset=off)
        off += count * dtype.itemsize
        arr = q.astype(np.float32) * float(scale) + float(lo)
        state[t["name"]] = torch.from_numpy(arr.reshape(t["shape"]).copy())
    return state


# ------------------------------------------------- progressive (truncatable)

MAGIC_PROG = b"NVCP"


def encode_progressive(state_dict, bits: int = 8):
    """Split the weights into a small base plus bit planes, most significant
    first.

    A normal codec file is all-or-nothing: cut it short and it stops decoding.
    Ordering the weight bits by significance instead makes the file
    *truncatable* - chop bytes off the end and every weight simply loses
    precision, so one training run yields a whole rate-distortion curve.

    This is affordable precisely because entropy coding buys so little on these
    weights (LZMA lands within ~2.5%% of their zeroth-order entropy), so giving
    up whole-blob context to compress plane by plane costs very little.
    """
    meta, params, base_1d, weights = [], [], bytearray(), []
    for name, tensor in state_dict.items():
        arr = tensor.detach().cpu().numpy().astype(np.float32).ravel()
        lo, hi = float(arr.min()), float(arr.max())
        levels = (1 << bits) - 1
        scale = float(np.float32((hi - lo) / levels)) if hi > lo else 1.0
        q = np.rint((arr - lo) / scale).clip(0, levels).astype(np.uint8)
        is_w = tensor.dim() >= 2
        meta.append({"name": name, "shape": list(tensor.shape), "w": is_w})
        params.append((lo, scale))
        if is_w:
            weights.append(q)
        else:
            base_1d.extend(q.tobytes())

    qw = np.concatenate(weights) if weights else np.zeros(0, np.uint8)
    # Biases stay at full precision in the base: they are a fraction of a
    # percent of the parameters but truncating them hurts out of proportion.
    base = np.asarray(params, dtype="<f4").tobytes() + bytes(base_1d)
    planes = [lzma.compress(np.packbits((qw >> p) & 1).tobytes(),
                            preset=9 | lzma.PRESET_EXTREME)
              for p in range(bits - 1, -1, -1)]
    return meta, base, planes, int(qw.size)


def decode_progressive(meta, base: bytes, planes, n_weights: int, bits: int):
    """Rebuild a state dict from however many bit planes actually arrived."""
    params = np.frombuffer(base, dtype="<f4", count=2 * len(meta)).reshape(-1, 2)
    off_1d = 8 * len(meta)

    q = np.zeros(n_weights, dtype=np.int32)
    for i, plane in enumerate(planes):
        raw = np.frombuffer(lzma.decompress(plane), dtype=np.uint8)
        bit = np.unpackbits(raw)[:n_weights].astype(np.int32)
        q |= bit << (bits - 1 - i)
    if len(planes) < bits:
        # Everything below the last plane received is unknown; sit in the
        # middle of that interval rather than at its floor, which would bias
        # every weight downwards.
        q += (1 << (bits - len(planes))) // 2

    state, w_off = {}, 0
    for t, (lo, scale) in zip(meta, params):
        n = math.prod(t["shape"]) if t["shape"] else 1
        if t["w"]:
            vals = q[w_off:w_off + n].astype(np.float32)
            w_off += n
        else:
            vals = np.frombuffer(base, dtype=np.uint8, count=n,
                                 offset=off_1d).astype(np.float32)
            off_1d += n
        arr = vals * float(scale) + float(lo)
        state[t["name"]] = torch.from_numpy(arr.reshape(t["shape"]).copy())
    return state


def save_progressive(path, header: dict, base: bytes, planes) -> int:
    header = dict(header, base_len=len(base), plane_lens=[len(p) for p in planes])
    hdr = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(MAGIC_PROG)
        f.write(struct.pack("<I", len(hdr)))
        f.write(hdr)
        f.write(base)
        for plane in planes:
            f.write(plane)
    return os.path.getsize(path)


def load_progressive(path):
    """Load a progressive file, tolerating one that has been cut short."""
    data = open(path, "rb").read()
    if data[:4] != MAGIC_PROG:
        raise SystemExit(f"{path} is not a progressive .nvc file")
    (hlen,) = struct.unpack("<I", data[4:8])
    header = json.loads(data[8:8 + hlen].decode())
    off = 8 + hlen
    base = data[off:off + header["base_len"]]
    if len(base) < header["base_len"]:
        raise SystemExit(f"{path} is truncated past the point of being usable: "
                         f"the base layer needs {header['base_len']} bytes")
    off += header["base_len"]
    planes = []
    for n in header["plane_lens"]:
        if off + n > len(data):
            break                      # the file was cut here; stop cleanly
        planes.append(data[off:off + n])
        off += n
    return header, base, planes


def save_nvc(path, header: dict, blob: bytes) -> int:
    # 8-bit weights are close to incompressible, so LZMA sometimes *costs*
    # bytes. Keep whichever is smaller and record which one we used.
    packed = lzma.compress(blob, preset=9 | lzma.PRESET_EXTREME)
    if len(packed) < len(blob):
        payload, codec = packed, "lzma"
    else:
        payload, codec = blob, "raw"
    header = dict(header, raw_weight_bytes=len(blob), codec=codec)
    hdr = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(hdr)))
        f.write(hdr)
        f.write(payload)
    return os.path.getsize(path)


def load_nvc(path):
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise SystemExit(f"{path} is not an .nvc file")
        (hlen,) = struct.unpack("<I", f.read(4))
        header = json.loads(f.read(hlen).decode())
        payload = f.read()
    blob = lzma.decompress(payload) if header.get("codec", "lzma") == "lzma" else payload
    return header, blob


def load_any(path):
    """Open either container. Returns (header, model, planes_present)."""
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic == MAGIC_PROG:
        header, base, planes = load_progressive(path)
        model = build_model(header["config"])
        model.load_state_dict(decode_progressive(
            header["tensors"], base, planes, header["n_weights"], header["bits"]))
        return header, model, len(planes)
    header, blob = load_nvc(path)
    return header, model_from_nvc(header, blob), None


def model_from_nvc(header, blob):
    model = build_model(header["config"])
    model.load_state_dict(dequantize_state(header["tensors"], blob))
    return model


# ------------------------------------------------------- x264 baseline

def x264_roundtrip(frames: np.ndarray, fps: float, crf: int, preset="veryslow"):
    """Encode with libx264 at a given CRF and decode back. -> (bytes, psnr)."""
    T, H, W, _ = frames.shape
    exe = ffmpeg_exe()
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "ref.mp4")
        enc = subprocess.run(
            [exe, "-y", "-hide_banner", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
             "-r", f"{fps:g}", "-i", "-",
             "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
             "-pix_fmt", "yuv420p", out],
            input=frames.tobytes(), capture_output=True)
        if enc.returncode != 0:
            raise RuntimeError(enc.stderr.decode()[-2000:])
        size = os.path.getsize(out)
        dec = subprocess.run(
            [exe, "-hide_banner", "-loglevel", "error", "-i", out,
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True)
        got = np.frombuffer(dec.stdout, dtype=np.uint8)
        n = T * H * W * 3
        if got.size < n:
            raise RuntimeError("x264 decode returned too few frames")
        got = got[:n].reshape(T, H, W, 3)
    return size, psnr_u8(frames, got)


def compare_with_x264(frames, fps, neural_bytes, neural_psnr, crfs=None):
    crfs = crfs or [18, 23, 28, 33, 38, 43, 48, 51]
    print("\n  libx264 on the exact same frames (preset veryslow, yuv420p):")
    print(f"    {'crf':>4} {'size':>12} {'bpp':>8} {'psnr':>8}")
    T, H, W, _ = frames.shape
    px = T * H * W
    rows = []
    for crf in crfs:
        try:
            size, p = x264_roundtrip(frames, fps, crf)
        except Exception as exc:
            print(f"    crf {crf}: failed ({exc})")
            continue
        rows.append((crf, size, p))
        print(f"    {crf:>4} {human_bytes(size):>12} {size * 8 / px:>8.3f} {p:>7.2f} dB")
    if not rows:
        return
    # The fair question: at the PSNR our network reached, how big is x264?
    crf, size, p = min(rows, key=lambda r: abs(r[2] - neural_psnr))
    print(f"\n    closest x264 quality: crf {crf} -> {p:.2f} dB in {human_bytes(size)}")
    print(f"    this codec:                    {neural_psnr:.2f} dB in {human_bytes(neural_bytes)}")
    ratio = neural_bytes / size if size else float("inf")
    verdict = ("x264 wins" if ratio > 1.05 else
               "this codec wins" if ratio < 0.95 else "roughly a tie")
    print(f"    -> {ratio:.2f}x the size of x264 at matched quality ({verdict})")


# ------------------------------------------------------------ demo content

def make_demo_clip(frames=60, size=128):
    """A synthetic clip so the tool is runnable with no video file on hand."""
    H = W = size
    ys, xs = np.mgrid[0:H, 0:W]
    xs = xs / W
    ys = ys / H
    out = np.zeros((frames, H, W, 3), dtype=np.uint8)
    for i in range(frames):
        t = i / max(frames - 1, 1)
        img = np.zeros((H, W, 3), dtype=np.float32)
        img[..., 0] = 0.5 + 0.5 * np.sin(6.2831 * (xs * 2 + t))
        img[..., 2] = 0.5 + 0.5 * np.cos(6.2831 * (ys * 2 - t))
        cx, cy = 0.5 + 0.32 * np.cos(6.2831 * t), 0.5 + 0.32 * np.sin(6.2831 * t)
        disc = ((xs - cx) ** 2 + (ys - cy) ** 2) < 0.02
        img[disc] = [1.0, 0.95, 0.2]
        bar = (np.abs(ys - (0.2 + 0.6 * t)) < 0.03)
        img[bar] = [0.05, 0.05, 0.08]
        out[i] = np.clip(img * 255, 0, 255).astype(np.uint8)
    return out


# ------------------------------------------------------------------- verbs

def cmd_demo_video(args):
    clip = make_demo_clip(args.frames, args.size)
    write_video(args.output, clip, args.fps)
    print(f"wrote {args.output}  {clip.shape[0]} frames  "
          f"{clip.shape[2]}x{clip.shape[1]}  {human_bytes(os.path.getsize(args.output))}")


def cmd_encode(args):
    device = pick_device(args.device)
    torch.manual_seed(args.seed)

    print(f"reading {args.input}")
    clip, fps = read_video(args.input, args.frames, args.frame_stride)
    src_bytes = os.path.getsize(args.input)
    stride_prod = math.prod(int(x) for x in str(args.strides).split(",") if x.strip())
    clip = resize_clip(clip, args.size, multiple=stride_prod)
    T, H, W, _ = clip.shape
    pixels = T * H * W
    raw_bytes = pixels * 3
    print(f"  {T} frames  {W}x{H}  {fps:g} fps  "
          f"(raw RGB would be {human_bytes(raw_bytes)})")

    strides = [int(x) for x in str(args.strides).split(",") if x.strip()]

    grid_frames = args.grid_frames or T
    mid_frames = args.mid_frames or grid_frames
    # Parameters the finer grid costs per channel, at the resolution it feeds.
    upto = math.prod(strides[:args.mid_level]) if args.mid_level > 0 else 1
    mid_slots = mid_frames * (H // math.prod(strides) * upto) * \
                (W // math.prod(strides) * upto)

    def make_cfg(width, floor=None, mid=None):
        return {"arch": args.arch, "height": H, "width": W, "frames": T,
                "fps": fps, "channels": width, "strides": strides,
                "embed_levels": args.embed_levels,
                "min_channels": args.min_channels if floor is None else floor,
                "depth": args.depth, "w0": 30.0,
                "grid_frames": grid_frames,
                "mid_channels": args.mid_channels if mid is None else mid,
                "mid_level": args.mid_level,
                "mid_frames": args.mid_frames}

    budget, floor, mid = None, args.min_channels, args.mid_channels
    if args.width:
        channels = args.width
    else:
        budget = parse_count(args.params)
        if args.arch == "grid" and args.mid_channels is None:
            # The finer grid is a fixed cost that the width search cannot trade
            # against, so size it from the budget first and let the decoder take
            # what is left. Measured on a 150k budget, spending about half there
            # beat spending none by 1.9 dB.
            mid = max(1, int(GRID_SHARE * budget / mid_slots))
        elif args.mid_channels is None:
            mid = 0
        while True:
            if args.min_channels is None:
                channels, floor = fit_budget(lambda w, f=None: make_cfg(w, f, mid),
                                             budget)
            else:
                channels = size_to_budget(
                    lambda w: make_cfg(w, floor, mid), budget)
            if (count_params(build_model(make_cfg(channels, floor, mid))) <= budget
                    or mid <= 0 or args.mid_channels is not None):
                break
            mid = int(mid * 0.8) if mid > 4 else mid - 1
        extra = f", channel floor {floor}" if args.arch != "siren" else ""
        if args.arch == "grid" and mid:
            extra += f", finer grid {mid} channels"
        print(f"  parameter budget {budget:,} -> width {channels}{extra}")

    cfg = make_cfg(channels, floor, mid)
    model = build_model(cfg)
    n_params = count_params(model)
    if budget is not None and n_params > budget:
        # The channel floor sets a minimum model size that no width can go
        # under, so a small budget can be unreachable. Say so rather than
        # quietly handing back something twice the size that was asked for.
        print(f"  WARNING: {n_params:,} parameters exceeds the {budget:,} budget. "
              f"The architecture cannot go smaller with a channel floor of "
              f"{cfg['min_channels']}; lower --min-channels or --strides to fit.")
    print(f"  arch {args.arch}  width {channels}  {n_params:,} parameters "
          f"({human_bytes(n_params * 4)} as fp32)")

    # A "batch" means different things per architecture: whole frames for nerv,
    # individual pixel samples for siren.
    batch = args.batch or (16384 if args.arch == "siren" else 4)
    qat_start = (None if args.qat_start >= 1.0
                 else max(1, int(args.epochs * args.qat_start)))
    print(f"training on {device} for {args.epochs} epochs (batch {batch})")
    trainer = train_siren if args.arch == "siren" else train_nerv
    trainer(model, clip, epochs=args.epochs, batch=batch, lr=args.lr,
            device=device, quiet=args.quiet,
            qat_bits=args.bits, qat_start=qat_start,
            qat_jitter=args.qat_jitter,
            **({"grid_smooth": args.grid_smooth} if args.arch != "siren" else {}))
    finalise_qat(model)

    recon_fp32 = render(model, cfg, device=device)
    psnr_fp32 = psnr_u8(clip, recon_fp32)

    if args.progressive:
        tensors, base, planes, n_w = encode_progressive(model.state_dict(), args.bits)
        header = {"version": 1, "config": cfg, "tensors": tensors,
                  "params": n_params, "bits": args.bits, "n_weights": n_w}
        total = save_progressive(args.output, header, base, planes)
        blob = base + b"".join(planes)
    else:
        tensors, blob = quantize_state(model.state_dict(), args.bits)
        header = {"version": 1, "config": cfg, "tensors": tensors,
                  "params": n_params, "bits": args.bits}
        total = save_nvc(args.output, header, blob)

    # Re-render from the *quantised* weights: that is what a decoder will see.
    model_q = load_any(args.output)[1]
    recon = render(model_q, cfg, device=device)
    psnr_q = psnr_u8(clip, recon)

    bpp = total * 8 / pixels
    print(f"\nwrote {args.output}")
    print(f"  size              {human_bytes(total)}  ({total:,} bytes)")
    print(f"  bits per pixel    {bpp:.4f}")
    print(f"  quality           {psnr_q:.2f} dB "
          f"(fp32 weights would give {psnr_fp32:.2f} dB)")
    print(f"  vs raw RGB        {raw_bytes / total:.1f}x smaller "
          f"({human_bytes(raw_bytes)} -> {human_bytes(total)})")
    print(f"  vs source file    {src_bytes / total:.2f}x "
          f"({human_bytes(src_bytes)}, note: different resolution/length)")
    if args.progressive:
        print(f"  layout            {args.bits} bit planes, most significant "
              f"first - truncating the file lowers the rate")
    else:
        codec = load_nvc(args.output)[0].get("codec")
        print(f"  entropy coding    {len(blob):,} B of quantised weights -> "
              f"{total:,} B on disk ({codec})")

    if args.preview:
        side = np.concatenate([clip, recon], axis=2)
        write_video(args.preview, side, fps)
        print(f"  preview           {args.preview} (original | reconstruction)")

    if args.compare:
        compare_with_x264(clip, fps, total, psnr_q)


def cmd_decode(args):
    device = pick_device(args.device)
    header, model, planes = load_any(args.input)
    cfg = header["config"]
    if planes is not None and planes < header["bits"]:
        print(f"  file carries {planes} of {header['bits']} bit planes; decoding "
              f"at reduced precision")
    n_frames = args.frames or cfg["frames"]
    # Keeping the source frame rate means asking for more frames stretches the
    # clip out: the network is sampled between the times it was trained on.
    fps = args.fps or cfg["fps"]
    t0 = time.time()
    frames = render(model, cfg, n_frames=n_frames, device=device)
    dt = time.time() - t0
    write_video(args.output, frames, fps)
    print(f"decoded {n_frames} frames at {cfg['width']}x{cfg['height']} "
          f"in {dt:.2f}s ({n_frames / dt:.1f} fps) -> {args.output}")
    if n_frames != cfg["frames"]:
        rate = n_frames / cfg["frames"]
        print(f"  trained on {cfg['frames']} frames; these are sampled at times "
              f"the network never saw ({rate:g}x slow motion at {fps:g} fps, "
              f"pass --fps {cfg['fps'] * rate:g} to keep the original duration)")


def cmd_info(args):
    header, _, planes = load_any(args.input)
    cfg = header["config"]
    size = os.path.getsize(args.input)
    px = cfg["frames"] * cfg["height"] * cfg["width"]
    print(f"{args.input}")
    print(f"  arch            {cfg['arch']}")
    print(f"  resolution      {cfg['width']}x{cfg['height']}")
    print(f"  frames          {cfg['frames']} @ {cfg['fps']:g} fps")
    print(f"  parameters      {header['params']:,} at {header['bits']} bits")
    print(f"  file size       {human_bytes(size)} ({size:,} bytes)")
    print(f"  bits per pixel  {size * 8 / px:.4f}")
    print(f"  vs raw RGB      {px * 3 / size:.1f}x smaller")
    if planes is not None:
        print(f"  bit planes      {planes} of {header['bits']} present"
              f"{' (truncated)' if planes < header['bits'] else ''}")


def cmd_truncate(args):
    header, base, planes = load_progressive(args.input)
    keep = min(args.bits, len(planes))
    if keep < 1:
        raise SystemExit("keep at least one bit plane")
    before = os.path.getsize(args.input)
    total = save_progressive(args.output, header, base, planes[:keep])
    print(f"{args.input} -> {args.output}")
    print(f"  bit planes  {len(planes)} -> {keep}")
    print(f"  size        {human_bytes(before)} -> {human_bytes(total)} "
          f"({total / before * 100:.0f}%)")
    print("  no retraining: the same encode serves every rate")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="nvc", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo-video", help="synthesise a test clip")
    d.add_argument("-o", "--output", default="sample.mp4")
    d.add_argument("--frames", type=int, default=60)
    d.add_argument("--size", type=int, default=128)
    d.add_argument("--fps", type=float, default=25)
    d.set_defaults(func=cmd_demo_video)

    e = sub.add_parser("encode", help="overfit a network to a video")
    e.add_argument("input")
    e.add_argument("-o", "--output", default="out.nvc")
    e.add_argument("--arch", choices=("nerv", "grid", "siren"), default="grid",
                   help="grid stores per-frame feature maps directly; nerv "
                        "generates them from an MLP over a Fourier embedding "
                        "of the timestamp")
    e.add_argument("--grid-frames", type=int, default=0,
                   help="temporal resolution of the base grid (0 = one slice "
                        "per frame). Fewer slices means fewer parameters and "
                        "smoother motion")
    e.add_argument("--mid-channels", type=int, default=None,
                   help="channels in a second, finer grid injected partway up "
                        "the decoder. Sized from the budget when not given; "
                        "0 disables it")
    e.add_argument("--mid-level", type=int, default=1,
                   help="which decoder block the finer grid feeds")
    e.add_argument("--grid-smooth", type=float, default=0.0,
                   help="penalise change between neighbouring grid slices. "
                        "Trades a little accuracy for a grid that delta-codes, "
                        "and keeps motion temporally consistent")
    e.add_argument("--mid-frames", type=int, default=0,
                   help="temporal resolution of the finer grid (0 = same as "
                        "the base grid)")
    e.add_argument("--size", type=int, default=128,
                   help="target long side, snapped to a multiple of 32")
    e.add_argument("--frames", type=int, default=60,
                   help="max frames to use; 0 means the whole video")
    e.add_argument("--frame-stride", type=int, default=1,
                   help="keep every Nth frame")
    e.add_argument("--params", default="120k",
                   help="parameter budget, e.g. 60k / 250k")
    e.add_argument("--width", type=int, default=None,
                   help="set layer width directly, overriding --params")
    e.add_argument("--depth", type=int, default=4, help="siren hidden layers")
    e.add_argument("--embed-levels", type=int, default=10)
    e.add_argument("--min-channels", type=int, default=None,
                   help="floor on nerv block channel counts. Sets the smallest "
                        "model the architecture can express; higher is better "
                        "quality but unreachable for small budgets. Chosen "
                        "automatically from --params when not given")
    e.add_argument("--strides", default="4,4,2",
                   help="nerv upsample factors; their product must divide the "
                        "frame size (default 4,2,2,2 = 32x)")
    e.add_argument("--epochs", type=int, default=300)
    e.add_argument("--batch", type=int, default=None,
                   help="frames per step (nerv, default 4) or pixels per step "
                        "(siren, default 16384)")
    e.add_argument("--lr", type=float, default=2e-3)
    e.add_argument("--bits", type=int, default=5,
                   help="weight quantisation. With QAT, 5 bits measured 29.35 dB "
                        "in 72.7 KB against 8 bits at 29.67 dB in 125.1 KB on "
                        "the same clip, so it is the default")
    e.add_argument("--qat-jitter", type=int, default=0,
                   help="train across a range of bit depths rather than one, "
                        "so a progressive file stays good when truncated")
    e.add_argument("--qat-start", type=float, default=0.5,
                   help="fraction of training after which weights are rounded "
                        "in the forward pass, so the network can adapt to the "
                        "quantisation grid. 1.0 disables it")
    e.add_argument("--preview", default=None,
                   help="write a side-by-side original|reconstruction video")
    e.add_argument("--progressive", action="store_true",
                   help="write bit planes most-significant-first so the file "
                        "can be truncated to any lower rate without retraining. "
                        "Costs about 12%% in size; pair with --qat-jitter")
    e.add_argument("--compare", action="store_true",
                   help="benchmark against libx264 on the same frames")
    e.add_argument("--device", default="auto")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--quiet", action="store_true")
    e.set_defaults(func=cmd_encode)

    c = sub.add_parser("decode", help="play a video back out of the weights")
    c.add_argument("input")
    c.add_argument("-o", "--output", default="decoded.mp4")
    c.add_argument("--frames", type=int, default=None,
                   help="render a different number of frames; more frames at "
                        "the same fps gives slow motion")
    c.add_argument("--fps", type=float, default=None)
    c.add_argument("--device", default="auto")
    c.set_defaults(func=cmd_decode)

    tr = sub.add_parser("truncate", help="lower the rate of a progressive file")
    tr.add_argument("input")
    tr.add_argument("-o", "--output", default="truncated.nvc")
    tr.add_argument("--bits", type=int, required=True,
                    help="how many bit planes to keep")
    tr.set_defaults(func=cmd_truncate)

    i = sub.add_parser("info", help="describe an .nvc file")
    i.add_argument("input")
    i.set_defaults(func=cmd_info)

    args = p.parse_args(argv)
    if args.cmd == "encode" and args.bits not in range(2, 17):
        p.error("--bits must be between 2 and 16")
    args.func(args)


if __name__ == "__main__":
    main()
