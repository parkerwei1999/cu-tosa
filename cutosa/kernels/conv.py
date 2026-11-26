"""CONV2D on int8 tensor cores, as a GEMM over the taps (k = (ky KW + kx) IC + ic, the weight layout); an INT16 input
is split into two int8 limbs. sum (x - izp)(w - wzp) = sum x w - wzp sum x - izp sum w + K izp wzp, a padded tap
reading izp. INT32 output wraps, INT48 saturates."""
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from ._launch import add_count, jit

MAX_K = 65280  # the largest K whose int32 row sums of an INT16 input cannot overflow


class Geometry(NamedTuple):
    kernel: tuple[int, int]
    stride: tuple[int, int]
    pad: tuple[int, int, int, int]  # top, bottom, left, right
    dilation: tuple[int, int]

    def out_hw(self, ih, iw):
        (kh, kw), (sy, sx), (pt, pb, pl, pr), (dy, dx) = self
        return (ih - 1 + pt + pb - (kh - 1) * dy) // sy + 1, (iw - 1 + pl + pr - (kw - 1) * dx) // sx + 1

    def is_linear(self):
        return self.kernel == (1, 1) and self.stride == (1, 1) and not any(self.pad)


@jit
def _conv(x_ptr, w_ptr, y_ptr, c_ptr, bias_ptr, wsum_ptr, M_per: tl.int64, N: tl.int64, K: tl.int64, c0: tl.int64,  # pyright: ignore[reportInvalidTypeForm]
          izp: tl.int64, wzp: tl.int64, lo: tl.int64, hi: tl.int64, IH: tl.int64, IW: tl.int64, IC: tl.int64,  # pyright: ignore[reportInvalidTypeForm]
          OH: tl.int64, OW: tl.int64, KW: tl.int64, sy: tl.int64, sx: tl.int64, pt: tl.int64, pl: tl.int64,  # pyright: ignore[reportInvalidTypeForm]
          dy: tl.int64, dx: tl.int64,  # pyright: ignore[reportInvalidTypeForm]
          WIDE: tl.constexpr, SAT: tl.constexpr, GATHER: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
          BK: tl.constexpr):
    """Without GATHER the input is read as [M, K] row major (a 1x1 kernel, stride 1, no padding)."""
    pid_m, pid_n, sid = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    row = sid.to(tl.int64) * M_per + rm
    if GATHER:
        img = row // (OH * OW)
        ih0 = (row // OW) % OH * sy - pt
        iw0 = row % OW * sx - pl
    px = x_ptr + row[:, None] * K + rk[None, :]
    pw = w_ptr + rk[:, None] + rn[None, :] * K  # w is [N, K] row major: element (k, n) of the right operand
    lo_acc = tl.zeros((BM, BN), dtype=tl.int32)
    hi_acc = tl.zeros((BM, BN), dtype=tl.int32)
    rs = tl.zeros((BM,), dtype=tl.int32)
    for k in range(0, K, BK):
        mx = (rm[:, None] < M_per) & (rk[None, :] + k < K)
        if GATHER:
            kk = k + rk
            tap = kk // IC
            ih = ih0[:, None] + (tap // KW * dy)[None, :]
            iw = iw0[:, None] + (tap % KW * dx)[None, :]
            pad = (ih < 0) | (ih >= IH) | (iw < 0) | (iw >= IW)
            px = x_ptr + ((img[:, None] * IH + ih) * IW + iw) * IC + (kk % IC)[None, :]
            mload = mx & ~pad
            pad = mx & pad
        else:
            mload = mx
        w = tl.load(pw, mask=(rk[:, None] + k < K) & (rn[None, :] < N), other=0)
        if WIDE:
            x = tl.load(px, mask=mload, other=128).to(tl.int32)
            if GATHER:
                x = tl.where(pad, izp.to(tl.int32), x)
            ux = x + 32768
            hi_acc = tl.dot(((ux >> 8) - 128).to(tl.int8), w, hi_acc, out_dtype=tl.int32)
            lo_acc = tl.dot(((ux & 255) - 128).to(tl.int8), w, lo_acc, out_dtype=tl.int32)
            rs += tl.sum(x - 128, axis=1)
        else:
            x = tl.load(px, mask=mload, other=0)
            if GATHER:
                x = tl.where(pad, izp.to(tl.int8), x)
            lo_acc = tl.dot(x, w, lo_acc, out_dtype=tl.int32)
            rs += tl.sum(x.to(tl.int32), axis=1)
        if not GATHER:
            px += BK
        pw += BK
    mn = rn < N
    wsum = tl.load(wsum_ptr + rn, mask=mn, other=0)
    bias = tl.load(bias_ptr + rn, mask=mn, other=0)
    if WIDE:
        sxw = (hi_acc.to(tl.int64) << 8) + lo_acc.to(tl.int64) + 128 * wsum[None, :]
        sx = rs.to(tl.int64) + 128 * K
    else:
        sxw = lo_acc.to(tl.int64)
        sx = rs.to(tl.int64)
    acc = sxw - wzp * sx[:, None] - izp * wsum[None, :] + K * izp * wzp + bias[None, :]
    m2 = (rm[:, None] < M_per) & mn[None, :]
    bad = (acc < lo) | (acc > hi)
    if SAT:
        acc = tl.minimum(tl.maximum(acc, lo), hi)
    tl.store(y_ptr + row[:, None] * N + rn[None, :], acc.to(y_ptr.dtype.element_ty), mask=m2)
    add_count(c_ptr, c0 + sid, bad, m2)


def conv2d(x, w, bias64, wsum64, geo, out_dtype, izp, wzp, lo, hi, saturate, cbuf, c0, parts):
    """x [N, IH, IW, IC] int8 or int16, w [OC, K] int8 (K = KH KW IC), bias64 and wsum64 (sum_k w) int64 [OC] ->
    y [N OH OW, OC] at out_dtype. Counts the accumulators outside [lo, hi]; the parts are row ranges."""
    n, ih, iw, ic = x.shape
    oc, K = w.shape
    oh, ow = geo.out_hw(ih, iw)
    M = n * oh * ow
    if K > MAX_K or oc * K >= 2**31 or triton.cdiv(oc, 64) > 65535 or x.numel() >= 2**31 or M * oc >= 2**31:
        raise ValueError(f"CONV2D with K = {K}, {oc} output channels is outside the kernel's limits (K <= {MAX_K} for "
                         f"the int32 row sums, fewer than 2^31 weights and elements)")
    y = torch.empty((M, oc), dtype=out_dtype, device=x.device)
    M_per = M // parts
    BM, BN, BK = 64, 64, 32
    _conv.launch((triton.cdiv(M_per, BM), triton.cdiv(oc, BN), parts),
                 (x, w, y, cbuf, bias64, wsum64, M_per, oc, K, c0, izp, wzp, lo, hi, ih, iw, ic, oh, ow,
                  geo.kernel[1], *geo.stride, geo.pad[0], geo.pad[2], *geo.dilation),
                 (x.dtype == torch.int16, saturate, not geo.is_linear(), BM, BN, BK), num_stages=2)
    return y
