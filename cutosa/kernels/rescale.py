"""RESCALE kernel: single rounding, signed, one multiplier or one per channel (last dimension), scale 32 / 16.
The int64 register cannot overflow: |x| <= 2^32 with mult < 2^31, or |x| < 2^48 with mult < 2^15, + <= 2^61."""
import torch
import triton
import triton.language as tl

from ._launch import jit

BLOCK = 1024


@jit
def _rescale(x_ptr, y_ptr, c_ptr, m_ptr, s_ptr, n_per: tl.int64, c0: tl.int64, c1: tl.int64, C: tl.int64,
             izp: tl.int64, m: tl.int64, s: tl.int64, ozp: tl.int64, olo: tl.int64, ohi: tl.int64,
             SCALE32: tl.constexpr, PER_CHANNEL: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    sid = tl.program_id(1)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)  # int64: a part may hold more than 2^31 elements
    mask = offs < n_per
    g = sid.to(tl.int64) * n_per + offs
    v = tl.load(x_ptr + g, mask=mask, other=0).to(tl.int64) - izp
    one = tl.full((BLOCK,), 1, tl.int64)
    if PER_CHANNEL:
        ch = g % C
        mm = tl.load(m_ptr + ch, mask=mask, other=0).to(tl.int64)
        ss = tl.load(s_ptr + ch, mask=mask, other=2).to(tl.int64)
    else:
        mm = one * m
        ss = one * s
    half = one << (ss - 1)
    r = (v * mm + half) >> ss
    if SCALE32:
        bad0 = (v < -half) | (v > half - 1)
    else:
        bad0 = (r < -2147483648) | (r > 2147483647)
    r += ozp
    bad1 = (r < -2147483648) | (r > 2147483647)
    r = tl.minimum(tl.maximum(r, olo), ohi)
    tl.store(y_ptr + g, r.to(y_ptr.dtype.element_ty), mask=mask)
    n0 = tl.sum((bad0 & mask).to(tl.int64), axis=0)
    n1 = tl.sum((bad1 & mask).to(tl.int64), axis=0)
    if n0 > 0:
        tl.atomic_add(c_ptr + c0 + sid, n0)
    if n1 > 0:
        tl.atomic_add(c_ptr + c1 + sid, n1)


def rescale(x, out_dtype, mult, shift, izp, ozp, olo, ohi, scale32, per_channel, cbuf, c0, c1, parts=1):
    """mult / shift: python ints, or int64 CUDA tensors of x.shape[-1] values (per_channel). Part p of `parts`
    equal parts of x adds its REQUIRE counts to cbuf[c0 + p] and cbuf[c1 + p]."""
    n_per = x.numel() // parts
    y = torch.empty(x.shape, dtype=out_dtype, device=x.device)
    if per_channel:
        m_t, s_t, m, s, C = mult, shift, 0, 0, x.shape[-1]
    else:
        m_t = s_t = cbuf  # not read
        m, s, C = mult, shift, 1
    _rescale.launch((triton.cdiv(n_per, BLOCK), parts), (x, y, cbuf, m_t, s_t, n_per, c0, c1, C, izp, m, s, ozp, olo, ohi),
                    (scale32, per_channel, BLOCK))
    return y
