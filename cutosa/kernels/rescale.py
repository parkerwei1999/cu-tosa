"""RESCALE as one fused kernel: read the input at its stored type, form the product in a 64-bit register, round,
shift, add the output zero point, count the two REQUIRE failures, clamp, write the output at its stored type.

Covers single rounding with signed input and output, one (multiplier, shift) pair or one pair per channel (channel =
last dimension), apply_scale_32 and apply_scale_16. The register never leaves int64: |value| < 2^48 with a multiplier
below 2^15 (apply_scale_16) or |value| <= 2^32 with a multiplier below 2^31 (apply_scale_32), plus a rounding term of
at most 2^61 (shift <= 62).

REQUIRE counts are added into two rows of the run's counter buffer (int64 [rows, samples]); see Emulator._slots.
"""
import torch
import triton
import triton.language as tl

BLOCK = 1024


@triton.jit
def _rescale(x_ptr, y_ptr, c_ptr, m_ptr, s_ptr, n_per, c0, c1, C, izp, m, s, ozp, olo, ohi,
             SCALE32: tl.constexpr, PER_CHANNEL: tl.constexpr, BLOCK: tl.constexpr):
    """one block of one sample. Counter c0: the elements that fail the first REQUIRE (SCALE32: the value entering
    apply_scale_32 outside [-2^(shift-1), 2^(shift-1)); else the result of apply_scale_16 outside INT32); counter c1:
    the elements whose result + output_zp is outside INT32."""
    pid = tl.program_id(0)
    sid = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
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
    """x: contiguous CUDA tensor. mult / shift: python ints, or (per_channel) int64 CUDA tensors of x.shape[-1] values.
    parts: x is split into this many equal parts in index order; part p adds its REQUIRE counts to the flat elements
    c0 + p and c1 + p of cbuf (int64). Returns y of x's shape and out_dtype."""
    n_per = x.numel() // parts
    y = torch.empty(x.shape, dtype=out_dtype, device=x.device)
    if per_channel:
        m_t, s_t, m, s, C = mult, shift, 0, 0, x.shape[-1]
    else:
        m_t = s_t = cbuf  # not read
        m, s, C = mult, shift, 1
    _rescale[(triton.cdiv(n_per, BLOCK), parts)](x, y, cbuf, m_t, s_t, n_per, c0, c1, C, izp, m, s, ozp, olo, ohi,
                                                 SCALE32=scale32, PER_CHANNEL=per_channel, BLOCK=BLOCK)
    return y
