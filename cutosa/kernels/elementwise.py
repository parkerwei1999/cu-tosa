"""ADD / SUB / MUL as one fused kernel: read both operands at their stored types, form the result in a 64-bit
register (a product of two INT32 values needs 62 bits), apply MUL's rounding shift, count the elements outside the
output type (the REQUIRE), and write the low bits of the result at the stored type (the wrap of the reference model).

Broadcasting: an operand whose dimensions other than 1 form one run of the output's dimensions is indexed in place as
(flat // inner) % n, inner = the product of the output dimensions after the run, n = the operand's element count. That
covers the same shape, a single value, (..., C) against (1, ..., 1, C) and (..., N, C) against (..., N, 1). Any other
pattern is expanded first.
"""
import math

import torch
import triton
import triton.language as tl

BLOCK = 1024
ADD, SUB, MUL = 0, 1, 2


@triton.jit
def _binary(a_ptr, b_ptr, y_ptr, c_ptr, n_per, nb, ia, na, ib, nb_b, shift, lo, hi,
            OP: tl.constexpr, A_FLAT: tl.constexpr, B_FLAT: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    sid = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_per
    g = sid.to(tl.int64) * n_per + offs
    if A_FLAT:
        ga = g
    else:
        ga = (g // ia) % na
    if B_FLAT:
        gb = g
    else:
        gb = (g // ib) % nb_b
    a = tl.load(a_ptr + ga, mask=mask, other=0).to(tl.int64)
    b = tl.load(b_ptr + gb, mask=mask, other=0).to(tl.int64)
    if OP == 0:
        r = a + b
    elif OP == 1:
        r = a - b
    else:
        one = tl.full((BLOCK,), 1, tl.int64)
        r = (a * b + ((one << shift) >> 1)) >> shift  # shift 0: the product itself
    bad = (r < lo) | (r > hi)
    tl.store(y_ptr + g, r.to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(c_ptr + sid * nb + pid, tl.sum((bad & mask).to(tl.int32), axis=0))


def _operand(x, shape):
    """x -> (contiguous tensor, inner, n, flat) for the index (flat // inner) % n over the output's flat index."""
    total = math.prod(shape)
    xs = (1,) * (len(shape) - x.dim()) + tuple(x.shape)
    x = x.contiguous()
    if xs == tuple(shape):
        return x, 1, total, True
    run = [i for i, d in enumerate(xs) if d != 1]
    if not run:
        return x, 1, 1, False
    i0, i1 = run[0], run[-1]
    if all(xs[i] == shape[i] for i in range(i0, i1 + 1)):
        return x, math.prod(shape[i1 + 1:]), math.prod(xs[i0:i1 + 1]), False
    return x.expand(shape).contiguous(), 1, total, True


def binary(op, a, b, out_dtype, lo, hi, shift=0, samples=1):
    """op: ADD / SUB / MUL. a, b: CUDA tensors that broadcast together. shift: MUL's rounding right shift (0..62).
    Returns (y, counts): y at out_dtype (low bits of the exact result), counts = int64 tensor, per part of `samples`
    equal parts of y in index order, of the elements whose exact result is outside [lo, hi]."""
    shape = tuple(torch.broadcast_shapes(a.shape, b.shape))
    n_per = math.prod(shape) // samples
    nb = triton.cdiv(n_per, BLOCK)
    a, ia, na, a_flat = _operand(a, shape)
    b, ib, nb_b, b_flat = _operand(b, shape)
    y = torch.empty(shape, dtype=out_dtype, device=a.device)
    c = torch.empty((samples, nb), dtype=torch.int32, device=a.device)
    _binary[(nb, samples)](a, b, y, c, n_per, nb, ia, na, ib, nb_b, shift, lo, hi,
                           OP=op, A_FLAT=a_flat, B_FLAT=b_flat, BLOCK=BLOCK)
    return y, c.sum(1)


@triton.jit
def _table(x_ptr, t_ptr, y_ptr, n, WIDE: tl.constexpr, BLOCK: tl.constexpr):
    """TABLE. WIDE: INT16 input, 513 INT16 entries, INT32 output = entry * 128 + slope * fraction (9-bit index, 7-bit
    fraction). Else: INT8 input, 256 INT8 entries."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.int32)
    if WIDE:
        idx = (x >> 7) + 256
        base = tl.load(t_ptr + idx, mask=mask, other=0).to(tl.int32)
        nxt = tl.load(t_ptr + idx + 1, mask=mask, other=0).to(tl.int32)
        y = base * 128 + (nxt - base) * (x & 127)
    else:
        y = tl.load(t_ptr + x + 128, mask=mask, other=0)
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


def table(x, tab, out_dtype, wide):
    x = x.contiguous()
    n = x.numel()
    y = torch.empty(x.shape, dtype=out_dtype, device=x.device)
    _table[(triton.cdiv(n, BLOCK),)](x, tab, y, n, WIDE=wide, BLOCK=BLOCK)
    return y


@triton.jit
def _reduce_sum(x_ptr, y_ptr, c_ptr, n_per, nb, L, inner, lo, hi, BLOCK: tl.constexpr):
    """sum over one axis of length L (inner = the element count of the dimensions after it). The REQUIRE of TOSA is on
    every running sum in index order, so the sum runs sequentially; an output element counts once if any of its
    running sums is outside [lo, hi]. The low bits of the total are stored."""
    pid = tl.program_id(0)
    sid = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_per
    j = sid.to(tl.int64) * n_per + offs
    base = (j // inner) * (L * inner) + (j % inner)
    acc = tl.zeros((BLOCK,), dtype=tl.int64)
    bad = tl.zeros((BLOCK,), dtype=tl.int32)
    for k in range(0, L):
        acc += tl.load(x_ptr + base + k * inner, mask=mask, other=0).to(tl.int64)
        bad = bad | ((acc < lo) | (acc > hi)).to(tl.int32)
    tl.store(y_ptr + j, acc.to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(c_ptr + sid * nb + pid, tl.sum(bad * mask.to(tl.int32), axis=0))


def reduce_sum(x, axis, out_dtype, lo, hi, samples=1):
    """Returns (y, counts): y = the sum over `axis` kept as a dimension of 1, at out_dtype."""
    x = x.contiguous()
    L = x.shape[axis]
    inner = math.prod(x.shape[axis + 1:])
    shape = tuple(1 if i == axis else d for i, d in enumerate(x.shape))
    n_per = math.prod(shape) // samples
    nb = triton.cdiv(n_per, BLOCK)
    y = torch.empty(shape, dtype=out_dtype, device=x.device)
    c = torch.empty((samples, nb), dtype=torch.int32, device=x.device)
    _reduce_sum[(nb, samples)](x, y, c, n_per, nb, L, inner, lo, hi, BLOCK=BLOCK)
    return y, c.sum(1)
