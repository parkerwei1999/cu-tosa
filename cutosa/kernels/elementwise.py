"""ADD / SUB / MUL, TABLE and REDUCE_SUM kernels.

ADD / SUB / MUL are one fused kernel: read both operands at their stored types, form the result in a 64-bit register
(a product of two INT32 values needs 62 bits), apply MUL's rounding shift, count the elements outside the output type
(the REQUIRE), and write the low bits of the result at the stored type (the wrap of the reference model).

Broadcasting: an operand whose dimensions other than 1 form one run of the output's dimensions is indexed in place as
(flat // inner) % n, inner = the product of the output dimensions after the run, n = the operand's element count. That
covers the same shape, a single value, (..., C) against (1, ..., 1, C) and (..., N, C) against (..., N, 1). Any other
pattern is expanded first.

REQUIRE counts are added into a row of the run's counter buffer (int64 [rows, samples]); see Emulator._slots.
"""
import math

import torch
import triton
import triton.language as tl

BLOCK = 1024
ADD, SUB, MUL = 0, 1, 2


@triton.jit
def _binary(a_ptr, b_ptr, y_ptr, c_ptr, n_per, c0, ia, na, ib, nb_b, shift, lo, hi,
            OP: tl.constexpr, A_FLAT: tl.constexpr, B_FLAT: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    sid = tl.program_id(1)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)  # int64 flat indices throughout
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
    n0 = tl.sum((bad & mask).to(tl.int64), axis=0)
    if n0 > 0:
        tl.atomic_add(c_ptr + c0 + sid, n0)


def broadcast_shape(a, b):
    """the broadcast of two shapes (torch.broadcast_shapes costs ~30 us a call, more than the kernel launch)."""
    a, b = tuple(a), tuple(b)
    if a == b:
        return a
    n = max(len(a), len(b))
    a, b = (1,) * (n - len(a)) + a, (1,) * (n - len(b)) + b
    if any(x != y and x != 1 and y != 1 for x, y in zip(a, b)):
        raise ValueError(f"shapes {a} and {b} do not broadcast")
    return tuple(y if x == 1 else x for x, y in zip(a, b))


def index_plan(xshape, shape):
    """how an operand of shape xshape is read over the flat index of `shape` (which it broadcasts to):
    -> (mode, inner, n): "flat" (the same elements), "index" ((flat // inner) % n) or "expand" (expand it first)."""
    total = math.prod(shape)
    xs = (1,) * (len(shape) - len(xshape)) + tuple(xshape)
    if xs == tuple(shape):
        return "flat", 1, total
    run = [i for i, d in enumerate(xs) if d != 1]
    if not run:
        return "index", 1, 1
    i0, i1 = run[0], run[-1]
    if all(xs[i] == shape[i] for i in range(i0, i1 + 1)):
        return "index", math.prod(shape[i1 + 1:]), math.prod(xs[i0:i1 + 1])
    return "expand", 1, total


def _operand(x, shape):
    """x -> (contiguous tensor, inner, n, flat) for the index (flat // inner) % n over the output's flat index."""
    mode, inner, n = index_plan(x.shape, shape)
    if mode == "expand":
        return x.expand(shape).contiguous(), 1, n, True
    return x.contiguous(), inner, n, mode == "flat"


def binary(op, a, b, out_dtype, lo, hi, cbuf, c0, shift=0, parts=1):
    """op: ADD / SUB / MUL. a, b: CUDA tensors that broadcast together. shift: MUL's rounding right shift (0..62).
    Returns y at out_dtype (the low bits of the exact result). Part p of `parts` equal parts of y in index order adds
    the count of its elements whose exact result is outside [lo, hi] to the flat element c0 + p of cbuf (int64)."""
    shape = broadcast_shape(a.shape, b.shape)
    n_per = math.prod(shape) // parts
    a, ia, na, a_flat = _operand(a, shape)
    b, ib, nb_b, b_flat = _operand(b, shape)
    y = torch.empty(shape, dtype=out_dtype, device=a.device)
    _binary[(triton.cdiv(n_per, BLOCK), parts)](a, b, y, cbuf, n_per, c0, ia, na, ib, nb_b, shift, lo, hi,
                                                OP=op, A_FLAT=a_flat, B_FLAT=b_flat, BLOCK=BLOCK)
    return y


@triton.jit
def _table(x_ptr, t_ptr, y_ptr, n, WIDE: tl.constexpr, BLOCK: tl.constexpr):
    """TABLE. WIDE: INT16 input, 513 INT16 entries, INT32 output = entry * 128 + slope * fraction (9-bit index, 7-bit
    fraction). Else: INT8 input, 256 INT8 entries."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
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
def _reduce_sum(x_ptr, y_ptr, c_ptr, n_per, c0, L, inner, lo, hi, BLOCK: tl.constexpr):
    """sum over one axis of length L that is not the last (inner = the element count of the dimensions after it): the
    lanes of a block are neighbours in memory. The REQUIRE of TOSA is on every running sum in index order, so the sum
    runs sequentially; an output element counts once if any of its running sums is outside [lo, hi]. The low bits of
    the total are stored."""
    pid = tl.program_id(0)
    sid = tl.program_id(1)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)  # int64 flat indices throughout
    mask = offs < n_per
    j = sid.to(tl.int64) * n_per + offs
    inner = inner.to(tl.int64)
    base = (j // inner) * (L * inner) + (j % inner)
    acc = tl.zeros((BLOCK,), dtype=tl.int64)
    bad = tl.zeros((BLOCK,), dtype=tl.int64)
    for k in range(0, L):
        acc += tl.load(x_ptr + base + k * inner, mask=mask, other=0).to(tl.int64)
        bad = bad | ((acc < lo) | (acc > hi)).to(tl.int64)
    tl.store(y_ptr + j, acc.to(y_ptr.dtype.element_ty), mask=mask)
    n0 = tl.sum(bad * mask.to(tl.int64), axis=0)
    if n0 > 0:
        tl.atomic_add(c_ptr + c0 + sid, n0)


@triton.jit
def _reduce_sum_last(x_ptr, y_ptr, c_ptr, n_per, c0, L, lo, hi, BJ: tl.constexpr, BK: tl.constexpr):
    """the same over the last axis: a tile of BJ outputs x BK elements is read along the axis, and its running sums
    are the carried sum plus a cumulative sum inside the tile. A padded element adds 0, which repeats a running sum
    that exists."""
    pid = tl.program_id(0)
    sid = tl.program_id(1)
    rj = pid.to(tl.int64) * BJ + tl.arange(0, BJ)
    mj = rj < n_per
    j = sid.to(tl.int64) * n_per + rj
    rk = tl.arange(0, BK)
    acc = tl.zeros((BJ,), dtype=tl.int64)
    bad = tl.zeros((BJ,), dtype=tl.int64)
    for k in range(0, L, BK):
        m2 = mj[:, None] & (rk[None, :] + k < L)
        t = tl.load(x_ptr + j[:, None] * L + k + rk[None, :], mask=m2, other=0).to(tl.int64)
        run = acc[:, None] + tl.cumsum(t, axis=1)
        bad = bad | tl.max(((run < lo) | (run > hi)).to(tl.int64), axis=1)
        acc += tl.sum(t, axis=1)
    tl.store(y_ptr + j, acc.to(y_ptr.dtype.element_ty), mask=mj)
    n0 = tl.sum(bad * mj.to(tl.int64), axis=0)
    if n0 > 0:
        tl.atomic_add(c_ptr + c0 + sid, n0)


def reduce_sum(x, axis, out_dtype, lo, hi, cbuf, c0, parts=1):
    """Returns y = the sum over `axis` kept as a dimension of 1, at out_dtype; REQUIRE counts as in binary()."""
    x = x.contiguous()
    L = x.shape[axis]
    inner = math.prod(x.shape[axis + 1:])
    shape = tuple(1 if i == axis else d for i, d in enumerate(x.shape))
    n_per = math.prod(shape) // parts
    y = torch.empty(shape, dtype=out_dtype, device=x.device)
    if inner == 1:
        BJ, BK = 64, 32
        _reduce_sum_last[(triton.cdiv(n_per, BJ), parts)](x, y, cbuf, n_per, c0, L, lo, hi, BJ=BJ, BK=BK)
    else:
        _reduce_sum[(triton.cdiv(n_per, BLOCK), parts)](x, y, cbuf, n_per, c0, L, inner, lo, hi, BLOCK=BLOCK)
    return y
