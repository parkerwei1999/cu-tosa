"""ADD / SUB / MUL, TABLE and REDUCE_SUM kernels. A result outside its type keeps its low bits, as in the
reference model. A broadcast operand is read in place as x[(g // inner) % n] when possible, else expanded."""
import math

import torch
import triton
import triton.language as tl

from ._launch import jit

BLOCK = 1024
ADD, SUB, MUL = 0, 1, 2


@jit
def _binary(a_ptr, b_ptr, y_ptr, c_ptr, n_per: tl.int64, c0: tl.int64, ia: tl.int64, na: tl.int64, ib: tl.int64,  # pyright: ignore[reportInvalidTypeForm]
            nb_b: tl.int64, shift: tl.int64, lo: tl.int64, hi: tl.int64,  # pyright: ignore[reportInvalidTypeForm]
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
        r = (a * b + ((one << shift) >> 1)) >> shift
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
    """how to read an operand of xshape at output index g of `shape`: ("flat", 1, total) reads x[g],
    ("index", inner, n) reads x[(g // inner) % n], ("expand", 1, total) needs an expanded copy."""
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
    """-> (contiguous tensor, inner, n, flat) to read x at output index g as x[(g // inner) % n]."""
    mode, inner, n = index_plan(x.shape, shape)
    if mode == "expand":
        return x.expand(shape).contiguous(), 1, n, True
    return x.contiguous(), inner, n, mode == "flat"


def binary(op, a, b, out_dtype, lo, hi, cbuf, c0, shift, parts):
    """op: ADD / SUB / MUL; shift: MUL's rounding shift (0..62). -> y at out_dtype, the low bits of the exact result.
    Part p of `parts` equal parts adds its count of results outside [lo, hi] to cbuf[c0 + p]."""
    shape = broadcast_shape(a.shape, b.shape)
    n_per = math.prod(shape) // parts
    a, ia, na, a_flat = _operand(a, shape)
    b, ib, nb_b, b_flat = _operand(b, shape)
    y = torch.empty(shape, dtype=out_dtype, device=a.device)
    _binary.launch((triton.cdiv(n_per, BLOCK), parts), (a, b, y, cbuf, n_per, c0, ia, na, ib, nb_b, shift, lo, hi),
                   (op, a_flat, b_flat, BLOCK))
    return y


@jit
def _table_int8(x_ptr, t_ptr, y_ptr, n: tl.int64, BLOCK: tl.constexpr):  # pyright: ignore[reportInvalidTypeForm]
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.int32)
    tl.store(y_ptr + offs, tl.load(t_ptr + x + 128, mask=mask, other=0), mask=mask)


@jit
def _table_int16(x_ptr, t_ptr, y_ptr, c_ptr, n_per: tl.int64, c0: tl.int64, BLOCK: tl.constexpr):  # pyright: ignore[reportInvalidTypeForm]
    sid = tl.program_id(1)
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_per
    g = sid.to(tl.int64) * n_per + offs
    x = tl.load(x_ptr + g, mask=mask, other=0).to(tl.int32)
    idx = (x >> 7) + 256
    base = tl.load(t_ptr + idx, mask=mask, other=0).to(tl.int32)
    slope = tl.load(t_ptr + idx + 1, mask=mask, other=0).to(tl.int32) - base
    tl.store(y_ptr + g, base * 128 + slope * (x & 127), mask=mask)
    n0 = tl.sum((((slope < -32768) | (slope > 32767)) & mask).to(tl.int64), axis=0)
    if n0 > 0:
        tl.atomic_add(c_ptr + c0 + sid, n0)


def table_int8(x, tab):
    """256 entries: y = tab[x + 128]."""
    x = x.contiguous()
    n = x.numel()
    y = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    _table_int8.launch((triton.cdiv(n, BLOCK),), (x, tab, y, n), (BLOCK,))
    return y


def table_int16(x, tab, cbuf, c0, parts):
    """513 entries: y (INT32) = entry * 128 + slope * (x & 127), entry = tab[(x >> 7) + 256], slope = the next entry -
    entry. Part p of `parts` equal parts adds its count of elements whose slope is outside INT16 to cbuf[c0 + p]."""
    x = x.contiguous()
    n_per = x.numel() // parts
    y = torch.empty(x.shape, dtype=torch.int32, device=x.device)
    _table_int16.launch((triton.cdiv(n_per, BLOCK), parts), (x, tab, y, cbuf, n_per, c0), (BLOCK,))
    return y


@jit
def _reduce_sum(x_ptr, y_ptr, c_ptr, n_per: tl.int64, c0: tl.int64, L: tl.int64, inner: tl.int64, lo: tl.int64,  # pyright: ignore[reportInvalidTypeForm]
                hi: tl.int64, BLOCK: tl.constexpr):  # pyright: ignore[reportInvalidTypeForm]
    """sum over an axis that is not the last; sequential because the REQUIRE is on every running sum (an output
    counts once if any of its running sums is outside [lo, hi])."""
    pid = tl.program_id(0)
    sid = tl.program_id(1)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)  # int64 flat indices throughout
    mask = offs < n_per
    j = sid.to(tl.int64) * n_per + offs
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


@jit
def _reduce_sum_last(x_ptr, y_ptr, c_ptr, n_per: tl.int64, c0: tl.int64, L: tl.int64, lo: tl.int64, hi: tl.int64,  # pyright: ignore[reportInvalidTypeForm]
                     BJ: tl.constexpr, BK: tl.constexpr):
    """the same over the last axis, in BJ x BK tiles: running sums = the carried sum + a cumsum inside the tile."""
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


def reduce_sum(x, axis, out_dtype, lo, hi, cbuf, c0, parts):
    """Returns y = the sum over `axis` kept as a dimension of 1, at out_dtype; REQUIRE counts as in binary()."""
    x = x.contiguous()
    L = x.shape[axis]
    inner = math.prod(x.shape[axis + 1:])
    shape = tuple(1 if i == axis else d for i, d in enumerate(x.shape))
    n_per = math.prod(shape) // parts
    y = torch.empty(shape, dtype=out_dtype, device=x.device)
    if inner == 1:
        # one tile spans the whole axis (<= 1024), one program ~2048 elements: smaller tiles were 20x slower
        BK = min(max(32, triton.next_power_of_2(L)), 1024)
        BJ = max(2, 2048 // BK)
        _reduce_sum_last.launch((triton.cdiv(n_per, BJ), parts), (x, y, cbuf, n_per, c0, L, lo, hi), (BJ, BK))
    else:
        _reduce_sum.launch((triton.cdiv(n_per, BLOCK), parts), (x, y, cbuf, n_per, c0, L, inner, lo, hi), (BLOCK,))
    return y
