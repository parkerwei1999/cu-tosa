"""CONV2D with a 1x1 kernel, stride 1 and no padding (a Linear layer) on the int8 tensor cores, one fused kernel.

Rows = the N * H * W input positions, K = the input channels, columns = the output channels; the weights are INT8
constants. With zero points:
    sum_k (x - izp)(w - wzp) = sum_k x w  -  wzp sum_k x  -  izp sum_k w  +  K izp wzp
sum_k x w is one int8 dot product for an INT8 input and two for an INT16 input (x = 256 hi + lo + 128, as in
matmul.py, so sum_k x w = 256 sum hi w + sum lo w + 128 sum_k w); sum_k x comes from the row sums in the kernel and
sum_k w is computed once from the constant. The bias is added, the accumulator REQUIRE is counted, and the result is
stored as the reference model does: the low 32 bits for an INT32 output, saturated for an INT48 output.
"""
import torch
import triton
import triton.language as tl

from ._launch import jit

MAX_K = 65280  # the int32 row sums of an INT16 input (proofs/limb_proof.py)


@jit
def _linear(x_ptr, w_ptr, y_ptr, c_ptr, bias_ptr, wsum_ptr, M_per: tl.int64, N: tl.int64, K: tl.int64, c0: tl.int64,
            izp: tl.int64, wzp: tl.int64, lo: tl.int64, hi: tl.int64,
            WIDE: tl.constexpr, SAT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n, sid = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    row = sid.to(tl.int64) * M_per + rm
    px = x_ptr + row[:, None] * K + rk[None, :]
    pw = w_ptr + rk[:, None] + rn[None, :] * K  # w is [N, K] row major: element (k, n) of the right operand
    lo_acc = tl.zeros((BM, BN), dtype=tl.int32)
    hi_acc = tl.zeros((BM, BN), dtype=tl.int32)
    rs = tl.zeros((BM,), dtype=tl.int32)
    for k in range(0, K, BK):
        mx = (rm[:, None] < M_per) & (rk[None, :] + k < K)
        w = tl.load(pw, mask=(rk[:, None] + k < K) & (rn[None, :] < N), other=0)
        if WIDE:
            x = tl.load(px, mask=mx, other=128).to(tl.int32)
            ux = x + 32768
            hi_acc = tl.dot(((ux >> 8) - 128).to(tl.int8), w, hi_acc, out_dtype=tl.int32)
            lo_acc = tl.dot(((ux & 255) - 128).to(tl.int8), w, lo_acc, out_dtype=tl.int32)
            rs += tl.sum(x - 128, axis=1)
        else:
            x = tl.load(px, mask=mx, other=0)
            lo_acc = tl.dot(x, w, lo_acc, out_dtype=tl.int32)
            rs += tl.sum(x.to(tl.int32), axis=1)
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
    n0 = tl.sum(tl.sum((bad & m2).to(tl.int64), axis=1), axis=0)
    if n0 > 0:
        tl.atomic_add(c_ptr + c0 + sid, n0)


def linear(x, w, bias64, wsum64, out_dtype, izp, wzp, lo, hi, saturate, cbuf, c0, parts=1):
    """x: contiguous [M, K] (int8, or int16 for an INT16 input). w: contiguous int8 [N, K]. bias64, wsum64: int64 [N]
    (the bias, and sum_k w). Returns y [M, N] at out_dtype; part p of `parts` equal row ranges adds the count of its
    accumulators outside [lo, hi] to the flat element c0 + p of cbuf (int64)."""
    M, K = x.shape
    N = w.shape[0]
    if K > MAX_K or N * K >= 2**31 or triton.cdiv(N, 64) > 65535:
        raise ValueError(f"CONV2D 1x1 with K = {K}, N = {N} is outside the kernel's limits (K <= {MAX_K} for the int32 "
                         f"row sums, fewer than 2^31 weights)")
    y = torch.empty((M, N), dtype=out_dtype, device=x.device)
    M_per = M // parts
    BM, BN, BK = 64, 64, 32
    _linear.launch((triton.cdiv(M_per, BM), triton.cdiv(N, BN), parts),
                   (x, w, y, cbuf, bias64, wsum64, M_per, N, K, c0, izp, wzp, lo, hi),
                   (x.dtype == torch.int16, saturate, BM, BN, BK), num_stages=2)
    return y
