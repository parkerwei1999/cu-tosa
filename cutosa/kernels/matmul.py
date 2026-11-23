"""MATMUL INT16 x INT16 -> INT48 on the int8 tensor cores, one fused kernel.

Tensor cores have no 16-bit integer multiply-accumulate (int8 is the widest integer type), so each INT16 operand is
split in registers into two int8 limbs: x = 256 hi + lo + 128 with hi, lo in [-128, 127]. Then
    x * y = 65536 hi_x hi_y + 256 (hi_x lo_y + lo_x hi_y) + lo_x lo_y + 128 (x - 128) + 128 (y - 128) + 16384
so a batched matmul is four int8 dot products (int32 accumulators) plus the row sums of x - 128 and the column sums of
y - 128, combined in int64 and written once. proofs/limb_proof.py proves the identity for every INT16 pair and the
accumulator bounds: the row / column sums overflow int32 first, at K > 65280.
"""
import torch
import triton
import triton.language as tl

MAX_K = 65280  # proofs/limb_proof.py: (2^31 - 1) // 32896


@triton.jit
def _bmm_i16(a_ptr, b_ptr, c_ptr, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """c[b, m, n] = sum_k a[b, m, k] * b[b, k, n], exact. A padded element loads 128: both limbs 0, x - 128 = 0."""
    pid_m, pid_n, pid_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    pa = a_ptr + pid_b * sab + rm[:, None] * sam + rk[None, :] * sak
    pb = b_ptr + pid_b * sbb + rk[:, None] * sbk + rn[None, :] * sbn
    hh = tl.zeros((BM, BN), dtype=tl.int32)
    hl = tl.zeros((BM, BN), dtype=tl.int32)
    ll = tl.zeros((BM, BN), dtype=tl.int32)
    rs = tl.zeros((BM,), dtype=tl.int32)
    cs = tl.zeros((BN,), dtype=tl.int32)
    for k in range(0, K, BK):
        x = tl.load(pa, mask=(rm[:, None] < M) & (rk[None, :] + k < K), other=128).to(tl.int32)
        y = tl.load(pb, mask=(rk[:, None] + k < K) & (rn[None, :] < N), other=128).to(tl.int32)
        ux = x + 32768
        uy = y + 32768
        xh = ((ux >> 8) - 128).to(tl.int8)
        xl = ((ux & 255) - 128).to(tl.int8)
        yh = ((uy >> 8) - 128).to(tl.int8)
        yl = ((uy & 255) - 128).to(tl.int8)
        hh = tl.dot(xh, yh, hh, out_dtype=tl.int32)
        hl = tl.dot(xh, yl, hl, out_dtype=tl.int32)
        hl = tl.dot(xl, yh, hl, out_dtype=tl.int32)
        ll = tl.dot(xl, yl, ll, out_dtype=tl.int32)
        rs += tl.sum(x - 128, axis=1)
        cs += tl.sum(y - 128, axis=0)
        pa += BK * sak
        pb += BK * sbk
    out = (hh.to(tl.int64) << 16) + (hl.to(tl.int64) << 8) + ll.to(tl.int64)
    out += 128 * rs.to(tl.int64)[:, None] + 128 * cs.to(tl.int64)[None, :] + 128 * 128 * K
    pc = c_ptr + pid_b * scb + rm[:, None] * scm + rn[None, :] * scn
    tl.store(pc, out, mask=(rm[:, None] < M) & (rn[None, :] < N))


def bmm_i16(a, b):
    """a [B, M, K], b [B, K, N]: CUDA tensors of INT16 values (int16 storage; any strides). Returns int64 [B, M, N]."""
    Bn, M, K = a.shape
    N = b.shape[2]
    if b.shape[0] != Bn or b.shape[1] != K:
        raise ValueError(f"MATMUL shapes {tuple(a.shape)} x {tuple(b.shape)}")
    if K > MAX_K:
        raise ValueError(f"limb MATMUL is exact only for K <= {MAX_K} (int32 row sums), got {K}")
    c = torch.empty((Bn, M, N), dtype=torch.int64, device=a.device)
    BM, BN, BK = 64, 64, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN), Bn)
    _bmm_i16[grid](a, b, c, M, N, K, *a.stride(), *b.stride(), *c.stride(), BM=BM, BN=BN, BK=BK, num_stages=2)
    return c
