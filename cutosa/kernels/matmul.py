"""MATMUL on int8 tensor cores. INT8 x INT8 -> INT32: one product plus the zero-point sums. INT16 x INT16 -> INT48
(no 16-bit integer MMA): x = 256 hi + lo + 128, so x y = 65536 hx hy + 256 (hx ly + lx hy) + lx ly + 128 (x - 128)
+ 128 (y - 128) + 16384 (proofs/limb_proof.py)."""
import torch
import triton
import triton.language as tl

from ._launch import jit

MAX_K = 65280  # (2^31 - 1) // 32896: the largest K whose int32 row sums cannot overflow (proofs/limb_proof.py)
# (2^31 - 1) // 255^2: |a - azp| and |b - bzp| are at most 255, so up to this K no INT32 result can wrap
MAX_K_INT8 = 33025
BM, BN, BK = 64, 64, 32


def fits(Bn, M, K, N, wide):
    """The kernel's limits: K within proofs/limb_proof.py (INT16) or the INT32 bound (INT8), launch grid axes 2 and 3
    <= 65535, int32 element offsets < 2^31."""
    return K <= (MAX_K if wide else MAX_K_INT8) and Bn <= 65535 and triton.cdiv(N, BN) <= 65535 \
        and max(Bn * M * K, Bn * K * N, Bn * M * N) < 2**31


@jit
def _bmm(a_ptr, b_ptr, c_ptr, M: tl.int64, N: tl.int64, K: tl.int64, azp: tl.int64, bzp: tl.int64,  # pyright: ignore[reportInvalidTypeForm]
         sab: tl.int64, sam: tl.int64, sak: tl.int64, sbb: tl.int64, sbk: tl.int64, sbn: tl.int64,  # pyright: ignore[reportInvalidTypeForm]
         scb: tl.int64, scm: tl.int64, scn: tl.int64,  # pyright: ignore[reportInvalidTypeForm]
         WIDE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """c[b, m, n] = sum_k (a[b, m, k] - azp) (b[b, k, n] - bzp), exact. A padded element loads 0, or 128 when WIDE:
    both limbs 0, x - 128 = 0."""
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
        ma = (rm[:, None] < M) & (rk[None, :] + k < K)
        mb = (rk[:, None] + k < K) & (rn[None, :] < N)
        if WIDE:
            x = tl.load(pa, mask=ma, other=128).to(tl.int32)
            y = tl.load(pb, mask=mb, other=128).to(tl.int32)
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
        else:
            x = tl.load(pa, mask=ma, other=0)
            y = tl.load(pb, mask=mb, other=0)
            ll = tl.dot(x, y, ll, out_dtype=tl.int32)
            rs += tl.sum(x.to(tl.int32), axis=1)
            cs += tl.sum(y.to(tl.int32), axis=0)
        pa += BK * sak
        pb += BK * sbk
    if WIDE:
        out = (hh.to(tl.int64) << 16) + (hl.to(tl.int64) << 8) + ll.to(tl.int64)
        out += 128 * rs.to(tl.int64)[:, None] + 128 * cs.to(tl.int64)[None, :] + 128 * 128 * K
    else:
        out = ll.to(tl.int64) - bzp * rs.to(tl.int64)[:, None] - azp * cs.to(tl.int64)[None, :] + K * azp * bzp
    pc = c_ptr + pid_b * scb + rm[:, None] * scm + rn[None, :] * scn
    tl.store(pc, out.to(c_ptr.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))


def bmm(a, b, azp, bzp):
    """a [B, M, K], b [B, K, N]: CUDA tensors, both int8 (INT8 -> int32 result) or both int16 with zero points 0
    (INT16 -> int64 result holding INT48); any strides."""
    Bn, M, K = a.shape
    N = b.shape[2]
    wide = a.dtype == torch.int16
    if b.shape[0] != Bn or b.shape[1] != K or b.dtype != a.dtype or (wide and (azp or bzp)):
        raise ValueError(f"MATMUL {a.dtype} {tuple(a.shape)} x {b.dtype} {tuple(b.shape)}, zero points {azp} {bzp}")
    if not fits(Bn, M, K, N, wide):
        raise ValueError(f"MATMUL {tuple(a.shape)} x {tuple(b.shape)} is outside the kernel's limits (K <= "
                         f"{MAX_K if wide else MAX_K_INT8}, batch <= 65535, fewer than 2^31 elements)")
    c = torch.empty((Bn, M, N), dtype=torch.int64 if wide else torch.int32, device=a.device)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN), Bn)
    _bmm.launch(grid, (a, b, c, M, N, K, azp, bzp, *a.stride(), *b.stride(), *c.stride()), (wide, BM, BN, BK),
                num_stages=2)
    return c
