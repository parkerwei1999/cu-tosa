"""Flat indices beyond 2^31 (needs ~5 GiB of GPU memory): ADD of an int8 (N, 1) with an int8 (1, 1024), 2^31 + 1024
output elements in one part, then RESCALE and REDUCE_SUM over that tensor, each against torch on slices, and the
generated kernel of the unit ADD + RESCALE against the two separate results.
usage: python tests/big_index_check.py
"""
import sys

import torch

from cutosa.kernels import fused
from cutosa.kernels.elementwise import ADD, binary, reduce_sum
from cutosa.kernels.rescale import rescale

N, C = 2**21 + 1, 1024
gen = torch.Generator().manual_seed(0)
a = torch.randint(-128, 128, (N, 1), generator=gen, dtype=torch.int8).cuda()
b = torch.randint(-128, 128, (1, C), generator=gen, dtype=torch.int8).cuda()
cbuf = torch.zeros(4, dtype=torch.int64, device="cuda")
y = binary(ADD, a, b, torch.int8, -128, 127, cbuf, 0)
ok, bad = True, 0
for i in range(0, N, 2**16):
    want = a[i:i + 2**16].to(torch.int16) + b.to(torch.int16)
    ok &= bool(torch.equal(y[i:i + 2**16], want.to(torch.int8)))
    bad += int(((want < -128) | (want > 127)).sum())
ok &= int(cbuf[0]) == bad
print(f"{'ok  ' if ok else 'FAIL'} ADD: {y.numel()} = 2^31 + {y.numel() - 2**31} elements, {bad} REQUIRE failures")
res = [ok]

r = rescale(y, torch.int8, 1 << 30, 31, 3, -5, -128, 127, True, False, cbuf, 1, 2)
ok = True
for i in range(0, N, 2**16):
    v = y[i:i + 2**16].to(torch.int64) - 3
    ok &= bool(torch.equal(r[i:i + 2**16], (((v << 30) + (1 << 30)) >> 31).add(-5).clamp(-128, 127).to(torch.int8)))
ok &= cbuf[1:3].tolist() == [0, 0]
print(f"{'ok  ' if ok else 'FAIL'} RESCALE over the same tensor")
res.append(ok)

prog = (("i", "i"), (("ADD", (8,), ("L0", "L1")), ("RESCALE", (True, False), ("V0",))))
params = torch.tensor([C, N, 1, C, -128, 127, 3, 1 << 30, 31, -5, -128, 127, 0], dtype=torch.int64, device="cuda")
cnt = torch.zeros(4, dtype=torch.int64, device="cuda")
u = torch.empty((N, C), dtype=torch.int8, device="cuda")
fused.launch(fused.kernel(prog), None, u, cnt, params, u.numel(), 1, 0, 1, [a.contiguous(), b.contiguous()], [])
ok = all(bool(torch.equal(u[i:i + 2**16], r[i:i + 2**16])) for i in range(0, N, 2**16)) and cnt[:3].tolist() == [bad, 0, 0]
print(f"{'ok  ' if ok else 'FAIL'} generated ADD + RESCALE kernel over the same elements")
res.append(ok)
del r, u

for axis in (0, 1):
    cbuf.zero_()
    s = reduce_sum(y, axis, torch.int32, -2**31, 2**31 - 1, cbuf, 0)
    want = torch.zeros_like(s, dtype=torch.int64)
    for i in range(0, N, 2**16):
        part = y[i:i + 2**16].sum(axis, keepdim=True, dtype=torch.int64)
        if axis == 0:
            want += part
        else:
            want[i:i + 2**16] = part
    ok = bool(torch.equal(s.to(torch.int64), want)) and int(cbuf[0]) == 0
    print(f"{'ok  ' if ok else 'FAIL'} REDUCE_SUM axis {axis}")
    res.append(ok)
print(f"RESULT: {'PASS' if all(res) else 'FAIL'}")
sys.exit(0 if all(res) else 1)
