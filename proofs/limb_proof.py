"""Z3 proof of the arithmetic of the fused INT16 x INT16 tensor-core kernel (mm_bench.py _bmm_i16_fused), for every
pair of INT16 codes, and the largest K for which each of its int32 accumulators cannot overflow.

Kernel: u = x + 32768, hi = (u >> 8) - 128, lo = (u & 255) - 128 (cast to int8); per k it accumulates in int32
hh += hi_x * hi_y, hl += hi_x * lo_y + lo_x * hi_y, ll += lo_x * lo_y, rs += x - 128, cs += y - 128, then writes
(hh << 16) + (hl << 8) + ll + 128 rs + 128 cs + 128 * 128 * K in int64.
Integer theory: u >= 0, so u >> 8 = u div 256 and u & 255 = u mod 256. (A 64-bit bit-vector version of the product
identity did not finish in 10 minutes: multiplier equivalence is hard for bit-blasting.)
usage: python limb_proof.py   (needs z3-solver)
"""
from z3 import And, Int, Not, Or, Solver, unsat


def limbs(name):
    x = Int(name)
    u = x + 32768
    return x, u / 256 - 128, u % 256 - 128, And(x >= -32768, x <= 32767)


def prove(name, assumptions, claim):
    s = Solver()
    s.add(assumptions, Not(claim))
    r = s.check()
    print(f"{'PROVED' if r == unsat else 'NOT PROVED: ' + str(r) + ' ' + str(s.model() if str(r) == 'sat' else '')}"
          f"  {name}", flush=True)
    return r == unsat


x, xh, xl, xr = limbs("x")
y, yh, yl, yr = limbs("y")
dom = And(xr, yr)
ok = prove("limbs fit int8: -128 <= hi, lo <= 127", xr,
           And(xh >= -128, xh <= 127, xl >= -128, xl <= 127))
ok &= prove("x == 256 hi + lo + 128", xr, x == 256 * xh + xl + 128)
# with x = 256 hx + lx + 128 (just proved) the product identity is a polynomial identity in hx, lx, hy, ly
hx, lx, hy, ly = Int("hx"), Int("lx"), Int("hy"), Int("ly")
lim = And(*[And(v >= -128, v <= 127) for v in (hx, lx, hy, ly)])
X, Y = 256 * hx + lx + 128, 256 * hy + ly + 128
ok &= prove("x*y == 65536 hh + 256 hl + ll + 128 (x-128) + 128 (y-128) + 16384 (one k)", lim,
            X * Y == 65536 * (hx * hy) + 256 * (hx * ly + lx * hy) + lx * ly + 128 * (X - 128) + 128 * (Y - 128) + 16384)
print("per-k term bounds (proved; each reached at x or y = -32768) and the largest K whose int32 accumulator cannot "
      "overflow (K * bound <= 2^31 - 1):")
for name, e, bound in (("hh term", hx * hy, 16384), ("hl term (two products)", hx * ly + lx * hy, 32768),
                       ("ll term", lx * ly, 16384), ("rs / cs term", X - 128, 32896)):
    ok &= prove(f"|{name}| <= {bound}", lim, And(e <= bound, e >= -bound))
    s = Solver()
    s.add(lim, Or(e == bound, e == -bound))
    print(f"    reached: {s.check()}   K_max = {((1 << 31) - 1) // bound}", flush=True)
print("RESULT:", "PROVED" if ok else "FAILED")
