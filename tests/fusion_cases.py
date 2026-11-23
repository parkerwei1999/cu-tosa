"""Fusion on graphs the real ones do not contain: small synthetic TOSA graphs, cutosa.Emulator with fusion against
cutosa.Int64Emulator and against cutosa.Emulator(fuse=False).

Fixed cases (an operand read twice by one operator, RESHAPE inside a unit, broadcast leaves, per-channel RESCALE
before a RESHAPE, wrapped results feeding the next member, a TABLE slope outside INT16, members without a REQUIRE,
one Emulator object run with several `keep` sets) and random graphs of pointwise operators. Every graph runs at
several sample counts with the REQUIRE check on and off; outputs, violation lists and kept tensors must be equal, or
both emulators must raise the same exception type.
usage: python tests/fusion_cases.py [--random 30] [--seed 0]
"""
import argparse
import sys

import numpy as np
import torch
import tosa_serializer as ts

from cutosa import RANGE, Emulator, Int64Emulator, TosaGraphIR

NP = {"BOOL": np.bool_, "INT8": np.int8, "INT16": np.int16, "INT32": np.int32, "INT48": np.int64, "SHAPE": np.int64}
SINGLE = ts.RoundingMode.SINGLE_ROUND
PROPAGATE = ts.NanPropagationMode.PROPAGATE


class Builder:
    """a TOSA graph under construction; tensors are handled by name, shapes and types are in self.info."""

    def __init__(self):
        self.g = ts.TosaSerializer("", targetMajor=1, targetMinor=0, targetPatch=0, targetDraft=False)
        self.block = self.g.currRegion.currBasicBlock
        self.info, self.feeds, self.k, self.made = {}, [], 0, []

    def _decl(self, shape, dt):
        self.k += 1
        n = f"t{self.k}"
        self.block.addTensor(n, list(shape), getattr(ts.DType, dt))
        self.info[n] = (tuple(shape), dt)
        self.made.append(n)
        return n

    def _op(self, kind, ins, outs, *attr_args):
        attr = ts.TosaSerializerAttribute()
        getattr(attr, "".join(w.title() for w in kind.split("_")) + "Attribute")(*attr_args)
        self.g.addOperator(getattr(ts.Op, kind), ins, outs, attr)

    def inp(self, array, dt):
        self.k += 1
        n = f"in{self.k}"
        array = np.asarray(array)
        self.g.addInputTensor(ts.TosaSerializerTensor(n, list(array.shape), getattr(ts.DType, dt), data=None))
        self.feeds.append(array.astype(np.int64))
        self.info[n] = (tuple(array.shape), dt)
        return n

    def const(self, values, dt, shape=None):
        self.k += 1
        n = f"c{self.k}"
        values = np.asarray(values)
        shape = tuple(values.shape) if shape is None else tuple(shape)
        self.g.addConst(list(shape), getattr(ts.DType, dt), values.astype(NP[dt]).reshape(shape), name=n)
        self.info[n] = (shape, dt)
        return n

    def shape(self, n):
        return self.info[n][0]

    def dt(self, n):
        return self.info[n][1]

    def rescale(self, x, out_dt, mult, shift, izp=0, ozp=0, scale32=True, per_channel=False):
        mult, shift = np.atleast_1d(np.asarray(mult, np.int64)), np.atleast_1d(np.asarray(shift, np.int64))
        ins = [x, self.const(mult, "INT32" if scale32 else "INT16"), self.const(shift, "INT8"),
               self.const([izp], self.dt(x)), self.const([ozp], out_dt)]
        y = self._decl(self.shape(x), out_dt)
        self._op("RESCALE", ins, [y], scale32, SINGLE, per_channel, False, False)
        return y

    def bin(self, kind, a, b, out_dt=None):
        shape = tuple(np.broadcast_shapes(self.shape(a), self.shape(b)))
        y = self._decl(shape, "BOOL" if kind.startswith("GREATER") else out_dt or self.dt(a))
        self._op(kind, [a, b], [y], *((PROPAGATE,) if kind in ("MINIMUM", "MAXIMUM") else ()))
        return y

    def mul(self, a, b, shift=0):
        y = self._decl(tuple(np.broadcast_shapes(self.shape(a), self.shape(b))), "INT32")
        self._op("MUL", [a, b, self.const([shift], "INT8")], [y])
        return y

    def table(self, x, values):
        wide = self.dt(x) == "INT16"
        y = self._decl(self.shape(x), "INT32" if wide else "INT8")
        self._op("TABLE", [x, self.const(np.asarray(values), "INT16" if wide else "INT8")], [y])
        return y

    def select(self, c, a, b):
        y = self._decl(tuple(np.broadcast_shapes(self.shape(c), self.shape(a), self.shape(b))), self.dt(a))
        self._op("SELECT", [c, a, b], [y])
        return y

    def reshape(self, x, shape):
        y = self._decl(shape, self.dt(x))
        self._op("RESHAPE", [x, self.const(np.asarray(shape, np.int64), "SHAPE")], [y])
        return y

    def reduce_sum(self, x, axis):
        y = self._decl(tuple(1 if i == axis else d for i, d in enumerate(self.shape(x))), self.dt(x))
        self._op("REDUCE_SUM", [x], [y], axis)
        return y

    def out(self, *names):
        for n in names:
            self.g.addOutputTensor(self.block.tensors[n])

    def serialize(self):
        return bytes(self.g.serialize())


def draw(rng, shape, dt, kind="full"):
    lo, hi = (0, 1) if dt == "BOOL" else RANGE[dt]
    if kind == "small":
        lo, hi = max(lo, -40), min(hi, 40)
    x = rng.integers(lo, hi + 1, shape, dtype=np.int64)
    if kind == "full" and x.size > 1:
        x.reshape(-1)[0], x.reshape(-1)[-1] = lo, hi
    return x


def run(emu, feeds, **kw):
    """-> ("ok", outputs, violations, kept tensors as int64 numpy) or ("raise", exception type name)."""
    try:
        outs = emu.run([f.copy() for f in feeds], **kw)
    except Exception as e:  # noqa: BLE001 - the exception type is the result
        return ("raise", type(e).__name__, str(e)[:120])
    return ("ok", outs, list(emu.violations), {n: t.to(torch.int64).cpu().numpy() for n, t in emu.env.items()})


def differ(a, b):
    """the first difference between two run() results, or None."""
    if a[0] != b[0]:
        return f"{a[0]} {a[1] if a[0] == 'raise' else ''} | {b[0]} {b[1:] if b[0] == 'raise' else ''}"
    if a[0] == "raise":
        return None if a[1] == b[1] else f"raises {a[1]} | {b[1]}"
    for i, (x, y) in enumerate(zip(a[1], b[1])):
        if x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y):
            return f"output {i} differs"
    if a[2] != b[2]:
        return f"violations {a[2][:2]} | {b[2][:2]}"
    if a[3].keys() != b[3].keys() or any(not np.array_equal(a[3][n], b[3][n]) for n in a[3]):
        return "kept tensors differ"
    return None


def check_graph(label, bd, samples=(1, 2, 3), keeps=((),), min_fused=1):
    """every (samples, check) on fresh objects, then the `keeps` sequence on one object. -> (ok, text)."""
    buf, problems, fused, viol = bd.serialize(), [], 0, 0
    for check in (True, False):
        for s in samples:
            ref = run(Int64Emulator(TosaGraphIR(buf), device="cuda", check=check), bd.feeds, samples=s)
            for fuse in (True, False):
                cu = Emulator(TosaGraphIR(buf), device="cuda", check=check, fuse=fuse)
                d = differ(ref, run(cu, bd.feeds, samples=s))
                if d:
                    problems.append(f"samples {s} check {check} fuse {fuse}: {d}")
                if fuse and ref[0] == "ok":
                    fused = max(fused, sum(v == "fused" for v in cu.how.values()))
            viol += len(ref[2]) if ref[0] == "ok" else 0
    one, ref = Emulator(TosaGraphIR(buf), device="cuda"), Int64Emulator(TosaGraphIR(buf), device="cuda")
    for keep in keeps:
        d = differ(run(ref, bd.feeds, keep=keep), run(one, bd.feeds, keep=keep))
        if d:
            problems.append(f"one object, keep {keep}: {d}")
    if fused < min_fused:
        problems.append(f"only {fused} operators ran fused")
    return not problems, f"{label}: {fused} operators fused, {viol} violation records" + "".join(
        f"\n      {p}" for p in problems[:4])


# -- fixed cases -------------------------------------------------------------------------------------------------------
def case_keep_sets(rng):
    b = Builder()
    x = b.inp(draw(rng, (2, 3), "INT32", "small"), "INT32")
    a = b.rescale(x, "INT32", 1 << 3, 4)  # |x| >= 8 fails apply_scale_32
    s = b.bin("ADD", a, b.const(np.arange(6).reshape(2, 3), "INT32"))
    y = b.rescale(s, "INT8", 1 << 30, 30)
    z = b.bin("ADD", x, b.const(np.arange(6).reshape(2, 3), "INT32"))  # a later reader of x
    b.out(y, z)
    return b, dict(keeps=((), (a,), (), (s,), (a, s), (a,), ()))


def case_no_require(rng):
    b = Builder()
    x = b.inp(draw(rng, (3, 4), "INT8"), "INT8")
    t = b.table(x, (np.arange(256) - 128)[::-1])
    b.out(b.bin("MAXIMUM", t, b.const(np.arange(12).reshape(3, 4) * 9 - 50, "INT8")))
    return b, dict(samples=(1, 2, 3, 5))


def case_operand_twice(rng):
    b = Builder()
    x = b.inp(draw(rng, (2, 6), "INT16"), "INT16")
    r = b.rescale(x, "INT16", 1 << 30, 31, ozp=7)
    sq = b.mul(r, r)
    c = b.bin("GREATER", sq, b.const([[2**20]], "INT32"))
    k = b.rescale(sq, "INT32", 3 << 28, 30)
    b.out(b.select(c, k, k), b.bin("SUB", sq, sq))
    return b, {}


def case_reshape_inside(rng):
    b = Builder()
    x = b.inp(draw(rng, (6, 2), "INT32"), "INT32")
    a = b.bin("ADD", x, b.const(draw(rng, (6, 2), "INT32"), "INT32"))
    r = b.reshape(b.reshape(a, (3, 4)), (2, 6))
    m = b.bin("SUB", r, b.const(draw(rng, (1, 6), "INT32", "small"), "INT32"))
    b.out(b.rescale(b.reshape(m, (12,)), "INT16", 1 << 29, 40))
    return b, dict(samples=(1, 2, 3, 6, 4))


def case_broadcast(rng):
    b = Builder()
    x = b.inp(draw(rng, (2, 1, 4), "INT8"), "INT8")
    y = b.inp(draw(rng, (1, 3, 1), "INT8"), "INT8")
    a = b.bin("ADD", x, y)  # both operands expand
    m = b.mul(a, b.const(draw(rng, (3, 4), "INT8"), "INT8"))  # (3, 4) read inside (2, 3, 4)
    s = b.bin("ADD", m, b.const([-5], "INT32"))  # one element
    b.out(b.bin("MINIMUM", s, b.const(draw(rng, (2, 1, 1), "INT32"), "INT32")))
    return b, {}


def case_per_channel(rng):
    b = Builder()
    x = b.inp(draw(rng, (2, 3, 4), "INT32"), "INT32")
    a = b.bin("ADD", x, b.const(draw(rng, (2, 3, 4), "INT32", "small"), "INT32"))
    r = b.rescale(a, "INT16", [1 << 30, 3 << 28, 1 << 20, 5], [34, 31, 22, 2], per_channel=True)
    t = b.reshape(r, (2, 4, 3))  # the channel of the RESCALE is not the last dimension of the root
    p = b.rescale(t, "INT8", [1 << 14, 3 << 12, 9], [16, 15, 4], ozp=-3, scale32=False, per_channel=True)
    b.out(b.bin("MAXIMUM", p, b.const([[[-100]]], "INT8")))
    return b, {}


def case_wrap(rng):
    b = Builder()
    x = b.inp(draw(rng, (4, 5), "INT8"), "INT8")
    a = b.bin("ADD", x, b.const(draw(rng, (4, 5), "INT8"), "INT8"))  # wraps in INT8
    d = b.bin("SUB", a, b.const(draw(rng, (1, 5), "INT8"), "INT8"))
    m = b.mul(d, a)
    w = b.inp(draw(rng, (4, 5), "INT32"), "INT32")
    q = b.mul(m, w, shift=3)  # wraps in INT32
    b.out(b.rescale(b.bin("ADD", q, w), "INT16", (1 << 31) - 1, 45), b.mul(w, w, shift=31))
    return b, dict(samples=(1, 2, 4))


def case_table_slope(rng):
    b = Builder()
    v = draw(rng, (2, 8), "INT32")
    v[:, 2:6] = (np.array([[100, 101, 102, 103], [101, 99, 102, 101]]) - 256) * 128 * 65536 + 5 * 65536  # those cells
    x = b.inp(v, "INT32")
    r = b.rescale(x, "INT16", 1 << 30, 46)
    tab = np.cumsum(rng.integers(-300, 300, 513))
    tab[100:104] = [-32768, 32767, -32768, 30000]  # slopes outside INT16
    t = b.table(r, tab.clip(-32768, 32767))
    b.out(b.rescale(t, "INT16", 1 << 30, 37, ozp=11))
    return b, dict(samples=(1, 2, 4))


def case_between(rng):
    """a unit whose members sit around an operator that runs on its own (record order)."""
    b = Builder()
    x = b.inp(draw(rng, (4, 6), "INT32", "small"), "INT32")
    a = b.rescale(x, "INT32", 1 << 3, 4)
    s = b.reduce_sum(x, 1)
    c = b.bin("ADD", a, b.const(draw(rng, (4, 6), "INT32"), "INT32"))
    d = b.bin("ADD", s, b.const(draw(rng, (4, 1), "INT32"), "INT32"))
    b.out(b.rescale(c, "INT8", 1 << 30, 30), d)
    return b, dict(samples=(1, 2, 4, 3))


def broadcasts_to(small, big):
    try:
        return np.broadcast_shapes(small, big) == tuple(big)
    except ValueError:
        return False


# -- random graphs ---------------------------------------------------------------------------------------------------
def random_graph(rng):
    b = Builder()
    base = [(2, 3, 4), (4, 6), (2, 2, 2, 3), (6, 4), (12,)][rng.integers(5)]
    pool = [b.inp(draw(rng, base, dt, rng.choice(["full", "small"])), dt)
            for dt in rng.choice(["INT8", "INT16", "INT32"], rng.integers(1, 3))]

    def other(x, same_shape_only=False):
        """a second operand of x's type: a tensor of the pool, or a constant that broadcasts to x."""
        fit = [n for n in pool if b.dt(n) == b.dt(x) and (b.shape(n) == b.shape(x) or (
            not same_shape_only and broadcasts_to(b.shape(n), b.shape(x))))]
        if fit and rng.random() < 0.6:
            return fit[rng.integers(len(fit))]
        sh = b.shape(x)
        sub = [sh, sh[1:], (1,) * (len(sh) - 1) + sh[-1:], (1,), sh[:1] + (1,) * (len(sh) - 1)][rng.integers(5)]
        return b.const(draw(rng, sub or (1,), b.dt(x), rng.choice(["full", "small"])), b.dt(x))

    for _ in range(int(rng.integers(4, 14))):
        x = pool[rng.integers(len(pool))]
        dt, kind = b.dt(x), rng.choice(["RESCALE", "ADD", "SUB", "MUL", "TABLE", "MINMAX", "SELECT", "RESHAPE"])
        if dt == "BOOL":
            num = [n for n in pool if b.dt(n) != "BOOL" and b.shape(n) == b.shape(x)]
            if not num:
                continue
            v = num[rng.integers(len(num))]
            y = b.select(x, v, other(v))
        elif kind == "RESCALE":
            out_dt = rng.choice(["INT8", "INT16", "INT32"])
            scale32 = bool(rng.integers(2))
            per_channel = bool(rng.random() < 0.25)
            n = b.shape(x)[-1] if per_channel else 1
            mult = rng.integers(0, 2**31 if scale32 else 2**15, n)
            shift = rng.integers(2, 63, n)
            zp = lambda t: int(rng.integers(-5, 6)) if t != "INT32" else 0
            y = b.rescale(x, out_dt, mult, shift, zp(dt), zp(out_dt), scale32, per_channel)
        elif kind in ("ADD", "SUB"):
            y = b.bin(kind, x, other(x))
        elif kind == "MUL":
            y = b.mul(x, other(x), int(rng.integers(0, 32)) if dt == "INT32" and rng.random() < 0.7 else 0)
        elif kind == "TABLE" and dt == "INT8":
            y = b.table(x, draw(rng, (256,), "INT8"))
        elif kind == "TABLE" and dt == "INT16":
            y = b.table(x, draw(rng, (513,), "INT16") if rng.random() < 0.3
                        else np.cumsum(rng.integers(-60, 61, 513)))
        elif kind == "MINMAX":
            y = b.bin(rng.choice(["MINIMUM", "MAXIMUM"]), x, other(x))
        elif kind == "SELECT":
            c = b.bin(rng.choice(["GREATER", "GREATER_EQUAL"]), x, other(x))
            pool.append(c)
            y = b.select(c, x, other(x, same_shape_only=True))
        elif kind == "RESHAPE":
            n = int(np.prod(b.shape(x)))
            y = b.reshape(x, [s for s in [(2, n // 2), (n,), (n // 4, 4), (2, 2, n // 4), (1, n)] if np.prod(s) == n][
                rng.integers(5) % 4])
        else:
            continue
        pool.append(y)
    made = [n for n in b.made if b.dt(n) != "BOOL"]
    outs = list(dict.fromkeys([made[-1]] + [made[i] for i in rng.integers(0, len(made), 1)]))
    b.out(*outs)
    inner = [n for n in b.made if n not in outs]
    keeps = ((),) + tuple(tuple(rng.choice(inner, min(len(inner), 2), replace=False)) for _ in range(2)) + ((),) \
        if inner else ((),)
    return b, dict(samples=(1, 2), keeps=keeps, min_fused=0)


def main(args):
    rng = np.random.default_rng(args.seed)
    results = []
    for case in (case_keep_sets, case_no_require, case_operand_twice, case_reshape_inside, case_broadcast,
                 case_per_channel, case_wrap, case_table_slope, case_between):
        bd, kw = case(rng)
        results.append(check_graph(case.__name__.removeprefix("case_"), bd, **kw))
        print(("ok   " if results[-1][0] else "FAIL ") + results[-1][1], flush=True)
    bad = 0
    for i in range(args.random):
        bd, kw = random_graph(np.random.default_rng([args.seed, i]))
        ok, text = check_graph(f"random {i}", bd, **kw)
        bad += not ok
        if not ok:
            print("FAIL " + text, flush=True)
    print(f"{'ok  ' if not bad else 'FAIL'} {args.random - bad}/{args.random} random graphs equal")
    passed = all(ok for ok, _ in results) and not bad
    print("RESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    sys.exit(main(ap.parse_args()))
