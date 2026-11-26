"""cu-tosa (CUDA) against the TOSA reference model, one operator at a time: hand-built edge cases and operators cut
out of deployed graphs; a row passes when every output bit and the graph status agree.
usage: PYTHONPATH=. python tests/reference_op_check.py [--graphs a.tosa ...] [--per-signature 3] [--seed 0]"""
import argparse
import collections
import os
import re
import sys
import tempfile
import time

import numpy as np
import torch
import tosa_reference_model as reference_model
import tosa_serializer as ts

from cutosa import RANGE, Emulator, TosaGraphIR, graph_status

NP = {"BOOL": np.bool_, "INT8": np.int8, "INT16": np.int16, "INT32": np.int32, "INT48": np.int64, "SHAPE": np.int64}
I31 = 1 << 31
# tosa_serializer names the other attribute builders CamelCase(operator) + "Attribute"
ATTR_METHOD = {"MATMUL": "MatMulAttribute", "CONV2D": "Conv2dAttribute"}
EVERY_INSTANCE_OPS = ("GREATER", "GREATER_EQUAL", "SELECT")


class Graph:
    """A TOSA 1.0 graph under construction; inputs in declaration order with the arrays that feed them."""

    def __init__(self):
        self.g = ts.TosaSerializer("", targetMajor=1, targetMinor=0, targetPatch=0, targetDraft=False)
        self.block = self.g.currRegion.currBasicBlock
        self.feeds = []

    def inp(self, name, array, dtype):
        self.g.addInputTensor(ts.TosaSerializerTensor(name, list(array.shape), getattr(ts.DType, dtype), data=None))
        self.feeds.append(np.ascontiguousarray(array.astype(NP[dtype])))

    def const(self, name, shape, dtype, values):
        self.g.addConst(list(shape), getattr(ts.DType, dtype), np.asarray(values).astype(NP[dtype]).reshape(shape),
                        name=name)

    def tensor(self, name, shape, dtype):
        self.block.addTensor(name, list(shape), getattr(ts.DType, dtype))

    def op(self, kind, ins, outs, *attr_args):
        attr = ts.TosaSerializerAttribute()
        method = ATTR_METHOD.get(kind, "".join(w.title() for w in kind.split("_")) + "Attribute")
        getattr(attr, method)(*attr_args)
        self.g.addOperator(getattr(ts.Op, kind), ins, outs, attr)

    def out(self, name):
        self.g.addOutputTensor(self.block.tensors[name])

    def serialize(self):
        return self.g.serialize()


def _rand(rng, shape, lo, hi):
    return rng.integers(lo, hi + 1, size=shape, dtype=np.int64)


def _zp(G, name, dtype, value=0):
    G.const(name, [1], dtype, [value])


def _shift(G, name, value):
    G.const(name, [1], "INT8", [value])


PROPAGATE = ts.NanPropagationMode.PROPAGATE
SINGLE, DOUBLE = ts.RoundingMode.SINGLE_ROUND, ts.RoundingMode.DOUBLE_ROUND


def _rescale_attr(scale32, mode=SINGLE, per_channel=False):
    return scale32, mode, per_channel, False, False


def _ties(rng, shape):
    """INT32 pairs (a, b): extremes, equal pairs and +-1 / +-2 neighbours, then random."""
    a = _rand(rng, shape, -I31, I31 - 1).reshape(-1)
    a[:8] = [-I31, I31 - 1, 0, -1, 1, -I31, I31 - 1, 12345]
    b = a + _rand(rng, a.shape, -2, 2)
    b[:8] = [-I31, I31 - 1, 0, 0, 0, -I31 + 1, I31 - 2, 12345]
    b = np.clip(b, -I31, I31 - 1)
    return a.reshape(shape), b.reshape(shape)


def s_compare(kind):
    def build(rng):
        G = Graph()
        a, b = _ties(rng, (4, 64))
        G.inp("a", a, "INT32")
        G.inp("b", b, "INT32")
        G.tensor("c", [4, 64], "BOOL")
        G.op(kind, ["a", "b"], ["c"])
        G.out("c")
        return G
    return build


def s_hard_mix(dtype):
    """GREATER_EQUAL(var, threshold) -> SELECT(cond, coarse, fine), the GA-LN hard-mix pattern."""
    def build(rng):
        G = Graph()
        shape = (1, 32, 1)
        var = _rand(rng, shape, 0, 1 << 20)
        var.reshape(-1)[:3] = [5000, 4999, 5001]
        lo, hi = RANGE[dtype]
        G.inp("var", var, "INT32")
        G.const("thr", [1, 1, 1], "INT32", [5000])
        G.inp("coarse", _rand(rng, shape, lo, hi), dtype)
        G.inp("fine", _rand(rng, shape, lo, hi), dtype)
        G.tensor("cond", shape, "BOOL")
        G.tensor("y", shape, dtype)
        G.op("GREATER_EQUAL", ["var", "thr"], ["cond"])
        G.op("SELECT", ["cond", "coarse", "fine"], ["y"])
        G.out("y")
        G.out("cond")
        return G
    return build


def s_clamp_guard(rng):
    """MAXIMUM(x, lo) -> MINIMUM(., hi) on INT32, the clamp-only INT32 saturation guard."""
    G = Graph()
    x = _rand(rng, (2, 128), -I31, I31 - 1)
    x.reshape(-1)[:6] = [-I31, I31 - 1, -(1 << 24), (1 << 24) - 1, -(1 << 24) - 1, 1 << 24]
    G.inp("x", x, "INT32")
    G.const("lo", [1, 1], "INT32", [-(1 << 24)])
    G.const("hi", [1, 1], "INT32", [(1 << 24) - 1])
    G.tensor("m", [2, 128], "INT32")
    G.tensor("y", [2, 128], "INT32")
    G.op("MAXIMUM", ["x", "lo"], ["m"], PROPAGATE)
    G.op("MINIMUM", ["m", "hi"], ["y"], PROPAGATE)
    G.out("y")
    return G


def s_clamp(dtype, lo, hi):
    def build(rng):
        G = Graph()
        r = RANGE[dtype]
        G.inp("x", _rand(rng, (3, 100), *r), dtype)
        G.tensor("y", [3, 100], dtype)
        nb = 1 if dtype == "INT8" else 2
        G.op("CLAMP", ["x"], ["y"], list(int(lo).to_bytes(nb, "little", signed=True)),
             list(int(hi).to_bytes(nb, "little", signed=True)), PROPAGATE)
        G.out("y")
        return G
    return build


def s_table8(rng):
    G = Graph()
    G.inp("x", np.arange(-128, 128).reshape(16, 16), "INT8")
    G.const("t", [256], "INT8", _rand(rng, (256,), -128, 127))
    G.tensor("y", [16, 16], "INT8")
    G.op("TABLE", ["x", "t"], ["y"])
    G.out("y")
    return G


def s_table16(max_step):
    """every INT16 code through a 513-entry table whose neighbour differences are at most max_step."""
    def build(rng):
        G = Graph()
        G.inp("x", np.arange(-32768, 32768).reshape(256, 256), "INT16")
        t = np.cumsum(_rand(rng, (513,), -max_step, max_step))
        t = np.clip(t - t.mean(), -32768, 32767)
        G.const("t", [513], "INT16", t)
        G.tensor("y", [256, 256], "INT32")
        G.op("TABLE", ["x", "t"], ["y"])
        G.out("y")
        return G
    return build


def s_rescale(in_dt, out_dt, mode, shifts, per_channel=False, izp=0, ozp=0, over_range=False):
    """over_range: INT32 inputs use every code, outside apply_scale_32's range for the shifts given."""
    def build(rng):
        G = Graph()
        C = 16
        shape = (4, C)
        sh = np.array([shifts[i % len(shifts)] for i in range(C if per_channel else 1)])
        m = _rand(rng, sh.shape, 1 << 30, I31 - 1)
        bound = min(RANGE[in_dt][1], (1 << (int(sh.min()) - 1)) - 1) if in_dt == "INT32" and not over_range \
            else RANGE[in_dt][1]
        x = _rand(rng, shape, max(RANGE[in_dt][0], -bound - 1 + izp), min(RANGE[in_dt][1], bound + izp))
        G.inp("x", x, in_dt)
        G.const("m", list(sh.shape), "INT32", m)
        G.const("s", list(sh.shape), "INT8", sh)
        _zp(G, "izp", in_dt, izp)
        _zp(G, "ozp", out_dt, ozp)
        G.tensor("y", list(shape), out_dt)
        G.op("RESCALE", ["x", "m", "s", "izp", "ozp"], ["y"], *_rescale_attr(True, mode, per_channel))
        G.out("y")
        return G
    return build


def _exact_bound(K, other_max):
    """largest operand magnitude b with K * b * other_max < 2^31 (other_max None: both operands bounded by b)."""
    if other_max is None:
        return int(np.sqrt((I31 - 1) / K))
    return int((I31 - 1) // (K * other_max))


def s_matmul(dtype, K, M=8, N=24, azp=0, bzp=0, exact=False, out_shift=None):
    """INT16: the reference model returns no INT48 output to numpy, so a RESCALE (multiplier 2^14) reads it: exact=True
    bounds the operands so every sum fits INT32 and the RESCALE is the identity (shift 14, every bit compared);
    otherwise operands use every code and shift 31 keeps the top bits. out_shift overrides the shift."""
    def build(rng):
        G = Graph()
        r = RANGE[dtype]
        if exact:
            b = _exact_bound(K, None)
            r = (-b, b)
        G.inp("a", _rand(rng, (2, M, K), *r), dtype)
        G.inp("b", _rand(rng, (2, K, N), *r), dtype)
        _zp(G, "azp", dtype, azp)
        _zp(G, "bzp", dtype, bzp)
        G.tensor("y", [2, M, N], "INT48" if dtype == "INT16" else "INT32")
        G.op("MATMUL", ["a", "b", "azp", "bzp"], ["y"])
        if dtype == "INT16":
            s = out_shift if out_shift is not None else (14 if exact else 31)
            G.const("m", [1], "INT16", [1 << 14])
            _shift(G, "s", s)
            _zp(G, "izp", "INT48")
            _zp(G, "ozp", "INT32")
            G.tensor("z", [2, M, N], "INT32")
            G.op("RESCALE", ["y", "m", "s", "izp", "ozp"], ["z"], *_rescale_attr(False))
            G.out("z")
        else:
            G.out("y")
        return G
    return build


def s_conv(in_dt, x_shape, k_hw, oc, pad, stride, izp=0):
    def build(rng):
        G = Graph()
        N, H, W, C = x_shape
        G.inp("x", _rand(rng, x_shape, *RANGE[in_dt]), in_dt)
        G.const("w", [oc, *k_hw, C], "INT8", _rand(rng, (oc, *k_hw, C), -128, 127))
        acc = "INT48" if in_dt == "INT16" else "INT32"
        G.const("bias", [oc], acc, _rand(rng, (oc,), -(1 << 20), 1 << 20))
        _zp(G, "izp", in_dt, izp)
        _zp(G, "wzp", "INT8")
        OH = (H + pad[0] + pad[1] - k_hw[0]) // stride[0] + 1
        OW = (W + pad[2] + pad[3] - k_hw[1]) // stride[1] + 1
        G.tensor("y", [N, OH, OW, oc], acc)
        G.op("CONV2D", ["x", "w", "bias", "izp", "wzp"], ["y"], list(pad), list(stride), [1, 1], False,
             getattr(ts.DType, acc))
        if acc == "INT48":
            G.const("m", [1], "INT16", [1 << 14])
            _shift(G, "s", 24)
            _zp(G, "izp2", "INT48")
            _zp(G, "ozp", "INT32")
            G.tensor("z", [N, OH, OW, oc], "INT32")
            G.op("RESCALE", ["y", "m", "s", "izp2", "ozp"], ["z"], *_rescale_attr(False))
            G.out("z")
        else:
            G.out("y")
        return G
    return build


def s_gather(rng):
    G = Graph()
    K, C, W = 300, 24, 17
    G.const("v", [1, K, C], "INT16", _rand(rng, (1, K, C), -32768, 32767))
    idx = _rand(rng, (1, W), 0, K - 1)
    idx.reshape(-1)[:2] = [0, K - 1]
    G.inp("i", idx, "INT32")
    G.tensor("y", [1, W, C], "INT16")
    G.op("GATHER", ["v", "i"], ["y"])
    G.out("y")
    return G


def s_reduce(rng):
    G = Graph()
    G.inp("x", _rand(rng, (1, 7, 384), -(1 << 15), 1 << 15), "INT32")
    G.inp("h", _rand(rng, (1, 7, 384), -32768, 32767), "INT16")
    G.tensor("s", [1, 7, 1], "INT32")
    G.tensor("m", [1, 1, 384], "INT16")
    G.op("REDUCE_SUM", ["x"], ["s"], 2)
    G.op("REDUCE_MAX", ["h"], ["m"], 1, PROPAGATE)
    G.out("s")
    G.out("m")
    return G


def s_shape_ops(rng):
    """RESHAPE -> 5-D TRANSPOSE -> SLICE of q (begin 0) and k -> CONCAT, INT8 (the qkv split pattern)."""
    G = Graph()
    B, N, H, d = 1, 10, 3, 8
    G.inp("x", _rand(rng, (B, N, 3 * H * d), -128, 127), "INT8")
    G.const("s5", [5], "SHAPE", [B, N, 3, H, d])
    G.tensor("r", [B, N, 3, H, d], "INT8")
    G.op("RESHAPE", ["x", "s5"], ["r"])
    G.tensor("t", [3, B, H, N, d], "INT8")
    G.op("TRANSPOSE", ["r"], ["t"], [2, 0, 3, 1, 4])
    for i, name in enumerate(("q", "k")):
        G.const(f"b{i}", [5], "SHAPE", [i, 0, 0, 0, 0])
        G.const(f"z{i}", [5], "SHAPE", [1, B, H, N, d])
        G.tensor(name, [1, B, H, N, d], "INT8")
        G.op("SLICE", ["t", f"b{i}", f"z{i}"], [name])
    G.tensor("qk", [2, B, H, N, d], "INT8")
    G.op("CONCAT", ["q", "k"], ["qk"], 0)
    G.out("qk")
    return G


SYNTHETIC = {
    "GREATER INT32 ties": s_compare("GREATER"),
    "GREATER_EQUAL INT32 ties": s_compare("GREATER_EQUAL"),
    "GE -> SELECT INT16": s_hard_mix("INT16"),
    "GE -> SELECT INT8": s_hard_mix("INT8"),
    "GE -> SELECT INT32": s_hard_mix("INT32"),
    "MAXIMUM -> MINIMUM INT32": s_clamp_guard,
    "CLAMP INT8": s_clamp("INT8", -100, 90),
    "CLAMP INT16": s_clamp("INT16", -20000, 30000),
    "TABLE INT8 all codes": s_table8,
    "TABLE INT16 all codes": s_table16(200),
    "TABLE INT16 step > INT16": s_table16(40000),
    "RESCALE INT32->INT16 single": s_rescale("INT32", "INT16", SINGLE, [20, 31, 38]),
    "RESCALE INT32->INT16 double": s_rescale("INT32", "INT16", DOUBLE, [32, 38, 45]),
    "RESCALE INT32->INT8 per channel": s_rescale("INT32", "INT8", SINGLE, [18, 25, 33, 40], per_channel=True, ozp=-7),
    "RESCALE INT8->INT16 izp": s_rescale("INT8", "INT16", SINGLE, [20], izp=13),
    "RESCALE INT16->INT32": s_rescale("INT16", "INT32", SINGLE, [17, 30]),
    "RESCALE INT32 input outside apply_scale_32 (reference aborts)": s_rescale("INT32", "INT16", SINGLE, [20],
                                                                                over_range=True),
    "MATMUL INT16 K=64 every bit": s_matmul("INT16", 64, exact=True),
    "MATMUL INT16 K=1500 every bit": s_matmul("INT16", 1500, exact=True),
    "MATMUL INT16 K=64 every code (top bits)": s_matmul("INT16", 64),
    "MATMUL INT16 K=1500 every code (top bits)": s_matmul("INT16", 1500),
    "RESCALE INT48 result outside INT32 (reference aborts)": s_matmul("INT16", 1500, out_shift=14),
    "MATMUL INT8 K=200 zp": s_matmul("INT8", 200, azp=-5, bzp=3),
    "CONV2D INT8 1x1 M=40000": s_conv("INT8", (1, 1, 40000, 16), (1, 1), 8, (0, 0, 0, 0), (1, 1), izp=-3),
    "CONV2D INT8 1x3 stride 2": s_conv("INT8", (1, 1, 101, 24), (1, 3), 16, (0, 0, 1, 1), (1, 2)),
    "CONV2D INT16 3x3 pad stride 2": s_conv("INT16", (1, 9, 11, 8), (3, 3), 12, (1, 1, 1, 1), (2, 2)),
    "GATHER INT16": s_gather,
    "REDUCE_SUM / REDUCE_MAX": s_reduce,
    "RESHAPE TRANSPOSE SLICE CONCAT": s_shape_ops,
}


def _attr_args(o):
    a = o.attrs
    k = o.op
    if k == "RESCALE":
        return (bool(a["Scale32"]), ts.RoundingMode(a["RoundingMode"]), bool(a["PerChannel"]), bool(a["InputUnsigned"]),
                bool(a["OutputUnsigned"]))
    if k == "CONV2D":
        return list(a["Pad"]), list(a["Stride"]), list(a["Dilation"]), bool(a["LocalBound"]), ts.DType(a["AccType"])
    if k == "CLAMP":
        return list(a["MinVal"]), list(a["MaxVal"]), ts.NanPropagationMode(a["NanMode"])
    if k in ("MAXIMUM", "MINIMUM"):
        return (ts.NanPropagationMode(a["NanMode"]),)
    if k == "REDUCE_MAX":
        return a["Axis"], ts.NanPropagationMode(a["NanMode"])
    if k in ("REDUCE_SUM", "CONCAT"):
        return (a["Axis"],)
    if k == "TRANSPOSE":
        return (list(a["Perms"]),)
    return ()


def signature(g, o):
    consts = tuple(g.dtype(n) + ("c" if n in g.shapes or g.tensors[n].data is not None else "") for n in o.inputs)
    extra = ()
    if o.op == "RESCALE":
        extra = tuple(o.attrs[k] for k in ("Scale32", "RoundingMode", "PerChannel", "InputUnsigned", "OutputUnsigned"))
    elif o.op == "CONV2D":
        extra = (tuple(o.attrs["Stride"]), tuple(o.attrs["Dilation"]), any(o.attrs["Pad"]))
    return o.op, consts, tuple(g.dtype(n) for n in o.outputs), extra


def pick(g, per_sig, rng):
    groups = collections.defaultdict(list)
    for o in g.ops:
        if o.op not in ("CONST", "CONST_SHAPE"):
            groups[signature(g, o)].append(o)
    chosen = []
    for sig, ops in groups.items():
        if sig[0] in EVERY_INSTANCE_OPS or len(ops) <= per_sig:
            chosen += [(sig, o) for o in ops]
        else:
            rest = rng.choice(len(ops) - 1, per_sig - 1, replace=False) + 1
            chosen += [(sig, ops[0])] + [(sig, ops[i]) for i in sorted(rest)]
    return groups, chosen


def _input_range(g, o, pos, dtype, draw):
    """(lo, hi) of a random non-constant input: narrow keeps INT32 math in range, full uses every code."""
    lo, hi = RANGE[dtype]
    if o.op == "GATHER" and pos == 1:
        return 0, g.tensors[o.inputs[0]].shape[1] - 1
    if draw == "narrow" and o.op in ("MATMUL", "CONV2D") and g.dtype(o.outputs[0]) == "INT48":
        # every INT48 sum within INT32, so the identity RESCALE that reads it compares every bit
        w = g.tensors[o.inputs[1]]
        if o.op == "MATMUL":
            b = _exact_bound(g.tensors[o.inputs[0]].shape[-1], None if w.data is None else int(np.abs(w.data).max()))
        else:
            bias = int(np.abs(g.tensors[o.inputs[2]].data).max())
            K = int(np.prod(w.shape[1:]))
            b = max(1, (I31 - 1 - bias) // (K * max(1, int(np.abs(w.data).max()))))
        return max(lo, -b), min(hi, b)
    if draw == "full" or dtype != "INT32":
        return lo, hi
    if o.op == "RESCALE" and pos == 0:
        sh = g.tensors[o.inputs[2]].data
        izp = int(g.tensors[o.inputs[3]].data.reshape(-1)[0])
        b = (1 << (int(sh.min()) - 1)) - 1
        return max(lo, izp - b - 1), min(hi, izp + b)
    return -(1 << 15), 1 << 15


def instance_graph(g, o, rng, draw):
    G = Graph()
    done = set()
    feed_index_by_name = {}
    for pos, n in enumerate(o.inputs):
        if n in done:
            continue
        done.add(n)
        if n in g.shapes:
            G.const(n, [len(g.shapes[n])], "SHAPE", g.shapes[n])
            continue
        t = g.tensors[n]
        if t.data is not None:
            G.const(n, t.shape, t.dtype, t.data)
        elif t.dtype == "INT48":  # the reference model takes no INT48 input from numpy: make it from INT16 codes
            K = 64
            G.inp(n + "__a", _rand(rng, (*t.shape[:-1], K), -32768, 32767), "INT16")
            _zp(G, n + "__azp", "INT16")
            G.tensor(n, t.shape, "INT48")
            if len(t.shape) == 4:
                oc = t.shape[-1]
                G.const(n + "__w", [oc, 1, 1, K], "INT8", _rand(rng, (oc, 1, 1, K), -128, 127))
                G.const(n + "__b", [oc], "INT48", np.zeros(oc))
                _zp(G, n + "__wzp", "INT8")
                G.op("CONV2D", [n + "__a", n + "__w", n + "__b", n + "__azp", n + "__wzp"], [n], [0, 0, 0, 0], [1, 1],
                     [1, 1], False, ts.DType.INT48)
            elif len(t.shape) == 3:
                G.const(n + "__w", [t.shape[0], K, t.shape[2]], "INT16",
                        _rand(rng, (t.shape[0], K, t.shape[2]), -32768, 32767))
                _zp(G, n + "__wzp", "INT16")
                G.op("MATMUL", [n + "__a", n + "__w", n + "__azp", n + "__wzp"], [n])
            else:
                raise NotImplementedError(f"INT48 input of rank {len(t.shape)}")
        elif t.dtype == "BOOL":  # nor a BOOL one: make it with a GREATER_EQUAL of two INT32 inputs
            a = _rand(rng, t.shape, -(1 << 15), 1 << 15)
            G.inp(n + "__a", a, "INT32")
            G.inp(n + "__b", a + _rand(rng, t.shape, -2, 2), "INT32")
            G.tensor(n, t.shape, "BOOL")
            G.op("GREATER_EQUAL", [n + "__a", n + "__b"], [n])
        else:
            x = _rand(rng, t.shape, *_input_range(g, o, pos, t.dtype, draw))
            first = feed_index_by_name.get(o.inputs[0])
            if o.op in ("GREATER", "GREATER_EQUAL") and pos == 1 and first is not None:
                a = G.feeds[first]
                if tuple(t.shape) == a.shape:  # GREATER and GREATER_EQUAL differ only on ties
                    x = np.clip(a.astype(np.int64) + _rand(rng, t.shape, -2, 2), *RANGE[t.dtype])
                else:  # a broadcast threshold (the GA-LN hard mix: [1, T, 1] against [1, 1, 1]): split the first
                    # input at its median and make every eighth element a tie
                    thr = int(np.median(a))
                    x = np.full(t.shape, thr)
                    a.reshape(-1)[::8] = thr
            feed_index_by_name[n] = len(G.feeds)
            G.inp(n, x, t.dtype)
    for n in o.outputs:
        G.tensor(n, g.tensors[n].shape, g.tensors[n].dtype)
    G.op(o.op, list(o.inputs), list(o.outputs), *_attr_args(o))
    for n in o.outputs:
        if g.tensors[n].dtype == "INT48":  # no INT48 output to numpy: rescale to INT32 inside the graph
            # narrow: identity (every bit; the inputs keep the sums within INT32); full: the top bits (shift 31)
            G.const(n + "__m", [1], "INT16", [1 << 14])
            _shift(G, n + "__s", 14 if draw == "narrow" else 31)
            _zp(G, n + "__izp", "INT48")
            _zp(G, n + "__ozp", "INT32")
            G.tensor(n + "__o", g.tensors[n].shape, "INT32")
            G.op("RESCALE", [n, n + "__m", n + "__s", n + "__izp", n + "__ozp"], [n + "__o"], *_rescale_attr(False))
            G.out(n + "__o")
        else:
            G.out(n)
    return G


# An operator whose eval() fails stops the reference model ("OpRescale failure: apply_scale_16() error: ..."); a
# per-element REQUIRE such as an ADD overflow ("REQUIRE() fails ... applyAdd()") wraps and evaluation goes on. Both
# end with "Graph result: UNPREDICTABLE." and "An error occurred when running the model.", so only the first marks
# a stop.
REF_ABORT = re.compile(r"\bOp\w+ failure\b")


def reference(binary, feeds):
    """reference model outputs, status, the number of its stderr lines and whether it stopped evaluating. Some REQUIRE
    failures (RESCALE: apply_scale_32 input or apply_scale_16 result out of range) stop the evaluation: the status is
    UNPREDICTABLE and the outputs are uninitialised memory (they change from run to run), so they are not compared."""
    with tempfile.TemporaryFile(mode="w+") as f:
        saved = os.dup(2)
        os.dup2(f.fileno(), 2)
        try:
            outs, st = reference_model.run(binary, feeds, verbosity="NONE", initialize_variable_tensor_from_numpy=True)
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        f.seek(0)
        lines = f.read().splitlines()
    return outs, st.name.removeprefix("TOSA_"), len(lines), any(REF_ABORT.search(ln) for ln in lines)


def check(name, G):
    """one reference run and one cu-tosa run -> a row."""
    binary = G.serialize()
    try:
        ref_outs, ref_status, n_msgs, aborted = reference(binary, G.feeds)
    except Exception as e:  # noqa: BLE001  the row reports it
        return {"name": name, "pass": False, "why": f"reference: {type(e).__name__}: {e}"}
    try:
        emu = Emulator(TosaGraphIR(binary), device="cuda")
        outs = emu.run(G.feeds)
    except Exception as e:  # noqa: BLE001  the row reports it
        return {"name": name, "pass": False, "why": f"emulator: {type(e).__name__}: {e}"}
    status = graph_status(emu.violations)
    same = len(outs) == len(ref_outs) and all(
        r.dtype == e.dtype and r.shape == e.shape and np.array_equal(r, e) for r, e in zip(ref_outs, outs, strict=True))
    row = {"name": name, "pass": (same or aborted) and status == ref_status, "status": ref_status,
           "emu_status": status, "bits": "n/a" if aborted else "equal" if same else "DIFFER", "ref_msgs": n_msgs,
           "checks": sorted({v.msg for v in emu.violations})}
    if aborted:
        row["why"] = "reference stopped evaluating: status compared, outputs not"
    elif not same:
        diffs = [(int((r.astype(np.int64) != e.astype(np.int64)).sum()), r.size) if r.shape == e.shape
                 else (f"{r.dtype}{r.shape} vs {e.dtype}{e.shape}") for r, e in zip(ref_outs, outs, strict=False)]
        row["why"] = f"differing elements per output {diffs}"
    if any(o.dtype == np.bool_ for o in ref_outs):
        row["true_frac"] = [round(float(o.mean()), 3) for o in ref_outs if o.dtype == np.bool_]
    return row


def show(row):
    extra = f" true={row['true_frac']}" if "true_frac" in row else ""
    st = row.get("status", "-")
    if row.get("emu_status", st) != st:
        st = f"{st}/{row['emu_status']}"
    print(f"{'PASS' if row['pass'] else 'FAIL'}  {row['name'][:78]:78s} {st:13s} "
          f"{row.get('bits', '-'):6s}{extra}  {row.get('why', '')}"
          f"{'  checks=' + str(row['checks']) if row.get('checks') else ''}", flush=True)


def main(args):
    if not torch.cuda.is_available():
        sys.exit("needs a CUDA device")
    rng = np.random.default_rng(args.seed)
    rows = []
    t0 = time.time()
    print(f"== synthetic ({len(SYNTHETIC)} graphs)")
    for name, build in SYNTHETIC.items():
        r = check(name, build(rng))
        show(r)
        rows.append(r)
    for path in args.graphs:
        g = TosaGraphIR(path)
        groups, chosen = pick(g, args.per_signature, rng)
        n_ops = sum(len(v) for v in groups.values())
        print(f"== {path}: {n_ops} operators, {len(groups)} signatures, {len(chosen)} instances x 2 draws")
        for sig, o in chosen:
            for draw in ("narrow", "full"):
                label = f"#{o.idx} {o.op} {','.join(sig[1])}->{','.join(sig[2])} {draw}"
                try:
                    G = instance_graph(g, o, rng, draw)
                except NotImplementedError as e:
                    r = {"name": label, "pass": False, "why": f"not built: {e}"}
                else:
                    r = check(label, G)
                r["graph"] = path
                show(r)
                rows.append(r)
    n_pass = sum(r["pass"] for r in rows)
    print(f"RESULT: {'PASS' if n_pass == len(rows) else 'FAIL'} ({n_pass}/{len(rows)} rows, "
          f"{time.time() - t0:.0f} s)")
    return 0 if n_pass == len(rows) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--graphs", nargs="*", default=[])
    ap.add_argument("--per-signature", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    sys.exit(main(ap.parse_args()))
