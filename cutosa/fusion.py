"""Which pointwise operators run together as one generated kernel (a fusion unit). The plan depends only on
the graph, never on input data."""
import math
from collections.abc import Iterator

import numpy as np

from .int64_ref import DATA_OPERANDS, RANGE, ROUND_SINGLE, OpInfo, TosaGraphIR

POINTWISE = ("RESCALE", "ADD", "SUB", "MUL", "TABLE", "MINIMUM", "MAXIMUM", "GREATER", "GREATER_EQUAL", "SELECT")
MAX_OPS = 24  # bounds the generated kernel's size


def _const(g, name):
    t = g.tensors.get(name)
    return None if t is None or t.data is None else np.asarray(t.data)


def mul_kernel_shift(in_dt, shift):
    return max(shift, 0) if in_dt == "INT32" else 0


def table_slope_can_fail(table):
    """whether an INT16 TABLE's slope, the difference of two neighbouring entries, can leave INT16."""
    slope = np.diff(table.reshape(-1))
    return bool(((slope < RANGE["INT16"][0]) | (slope > RANGE["INT16"][1])).any())


def eligible(g, o):
    """whether cu-tosa's pointwise kernels, fused or alone, compute `o` exactly."""
    if o.op not in POINTWISE:
        return False
    out_dt = g.dtype(o.outputs[0])
    shape = g.tensors[o.outputs[0]].shape
    if math.prod(shape) == 0 or any(math.prod(g.tensors[n].shape) == 0 for n in o.inputs[:DATA_OPERANDS[o.op]]):
        return False
    if o.op == "RESCALE":
        a = o.attrs
        mult, shift = _const(g, o.inputs[1]), _const(g, o.inputs[2])
        if mult is None or shift is None or _const(g, o.inputs[3]) is None or _const(g, o.inputs[4]) is None:
            return False
        mult, shift = mult.reshape(-1).astype(np.int64), shift.reshape(-1).astype(np.int64)
        if a["InputUnsigned"] or a["OutputUnsigned"] or (a["Scale32"] and a["RoundingMode"] != ROUND_SINGLE):
            return False
        if (mult < 0).any() or (shift < 2).any() or (shift > 62).any() or mult.size != shift.size:
            return False
        if a["PerChannel"]:
            return mult.size == g.tensors[o.inputs[0]].shape[-1]
        return mult.size == 1
    if o.op in ("ADD", "SUB"):
        return out_dt in ("INT8", "INT16", "INT32")
    if o.op == "MUL":
        shift = _const(g, o.inputs[2])
        if shift is None or shift.size != 1 or out_dt not in ("INT8", "INT16", "INT32"):
            return False
        shift = int(shift.reshape(-1)[0])
        # shift > 62: a * b + 2^(shift-1) would overflow the kernels' int64 registers (|a * b| <= 2^62)
        return shift <= 62 if g.dtype(o.inputs[0]) == "INT32" else shift == 0
    if o.op == "TABLE":
        tab = _const(g, o.inputs[1])
        sig = (g.dtype(o.inputs[0]), g.dtype(o.inputs[1]), out_dt, None if tab is None else tab.size)
        if sig == ("INT8", "INT8", "INT8", 256):
            return True
        return sig == ("INT16", "INT16", "INT32", 513)
    return True


class Unit:
    """Operators computed by one generated kernel. members: in graph order; the last (the root) is the only one
    whose result is stored. reshapes: RESHAPEs between members. leaves: (member index, tensor) pairs the kernel
    reads from memory."""

    def __init__(self, root: OpInfo) -> None:
        self.members = [root]
        self.reshapes: list[OpInfo] = []
        self.leaves: list[tuple[int, str]] = []

    @property
    def root(self) -> OpInfo:
        return self.members[-1]

    @property
    def key(self) -> tuple[int, ...]:
        return tuple(m.idx for m in self.members)


def ops_with_input_dependency(g: TosaGraphIR) -> Iterator[tuple[OpInfo, bool]]:
    from_inputs = set(g.inputs)
    for op in g.ops:
        reads_inputs = any(name in from_inputs for name in op.inputs)
        if reads_inputs:
            from_inputs.update(op.outputs)
        yield op, reads_inputs


def plan(g, keep=()):
    """-> (units {root idx: Unit}, absorbed {member or RESHAPE idx: root idx}). Every eligible operator that reads
    data is in exactly one unit, alone if nothing joins it; `keep` and the graph outputs stay stored."""
    producer, consumers = {}, {}
    for o in g.ops:
        for n in o.outputs:
            producer[n] = o
        for n in o.inputs:
            consumers.setdefault(n, []).append(o)
    reads_data = {o.idx: reads for o, reads in ops_with_input_dependency(g)}
    pinned = set(keep) | set(g.outputs)
    ok = {o.idx: eligible(g, o) and reads_data[o.idx] for o in g.ops}

    def absorbable(name, total):
        """the producer of `name` if it can join the unit, and the RESHAPEs crossed to reach it."""
        chain = []
        while True:
            p = producer.get(name)
            if p is None or name in pinned or len(consumers.get(name, ())) != 1 \
                    or math.prod(g.tensors[name].shape) != total:
                return None, chain
            if p.op == "RESHAPE" and reads_data[p.idx]:
                chain.append(p)
                name = p.inputs[0]
                continue
            return (p, chain) if ok[p.idx] else (None, chain)

    units, absorbed = {}, {}
    for o in reversed(g.ops):  # roots are found last-first, so a member is never also taken as a root
        if not ok[o.idx] or o.idx in absorbed:
            continue
        u, total = Unit(o), math.prod(g.tensors[o.outputs[0]].shape)
        todo = [o]
        while todo:
            m = todo.pop()
            for n in m.inputs[:DATA_OPERANDS[m.op]]:
                p, chain = absorbable(n, total)
                if p is None or len(u.members) >= MAX_OPS:
                    continue
                u.members.append(p)
                u.reshapes += chain
                todo.append(p)
        u.members.sort(key=lambda m: m.idx)
        assert u.root is o
        inside = {m.outputs[0] for m in u.members} | {r.outputs[0] for r in u.reshapes}
        for m in u.members:
            for n in m.inputs[:DATA_OPERANDS[m.op]]:
                if n not in inside:
                    u.leaves.append((m.idx, n))
        units[o.idx] = u
        for m in u.members[:-1] + u.reshapes:
            absorbed[m.idx] = o.idx
    return units, absorbed
