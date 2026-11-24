"""Which pointwise operators run together as one generated kernel (a fusion unit). The plan depends only on
the graph, never on input data."""
import math

import numpy as np

from .int64_ref import ROUND_SINGLE

POINTWISE = ("RESCALE", "ADD", "SUB", "MUL", "TABLE", "MINIMUM", "MAXIMUM", "GREATER", "GREATER_EQUAL", "SELECT")
DATA_IN = {"RESCALE": 1, "ADD": 2, "SUB": 2, "MUL": 2, "TABLE": 1, "MINIMUM": 2, "MAXIMUM": 2, "GREATER": 2,
           "GREATER_EQUAL": 2, "SELECT": 3}
MAX_OPS = 24  # bounds the generated kernel's size


def _const(g, name):
    t = g.tensors.get(name)
    return None if t is None or t.data is None else np.asarray(t.data)


def eligible(g, o):
    """whether the fused kernel computes `o` exactly."""
    if o.op not in POINTWISE:
        return False
    out_dt = g.dtype(o.outputs[0])
    shape = g.tensors[o.outputs[0]].shape
    if math.prod(shape) == 0 or any(math.prod(g.tensors[n].shape) == 0 for n in o.inputs[:DATA_IN[o.op]]):
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
        return shift <= 62 if g.dtype(o.inputs[0]) == "INT32" else shift == 0
    if o.op == "TABLE":
        tab = _const(g, o.inputs[1])
        sig = (g.dtype(o.inputs[0]), g.dtype(o.inputs[1]), out_dt, None if tab is None else tab.size)
        if sig == ("INT8", "INT8", "INT8", 256):
            return True
        return sig == ("INT16", "INT16", "INT32", 513)  # the fused kernel counts a slope outside INT16 itself
    return True


class Unit:
    """Operators computed by one generated kernel. members: in graph order; the last (the root) is the only one
    whose result is stored. reshapes: RESHAPEs between members. leaves: (member index, tensor) pairs the kernel
    reads from memory. key: the member indices."""

    def __init__(self, root):
        self.root, self.members, self.reshapes, self.leaves, self.key = root, [root], [], [], None


def plan(g, keep=()):
    """-> (units {root idx: Unit}, absorbed {member or RESHAPE idx: root idx}). Every eligible operator that reads
    data is in exactly one unit, alone if nothing joins it; `keep` and the graph outputs stay stored."""
    producer, consumers = {}, {}
    for o in g.ops:
        for n in o.outputs:
            producer[n] = o
        for n in o.inputs:
            consumers.setdefault(n, []).append(o)
    from_inputs = set(g.inputs)
    reads_data = {}
    for o in g.ops:
        reads_data[o.idx] = any(n in from_inputs for n in o.inputs)
        if reads_data[o.idx]:
            from_inputs.update(o.outputs)
    pinned = set(keep) | set(g.outputs)
    ok = {o.idx: eligible(g, o) and reads_data[o.idx] for o in g.ops}
    numel = lambda n: math.prod(g.tensors[n].shape)

    def absorbable(name, total):
        """the producer of `name` if it can join the unit, and the RESHAPEs crossed to reach it."""
        chain = []
        while True:
            p = producer.get(name)
            if p is None or name in pinned or len(consumers.get(name, ())) != 1 or numel(name) != total:
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
        u, total = Unit(o), numel(o.outputs[0])
        todo = [o]
        while todo:
            m = todo.pop()
            for n in m.inputs[:DATA_IN[m.op]]:
                p, chain = absorbable(n, total)
                if p is None or len(u.members) >= MAX_OPS:
                    continue
                u.members.append(p)
                u.reshapes += chain
                todo.append(p)
        u.members.sort(key=lambda m: m.idx)
        u.key = ("unit", *[m.idx for m in u.members])
        inside = {m.outputs[0] for m in u.members} | {r.outputs[0] for r in u.reshapes}
        for m in u.members:
            for n in m.inputs[:DATA_IN[m.op]]:
                if n not in inside:
                    u.leaves.append((m.idx, n))
        units[o.idx] = u
        for m in u.members[:-1] + u.reshapes:
            absorbed[m.idx] = o.idx
    return units, absorbed
