"""How much the fusion plan removes: operators, kernel launches and stored bytes per graph, and how many distinct
kernel structures the units have.  usage: python tests/fusion_stats.py name=graph.tosa ..."""
import math
import sys
from collections import Counter

from cutosa import TosaGraphIR
from cutosa.fusion import POINTWISE, plan

BYTES = {"BOOL": 1, "INT8": 1, "INT16": 2, "INT32": 4, "INT48": 8}
allsig = Counter()
for spec in sys.argv[1:]:
    name, path = spec.split("=", 1)
    g = TosaGraphIR(path)
    units, absorbed = plan(g)
    ops = [o for o in g.ops if o.op not in ("CONST", "CONST_SHAPE")]
    launch = [o for o in ops if o.op in POINTWISE or o.op in ("MATMUL", "CONV2D", "REDUCE_SUM")]
    after = [o for o in launch if o.idx not in absorbed]
    stored = lambda os: sum(math.prod(g.tensors[o.outputs[0]].shape) * BYTES[g.dtype(o.outputs[0])] for o in os)
    sizes = Counter(len(u.members) for u in units.values())
    sig = Counter()
    for u in units.values():
        s = []
        for m in u.members:
            a = m.attrs
            s.append(m.op + (f"/{int(bool(a['Scale32']))}{int(bool(a['PerChannel']))}" if m.op == "RESCALE" else "")
                     + ":" + g.dtype(m.outputs[0]))
        sig[tuple(s)] += 1
    allsig.update(sig)
    solo = Counter(o.op for o in after if o.idx not in units)
    print(f"{name}: {len(ops)} operators, kernel launches {len(launch)} -> {len(after)} "
          f"({len(launch) / len(after):.2f}x fewer), pointwise bytes stored {stored([o for o in launch if o.op in POINTWISE]) / 2**20:.0f}"
          f" -> {stored([o for o in after if o.op in POINTWISE]) / 2**20:.0f} MiB")
    print(f"   {len(units)} units, {len(sig)} structures, sizes {dict(sorted(sizes.items()))}")
    print(f"   launches left outside units: {dict(solo.most_common(8))}")
print(f"all graphs: {len(allsig)} distinct unit structures; most common:")
for s, n in allsig.most_common(6):
    print(f"   {n:5d} x {' '.join(s)}")
