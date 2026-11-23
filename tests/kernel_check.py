"""The kernels one at a time against the int64 emulator's operator on the same tensors (CUDA).

RESCALE: every real RESCALE operator of the given graphs that the kernel covers, grouped by signature (types, scale32,
per channel), with inputs over the operator's declared input range and over a range that breaks its REQUIREs, as one
sample and as 4 samples; outputs and both REQUIRE counts must be equal.
MATMUL: INT16 x INT16 shapes against an exact int64 matmul on the CPU.
usage: python tests/kernel_check.py name=graph.tosa ...
"""
import sys
from collections import defaultdict

import torch

from cutosa import RANGE, STORE, Emulator, Int64Emulator, TosaGraphIR
from cutosa.kernels.matmul import bmm_i16


def rescale_rows(name, path, per_sig, gen):
    g = TosaGraphIR(path)
    ref, cu = Int64Emulator(g, device="cuda"), Emulator(g, device="cuda")
    groups = defaultdict(list)
    for o in g.ops:
        if o.op == "RESCALE":
            a = o.attrs
            groups[(g.dtype(o.inputs[0]), g.dtype(o.outputs[0]), bool(a["Scale32"]), bool(a["PerChannel"]))].append(o)
    rows = []
    for sig, ops in sorted(groups.items()):
        for o in ops[:: max(1, len(ops) // per_sig)][:per_sig]:
            in_dt = sig[0]
            shape = g.tensors[o.inputs[0]].shape
            if not shape or int(torch.tensor(shape).prod()) == 0:
                continue
            lead = next((i for i, d in enumerate(shape) if d != 1), 0)
            shape4 = tuple(4 * d if i == lead else d for i, d in enumerate(shape))  # 4 samples on the outermost dim
            if sig[3] and lead == len(shape) - 1:
                shape4 = (4,) + tuple(shape)
            for label, (lo, hi), shp, samples in (("declared", RANGE[in_dt], shape, 1),
                                                  ("wide", RANGE["INT48" if in_dt == "INT48" else "INT32"], shape, 1),
                                                  ("wide x4", RANGE["INT48" if in_dt == "INT48" else "INT32"], shape4, 4)):
                x = torch.randint(lo, hi + 1, shp, generator=gen, dtype=torch.int64)
                x.view(-1)[0], x.view(-1)[-1] = lo, hi
                x = x.cuda()
                fits = RANGE[in_dt][0] <= lo and hi <= RANGE[in_dt][1]
                for e in (ref, cu):
                    e.violations, e._pending, e.samples, e._from_inputs = [], [], samples, True
                want = ref.op_RESCALE(o, x, None, None, None, None)[0]
                # a wide draw does not fit the stored input type: feed the kernel int64 storage for it
                got = cu.k_RESCALE(o, x.to(STORE[in_dt]) if fits else x, None, None, None, None)
                if got is None:
                    rows.append((name, o.idx, sig, label, "not covered by the kernel", True))
                    continue
                wc = [(m, c.tolist()) for _, _, m, c in ref._pending]
                gc = [(m, c.tolist()) for _, _, m, c in cu._pending]
                ok = bool(torch.equal(got[0].to(torch.int64), want)) and wc == gc and got[0].dtype == STORE[sig[1]]
                nbad = sum(sum(c) if isinstance(c, list) else c for _, c in wc)
                rows.append((name, o.idx, sig, f"{label}, {x.numel()} el, {nbad} REQUIRE failures", ok))
    return rows


REPLAY = ("ADD", "SUB", "MUL", "TABLE", "REDUCE_SUM")
DATA = {"MUL": 2, "TABLE": 1}  # the operands that carry data; default all


def replay_rows(name, path, per_sig, gen):
    """real ADD / SUB / MUL / TABLE / REDUCE_SUM operators: constants as they are, the other inputs drawn over the
    whole declared range, over 3/4 of it (some results overflow) and over small values (none do); the kernel against
    the int64 emulator's operator on the widened tensors. Operators whose inputs all have one shape also run as 4
    samples."""
    g = TosaGraphIR(path)
    ref, cu = Int64Emulator(g, device="cuda"), Emulator(g, device="cuda")
    groups = defaultdict(list)
    for o in g.ops:
        if o.op in REPLAY:
            data = o.inputs[:DATA.get(o.op, len(o.inputs))]
            sig = (o.op, tuple(g.dtype(n) for n in data), tuple("const" if g.tensors[n].data is not None else
                   "x".join(str(int(d != 1)) for d in g.tensors[n].shape) for n in data),
                   o.attrs.get("Axis"), ref._scalar(o.inputs[2]) > 0 if o.op == "MUL" else None)
            groups[sig].append(o)
    rows = []
    for sig, ops in sorted(groups.items(), key=str):
        for o in ops[:: max(1, len(ops) // per_sig)][:per_sig]:
            data = o.inputs[:DATA.get(o.op, len(o.inputs))]
            var = [n for n in data if g.tensors[n].data is None]
            shapes = {g.tensors[n].shape for n in var}
            x4 = len(shapes) == 1 and len(var) == len(data) and len(next(iter(shapes))) > 0 and o.op != "TABLE"
            for draw in ("full", "3/4", "small") + (("3/4 x4",) if x4 else ()):
                samples, wide, narrow = (4 if draw.endswith("x4") else 1), [], []
                for n in o.inputs:
                    if n in cu.consts or n not in data:
                        wide.append(ref.consts.get(n))
                        narrow.append(cu.consts.get(n))
                        continue
                    lo, hi = RANGE[g.dtype(n)]
                    if draw.startswith("3/4"):
                        lo, hi = 3 * lo // 4, 3 * hi // 4
                    elif draw == "small":
                        lo, hi = max(lo, -1000), min(hi, 1000)
                    shape = g.tensors[n].shape
                    if samples > 1:
                        lead = next((i for i, d in enumerate(shape) if d != 1), 0)
                        shape = tuple(samples * d if i == lead else d for i, d in enumerate(shape))
                    x = torch.randint(lo, hi + 1, shape, generator=gen, dtype=torch.int64)
                    if x.numel() > 1:
                        x.view(-1)[0], x.view(-1)[-1] = lo, hi
                    wide.append(x.cuda())
                    narrow.append(x.to(STORE[g.dtype(n)]).cuda())
                for e in (ref, cu):
                    e.violations, e._pending, e.samples, e._from_inputs = [], [], samples, True
                want = getattr(ref, "op_" + o.op)(o, *wide)[0]
                got = getattr(cu, "k_" + o.op)(o, *narrow)
                if got is None:
                    rows.append((name, o.idx, sig, f"{draw}: not covered by the kernel", True))
                    continue
                wc = [(m, c.tolist()) for _, _, m, c in ref._pending]
                gc = [(m, c.tolist()) for _, _, m, c in cu._pending]
                wc = [(m, c) for m, c in wc if (any(c) if isinstance(c, list) else c)]  # run() drops zero counts
                gc = [(m, c) for m, c in gc if (any(c) if isinstance(c, list) else c)]
                ok = bool(torch.equal(got[0].to(torch.int64), want)) and wc == gc \
                    and got[0].dtype == STORE[g.dtype(o.outputs[0])]
                nbad = sum(sum(c) if isinstance(c, list) else c for _, c in wc)
                rows.append((name, o.idx, sig, f"{draw}, {want.numel()} el, {nbad} REQUIRE failures", ok))
    return rows


def matmul_rows(gen):
    rows = []
    for (N, M, K, P) in ((16, 1500, 1500, 64), (16, 228, 64, 1500), (3, 7, 1, 5), (1, 65, 129, 33), (2, 64, 4097, 64)):
        a = torch.randint(-32768, 32768, (N, M, K), generator=gen, dtype=torch.int64)
        b = torch.randint(-32768, 32768, (N, K, P), generator=gen, dtype=torch.int64)
        a[0, 0], b[0, :, 0] = -32768, -32768
        want = torch.matmul(a, b)
        a16, b16 = a.to(torch.int16).cuda(), b.to(torch.int16).cuda()
        ok = bool(torch.equal(bmm_i16(a16, b16).cpu(), want))
        # strided operands: the kernel is given transposed views, as the lowering's TRANSPOSE leaves them
        at = a16.transpose(1, 2).contiguous().transpose(1, 2)
        bt = b16.transpose(1, 2).contiguous().transpose(1, 2)
        ok2 = bool(torch.equal(bmm_i16(at, bt).cpu(), want))
        rows.append(("matmul", 0, (N, M, K, P), "contiguous", ok))
        rows.append(("matmul", 0, (N, M, K, P), "strided views", ok2))
    return rows


def main(specs):
    gen = torch.Generator().manual_seed(0)
    rows = matmul_rows(gen)
    for spec in specs:
        name, path = spec.split("=", 1)
        rows += rescale_rows(name, path, 3, gen)
        rows += replay_rows(name, path, 3, gen)
    bad = [r for r in rows if not r[-1]]
    for r in rows:
        print("ok  " if r[-1] else "FAIL", *r[:-1], flush=True)
    print(f"RESULT: {'PASS' if not bad else 'FAIL'} ({len(rows) - len(bad)}/{len(rows)} rows)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
