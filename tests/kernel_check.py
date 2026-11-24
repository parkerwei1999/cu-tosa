"""Each kernel and fusion unit against the int64 emulator's operators on the same tensors: declared input range,
a range that breaks the REQUIREs and small values, as 1 and several samples.
usage: python tests/kernel_check.py name=graph.tosa ..."""
import sys
from collections import defaultdict

import torch

from cutosa import RANGE, STORE, Emulator, Int64Emulator, TosaGraphIR
from cutosa.kernels.matmul import bmm_i16


def begin(ref, cu, samples):
    """what run() sets before the operators: empty REQUIRE records, the sample count, an operator that reads data."""
    ref.violations, ref._pending, ref.samples, ref._from_inputs = [], [], samples, True
    cu._reset_require_counts(samples)
    cu._from_inputs = True


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
                begin(ref, cu, samples)
                want = ref.op_RESCALE(o, x, None, None, None, None)[0]
                # a wide draw does not fit the stored input type: feed the kernel int64 storage for it
                got = cu.k_RESCALE(o, x.to(STORE[in_dt]) if fits else x, None, None, None, None)
                if got is None:
                    rows.append((name, o.idx, sig, label, "not covered by the kernel", True))
                    continue
                wc = [(m, c.tolist()) for _, _, m, c in ref._pending]
                gc = [(m, c) for _, _, m, c in cu._require_count_records()]
                ok = bool(torch.equal(got[0].to(torch.int64), want)) and wc == gc and got[0].dtype == STORE[sig[1]]
                nbad = sum(sum(c) if isinstance(c, list) else c for _, c in wc)
                rows.append((name, o.idx, sig, f"{label}, {x.numel()} el, {nbad} REQUIRE failures", ok))
    return rows


REPLAY = ("ADD", "SUB", "MUL", "TABLE", "REDUCE_SUM", "CONV2D")
DATA = {"MUL": 2, "TABLE": 1, "CONV2D": 3}  # the operands that carry data; default all


def replay_rows(name, path, per_sig, gen):
    """real ADD / SUB / MUL / TABLE / REDUCE_SUM operators: kernel vs int64 operator, inputs over three ranges."""
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
            x4 = len(shapes) == 1 and (len(var) == len(data) or o.op == "CONV2D") and len(next(iter(shapes))) > 0 \
                and o.op != "TABLE" and not (o.op == "CONV2D" and all(d == 1 for d in next(iter(shapes))[:3]))
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
                begin(ref, cu, samples)
                want = getattr(ref, "op_" + o.op)(o, *wide)[0]
                got = getattr(cu, "k_" + o.op)(o, *narrow)
                if got is None:
                    rows.append((name, o.idx, sig, f"{draw}: not covered by the kernel", True))
                    continue
                wc = [(m, c.tolist()) for _, _, m, c in ref._pending]
                gc = [(m, c) for _, _, m, c in cu._require_count_records()]
                wc = [(m, c) for m, c in wc if (any(c) if isinstance(c, list) else c)]  # run() drops zero counts
                gc = [(m, c) for m, c in gc if (any(c) if isinstance(c, list) else c)]
                ok = bool(torch.equal(got[0].to(torch.int64), want)) and wc == gc \
                    and got[0].dtype == STORE[g.dtype(o.outputs[0])]
                nbad = sum(sum(c) if isinstance(c, list) else c for _, c in wc)
                rows.append((name, o.idx, sig, f"{draw}, {want.numel()} el, {nbad} REQUIRE failures", ok))
    return rows


def unit_rows(name, path, per_struct, gen):
    """real fusion units, one per kernel structure: generated kernel vs the int64 operators member by member."""
    from cutosa.fusion import plan
    g = TosaGraphIR(path)
    ref, cu = Int64Emulator(g, device="cuda"), Emulator(g, device="cuda")
    units, _ = plan(g)
    groups = defaultdict(list)
    for u in units.values():
        groups[cu._unit_prep(u)["prog"]].append(u)
    rows = []
    for prog, us in sorted(groups.items(), key=str):
        label = " ".join(k for k, _, _ in prog[1])
        for u in us[:: max(1, len(us) // per_struct)][:per_struct]:
            shapes = [g.tensors[m.outputs[0]].shape for m in u.members]
            lead = [next((d for d in s_ if d != 1), 1) for s_ in shapes]
            many = next((k for k in (4, 2, 3, 5) if all(d % k == 0 for d in lead)), None)
            for draw in ("full", "3/4", "small") + (("3/4 samples",) if many else ()):
                samples = many if draw.endswith("samples") else 1
                wide, narrow = dict(ref.consts), dict(cu.consts)
                for _, n in u.leaves:
                    if n in wide:
                        continue
                    lo, hi = RANGE[g.dtype(n)] if g.dtype(n) != "BOOL" else (0, 1)
                    if draw.startswith("3/4"):
                        lo, hi = 3 * lo // 4, 3 * hi // 4
                    elif draw == "small":
                        lo, hi = max(lo, -1000), min(hi, 1000)
                    x = torch.randint(lo, hi + 1, g.tensors[n].shape, generator=gen, dtype=torch.int64)
                    if x.numel() > 1:
                        x.view(-1)[0], x.view(-1)[-1] = lo, hi
                    wide[n], narrow[n] = x.cuda(), x.to(STORE[g.dtype(n)]).cuda()
                begin(ref, cu, samples)
                for m in sorted(u.members + u.reshapes, key=lambda m: m.idx):
                    wide[m.outputs[0]] = getattr(ref, "op_" + m.op)(m, *[wide.get(n) for n in m.inputs])[0]
                want = wide[u.root.outputs[0]]
                got = cu._run_unit(u, narrow)
                if got is None:
                    rows.append((name, u.root.idx, label, f"{draw}: not run as a unit", True))
                    continue
                wc = [(i, m_, c.tolist()) for i, _, m_, c in ref._pending]
                gc = [(i, m_, c) for i, _, m_, c in cu._require_count_records()]
                nz = lambda rs: sorted((i, m_, c) for i, m_, c in rs if (any(c) if isinstance(c, list) else c))
                ok = bool(torch.equal(got.to(torch.int64), want)) and nz(wc) == nz(gc) \
                    and got.dtype == STORE[g.dtype(u.root.outputs[0])]
                nbad = sum(sum(c) if isinstance(c, list) else c for _, _, c in wc)
                rows.append((name, u.root.idx, label, f"{draw}{f' x{samples}' if samples > 1 else ''}, "
                             f"{want.numel()} el, {nbad} REQUIRE failures in {len(nz(wc))} records", ok))
    return rows


def linear_rows(gen):
    """1x1 CONV2D on synthetic operands vs an exact int64 product: zero points, a bias that pushes past the type."""
    from cutosa.kernels.conv import linear
    rows = []
    for dt, out_dt, M, K, N, izp, wzp, big in (
            ("INT8", "INT32", 8, 1, 3, 0, 0, 0), ("INT8", "INT32", 196, 192, 768, -128, 0, 0),
            ("INT8", "INT32", 200, 77, 65, 5, -3, 2**31 - 100000), ("INT8", "INT32", 64, 1000, 130, 127, 127, -2**31),
            ("INT16", "INT48", 8, 1, 3, 0, 0, 0), ("INT16", "INT48", 1500, 384, 1536, 0, 0, 0),
            ("INT16", "INT48", 200, 77, 65, 0, 2, 2**47 - 10**8), ("INT16", "INT48", 64, 3000, 130, 0, -7, -2**47)):
        lo, hi = RANGE[dt]
        x = torch.randint(lo, hi + 1, (M, K), generator=gen, dtype=torch.int64)
        x.view(-1)[0], x.view(-1)[-1] = lo, hi
        x[M // 2] = hi  # a row and a column at the extremes
        w = torch.randint(-128, 128, (N, K), generator=gen, dtype=torch.int64)
        w[N // 2] = -128
        bias = torch.randint(-1000, 1000, (N,), generator=gen, dtype=torch.int64) + big
        acc = (x - izp) @ (w - wzp).t() + bias
        olo, ohi = RANGE[out_dt]
        bad = (acc < olo) | (acc > ohi)
        want = acc.clamp(olo, ohi) if out_dt == "INT48" else ((acc + 2**31) % 2**32) - 2**31
        for parts in (1, 4):
            cbuf = torch.zeros(8, dtype=torch.int64, device="cuda")
            got = linear(x.to(STORE[dt]).cuda(), w.to(torch.int8).cuda(), bias.cuda(), w.sum(1).cuda(), STORE[out_dt],
                         izp, wzp, olo, ohi, out_dt == "INT48", cbuf, 2, parts)
            ok = bool(torch.equal(got.cpu().to(torch.int64), want)) and got.dtype == STORE[out_dt] \
                and cbuf.cpu().tolist() == [0, 0] + bad.reshape(parts, -1).sum(1).tolist() + [0] * (6 - parts)
            rows.append(("synthetic", "-", ("CONV2D 1x1", dt, (M, K, N), f"zp {izp} {wzp}"),
                         f"{parts} parts, {int(bad.sum())} REQUIRE failures", ok))
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
    rows = matmul_rows(gen) + linear_rows(gen)
    for spec in specs:
        name, path = spec.split("=", 1)
        rows += rescale_rows(name, path, 3, gen)
        rows += replay_rows(name, path, 3, gen)
        rows += unit_rows(name, path, 3, gen)
    bad = [r for r in rows if not r[-1]]
    for r in rows:
        print("ok  " if r[-1] else "FAIL", *r[:-1], flush=True)
    print(f"RESULT: {'PASS' if not bad else 'FAIL'} ({len(rows) - len(bad)}/{len(rows)} rows)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
