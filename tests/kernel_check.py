"""Each kernel and fusion unit against the int64 emulator's operators on the same tensors: declared input range,
a range that breaks the REQUIREs and small values, as 1 and several samples.
usage: python tests/kernel_check.py name=graph.tosa ..."""
import sys
from collections import defaultdict

import torch

from cutosa import RANGE, STORE, Emulator, Int64Emulator, TosaGraphIR
from cutosa.int64_ref import DATA_OPERANDS
from cutosa.kernels.matmul import MAX_K_INT8, bmm


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
                want = ref.op_RESCALE(o, x)[0]
                # a wide draw does not fit the stored input type: feed the kernel int64 storage for it
                got = cu.k_RESCALE(o, x.to(STORE[in_dt]) if fits else x)
                if got is None:
                    rows.append((name, o.idx, sig, label, "not covered by the kernel", True))
                    continue
                wc = [(msg, t.tolist()) for _, _, msg, t in ref._pending]
                gc = [(v.msg, v.failed) for v in cu._require_count_records()]
                ok = bool(torch.equal(got[0].to(torch.int64), want)) and wc == gc and got[0].dtype == STORE[sig[1]]
                nbad = sum(sum(c) if isinstance(c, list) else c for _, c in wc)
                rows.append((name, o.idx, sig, f"{label}, {x.numel()} el, {nbad} REQUIRE failures", ok))
    return rows


REPLAY = ("ADD", "SUB", "MUL", "TABLE", "REDUCE_SUM", "CONV2D", "MATMUL")


def replay_rows(name, path, per_sig, gen):
    """real ADD / SUB / MUL / TABLE / REDUCE_SUM / CONV2D / MATMUL operators: kernel vs int64 operator, inputs over
    three ranges."""
    g = TosaGraphIR(path)
    ref, cu = Int64Emulator(g, device="cuda"), Emulator(g, device="cuda")
    groups = defaultdict(list)
    for o in g.ops:
        if o.op in REPLAY:
            data = o.inputs[:DATA_OPERANDS[o.op]]
            sig = (o.op, tuple(g.dtype(n) for n in data), tuple("const" if g.tensors[n].data is not None else
                   "x".join(str(int(d != 1)) for d in g.tensors[n].shape) for n in data),
                   o.attrs.get("Axis"), ref._scalar(o.inputs[2]) > 0 if o.op == "MUL" else None)
            groups[sig].append(o)
    rows = []
    for sig, ops in sorted(groups.items(), key=str):
        for o in ops[:: max(1, len(ops) // per_sig)][:per_sig]:
            data = o.inputs[:DATA_OPERANDS[o.op]]
            var = [n for n in data if g.tensors[n].data is None]
            shapes = {g.tensors[n].shape for n in var}
            x4 = len(shapes) == 1 and len(var) == len(data) and len(next(iter(shapes))) > 0 \
                and o.op != "TABLE" and not (o.op == "CONV2D" and all(d == 1 for d in next(iter(shapes))[:3]))
            for draw in ("full", "3/4", "small") + (("3/4 x4",) if x4 else ()):
                samples, wide, narrow = (4 if draw.endswith("x4") else 1), [], []
                for n in data:
                    if n in cu.consts:
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
                wc = [(msg, t.tolist()) for _, _, msg, t in ref._pending]
                gc = [(v.msg, v.failed) for v in cu._require_count_records()]
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
        groups[cu._compile_unit(u).prog].append(u)
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
                    data = [wide[n] for n in m.inputs[:DATA_OPERANDS[m.op]]]
                    wide[m.outputs[0]] = getattr(ref, "op_" + m.op)(m, *data)[0]
                want = wide[u.root.outputs[0]]
                got = cu._run_unit(u, narrow)
                if got is None:
                    rows.append((name, u.root.idx, label, f"{draw}: not run as a unit", True))
                    continue
                wc = [(op.idx, msg, t.tolist()) for op, _, msg, t in ref._pending]
                gc = [(v.op_idx, v.msg, v.failed) for v in cu._require_count_records()]
                nz = lambda rs: sorted((i, m_, c) for i, m_, c in rs if (any(c) if isinstance(c, list) else c))
                ok = bool(torch.equal(got.to(torch.int64), want)) and nz(wc) == nz(gc) \
                    and got.dtype == STORE[g.dtype(u.root.outputs[0])]
                nbad = sum(sum(c) if isinstance(c, list) else c for _, _, c in wc)
                rows.append((name, u.root.idx, label, f"{draw}{f' x{samples}' if samples > 1 else ''}, "
                             f"{want.numel()} el, {nbad} REQUIRE failures in {len(nz(wc))} records", ok))
    return rows


def conv_int64(x, w, bias, geo, izp, wzp):
    """x [N, IH, IW, IC], w [OC, KH, KW, IC] int64 -> the exact accumulator [N, OH, OW, OC], one tap at a time."""
    (kh, kw), (sy, sx), (pt, pb, pl, pr), (dy, dx) = geo
    n, ih, iw, ic = x.shape
    oh, ow = geo.out_hw(ih, iw)
    xp = torch.full((n, ih + pt + pb, iw + pl + pr, ic), izp, dtype=torch.int64)
    xp[:, pt:pt + ih, pl:pl + iw] = x
    acc = bias.expand(n, oh, ow, w.shape[0]).clone()
    for ky in range(kh):
        for kx in range(kw):
            tap = xp[:, ky * dy: ky * dy + (oh - 1) * sy + 1: sy, kx * dx: kx * dx + (ow - 1) * sx + 1: sx]
            acc += (tap - izp) @ (w[:, ky, kx] - wzp).t()
    return acc


def conv_rows(gen):
    """CONV2D on synthetic operands vs an exact int64 convolution: 1x1 and gathered taps (padding, stride, dilation),
    zero points, a bias that pushes past the type."""
    from cutosa.kernels.conv import Geometry, conv2d
    one = Geometry((1, 1), (1, 1), (0, 0, 0, 0), (1, 1))
    rows = []
    for dt, out_dt, xshape, oc, geo, izp, wzp, big in (
            ("INT8", "INT32", (1, 1, 8, 1), 3, one, 0, 0, 0), ("INT8", "INT32", (1, 1, 196, 192), 768, one, -128, 0, 0),
            ("INT8", "INT32", (1, 1, 200, 77), 65, one, 5, -3, 2**31 - 100000),
            ("INT8", "INT32", (1, 1, 64, 1000), 130, one, 127, 127, -2**31),
            ("INT16", "INT48", (1, 1, 8, 1), 3, one, 0, 0, 0),
            ("INT16", "INT48", (1, 1, 1500, 384), 1536, one, 0, 0, 0),
            ("INT16", "INT48", (1, 1, 200, 77), 65, one, 0, 2, 2**47 - 10**8),
            ("INT16", "INT48", (1, 1, 64, 3000), 130, one, 0, -7, -2**47),
            ("INT8", "INT32", (2, 9, 11, 8), 12, Geometry((3, 3), (2, 2), (1, 1, 1, 1), (1, 1)), -3, 2, 0),
            ("INT8", "INT32", (1, 32, 64, 3), 20, Geometry((16, 16), (16, 16), (0, 0, 0, 0), (1, 1)), 5, 0, 0),
            ("INT8", "INT32", (1, 1, 103, 24), 16, Geometry((1, 3), (1, 2), (0, 0, 1, 1), (1, 1)), -128, 127,
             2**31 - 1000),
            ("INT8", "INT32", (1, 11, 13, 5), 7, Geometry((3, 3), (1, 2), (2, 1, 0, 3), (2, 2)), 127, -128, -2**31),
            ("INT8", "INT32", (4, 30, 7, 33), 70, Geometry((3, 1), (1, 1), (1, 1, 0, 0), (1, 1)), -1, 1, 0),
            ("INT16", "INT48", (2, 9, 11, 8), 12, Geometry((3, 3), (2, 2), (1, 1, 1, 1), (1, 1)), 0, -7, 2**47 - 10**8),
            ("INT16", "INT48", (1, 1, 300, 384), 70, Geometry((1, 3), (1, 1), (0, 0, 1, 1), (1, 1)), 0, 0, 0)):
        lo, hi = RANGE[dt]
        x = torch.randint(lo, hi + 1, xshape, generator=gen, dtype=torch.int64)
        x.view(-1)[0], x.view(-1)[-1] = lo, hi
        x[..., xshape[2] // 2, :] = hi  # a column of pixels and a weight row at the extremes
        w = torch.randint(-128, 128, (oc, *geo.kernel, xshape[3]), generator=gen, dtype=torch.int64)
        w[oc // 2] = -128
        bias = torch.randint(-1000, 1000, (oc,), generator=gen, dtype=torch.int64) + big
        acc = conv_int64(x, w, bias, geo, izp, wzp).reshape(-1, oc)
        olo, ohi = RANGE[out_dt]
        bad = (acc < olo) | (acc > ohi)
        want = acc.clamp(olo, ohi) if out_dt == "INT48" else ((acc + 2**31) % 2**32) - 2**31
        w2 = w.reshape(oc, -1)
        for parts in (1, 4) if acc.shape[0] % 4 == 0 else (1,):
            cbuf = torch.zeros(8, dtype=torch.int64, device="cuda")
            got = conv2d(x.to(STORE[dt]).cuda(), w2.to(torch.int8).cuda(), bias.cuda(), w2.sum(1).cuda(), geo,
                         STORE[out_dt], izp, wzp, olo, ohi, out_dt == "INT48", cbuf, 2, parts)
            ok = bool(torch.equal(got.cpu().to(torch.int64), want)) and got.dtype == STORE[out_dt] \
                and cbuf.cpu().tolist() == [0, 0] + bad.reshape(parts, -1).sum(1).tolist() + [0] * (6 - parts)
            label = "1x1" if geo.is_linear() else f"{geo.kernel} stride {geo.stride} pad {geo.pad} dil {geo.dilation}"
            rows.append(("synthetic", "-", (f"CONV2D {label}", dt, xshape, oc, f"zp {izp} {wzp}"),
                         f"{parts} parts, {int(bad.sum())} REQUIRE failures", ok))
    return rows


def matmul_rows(gen):
    """MATMUL vs an exact int64 product, contiguous and as transposed views (as the lowering's TRANSPOSE leaves them).
    INT8 at K = MAX_K_INT8 with every product at +-255^2 reaches the INT32 bound."""
    rows = []
    cases = [("INT16", shape, 0, 0) for shape in ((16, 1500, 1500, 64), (16, 228, 64, 1500), (3, 7, 1, 5),
                                                   (1, 65, 129, 33), (2, 64, 4097, 64))]
    cases += [("INT8", (3, 197, 64, 197), -5, 3), ("INT8", (192, 49, 32, 49), 0, 0), ("INT8", (3, 7, 1, 5), 0, -1),
              ("INT8", (1, 65, 129, 33), 127, -128), ("INT8", (2, 8, MAX_K_INT8, 8), 127, -128)]
    for dt, (N, M, K, P), azp, bzp in cases:
        lo, hi = RANGE[dt]
        a = torch.randint(lo, hi + 1, (N, M, K), generator=gen, dtype=torch.int64)
        b = torch.randint(lo, hi + 1, (N, K, P), generator=gen, dtype=torch.int64)
        a[0, 0], b[0, :, 0] = lo, lo
        a[0, 1], b[0, :, 1] = lo, hi
        want = torch.matmul(a - azp, b - bzp)
        sa, sb = a.to(STORE[dt]).cuda(), b.to(STORE[dt]).cuda()
        at = sa.transpose(1, 2).contiguous().transpose(1, 2)
        bt = sb.transpose(1, 2).contiguous().transpose(1, 2)
        for label, (x, y) in (("contiguous", (sa, sb)), ("strided views", (at, bt))):
            got = bmm(x, y, azp, bzp)
            ok = bool(torch.equal(got.cpu().to(torch.int64), want)) \
                and got.dtype == (torch.int64 if dt == "INT16" else torch.int32)
            rows.append(("matmul", 0, (dt, (N, M, K, P), f"zp {azp} {bzp}"),
                         f"{label}, max |sum| {int(want.abs().max())}", ok))
    return rows


def main(specs):
    gen = torch.Generator().manual_seed(0)
    rows = matmul_rows(gen) + conv_rows(gen)
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
