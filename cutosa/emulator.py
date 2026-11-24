"""The int64 emulator's results, with tensors stored at their TOSA type and computed by GPU kernels."""
import math

import numpy as np
import torch

from .int64_ref import BITS, RANGE, ROUND_SINGLE, TosaGraphIR
from .int64_ref import Emulator as Int64Emulator

STORE = {"BOOL": torch.int8, "INT8": torch.int8, "INT16": torch.int16, "INT32": torch.int32, "INT48": torch.int64}
NO_WIDEN_OPS = frozenset(("MINIMUM", "MAXIMUM", "SELECT", "CLAMP", "REDUCE_MAX", "TRANSPOSE", "RESHAPE", "SLICE",
                       "CONCAT"))


class Emulator(Int64Emulator):
    def __init__(self, graph, device, check=True, mm_backend="intmm", fuse=True):
        """fuse: run each tree of pointwise operators as one generated kernel."""
        super().__init__(graph, device=device, check=check, mm_backend=mm_backend)
        self.native = self.device.type == "cuda"
        self.fuse = fuse
        self._plans = {}
        self.how = {}
        self._prep = {}
        self._counters, self._cbuf = [], None
        if self.native:
            for n, t in list(self.consts.items()):
                self.consts[n] = self._narrow_to_storage(n, t, verify=True)
            self._reset_require_counts(1)

    def _narrow_to_storage(self, name, t, verify):
        store = STORE[self.g.dtype(name)]
        if t.dtype == store:
            return t
        if verify and t.numel():
            lo, hi = RANGE[self.g.dtype(name)]
            mn, mx = (int(v) for v in torch.aminmax(t))
            if mn < lo or mx > hi:
                raise NotImplementedError(f"{name}: values [{mn}, {mx}] do not fit the stored {self.g.dtype(name)} "
                                          f"(an unsigned RESCALE output?)")
        return t.to(store)

    @staticmethod
    def _widen_to_int64(t):
        return t if t is None or t.dtype == torch.int64 else t.to(torch.int64)

    def _reset_require_counts(self, samples):
        """reset the REQUIRE records for a run of `samples` samples."""
        if samples > 65535:
            raise NotImplementedError(f"{samples} samples in one run (the kernels take at most 65535)")
        self.violations, self._pending, self.samples, self._counters = [], [], samples, []
        # each operator uses at most 2 counter rows; the last row collects the counts when check=False
        rows = 2 * len(self.g.ops) + 1
        if self._cbuf is None or self._cbuf.numel() != rows * samples:
            self._cbuf = torch.zeros(rows * samples, dtype=torch.int64, device=self.device)
        else:
            self._cbuf.zero_()

    def _new_require_counter(self, o, msg, parts):
        """-> where a new counter row starts in self._cbuf; sample p adds its count at that offset + p."""
        if not self.check:
            return self._cbuf.numel() - self.samples
        self._counters.append((o.idx, o.op, msg, parts == self.samples))
        return (len(self._counters) - 1) * self.samples

    def _require_count_records(self):
        """-> [(op index, op, message, count or per-sample counts)], zero counts included, one device transfer."""
        if not self._counters:
            return []
        n = len(self._counters)
        counts = self._cbuf[:n * self.samples].view(n, self.samples).cpu().tolist()
        return [(i, op, msg, (c if per_part else [c[0]] * self.samples) if self.samples > 1 else c[0])
                for (i, op, msg, per_part), c in zip(self._counters, counts)]

    def _sample_parts(self, o, shape):
        """how many per-sample parts a checked tensor of this shape splits into."""
        if self.samples == 1 or len(shape) == 0 or not self._from_inputs or not self.check:
            return 1
        if next((d for d in shape if d != 1), 1) % self.samples:
            raise ValueError(f"#{o.idx} {o.op}: checked tensor of shape {tuple(shape)} has no outermost dimension "
                             f"that splits into {self.samples} samples")
        return self.samples

    def k_RESCALE(self, o, x, _m, _s, _izp, _ozp):
        from .kernels.rescale import rescale
        a = o.attrs
        scale32 = bool(a["Scale32"])
        if a["InputUnsigned"] or a["OutputUnsigned"] or (scale32 and a["RoundingMode"] != ROUND_SINGLE) \
                or x.numel() == 0:
            return None
        mult, shift = self._host(o.inputs[1]).reshape(-1), self._host(o.inputs[2]).reshape(-1)
        per_channel = bool(a["PerChannel"])
        if (mult < 0).any() or (shift < 2).any() or (shift > 62).any() or mult.size != shift.size \
                or (per_channel and mult.size != x.shape[-1]):
            return None  # REQUIRE failures and malformed operands stay with the int64 emulator
        out_dt = self.g.dtype(o.outputs[0])
        izp, ozp = self._scalar(o.inputs[3]), self._scalar(o.inputs[4])
        if per_channel:
            if o.idx not in self._prep:
                self._prep[o.idx] = (torch.from_numpy(mult).to(self.device), torch.from_numpy(shift).to(self.device))
            m, s = self._prep[o.idx]
        else:
            m, s = int(mult[0]), int(shift[0])
        parts = self._sample_parts(o, x.shape)
        c0 = self._new_require_counter(o, "apply_scale_32 value outside [-2^(shift-1), 2^(shift-1))" if scale32
                        else "apply_scale_16 result outside INT32", parts)
        c1 = self._new_require_counter(o, "RESCALE result + output_zp outside INT32", parts)
        return [rescale(x.contiguous(), STORE[out_dt], m, s, izp, ozp, *RANGE[out_dt], scale32, per_channel,
                        self._cbuf, c0, c1, parts)]

    def k_MATMUL(self, o, A, B, _azp, _bzp):
        from .kernels.matmul import bmm_i16, fits
        if self.g.dtype(o.inputs[0]) != "INT16" or self.g.dtype(o.inputs[1]) != "INT16" \
                or self.g.dtype(o.outputs[0]) != "INT48" or self._scalar(o.inputs[2]) or self._scalar(o.inputs[3]) \
                or A.numel() == 0 or B.numel() == 0 or not fits(*A.shape, B.shape[2]):
            return None
        # called only to raise, like the int64 emulator, when the samples cannot split the output
        self._sample_parts(o, (A.shape[0], A.shape[1], B.shape[2]))
        # no INT48 overflow count needed: |sum| <= K * 2^30 < 2^47 for K <= MAX_K
        return [bmm_i16(A, B)]

    def _k_binary(self, o, code, a, b, what, shift=0):
        from .kernels.elementwise import binary, broadcast_shape
        out_dt = self.g.dtype(o.outputs[0])
        shape = broadcast_shape(a.shape, b.shape)
        if out_dt not in ("INT8", "INT16", "INT32") or a.numel() == 0 or b.numel() == 0:
            return None
        parts = self._sample_parts(o, shape)
        c0 = self._new_require_counter(o, f"{what} outside {out_dt}", parts)
        return [binary(code, a, b, STORE[out_dt], *RANGE[out_dt], self._cbuf, c0, shift=shift, parts=parts)]

    def k_ADD(self, o, a, b):
        from .kernels.elementwise import ADD
        return self._k_binary(o, ADD, a, b, "ADD result")

    def k_SUB(self, o, a, b):
        from .kernels.elementwise import SUB
        return self._k_binary(o, SUB, a, b, "SUB result")

    def k_MUL(self, o, a, b, _shift):
        from .kernels.elementwise import MUL
        shift = self._scalar(o.inputs[2])
        if self.g.dtype(o.inputs[0]) != "INT32":
            if shift != 0:
                return None  # a REQUIRE failure: stays with the int64 emulator
            eff = 0
        else:
            eff = max(shift, 0)
        if eff > 62:  # the product would overflow int64
            return None
        return self._k_binary(o, MUL, a, b, f"MUL result (shift {shift})", shift=eff)

    def k_TABLE(self, o, x, table):
        from .kernels.elementwise import table as table_kernel
        dt, out_dt = self.g.dtype(o.inputs[0]), self.g.dtype(o.outputs[0])
        host = self.g.tensors[o.inputs[1]].data
        if host is None or x.numel() == 0 or (dt, self.g.dtype(o.inputs[1]), out_dt, host.size) not in (
                ("INT8", "INT8", "INT8", 256), ("INT16", "INT16", "INT32", 513)):
            return None
        if dt == "INT16":
            slope = np.diff(host.reshape(-1))
            if ((slope < RANGE["INT16"][0]) | (slope > RANGE["INT16"][1])).any():
                return None  # the slope REQUIRE can fail: stays with the int64 emulator
        return [table_kernel(x, table.reshape(-1).contiguous(), STORE[out_dt], dt == "INT16")]

    def k_REDUCE_SUM(self, o, x):
        from .kernels.elementwise import reduce_sum
        out_dt = self.g.dtype(o.outputs[0])
        if out_dt not in ("INT8", "INT16", "INT32") or x.numel() == 0:
            return None
        axis = o.attrs["Axis"]
        shape = tuple(1 if i == axis else d for i, d in enumerate(x.shape))
        parts = self._sample_parts(o, shape)
        c0 = self._new_require_counter(o, f"REDUCE_SUM running sum outside {out_dt}", parts)
        return [reduce_sum(x, axis, STORE[out_dt], *RANGE[out_dt], self._cbuf, c0, parts=parts)]

    def k_CONV2D(self, o, x, w, bias, _izp, _wzp):
        from .kernels.conv import MAX_K, linear
        a, g = o.attrs, self.g
        in_dt, out_dt = g.dtype(o.inputs[0]), g.dtype(o.outputs[0])
        N, IH, IW, IC = x.shape
        OC, KH, KW, _ = w.shape
        M = N * IH * IW
        if KH != 1 or KW != 1 or tuple(a["Stride"]) != (1, 1) or any(a["Pad"]) or g.dtype(o.inputs[1]) != "INT8" \
                or (in_dt, out_dt) not in (("INT8", "INT32"), ("INT16", "INT48")) or IC > MAX_K or M == 0 or OC == 0 \
                or OC * IC >= 2**31 or OC > 65535 * 64:
            return None
        parts = self._sample_parts(o, (N, IH, IW, OC))
        if parts > 1 and M % parts:  # the samples would split the output channels, but this kernel counts per row
            return None
        prep = self._prep.get(o.idx)
        if prep is None:
            w2 = w.reshape(OC, IC).contiguous()
            prep = (w2, w2.sum(1, dtype=torch.int64), bias.reshape(-1).to(torch.int64).expand(OC).contiguous())
            if o.inputs[1] in self.consts and o.inputs[2] in self.consts:
                self._prep[o.idx] = prep
        w2, wsum, bias64 = prep
        saturate = out_dt == "INT48"
        what = "int48 accumulator (saturated)" if saturate else "int32 accumulator overflow (wrapped)"
        c0 = self._new_require_counter(o, f"{what} outside {out_dt}", parts)
        y = linear(x.reshape(M, IC).contiguous(), w2, bias64, wsum, STORE[out_dt], self._scalar(o.inputs[3]),
                   self._scalar(o.inputs[4]), *RANGE[out_dt], saturate, self._cbuf, c0, parts)
        return [y.reshape(N, IH, IW, OC)]

    def k_GREATER(self, o, a, b):
        return [(a > b).to(torch.int8)]

    def k_GREATER_EQUAL(self, o, a, b):
        return [(a >= b).to(torch.int8)]

    def k_GATHER(self, o, values, idx):
        # torch.gather needs an int64 index; the values stay int8 / int16
        return self.op_GATHER(o, values, self._widen_to_int64(idx))

    def _plan(self, keep, fuse):
        """-> (absorbed, steps), cached per (fuse, keep). absorbed: operator computed inside a fused kernel -> the
        operator that stores its unit's result. steps: (operator, its unit or None, whether it reads graph-input data,
        tensors to free after it)."""
        key = (fuse, frozenset(keep))
        if key not in self._plans:
            from .fusion import plan
            g = self.g
            units, absorbed = plan(g, keep) if fuse else ({}, {})
            last = {}
            for o in g.ops:
                at = absorbed.get(o.idx, o.idx)  # an absorbed operator's inputs are read when its unit's kernel runs
                for n in o.inputs:
                    last[n] = max(last.get(n, -1), at)
            free = {}
            for n, i in last.items():
                if n not in keep and n not in self.consts:
                    free.setdefault(i, []).append(n)
            from_inputs, steps = set(g.inputs), []
            for o in g.ops:
                if o.op in ("CONST", "CONST_SHAPE"):
                    continue
                reads = any(n in from_inputs for n in o.inputs)
                if reads:
                    from_inputs.update(o.outputs)
                if o.idx not in absorbed:
                    steps.append((o, units.get(o.idx), reads, tuple(free.get(o.idx, ()))))
            self._plans[key] = (absorbed, steps)
        return self._plans[key]

    def _unit_prep(self, u):
        """a unit's kernel structure and everything constant about it (built once per unit)."""
        from .kernels import fused
        from .kernels.elementwise import index_plan
        from .fusion import DATA_IN
        g = self.g
        alias = {r.outputs[0]: r.inputs[0] for r in u.reshapes}
        reg = {m.outputs[0]: f"V{j}" for j, m in enumerate(u.members)}
        leaves, modes, lead, ops, tail, aux, checks = [], [], [], [], [], [], []
        for m in u.members:
            out_dt, shape = g.dtype(m.outputs[0]), g.tensors[m.outputs[0]].shape
            refs = []
            for n in m.inputs[:DATA_IN[m.op]]:
                while n in alias:
                    n = alias[n]
                if n in reg:
                    refs.append(reg[n])
                    continue
                mode, inner, cnt = index_plan(g.tensors[n].shape, shape)
                refs.append(f"L{len(leaves)}")
                leaves.append((n, shape if mode == "expand" else None))
                modes.append("i" if mode == "index" else "f")
                if mode == "index":
                    lead += [inner, cnt]
            flags = ()
            if m.op == "RESCALE":
                a = m.attrs
                scale32, per_channel = bool(a["Scale32"]), bool(a["PerChannel"])
                flags = (scale32, per_channel)
                izp, ozp = self._scalar(m.inputs[3]), self._scalar(m.inputs[4])
                mult, shift = self._host(m.inputs[1]).reshape(-1), self._host(m.inputs[2]).reshape(-1)
                if per_channel:
                    tail += [izp, ozp, *RANGE[out_dt], g.tensors[m.inputs[0]].shape[-1]]
                    aux += [torch.from_numpy(mult).to(self.device), torch.from_numpy(shift).to(self.device)]
                else:
                    tail += [izp, int(mult[0]), int(shift[0]), ozp, *RANGE[out_dt]]
                checks.append((m, "apply_scale_32 value outside [-2^(shift-1), 2^(shift-1))" if scale32
                               else "apply_scale_16 result outside INT32"))
                checks.append((m, "RESCALE result + output_zp outside INT32"))
            elif m.op in ("ADD", "SUB"):
                flags = (BITS[out_dt],)
                tail += list(RANGE[out_dt])
                checks.append((m, f"{m.op} result outside {out_dt}"))
            elif m.op == "MUL":
                shift = self._scalar(m.inputs[2])
                flags = (BITS[out_dt],)
                tail += [max(shift, 0) if g.dtype(m.inputs[0]) == "INT32" else 0, *RANGE[out_dt]]
                checks.append((m, f"MUL result (shift {shift}) outside {out_dt}"))
            elif m.op == "TABLE":
                wide = g.dtype(m.inputs[0]) == "INT16"
                slope = np.diff(self._host(m.inputs[1]).reshape(-1)) if wide else np.zeros(1, np.int64)
                flags = (wide, bool(((slope < RANGE["INT16"][0]) | (slope > RANGE["INT16"][1])).any()))
                if flags[1]:
                    checks.append((m, "TABLE slope outside INT16"))
            ops.append((m.op, flags, tuple(refs)))
        # same order as the generated kernel's arguments: tables, then per-channel (mult, shift)
        tables = [self.consts[m.inputs[1]].reshape(-1).contiguous() for m in u.members if m.op == "TABLE"]
        aux = tables + aux
        prog = (tuple(modes), tuple(ops))
        # a trailing 0 so that the tensor is never empty
        params = torch.tensor(lead + tail + [0], dtype=torch.int64, device=self.device)
        assert params.numel() - 1 == fused.n_params(prog) and len(checks) == fused.n_counters(prog)
        root = u.members[-1].outputs[0]
        store = STORE[g.dtype(root)]
        # the kernel is compiled for these dtypes without looking at the tensors; _run stores each as STORE[type]
        types = (store, torch.int64, torch.int64,
                 *[self.consts[n].dtype if n in self.consts else STORE[g.dtype(n)] for n, _ in leaves],
                 *[t.dtype for t in aux])
        return dict(prog=prog, kernel=fused.kernel(prog), types=types, params=params, leaves=leaves, aux=aux,
                    shape=g.tensors[root].shape, numel=math.prod(g.tensors[root].shape), store=store,
                    shapes=list({m.idx: (m, g.tensors[m.outputs[0]].shape) for m, _ in checks}.values()), parts={},
                    counters={per_part: [(m.idx, m.op, msg, per_part) for m, msg in checks]
                              for per_part in (True, False)})

    def _run_unit(self, u, env):
        """runs the unit as one kernel and returns the root's result, or None when its members need different
        per-sample splits (the caller then runs them one by one)."""
        from .kernels import fused
        # keyed by the members: with another keep set, the same root can have other members
        prep = self._prep.get(u.key)
        if prep is None:
            prep = self._prep[u.key] = self._unit_prep(u)
        mode = (self.samples, self.check)
        parts = prep["parts"].get(mode, 0)
        if parts == 0:  # computed once per (samples, check): it depends on shapes only
            self._from_inputs = True
            # prep["shapes"] has only the members with a REQUIRE check
            found = {self._sample_parts(m, shape) for m, shape in prep["shapes"]}
            parts = prep["parts"][mode] = 1 if not found else found.pop() if len(found) == 1 else None
        if parts is None:
            return None
        if self.check:
            cbase, cstride = len(self._counters) * self.samples, self.samples
            self._counters += prep["counters"][parts == self.samples]
        else:
            cbase, cstride = self._cbuf.numel() - self.samples, 0
        leaves = [env[n].contiguous() if shape is None else env[n].expand(shape).contiguous()
                  for n, shape in prep["leaves"]]
        y = torch.empty(prep["shape"], dtype=prep["store"], device=self.device)
        fused.launch(prep["kernel"], prep["types"], y, self._cbuf, prep["params"], prep["numel"] // parts, parts,
                     cbase, cstride, leaves, prep["aux"])
        return y

    def _exec(self, o, env, from_inputs):
        """runs one operator unfused -> (outputs, how). how: "kernel" = a cu-tosa GPU kernel; "stored" = the int64
        emulator's method directly on the int8 / int16 tensors; "fallback" = widen to int64, int64 method, narrow
        back."""
        if o.op == "RESHAPE":  # reads the constant target shape once and reuses it
            shape = self._prep.get(o.idx)
            if shape is None and (o.inputs[1] in self.g.shapes or self.g.tensors[o.inputs[1]].data is not None):
                shape = self._prep[o.idx] = tuple(int(v) for v in self._host(o.inputs[1]))
            if shape is not None:
                return [env[o.inputs[0]].reshape(shape)], "stored"
        ref = getattr(self, "op_" + o.op, None)
        if ref is None:
            raise NotImplementedError(f"op {o.op} (#{o.idx})")
        ins = [env.get(n) for n in o.inputs]
        self._from_inputs = from_inputs
        kernel = getattr(self, "k_" + o.op, None)
        outs = kernel(o, *ins) if kernel is not None else None
        if outs is not None:
            return outs, "kernel"
        if o.op in NO_WIDEN_OPS:
            return ref(o, *ins), "stored"
        outs = ref(o, *[self._widen_to_int64(t) for t in ins])
        # unsigned values do not fit the signed stored type: _narrow_to_storage raises
        unsigned = o.op == "RESCALE" and bool(o.attrs["OutputUnsigned"])
        return [self._narrow_to_storage(n, t, verify=unsigned) for n, t in zip(o.outputs, outs)], "fallback"

    def run(self, inputs, hook=None, keep=(), samples=1):
        if not self.native:
            return super().run(inputs, hook=hook, keep=keep, samples=samples)
        with torch.cuda.device(self.device):  # Triton launches on torch's current device
            return self._run(inputs, hook, keep, samples)

    def _run(self, inputs, hook, keep, samples):
        g = self.g
        if len(inputs) != len(g.inputs):
            raise ValueError(f"graph has {len(g.inputs)} inputs, got {len(inputs)}")
        self._reset_require_counts(samples)
        keep = set(keep) | set(g.outputs)
        # test faults can target any operator, so run unfused
        absorbed, steps = self._plan(keep, self.fuse and not self.faults)
        env = dict(self.consts)
        for name, x in zip(g.inputs, inputs):
            t = torch.as_tensor(np.asarray(x)) if not isinstance(x, torch.Tensor) else x
            if tuple(t.shape) != g.tensors[name].shape:
                raise ValueError(f"input {name}: shape {tuple(t.shape)} != {g.tensors[name].shape}")
            # through int64, like the int64 emulator, whatever the numpy integer type
            if t.dtype != STORE[g.dtype(name)]:
                t = t.to(torch.int64)
            env[name] = self._narrow_to_storage(name, t.to(self.device), verify=True)
        how = self.how
        how.update(dict.fromkeys(absorbed, "fused"))
        for o, u, reads, drop in steps:
            y = self._run_unit(u, env) if u is not None else None
            if y is not None:  # the kernel wrote the declared shape and type: no _narrow_to_storage check
                how[o.idx] = "fused"
                env[o.outputs[0]] = y
                if hook is not None:
                    hook(o, [y])
            else:
                if u is not None:
                    inner = sorted(u.members[:-1] + u.reshapes, key=lambda m: m.idx)
                    for m in inner:
                        mo, how[m.idx] = self._exec(m, env, True)
                        env.update(zip(m.outputs, mo))
                    outs, how[o.idx] = self._exec(o, env, reads)
                    for m in inner:
                        env.pop(m.outputs[0], None)
                else:
                    outs, how[o.idx] = self._exec(o, env, reads)
                if o.idx in self.faults:
                    outs = self.faults[o.idx](outs)
                for n, t in zip(o.outputs, outs):
                    want = g.tensors[n].shape
                    if tuple(t.shape) != want:
                        raise RuntimeError(f"#{o.idx} {o.op} -> {n}: shape {tuple(t.shape)} != declared {want}")
                    if t.dtype != STORE[g.dtype(n)]:
                        raise RuntimeError(f"#{o.idx} {o.op} -> {n}: stored as {t.dtype}, want {STORE[g.dtype(n)]}")
                    env[n] = t
                if hook is not None:
                    hook(o, outs)
            for n in drop:
                env.pop(n, None)
        self.env = {n: env[n] for n in keep if n in env}
        records = self._require_count_records()  # stable sort: an operator's records stay in order
        if self._pending:
            counts = torch.stack([c for *_, c in self._pending]).cpu().tolist()
            records += [(i, op, msg, n) for (i, op, msg, _), n in zip(self._pending, counts)]
            self._pending = []
        self.violations += [r for r in records if (any(r[3]) if self.samples > 1 else r[3])]
        self.violations.sort(key=lambda v: v[0])
        return [self.to_numpy(n, env[n]) for n in g.outputs]


__all__ = ["Emulator", "TosaGraphIR", "STORE", "BITS"]
