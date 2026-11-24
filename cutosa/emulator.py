"""The int64 emulator's results, with tensors stored at their TOSA type and computed by GPU kernels."""
import math
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import torch

from .emulator_types import CompiledUnit, How, PlanStep, RequireCounter
from .fusion import Unit, eligible, mul_kernel_shift, plan, table_slope_can_fail
from .int64_ref import BITS, DATA_OPERANDS, RANGE, OpInfo, Require, TosaGraphIR, Violation
from .int64_ref import Emulator as Int64Emulator

STORE = {"BOOL": torch.int8, "INT8": torch.int8, "INT16": torch.int16, "INT32": torch.int32, "INT48": torch.int64}
NO_WIDEN_OPS = frozenset(("MINIMUM", "MAXIMUM", "SELECT", "CLAMP", "REDUCE_MAX", "TRANSPOSE", "RESHAPE", "SLICE",
                       "CONCAT"))


class Emulator(Int64Emulator):
    def __init__(self, graph: TosaGraphIR | str | bytes, device: str | torch.device, fuse: bool = True) -> None:
        """fuse: run each tree of pointwise operators as one generated kernel."""
        super().__init__(graph, device)
        self.on_cuda = self.device.type == "cuda"
        self.fuse = fuse
        self._plans: dict[tuple[bool, frozenset[str]], tuple[dict[int, int], list[PlanStep]]] = {}
        self.how: dict[int, How] = {}
        self._per_channel_scales: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._conv_weights: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._reshape_shapes: dict[int, tuple[int, ...]] = {}
        self._compiled_units: dict[tuple[int, ...], CompiledUnit] = {}
        self._counters: list[RequireCounter] = []
        self._cbuf = torch.zeros(0, dtype=torch.int64, device=self.device)
        if self.on_cuda:
            for n, t in list(self.consts.items()):
                self.consts[n] = self._narrow_to_storage(n, t, verify=True)
            self._reset_require_counts(1)

    def _narrow_to_storage(self, name: str, t: torch.Tensor, verify: bool) -> torch.Tensor:
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

    def _const(self, o: OpInfo, j: int) -> torch.Tensor:
        # the int64 methods compute in int64; self.consts holds the stored types
        return self._stored_const(o, j).to(torch.int64)

    def _stored_const(self, o: OpInfo, j: int) -> torch.Tensor:
        return super()._const(o, j)

    def _reset_require_counts(self, samples: int) -> None:
        """reset the REQUIRE records for a run of `samples` samples."""
        if samples > 65535:
            raise NotImplementedError(f"{samples} samples in one run (the kernels take at most 65535)")
        self.violations, self._pending, self.samples, self._counters = [], [], samples, []
        rows = 2 * len(self.g.ops)  # an operator uses at most 2 counter rows
        if self._cbuf.numel() != rows * samples:
            self._cbuf = torch.zeros(rows * samples, dtype=torch.int64, device=self.device)
        else:
            self._cbuf.zero_()

    def _new_require_counter(self, o: OpInfo, req: Require, parts: int, **fmt: object) -> int:
        """-> where a new counter row starts in self._cbuf; sample p adds its count at that offset + p."""
        self._counters.append(RequireCounter(o.idx, o.op, req, req.value.format(**fmt), parts == self.samples))
        return (len(self._counters) - 1) * self.samples

    def _require_count_records(self) -> list[Violation]:
        """zero counts included; one device transfer."""
        if not self._counters:
            return []
        rows = len(self._counters)
        counts = self._cbuf[:rows * self.samples].view(rows, self.samples).cpu().tolist()
        return [c.violation((row if c.per_sample else [row[0]] * self.samples) if self.samples > 1 else row[0])
                for c, row in zip(self._counters, counts, strict=True)]

    def _sample_parts(self, o: OpInfo, shape: Sequence[int]) -> int:
        """how many per-sample parts a checked tensor of this shape splits into."""
        if self.samples == 1 or len(shape) == 0 or not self._from_inputs:
            return 1
        if next((d for d in shape if d != 1), 1) % self.samples:
            raise ValueError(f"#{o.idx} {o.op}: checked tensor of shape {tuple(shape)} has no outermost dimension "
                             f"that splits into {self.samples} samples")
        return self.samples

    def _check_sample_split(self, o: OpInfo, shape: Sequence[int]) -> None:
        """raises where the int64 emulator raises: a checked tensor of this shape cannot be split by sample."""
        self._sample_parts(o, shape)

    def k_RESCALE(self, o: OpInfo, x: torch.Tensor) -> list[torch.Tensor] | None:
        from .kernels.rescale import rescale
        if not eligible(self.g, o):
            return None
        scale32, per_channel = bool(o.attrs["Scale32"]), bool(o.attrs["PerChannel"])
        mult, shift = self._host(o.inputs[1]).reshape(-1), self._host(o.inputs[2]).reshape(-1)
        out_dt = self.g.dtype(o.outputs[0])
        izp, ozp = self._scalar(o.inputs[3]), self._scalar(o.inputs[4])
        if per_channel:
            if o.idx not in self._per_channel_scales:
                self._per_channel_scales[o.idx] = (torch.from_numpy(mult).to(self.device),
                                                   torch.from_numpy(shift).to(self.device))
            m, s = self._per_channel_scales[o.idx]
        else:
            m, s = int(mult[0]), int(shift[0])
        parts = self._sample_parts(o, x.shape)
        c0 = self._new_require_counter(o, Require.APPLY_SCALE_32 if scale32 else Require.APPLY_SCALE_16, parts)
        c1 = self._new_require_counter(o, Require.RESCALE_RESULT, parts)
        return [rescale(x.contiguous(), STORE[out_dt], m, s, izp, ozp, *RANGE[out_dt], scale32, per_channel,
                        self._cbuf, c0, c1, parts)]

    def k_MATMUL(self, o: OpInfo, A: torch.Tensor, B: torch.Tensor) -> list[torch.Tensor] | None:
        from .kernels.matmul import bmm_i16, fits
        if self.g.dtype(o.inputs[0]) != "INT16" or self.g.dtype(o.inputs[1]) != "INT16" \
                or self.g.dtype(o.outputs[0]) != "INT48" or self._scalar(o.inputs[2]) or self._scalar(o.inputs[3]) \
                or A.numel() == 0 or B.numel() == 0 or not fits(*A.shape, B.shape[2]):  # pyright: ignore[reportCallIssue]
            return None
        self._check_sample_split(o, (A.shape[0], A.shape[1], B.shape[2]))
        # no INT48 overflow count needed: |sum| <= K * 2^30 < 2^47 for K <= MAX_K
        return [bmm_i16(A, B)]

    def _k_binary(self, o: OpInfo, code: int, a: torch.Tensor, b: torch.Tensor, req: Require, kernel_shift: int,
                  **fmt: object) -> list[torch.Tensor] | None:
        from .kernels.elementwise import binary, broadcast_shape
        if not eligible(self.g, o):
            return None
        out_dt = self.g.dtype(o.outputs[0])
        parts = self._sample_parts(o, broadcast_shape(a.shape, b.shape))
        c0 = self._new_require_counter(o, req, parts, dt=out_dt, **fmt)
        return [binary(code, a, b, STORE[out_dt], *RANGE[out_dt], self._cbuf, c0, kernel_shift, parts)]

    def k_ADD(self, o: OpInfo, a: torch.Tensor, b: torch.Tensor) -> list[torch.Tensor] | None:
        from .kernels.elementwise import ADD
        return self._k_binary(o, ADD, a, b, Require.ADD_RESULT, 0)

    def k_SUB(self, o: OpInfo, a: torch.Tensor, b: torch.Tensor) -> list[torch.Tensor] | None:
        from .kernels.elementwise import SUB
        return self._k_binary(o, SUB, a, b, Require.SUB_RESULT, 0)

    def k_MUL(self, o: OpInfo, a: torch.Tensor, b: torch.Tensor) -> list[torch.Tensor] | None:
        from .kernels.elementwise import MUL
        shift = self._scalar(o.inputs[2])
        req = Require.MUL_RESULT if shift else Require.MUL_RESULT_SHIFT0
        return self._k_binary(o, MUL, a, b, req, mul_kernel_shift(self.g.dtype(o.inputs[0]), shift), shift=shift)

    def k_TABLE(self, o: OpInfo, x: torch.Tensor) -> list[torch.Tensor] | None:
        from .kernels.elementwise import table_int8, table_int16
        if not eligible(self.g, o):
            return None
        table = self._stored_const(o, 1).reshape(-1).contiguous()
        if self.g.dtype(o.inputs[0]) == "INT8":
            return [table_int8(x, table)]
        can_fail = table_slope_can_fail(self._host(o.inputs[1]))
        # the int64 emulator splits the count by sample, and raises when it cannot, only for such a table
        parts = self._sample_parts(o, x.shape) if can_fail else 1
        c0 = self._new_require_counter(o, Require.TABLE_SLOPE, parts)
        return [table_int16(x, table, self._cbuf, c0, parts)]

    def k_REDUCE_SUM(self, o: OpInfo, x: torch.Tensor) -> list[torch.Tensor] | None:
        from .kernels.elementwise import reduce_sum
        out_dt = self.g.dtype(o.outputs[0])
        if out_dt not in ("INT8", "INT16", "INT32") or x.numel() == 0:
            return None
        axis = o.attrs["Axis"]
        shape = tuple(1 if i == axis else d for i, d in enumerate(x.shape))
        parts = self._sample_parts(o, shape)
        c0 = self._new_require_counter(o, Require.REDUCE_SUM, parts, dt=out_dt)
        return [reduce_sum(x, axis, STORE[out_dt], *RANGE[out_dt], self._cbuf, c0, parts)]

    def k_CONV2D(self, o: OpInfo, x: torch.Tensor) -> list[torch.Tensor] | None:
        from .kernels.conv import MAX_K, linear
        a, g = o.attrs, self.g
        w = self._stored_const(o, 1)
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
        weights = self._conv_weights.get(o.idx)
        if weights is None:
            w2 = w.reshape(OC, IC).contiguous()
            bias = self._stored_const(o, 2).reshape(-1).to(torch.int64).expand(OC).contiguous()
            weights = self._conv_weights[o.idx] = (w2, w2.sum(1, dtype=torch.int64), bias)
        w2, wsum, bias64 = weights
        saturate = out_dt == "INT48"
        c0 = self._new_require_counter(o, Require.ACC_INT48 if saturate else Require.ACC_INT32, parts)
        y = linear(x.reshape(M, IC).contiguous(), w2, bias64, wsum, STORE[out_dt], self._scalar(o.inputs[3]),
                   self._scalar(o.inputs[4]), *RANGE[out_dt], saturate, self._cbuf, c0, parts)
        return [y.reshape(N, IH, IW, OC)]

    def k_GREATER(self, _: OpInfo, a: torch.Tensor, b: torch.Tensor) -> list[torch.Tensor] | None:
        return [(a > b).to(torch.int8)]

    def k_GREATER_EQUAL(self, _: OpInfo, a: torch.Tensor, b: torch.Tensor) -> list[torch.Tensor] | None:
        return [(a >= b).to(torch.int8)]

    def k_GATHER(self, o: OpInfo, values: torch.Tensor, idx: torch.Tensor) -> list[torch.Tensor] | None:
        # torch.gather needs an int64 index; the values stay int8 / int16
        return self.op_GATHER(o, values, idx.to(torch.int64))

    def _plan(self, keep: set[str], fuse: bool) -> tuple[dict[int, int], list[PlanStep]]:
        """-> (absorbed, steps), cached per (fuse, keep). absorbed: operator computed inside a fused kernel -> the
        operator that stores its unit's result."""
        key = (fuse, frozenset(keep))
        if key not in self._plans:
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
                    steps.append(PlanStep(o, units.get(o.idx), reads, tuple(free.get(o.idx, ()))))
            self._plans[key] = (absorbed, steps)
        return self._plans[key]

    def _unit_operands(self, u: Unit) -> tuple[list[tuple[str, ...]], list[tuple[str, tuple[int, ...] | None]],
                                               list[str], list[int]]:
        """-> (each member's operand refs, the leaves read from memory, their read modes, the index parameters)."""
        from .kernels.elementwise import index_plan
        g = self.g
        alias = {r.outputs[0]: r.inputs[0] for r in u.reshapes}
        reg = {m.outputs[0]: f"V{j}" for j, m in enumerate(u.members)}
        refs, leaves, modes, index_params = [], [], [], []
        for m in u.members:
            shape = g.tensors[m.outputs[0]].shape
            member_refs = []
            for n in m.inputs[:DATA_OPERANDS[m.op]]:
                while n in alias:
                    n = alias[n]
                if n in reg:
                    member_refs.append(reg[n])
                    continue
                mode, inner, count = index_plan(g.tensors[n].shape, shape)
                member_refs.append(f"L{len(leaves)}")
                leaves.append((n, shape if mode == "expand" else None))
                modes.append("i" if mode == "index" else "f")
                if mode == "index":
                    index_params += [inner, count]
            refs.append(tuple(member_refs))
        return refs, leaves, modes, index_params

    def _member_params(self, m: OpInfo) -> tuple[tuple[object, ...], list[int], list[torch.Tensor],
                                                 list[RequireCounter]]:
        """-> (flags, params, per-channel scale tensors, REQUIRE counters) of one unit member."""
        g = self.g
        out_dt = g.dtype(m.outputs[0])

        def counter(req: Require, **fmt: object) -> RequireCounter:
            return RequireCounter(m.idx, m.op, req, req.value.format(**fmt), False)

        if m.op == "RESCALE":
            scale32, per_channel = bool(m.attrs["Scale32"]), bool(m.attrs["PerChannel"])
            izp, ozp = self._scalar(m.inputs[3]), self._scalar(m.inputs[4])
            mult, shift = self._host(m.inputs[1]).reshape(-1), self._host(m.inputs[2]).reshape(-1)
            counters = [counter(Require.APPLY_SCALE_32 if scale32 else Require.APPLY_SCALE_16),
                        counter(Require.RESCALE_RESULT)]
            if per_channel:
                return ((scale32, per_channel), [izp, ozp, *RANGE[out_dt], g.tensors[m.inputs[0]].shape[-1]],
                        [torch.from_numpy(mult).to(self.device), torch.from_numpy(shift).to(self.device)], counters)
            return (scale32, per_channel), [izp, int(mult[0]), int(shift[0]), ozp, *RANGE[out_dt]], [], counters
        if m.op in ("ADD", "SUB"):
            req = Require.ADD_RESULT if m.op == "ADD" else Require.SUB_RESULT
            return (BITS[out_dt],), list(RANGE[out_dt]), [], [counter(req, dt=out_dt)]
        if m.op == "MUL":
            shift = self._scalar(m.inputs[2])
            req = Require.MUL_RESULT if shift else Require.MUL_RESULT_SHIFT0
            return ((BITS[out_dt],), [mul_kernel_shift(g.dtype(m.inputs[0]), shift), *RANGE[out_dt]], [],
                    [counter(req, dt=out_dt, shift=shift)])
        if m.op == "TABLE":
            wide = g.dtype(m.inputs[0]) == "INT16"
            slope_check = wide and table_slope_can_fail(self._host(m.inputs[1]))
            return (wide, slope_check), [], [], [counter(Require.TABLE_SLOPE)] if slope_check else []
        return (), [], [], []

    def _compile_unit(self, u: Unit) -> CompiledUnit:
        """a unit's kernel structure and everything constant about it (built once per unit)."""
        from .kernels import fused
        g = self.g
        refs, leaves, modes, index_params = self._unit_operands(u)
        ops, member_params, channel_scales, checks = [], [], [], []
        for m, member_refs in zip(u.members, refs, strict=True):
            flags, params, scales, counters = self._member_params(m)
            ops.append((m.op, flags, member_refs))
            member_params += params
            channel_scales += scales
            checks += [(m, c) for c in counters]
        # same order as the generated kernel's arguments: tables, then per-channel (mult, shift)
        tables = [self._stored_const(m, 1).reshape(-1).contiguous() for m in u.members if m.op == "TABLE"]
        aux = tables + channel_scales
        prog = (tuple(modes), tuple(ops))
        # a trailing 0 so that the tensor is never empty
        params = torch.tensor(index_params + member_params + [0], dtype=torch.int64, device=self.device)
        kernel, n_params, n_counters = fused.kernel(prog)
        assert params.numel() - 1 == n_params and len(checks) == n_counters
        root = u.members[-1].outputs[0]
        store = STORE[g.dtype(root)]
        # the kernel is compiled for these dtypes without looking at the tensors; _run_on_cuda keeps every graph tensor
        # that is not a constant at STORE[type]
        types = (store, torch.int64, torch.int64,
                 *[self.consts[n].dtype if n in self.consts else STORE[g.dtype(n)] for n, _ in leaves],
                 *[t.dtype for t in aux])
        return CompiledUnit(prog=prog, kernel=kernel, types=types, params=params, leaves=leaves, aux=aux,
                            shape=g.tensors[root].shape, numel=math.prod(g.tensors[root].shape), store=store,
                            checked_shapes=list({m.idx: (m, g.tensors[m.outputs[0]].shape)
                                                 for m, _ in checks}.values()),
                            counters_per_sample=[c._replace(per_sample=True) for _, c in checks],
                            counters_whole=[c for _, c in checks], parts_by_samples={})

    def _unit_sample_parts(self, compiled: CompiledUnit) -> int:
        """the unit's per-sample split; raises where the int64 emulator raises."""
        self._from_inputs = True  # fusion.plan admits only operators that read graph-input data
        found = {self._sample_parts(m, shape) for m, shape in compiled.checked_shapes}
        # fusion.plan gives every member the root's element count: a scalar (split 1) shares a unit only with
        # scalars or with other 1-element tensors, and those raise above when samples > 1
        assert len(found) <= 1
        return found.pop() if found else 1

    def _run_unit(self, u: Unit, env: dict[str, torch.Tensor]) -> torch.Tensor:
        """runs the unit as one kernel and returns the root's result."""
        from .kernels import fused
        # keyed by the members: with another keep set, the same root can have other members
        compiled = self._compiled_units.get(u.key)
        if compiled is None:
            compiled = self._compiled_units[u.key] = self._compile_unit(u)
        if not compiled.has_cached_parts(self.samples):  # it depends on shapes only
            compiled.write_cached_parts(self.samples, self._unit_sample_parts(compiled))
        parts = compiled.read_cached_parts(self.samples)
        cbase, cstride = len(self._counters) * self.samples, self.samples
        self._counters += compiled.counters_per_sample if parts == self.samples else compiled.counters_whole
        leaves = [env[n].contiguous() if shape is None else env[n].expand(shape).contiguous()
                  for n, shape in compiled.leaves]
        y = torch.empty(compiled.shape, dtype=compiled.store, device=self.device)
        fused.launch(compiled.kernel, compiled.types, y, self._cbuf, compiled.params, compiled.numel // parts, parts,
                     cbase, cstride, leaves, compiled.aux)
        return y

    def _run_op(self, o: OpInfo, env: dict[str, torch.Tensor], from_inputs: bool) -> tuple[list[torch.Tensor], How]:
        """runs one operator unfused."""
        if o.op == "RESHAPE":  # reads the constant target shape once and reuses it
            shape = self._reshape_shapes.get(o.idx)
            if shape is None and (o.inputs[1] in self.g.shapes or self.g.tensors[o.inputs[1]].data is not None):
                shape = self._reshape_shapes[o.idx] = tuple(int(v) for v in self._host(o.inputs[1]))
            if shape is not None:
                return [env[o.inputs[0]].reshape(shape)], How.METHOD
        ref = getattr(self, "op_" + o.op, None)
        if ref is None:
            raise NotImplementedError(f"op {o.op} (#{o.idx})")
        ins = [env[n] for n in o.inputs[:DATA_OPERANDS[o.op]]]
        self._from_inputs = from_inputs
        kernel = getattr(self, "k_" + o.op, None)
        outs = kernel(o, *ins) if kernel is not None else None
        if outs is not None:
            return outs, How.KERNEL
        if o.op in NO_WIDEN_OPS:
            return ref(o, *ins), How.METHOD
        outs = ref(o, *[t.to(torch.int64) for t in ins])
        # unsigned values do not fit the signed stored type: _narrow_to_storage raises
        unsigned = o.op == "RESCALE" and bool(o.attrs["OutputUnsigned"])
        return [self._narrow_to_storage(n, t, verify=unsigned) for n, t in zip(o.outputs, outs, strict=True)], \
            How.METHOD_I64

    def run(self, inputs: Sequence[np.ndarray | torch.Tensor],
            hook: Callable[[OpInfo, list[torch.Tensor]], None] | None = None, keep: Iterable[str] = (),
            samples: int = 1) -> list[np.ndarray]:
        if not self.on_cuda:
            return super().run(inputs, hook=hook, keep=keep, samples=samples)
        with torch.cuda.device(self.device):  # Triton launches on torch's current device
            return self._run_on_cuda(inputs, hook, keep, samples)

    def _run_on_cuda(self, inputs: Sequence[np.ndarray | torch.Tensor],
                     hook: Callable[[OpInfo, list[torch.Tensor]], None] | None, keep: Iterable[str],
                     samples: int) -> list[np.ndarray]:
        g = self.g
        if len(inputs) != len(g.inputs):
            raise ValueError(f"graph has {len(g.inputs)} inputs, got {len(inputs)}")
        self._reset_require_counts(samples)
        keep = set(keep) | set(g.outputs)
        # test faults can target any operator, so run unfused
        absorbed, steps = self._plan(keep, self.fuse and not self.faults)
        env = dict(self.consts)
        for name, x in zip(g.inputs, inputs, strict=True):
            t = torch.as_tensor(np.asarray(x)) if not isinstance(x, torch.Tensor) else x
            if tuple(t.shape) != g.tensors[name].shape:
                raise ValueError(f"input {name}: shape {tuple(t.shape)} != {g.tensors[name].shape}")
            # through int64, like the int64 emulator, whatever the numpy integer type
            if t.dtype != STORE[g.dtype(name)]:
                t = t.to(torch.int64)
            env[name] = self._narrow_to_storage(name, t.to(self.device), verify=True)
        how = self.how
        how.update(dict.fromkeys(absorbed, How.FUSED))
        for o, u, reads, drop in steps:
            if u is not None:  # the kernel wrote the declared shape and type: no _narrow_to_storage check
                y = self._run_unit(u, env)
                how[o.idx] = How.FUSED
                env[o.outputs[0]] = y
                if hook is not None:
                    hook(o, [y])
            else:
                outs, how[o.idx] = self._run_op(o, env, reads)
                if o.idx in self.faults:
                    outs = self.faults[o.idx](outs)
                for n, t in zip(o.outputs, outs, strict=True):
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
            counts = torch.stack([t for *_, t in self._pending]).cpu().tolist()
            records += [Violation(o.idx, o.op, req, msg, n)
                        for (o, req, msg, _), n in zip(self._pending, counts, strict=True)]
            self._pending = []
        self.violations += [v for v in records if (any(v.failed) if isinstance(v.failed, list) else v.failed)]
        self.violations.sort(key=lambda v: v.op_idx)
        return [self.to_numpy(n, env[n]) for n in g.outputs]

