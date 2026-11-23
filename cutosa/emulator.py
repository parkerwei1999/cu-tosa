"""cu-tosa emulator: the int64 emulator's semantics with every tensor stored at its TOSA type and fused GPU kernels.

Storage: INT8 -> int8, INT16 -> int16, INT32 -> int32, INT48 -> int64, BOOL -> int8 (0 / 1).
An operator runs one of three ways, chosen per operator instance:
  kernel       a k_<OP> method that reads and writes the stored types (RESCALE, MATMUL INT16 x INT16, ADD, SUB, MUL,
               TABLE, REDUCE_SUM, the compares, GATHER); it returns None for an attribute combination it does not cover
  as stored    the int64 emulator's op_<OP> applied to the stored tensors unchanged, for operators whose result does
               not depend on the element width (MINIMUM, MAXIMUM, SELECT, CLAMP, REDUCE_MAX and the shape operators)
  fallback     the int64 emulator's op_<OP> on tensors widened to int64, its result narrowed to the stored type
On a CPU device everything is the int64 emulator (the kernels are CUDA only).

Interface and results are those of int64_ref.Emulator: run(), violations, graph_status().
"""
import numpy as np
import torch

from .int64_ref import BITS, RANGE, ROUND_SINGLE, TosaGraphIR
from .int64_ref import Emulator as Int64Emulator

STORE = {"BOOL": torch.int8, "INT8": torch.int8, "INT16": torch.int16, "INT32": torch.int32, "INT48": torch.int64}
AS_STORED = frozenset(("MINIMUM", "MAXIMUM", "SELECT", "CLAMP", "REDUCE_MAX", "TRANSPOSE", "RESHAPE", "SLICE",
                       "CONCAT"))


class Emulator(Int64Emulator):
    def __init__(self, graph, device="cpu", check=True, mm_backend="intmm", verify_narrow=False):
        """verify_narrow: check that every fallback result fits its stored type before narrowing it (a test aid; one
        extra pass over each such tensor)."""
        super().__init__(graph, device=device, check=check, mm_backend=mm_backend)
        self.native = self.device.type == "cuda"
        self.verify_narrow = verify_narrow
        self.how = {}  # op index -> "kernel" / "stored" / "fallback" (filled by run)
        if self.native:
            for n, t in list(self.consts.items()):
                self.consts[n] = self._narrow(n, t, always_verify=True)

    # -- storage -----------------------------------------------------------------------------------------------------
    def _narrow(self, name, t, always_verify=False):
        store = STORE[self.g.dtype(name)]
        if t.dtype == store:
            return t
        if (always_verify or self.verify_narrow) and t.numel():
            lo, hi = RANGE[self.g.dtype(name)]
            mn, mx = (int(v) for v in torch.aminmax(t))
            if mn < lo or mx > hi:
                raise NotImplementedError(f"{name}: values [{mn}, {mx}] do not fit the stored {self.g.dtype(name)} "
                                          f"(an unsigned RESCALE output?)")
        return t.to(store)

    @staticmethod
    def _widen(t):
        return t if t is None or t.dtype == torch.int64 else t.to(torch.int64)

    def _counts(self, c):
        """a kernel's per-part REQUIRE counts in the form Int64Emulator._count_bad queues them."""
        if self.samples > 1:
            return c if c.numel() == self.samples else c.sum().expand(self.samples)
        return c.sum()

    def _parts(self, o, shape):
        """how many per-sample parts a checked tensor of this shape splits into (Int64Emulator._per_sample)."""
        if self.samples == 1 or len(shape) == 0 or not self._from_inputs:
            return 1
        if next((d for d in shape if d != 1), 1) % self.samples:
            raise ValueError(f"#{o.idx} {o.op}: checked tensor of shape {tuple(shape)} has no outermost dimension "
                             f"that splits into {self.samples} samples")
        return self.samples

    # -- kernels -----------------------------------------------------------------------------------------------------
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
            m = torch.from_numpy(mult).to(self.device)
            s = torch.from_numpy(shift).to(self.device)
        else:
            m, s = int(mult[0]), int(shift[0])
        parts = self._parts(o, x.shape)
        y, c0, c1 = rescale(x.contiguous(), STORE[out_dt], m, s, izp, ozp, *RANGE[out_dt], scale32, per_channel, parts)
        if self.check:
            first = "apply_scale_32 value outside [-2^(shift-1), 2^(shift-1))" if scale32 \
                else "apply_scale_16 result outside INT32"
            self._pending.append((o.idx, o.op, first, self._counts(c0)))
            self._pending.append((o.idx, o.op, "RESCALE result + output_zp outside INT32", self._counts(c1)))
        return [y]

    def k_MATMUL(self, o, A, B, _azp, _bzp):
        from .kernels.matmul import MAX_K, bmm_i16
        if self.g.dtype(o.inputs[0]) != "INT16" or self.g.dtype(o.inputs[1]) != "INT16" \
                or self.g.dtype(o.outputs[0]) != "INT48" or self._scalar(o.inputs[2]) or self._scalar(o.inputs[3]) \
                or A.shape[-1] > MAX_K or A.numel() == 0 or B.numel() == 0:
            return None
        # |sum| <= K * 2^30 < 2^47 for K <= MAX_K: the INT48 accumulator check of the int64 emulator cannot fail
        return [bmm_i16(A, B)]

    def _k_binary(self, o, code, a, b, what, shift=0):
        from .kernels.elementwise import binary
        out_dt = self.g.dtype(o.outputs[0])
        shape = torch.broadcast_shapes(a.shape, b.shape)
        if out_dt not in ("INT8", "INT16", "INT32") or a.numel() == 0 or b.numel() == 0:
            return None
        y, c = binary(code, a, b, STORE[out_dt], *RANGE[out_dt], shift=shift, samples=self._parts(o, shape))
        if self.check:
            self._pending.append((o.idx, o.op, f"{what} outside {out_dt}", self._counts(c)))
        return [y]

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
        if eff > 62:  # the register would leave int64
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
        y, c = reduce_sum(x, axis, STORE[out_dt], *RANGE[out_dt], samples=self._parts(o, shape))
        if self.check:
            self._pending.append((o.idx, o.op, f"REDUCE_SUM running sum outside {out_dt}", self._counts(c)))
        return [y]

    def k_GREATER(self, o, a, b):
        return [(a > b).to(torch.int8)]

    def k_GREATER_EQUAL(self, o, a, b):
        return [(a >= b).to(torch.int8)]

    def k_GATHER(self, o, values, idx):
        return self.op_GATHER(o, values, self._widen(idx))  # torch.gather wants an int64 index; values stay as stored

    # -- driver ------------------------------------------------------------------------------------------------------
    def run(self, inputs, hook=None, keep=(), samples=1):
        if not self.native:
            return super().run(inputs, hook=hook, keep=keep, samples=samples)
        g = self.g
        if len(inputs) != len(g.inputs):
            raise ValueError(f"graph has {len(g.inputs)} inputs, got {len(inputs)}")
        self.violations, self._pending, self.samples = [], [], samples
        from_inputs = set(g.inputs)
        env = dict(self.consts)
        for name, x in zip(g.inputs, inputs):
            t = torch.as_tensor(np.asarray(x)) if not isinstance(x, torch.Tensor) else x
            if tuple(t.shape) != g.tensors[name].shape:
                raise ValueError(f"input {name}: shape {tuple(t.shape)} != {g.tensors[name].shape}")
            env[name] = self._narrow(name, t.to(self.device), always_verify=True)
        keep = set(keep) | set(g.outputs)
        for o in g.ops:
            if o.op in ("CONST", "CONST_SHAPE"):
                continue
            ref = getattr(self, "op_" + o.op, None)
            if ref is None:
                raise NotImplementedError(f"op {o.op} (#{o.idx})")
            ins = [env.get(n) for n in o.inputs]
            self._from_inputs = any(n in from_inputs for n in o.inputs)
            if self._from_inputs:
                from_inputs.update(o.outputs)
            kernel = getattr(self, "k_" + o.op, None)
            outs = kernel(o, *ins) if kernel is not None else None
            how = "kernel"
            if outs is None and o.op in AS_STORED:
                outs, how = ref(o, *ins), "stored"
            elif outs is None:
                outs, how = ref(o, *[self._widen(t) for t in ins]), "fallback"
                unsigned = o.op == "RESCALE" and bool(o.attrs["OutputUnsigned"])  # 0..255 in an INT8: refused
                outs = [self._narrow(n, t, always_verify=unsigned) for n, t in zip(o.outputs, outs)]
            self.how[o.idx] = how
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
            for n in o.inputs:
                if self.last_use.get(n) == o.idx and n not in keep and n not in self.consts:
                    env.pop(n, None)
        self.env = {n: env[n] for n in keep if n in env}
        if self._pending:
            counts = torch.stack([c for *_, c in self._pending]).cpu().tolist()
            self.violations += [(i, op, msg, n) for (i, op, msg, _), n in zip(self._pending, counts) if any(n)] \
                if self.samples > 1 else [(i, op, msg, n) for (i, op, msg, _), n in zip(self._pending, counts) if n]
            self.violations.sort(key=lambda v: v[0])
            self._pending = []
        return [self.to_numpy(n, env[n]) for n in g.outputs]


__all__ = ["Emulator", "TosaGraphIR", "STORE", "BITS"]
