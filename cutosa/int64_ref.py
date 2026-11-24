"""Integer emulator of TOSA 1.0 INT-profile graphs on torch (CPU or CUDA); target = bit-exact with tosa_reference_model.

Data path: every tensor is a torch.int64 tensor holding values of its TOSA dtype (INT8 / INT16 / INT32 / INT48).
No floating point anywhere: wide math is int64, matmul / conv accumulate through torch._int_mm (int8 x int8 -> int32,
exact) on int8 limbs of the operands, so CPU and CUDA give identical bits by construction.

Semantics follow the TOSA 1.0 pseudo-code and, where the spec says REQUIRE / leaves overflow unpredictable, the
reference model's observable behaviour: an ADD / SUB / MUL (any shift) / REDUCE_SUM result outside INT32 wraps to its
low 32 bits, int32 conv / matmul accumulators wrap, int48 accumulators saturate. REQUIRE violations are recorded in
Emulator.violations (op index, op, message, element count); graph_status() turns them into the reference model's graph
status (an INT32 MUL with shift 0 is recorded but, as in the reference model, leaves the graph VALID). A run of a graph
lowered for a batch of independent samples (run(samples=n)) counts the elements per sample instead (a list of n
counts), so graph_status(violations, sample) is the status of one sample: an op computed from the graph inputs splits
its checked tensor along the outermost dimension other than 1, an op on constants only (a positional embedding, a bias
table) counts its elements for every sample, as each sample's own run computes it. The split assumes the batch is the
outermost factor of that dimension; check that before relying on it for a graph (a dimension that is not a multiple
of n raises).
Ops other than the ones with an op_<NAME> method below raise NotImplementedError.

usage (library):
    g = TosaGraphIR("graph.tosa"); emu = Emulator(g, device="cpu")
    outs = emu.run([np_input0, ...])          # list of numpy arrays, dtype of each graph output (INT48 -> int64)
    status = graph_status(emu.violations)     # "VALID" or "UNPREDICTABLE"
"""
import importlib
import inspect
from dataclasses import dataclass, field

import numpy as np
import torch
from tosa import Attribute as _attr_mod
from tosa import DType, Op
from tosa.TosaGraph import TosaGraph

OP_NAMES = {v: k for k, v in vars(Op.Op).items() if not k.startswith("_")}
DTYPE_NAMES = {v: k for k, v in vars(DType.DType).items() if not k.startswith("_")}
ATTR_NAMES = {v: k for k, v in vars(_attr_mod.Attribute).items() if not k.startswith("_")}

RANGE = {"BOOL": (0, 1), "INT8": (-(1 << 7), (1 << 7) - 1), "INT16": (-(1 << 15), (1 << 15) - 1),
         "INT32": (-(1 << 31), (1 << 31) - 1), "INT48": (-(1 << 47), (1 << 47) - 1)}
BITS = {"INT8": 8, "INT16": 16, "INT32": 32, "INT48": 48}
NP_DTYPE = {"BOOL": np.bool_, "INT8": np.int8, "INT16": np.int16, "INT32": np.int32, "INT48": np.int64}
ROUND_SINGLE, ROUND_INEXACT, ROUND_DOUBLE = 1, 2, 3  # tosa.RoundingMode
VALID, UNPREDICTABLE = "VALID", "UNPREDICTABLE"  # tosa_reference_model GraphStatus names without the TOSA_ prefix
# REQUIRE checks (message prefixes) that the reference model does not make: an INT32 MUL with shift 0 keeps the low 32
# bits of the product and the graph stays VALID there. Recorded, but they leave graph_status() VALID too.
UNCHECKED_BY_REFERENCE = ("MUL result (shift 0)",)


def graph_status(violations: list, sample: int | None = None) -> str:
    """The reference model's graph status for an emulator run's violations: UNPREDICTABLE when a REQUIRE check that the
    reference model also makes failed, VALID otherwise. sample: of a run over samples, the status of that sample."""
    return UNPREDICTABLE if any(not v[2].startswith(UNCHECKED_BY_REFERENCE) and (sample is None or v[3][sample])
                                for v in violations) else VALID


# ----------------------------------------------------------------------------------------------------------------------
# flatbuffer -> plain python IR
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class TensorInfo:
    name: str
    dtype: str
    shape: tuple
    data: np.ndarray = None  # int64 values (constants only)


@dataclass
class OpInfo:
    idx: int
    op: str
    inputs: list
    outputs: list
    attrs: dict = field(default_factory=dict)


def _decode_const(t, dtype, shape):
    raw = t.DataAsNumpy().tobytes()
    if dtype == "INT48":  # 6 little-endian bytes per element, sign-extended from bit 47
        b = np.frombuffer(raw, np.uint8).reshape(-1, 6).astype(np.int64)
        v = np.zeros(b.shape[0], np.int64)
        for i in range(6):
            v |= b[:, i] << (8 * i)
        v = np.where(v >= (1 << 47), v - (1 << 48), v)
    elif dtype == "BOOL":
        v = np.frombuffer(raw, np.uint8).astype(np.int64)
    else:
        v = np.frombuffer(raw, NP_DTYPE[dtype]).astype(np.int64)
    n = int(np.prod(shape)) if len(shape) else 1
    if v.size > n and n == 1 and (v == v[0]).all():
        # ExecuTorch 1.x writes CONV2D weight_zp as an OC-long all-equal vector under declared shape [1]:
        # keep the declared shape (value unambiguous); anything else is rejected below
        v = v[:1]
    if v.size != n:
        raise ValueError(f"const {t.Name().decode()}: {v.size} values for shape {shape}")
    return v.reshape(shape)


def _attr_values(o):
    """Generic read of an operator's attribute table -> {field: value}."""
    name = ATTR_NAMES.get(o.AttributeType(), "NONE")
    if name == "NONE" or o.Attribute() is None:
        return {}
    cls = getattr(importlib.import_module(f"tosa.{name}"), name)
    a = cls()
    tab = o.Attribute()
    a.Init(tab.Bytes, tab.Pos)
    out = {}
    for meth, fn in inspect.getmembers(cls, inspect.isfunction):
        if meth.startswith(("_", "Init", "GetRootAs")) or meth.endswith(("Length", "IsNone", "AsNumpy")) \
                or "BufferHasIdentifier" in meth:
            continue
        params = list(inspect.signature(fn).parameters)
        if params == ["self"]:
            out[meth] = getattr(a, meth)()
        elif params == ["self", "j"]:
            v = getattr(a, meth + "AsNumpy")()
            out[meth] = [int(x) for x in v] if isinstance(v, np.ndarray) else []
    return out


class TosaGraphIR:
    """Single-region / single-block TOSA 1.0 graph as python objects (tensors, shape constants, ops in file order)."""

    def __init__(self, src):
        buf = src if isinstance(src, (bytes, bytearray)) else open(src, "rb").read()
        self.raw = bytes(buf)
        g = TosaGraph.GetRootAsTosaGraph(self.raw, 0)
        if g.RegionsLength() != 1 or g.Regions(0).BlocksLength() != 1:
            raise NotImplementedError("only single-region single-block graphs (no COND_IF / WHILE_LOOP)")
        v = g.Version()
        self.version = (v._Major(), v._Minor(), v._Patch(), v._Draft())
        blk = g.Regions(0).Blocks(0)
        self.tensors = {}
        for i in range(blk.TensorsLength()):
            t = blk.Tensors(i)
            name = t.Name().decode()
            dt = DTYPE_NAMES.get(t.Type(), str(t.Type()))
            shape = tuple(int(s) for s in t.ShapeAsNumpy()) if t.ShapeLength() else ()
            if t.Variable():
                raise NotImplementedError(f"variable tensor {name}")
            data = _decode_const(t, dt, shape) if t.DataLength() else None
            self.tensors[name] = TensorInfo(name, dt, shape, data)
        self.shapes = {}  # CONST_SHAPE values: name -> tuple of python ints
        for i in range(blk.ShapesLength()):
            s = blk.Shapes(i)
            raw = s.DataAsNumpy().tobytes() if s.DataLength() else b""
            self.shapes[s.Name().decode()] = tuple(int(x) for x in np.frombuffer(raw, "<i8"))
        self.ops = []
        for i in range(blk.OperatorsLength()):
            o = blk.Operators(i)
            self.ops.append(OpInfo(i, OP_NAMES.get(o.Op(), str(o.Op())),
                                   [o.Inputs(j).decode() for j in range(o.InputsLength())],
                                   [o.Outputs(j).decode() for j in range(o.OutputsLength())], _attr_values(o)))
        self.inputs = [blk.Inputs(j).decode() for j in range(blk.InputsLength())]
        self.outputs = [blk.Outputs(j).decode() for j in range(blk.OutputsLength())]

    def dtype(self, name):
        return self.tensors[name].dtype if name in self.tensors else "SHAPE"


# ----------------------------------------------------------------------------------------------------------------------
# exact integer matmul through int8 limbs + torch._int_mm
# ----------------------------------------------------------------------------------------------------------------------
def _limbs(x, nbits):
    """x (int64, values in [-2^(nbits-1), 2^(nbits-1))) -> (int8 limbs low..high, offset c) with
    x == sum_i 256^i * limb_i + c exactly. 8-bit operands are their own single limb (c = 0)."""
    nl = (nbits + 7) // 8
    if nl == 1:
        return [x.to(torch.int8)], 0
    off = 1 << (8 * nl - 1)
    u = x + off  # in [0, 256^nl)
    limbs = [(((u >> (8 * i)) & 255) - 128).to(torch.int8) for i in range(nl)]
    c = 128 * sum(256 ** i for i in range(nl)) - off
    return limbs, c


# cuBLASLt's int8 GEMM kernel reads past the end of an operand (compute-sanitizer memcheck on RTX 4000 Ada,
# CUDA 13.0, torch 2.12: 4-byte reads up to 10 781 bytes after a [1504, 384] operand, about one 32-row tile). A read
# that crosses into an unmapped page faults ("an illegal memory access"), so every operand is followed by this many
# zero rows of its own width in the same allocation.
_CUDA_SLACK_ROWS = 256


def _cuda_operand(x, rows, cols):
    """x zero-padded to [rows, cols], contiguous, followed by _CUDA_SLACK_ROWS zero rows in the same allocation."""
    buf = torch.zeros(rows + _CUDA_SLACK_ROWS, cols, dtype=torch.int8, device=x.device)
    buf[:x.shape[0], :x.shape[1]] = x
    return buf[:rows]


# cuBLASLt's int8 GEMM returns CUBLAS_STATUS_NOT_SUPPORTED from M = 65536 rows on (RTX 4000 Ada, torch 2.12; 65504
# works): Swin-T's patch embedding and stage-1 MLP reach M = 200704 at batch 64. Rows are independent, so a larger M
# runs in blocks of this many rows.
_CUDA_MAX_ROWS = 32768


def _int_mm(a8, b8):
    """int8 [M,K] x int8 [K,N] -> int64 [M,N] via torch._int_mm (int32 accumulation, exact for K < 2^17)."""
    m, k = a8.shape
    n = b8.shape[1]
    if a8.is_cuda:  # cuBLASLt int8 GEMM: zero-pad M to a multiple of 32 (some GPUs reject others), K and N to 8
        kp, np_ = -(-k // 8) * 8, -(-n // 8) * 8
        b = _cuda_operand(b8, kp, np_)
        out = []
        for r in range(0, m, _CUDA_MAX_ROWS):
            ab = a8[r:r + _CUDA_MAX_ROWS]
            mp = -(-ab.shape[0] // 32) * 32
            out.append(torch._int_mm(_cuda_operand(ab, mp, kp), b)[:ab.shape[0], :n].to(torch.int64))
        return torch.cat(out) if len(out) != 1 else out[0]
    return torch._int_mm(a8.contiguous(), b8.contiguous()).to(torch.int64)


def imatmul(a, b, a_bits, b_bits, a_zp=0, b_zp=0, backend="intmm"):
    """Exact sum_k (a[m,k] - a_zp) * (b[k,n] - b_zp) as int64. a: [M,K], b: [K,N] int64 tensors whose values fit
    a_bits / b_bits signed. backend 'int64' = plain int64 torch.matmul (CPU only, used as an independent oracle)."""
    if a.shape[1] >= (1 << 17):
        raise NotImplementedError("K >= 2^17 would overflow the int32 _int_mm accumulator")
    if backend == "int64":
        return torch.matmul(a - a_zp, b - b_zp)
    la, ca = _limbs(a, a_bits)
    lb, cb = _limbs(b, b_bits)
    K = a.shape[1]
    acc = None
    for i, ai in enumerate(la):
        for j, bj in enumerate(lb):
            p = _int_mm(ai, bj)
            if i + j:
                p = p << (8 * (i + j))
            acc = p if acc is None else acc + p
    oa, ob = ca - a_zp, cb - b_zp  # (a - a_zp) = A' + oa with A' = sum 256^i limb_i = a - ca
    if ob:
        acc = acc + ob * (a.sum(1, keepdim=True) - K * ca)
    if oa:
        acc = acc + oa * (b.sum(0, keepdim=True) - K * cb)
    if oa and ob:
        acc = acc + K * oa * ob
    return acc


def ibmm(a, b, a_bits, b_bits, a_zp=0, b_zp=0):
    """imatmul over a batch: exact sum_k (a[n,m,k] - a_zp) * (b[n,k,p] - b_zp) as int64 for a: [N,M,K], b: [N,K,P]
    int64 tensors whose values fit a_bits / b_bits signed, one batched GEMM per pair of 8-bit limbs instead of one
    small GEMM per n (Swin's window attention: batch x windows x heads matrices, 1824 per image on Swin-T). A limb
    product is at most 2^14 in magnitude, so every partial sum over K <= 1024 stays below 2^24 and float32 is exact;
    float64 beyond that, or when TF32 may round the operands."""
    K = a.shape[-1]
    ft = torch.float32 if K <= 1024 and not torch.backends.cuda.matmul.allow_tf32 else torch.float64
    la, ca = _limbs(a, a_bits)
    lb, cb = _limbs(b, b_bits)
    acc = None
    for i, ai in enumerate(la):
        for j, bj in enumerate(lb):
            p = torch.bmm(ai.to(ft), bj.to(ft)).to(torch.int64)
            if i + j:
                p = p << (8 * (i + j))
            acc = p if acc is None else acc + p
    oa, ob = ca - a_zp, cb - b_zp  # as imatmul, the sums over K along the batched axes
    if ob:
        acc = acc + ob * (a.sum(2, keepdim=True) - K * ca)
    if oa:
        acc = acc + oa * (b.sum(1, keepdim=True) - K * cb)
    if oa and ob:
        acc = acc + K * oa * ob
    return acc


def _wrap(x, bits):
    """two's-complement wrap of int64 values to `bits` (what Eigen int32 arithmetic does on overflow)."""
    half = 1 << (bits - 1)
    return ((x + half) & ((1 << bits) - 1)) - half


# ----------------------------------------------------------------------------------------------------------------------
# emulator
# ----------------------------------------------------------------------------------------------------------------------
class Emulator:
    def __init__(self, graph, device="cpu", check=True, mm_backend="intmm"):
        self.g = graph if isinstance(graph, TosaGraphIR) else TosaGraphIR(graph)
        self.device = torch.device(device)
        self.check = check
        self.mm_backend = mm_backend
        self.consts = {n: torch.from_numpy(t.data).to(self.device) for n, t in self.g.tensors.items()
                       if t.data is not None}
        self.violations = []
        self.samples = 1  # run(samples=n): REQUIRE counts per sample
        self._from_inputs = True  # run(): the current op reads a tensor computed from the graph inputs
        self.faults = {}  # op index -> f(list of outputs) -> outputs; test hook that perturbs an op's outputs
        # last use of every tensor (to free intermediates early)
        self.last_use = {}
        for o in self.g.ops:
            for n in o.inputs:
                self.last_use[n] = o.idx

    # -- helpers ---------------------------------------------------------------------------------------------------
    def _per_sample(self, o, bad):
        """bad (bool tensor) -> int64 counts per sample. An op computed from the graph inputs splits its outermost
        dimension other than 1 (the lowering adds leading 1s, e.g. (1, B, N, C)) into self.samples equal parts,
        sample-major; an op on constants only, or a scalar check (one that does not depend on the data), counts its
        elements once for every sample."""
        if bad.dim() == 0 or not self._from_inputs:
            return bad.sum().to(torch.int64).expand(self.samples)
        if next((d for d in bad.shape if d != 1), 1) % self.samples:
            raise ValueError(f"#{o.idx} {o.op}: checked tensor of shape {tuple(bad.shape)} has no outermost dimension "
                             f"that splits into {self.samples} samples")
        return bad.reshape(self.samples, -1).sum(1)

    def _require(self, o, ok, msg):
        """record a TOSA REQUIRE violation (ok = bool tensor or python bool)."""
        if not self.check:
            return
        if self.samples > 1:
            counts = self._per_sample(o, ~ok) if isinstance(ok, torch.Tensor) else torch.full((self.samples,), int(not ok))
            if counts.any():
                self.violations.append((o.idx, o.op, msg, counts.tolist()))
        elif isinstance(ok, torch.Tensor):
            bad = int((~ok).sum())
            if bad:
                self.violations.append((o.idx, o.op, msg, bad))
        elif not ok:
            self.violations.append((o.idx, o.op, msg, 1))

    def _count_bad(self, o, x, lo, hi, msg):
        """record a REQUIRE violation for the elements of x outside [lo, hi] (scalars, or tensors broadcastable to x).
        CUDA: the count stays on the device until the end of run() (no per-op host sync). CPU: scalar bounds take a
        single min/max reduction on the common (all-valid) path."""
        if not self.check or x.numel() == 0:
            return
        if self.samples > 1:
            counts = self._per_sample(o, (x < lo) | (x > hi))
            if x.is_cuda:
                self._pending.append((o.idx, o.op, msg, counts))
            elif counts.any():
                self.violations.append((o.idx, o.op, msg, counts.tolist()))
            return
        if x.is_cuda:
            self._pending.append((o.idx, o.op, msg, ((x < lo) | (x > hi)).sum()))
            return
        if not isinstance(lo, torch.Tensor) and not isinstance(hi, torch.Tensor):
            mn, mx = torch.aminmax(x)
            if int(mn) >= lo and int(mx) <= hi:
                return
        bad = int(((x < lo) | (x > hi)).sum())
        if bad:
            self.violations.append((o.idx, o.op, msg, bad))

    def _in_range(self, o, x, dt, what):
        lo, hi = RANGE[dt]
        self._count_bad(o, x, lo, hi, f"{what} outside {dt}")

    def _scalar(self, name):
        v = self.g.tensors[name].data
        if v is None or v.size != 1:
            raise NotImplementedError(f"{name}: expected a constant scalar (zero point / shift)")
        return int(v.reshape(-1)[0])

    def _host(self, name):
        """constant operand on host as numpy int64 (shift / multiplier / zero points / shapes)."""
        if name in self.g.shapes:
            return np.array(self.g.shapes[name], np.int64)
        v = self.g.tensors[name].data
        if v is None:
            raise NotImplementedError(f"{name}: non-constant operand where a constant is required")
        return v

    # -- driver ----------------------------------------------------------------------------------------------------
    def run(self, inputs, hook=None, keep=(), samples=1):
        """inputs: list (graph input order) of numpy / torch integer arrays. hook(op, [out tensors]) is called after
        every non-constant op. keep: tensor names to retain in self.env after the run. samples: the independent samples
        of a graph lowered at a batch (its leading dimension); > 1 counts REQUIRE violations per sample. Returns numpy
        outputs."""
        g = self.g
        if len(inputs) != len(g.inputs):
            raise ValueError(f"graph has {len(g.inputs)} inputs, got {len(inputs)}")
        self.violations, self._pending, self.samples = [], [], samples
        from_inputs = set(g.inputs)  # tensors computed from the graph inputs; the others come from constants only
        env = dict(self.consts)
        for name, x in zip(g.inputs, inputs):
            t = torch.as_tensor(np.asarray(x)) if not isinstance(x, torch.Tensor) else x
            t = t.to(device=self.device, dtype=torch.int64)
            if tuple(t.shape) != g.tensors[name].shape:
                raise ValueError(f"input {name}: shape {tuple(t.shape)} != {g.tensors[name].shape}")
            env[name] = t
        keep = set(keep) | set(g.outputs)
        for o in g.ops:
            if o.op in ("CONST", "CONST_SHAPE"):
                continue
            fn = getattr(self, "op_" + o.op, None)
            if fn is None:
                raise NotImplementedError(f"op {o.op} (#{o.idx})")
            ins = [env.get(n) for n in o.inputs]
            self._from_inputs = any(n in from_inputs for n in o.inputs)
            if self._from_inputs:
                from_inputs.update(o.outputs)
            outs = fn(o, *ins)
            if o.idx in self.faults:
                outs = self.faults[o.idx](outs)
            for n, t in zip(o.outputs, outs):
                want = g.tensors[n].shape
                if tuple(t.shape) != want:
                    raise RuntimeError(f"#{o.idx} {o.op} -> {n}: shape {tuple(t.shape)} != declared {want}")
                env[n] = t
            if hook is not None:
                hook(o, outs)
            for n in o.inputs:  # free intermediates after their last use
                if self.last_use.get(n) == o.idx and n not in keep and n not in self.consts:
                    env.pop(n, None)
        self.env = {n: env[n] for n in keep if n in env}
        if self._pending:  # resolve device-side REQUIRE counts with one transfer, keep op order
            counts = torch.stack([c for *_, c in self._pending]).cpu().tolist()
            self.violations += [(i, op, msg, n) for (i, op, msg, _), n in zip(self._pending, counts) if any(n)] \
                if self.samples > 1 else [(i, op, msg, n) for (i, op, msg, _), n in zip(self._pending, counts) if n]
            self.violations.sort(key=lambda v: v[0])
            self._pending = []
        return [self.to_numpy(n, env[n]) for n in g.outputs]

    def to_numpy(self, name, t):
        # Native quantize/dequantize glue expects the reference runner's C-order output.
        return t.cpu().numpy().astype(NP_DTYPE[self.g.dtype(name)], order="C")

    # -- elementwise -----------------------------------------------------------------------------------------------
    def _checked_wrap(self, o, r, what):
        """REQUIRE r within the output type, then wrap it to the type's low bits (the reference model's result)."""
        dt = self.g.dtype(o.outputs[0])
        self._in_range(o, r, dt, what)
        return _wrap(r, BITS[dt])

    def op_ADD(self, o, a, b):
        return [self._checked_wrap(o, a + b, "ADD result")]

    def op_SUB(self, o, a, b):
        return [self._checked_wrap(o, a - b, "SUB result")]

    def op_MINIMUM(self, o, a, b):
        return [torch.minimum(a, b)]

    def op_MAXIMUM(self, o, a, b):
        return [torch.maximum(a, b)]

    def op_GREATER(self, o, a, b):
        return [(a > b).to(torch.int64)]

    def op_GREATER_EQUAL(self, o, a, b):
        return [(a >= b).to(torch.int64)]

    def op_SELECT(self, o, condition, a, b):
        return [torch.where(condition.to(torch.bool), a, b)]

    def op_CLAMP(self, o, x):
        dt = self.g.dtype(o.inputs[0])
        if dt not in ("INT8", "INT16"):
            raise NotImplementedError(f"CLAMP input dtype {dt}")
        lo = int.from_bytes(bytes(o.attrs["MinVal"]), "little", signed=True)
        hi = int.from_bytes(bytes(o.attrs["MaxVal"]), "little", signed=True)
        if lo > hi:
            raise ValueError("CLAMP maximum is smaller than minimum")
        return [x.clamp(lo, hi)]

    def op_MUL(self, o, a, b, _shift):
        shift = self._scalar(o.inputs[2])
        r = a * b  # |a*b| <= 2^62 for int32 operands: exact in int64
        if self.g.dtype(o.inputs[0]) == "INT32":
            if shift > 0:
                r += 1 << (shift - 1)
                r >>= shift
        elif shift != 0:
            self._require(o, False, "MUL shift != 0 for int8/int16 inputs")
        return [self._checked_wrap(o, r, f"MUL result (shift {shift})")]

    def op_TABLE(self, o, x, table):
        dt = self.g.dtype(o.inputs[0])
        table = table.reshape(-1)
        if dt == "INT8":
            return [table[x + 128]]
        if dt != "INT16":
            raise NotImplementedError(f"TABLE input {dt}")
        tab_slope = table[1:] - table[:-1]  # 512 slopes of the 513-entry table
        idx = x >> 7  # 9-bit index, 7-bit fraction (TOSA apply_lookup_s)
        idx += 256
        frac = x & 0x7F
        host_tab = self.g.tensors[o.inputs[1]].data
        host_slope = None if host_tab is None else np.diff(host_tab.reshape(-1))
        if self.check and (host_slope is None or ((host_slope < RANGE["INT16"][0]) |
                                                  (host_slope > RANGE["INT16"][1])).any()):
            bad = (tab_slope < RANGE["INT16"][0]) | (tab_slope > RANGE["INT16"][1])
            self._count_bad(o, bad[idx].to(torch.int64), 0, 0, "TABLE slope outside INT16")
        out = table[idx] * 128  # a new tensor: with a rank-0 x, table[idx] is a view of the table constant
        out += tab_slope[idx] * frac
        return [out]

    def op_RESCALE(self, o, x, _m, _s, _izp, _ozp):
        a = o.attrs
        in_dt, out_dt = self.g.dtype(o.inputs[0]), self.g.dtype(o.outputs[0])
        mult, shift = self._host(o.inputs[1]).reshape(-1), self._host(o.inputs[2]).reshape(-1)
        izp, ozp = self._scalar(o.inputs[3]), self._scalar(o.inputs[4])
        scale32, mode = bool(a["Scale32"]), a["RoundingMode"]
        if a["InputUnsigned"]:
            mask = (1 << BITS[in_dt]) - 1
            x, izp = x & mask, izp & mask
        v = x - izp if izp else x
        if a["PerChannel"]:  # channel = last dimension
            if mult.size != x.shape[-1]:
                raise RuntimeError(f"per-channel RESCALE: {mult.size} multipliers for last dim {x.shape[-1]}")
            m = torch.from_numpy(mult).to(self.device)
            s = torch.from_numpy(shift).to(self.device)
            half = torch.ones((), dtype=torch.int64, device=self.device) << (s - 1)
        else:
            m, s = int(mult[0]), int(shift[0])
            half = 1 << (s - 1)
        self._require(o, bool((mult >= 0).all()), "RESCALE multiplier < 0")
        self._require(o, bool(((shift >= 2) & (shift <= 62)).all()), "RESCALE shift outside [2, 62]")
        r = v * m  # |v| < 2^48, |m| < 2^31 (and |v| < 2^32 when scale32): exact in int64
        if scale32:
            self._count_bad(o, v, -half, half - 1, "apply_scale_32 value outside [-2^(shift-1), 2^(shift-1))")
            if mode == ROUND_DOUBLE:
                adj = torch.where(v >= 0, 1 << 30, -(1 << 30))
                r += adj * (s > 31) if isinstance(s, torch.Tensor) else (adj if s > 31 else 0)
            elif mode != ROUND_SINGLE:
                raise NotImplementedError(f"RESCALE rounding_mode {mode}")
            r += half
            r >>= s
        else:  # apply_scale_16 (value may be int48)
            r += half
            r >>= s
            self._in_range(o, r, "INT32", "apply_scale_16 result")
        if a["OutputUnsigned"]:
            r += ozp & ((1 << BITS[out_dt]) - 1)
            return [r.clamp_(0, (1 << BITS[out_dt]) - 1)]
        if ozp:
            r += ozp
        self._in_range(o, r, "INT32", "RESCALE result + output_zp")
        lo, hi = RANGE[out_dt]
        return [r.clamp_(lo, hi)]

    # -- reductions ------------------------------------------------------------------------------------------------
    def op_REDUCE_SUM(self, o, x):
        """REQUIRE as TOSA 1.0 states it (apply_add_s per element: every running sum in index order along the axis,
        not only the total, must fit the output type); the total wraps to the type's low bits like the reference
        model's int32 accumulation."""
        axis, dt = o.attrs["Axis"], self.g.dtype(o.outputs[0])
        if self.check:
            lo, hi = RANGE[dt]
            run = x.cumsum(dim=axis)
            bad = ((run < lo) | (run > hi)).any(dim=axis, keepdim=True)
            self._count_bad(o, bad.to(torch.int64), 0, 0, f"REDUCE_SUM running sum outside {dt}")
        return [_wrap(x.sum(dim=axis, keepdim=True), BITS[dt])]

    def op_REDUCE_MAX(self, o, x):
        return [x.amax(dim=o.attrs["Axis"], keepdim=True)]

    # -- tensor ops ------------------------------------------------------------------------------------------------
    def _acc_out(self, o, acc):
        dt = self.g.dtype(o.outputs[0])
        if dt == "INT32":  # reference model: Eigen int32 accumulation -> wraps on overflow
            self._in_range(o, acc, "INT32", "int32 accumulator overflow (wrapped)")
            return _wrap(acc, 32)
        if dt == "INT48":  # reference model: int64 accumulation, then saturate to int48
            lo, hi = RANGE["INT48"]
            self._in_range(o, acc, "INT48", "int48 accumulator (saturated)")
            return acc.clamp(lo, hi)
        raise NotImplementedError(f"{o.op} output {dt}")

    def op_MATMUL(self, o, A, B, _azp, _bzp):
        azp, bzp = self._scalar(o.inputs[2]), self._scalar(o.inputs[3])
        abits, bbits = BITS[self.g.dtype(o.inputs[0])], BITS[self.g.dtype(o.inputs[1])]
        if self.mm_backend == "int64":  # the independent oracle keeps one plain int64 matmul per batch entry
            out = torch.stack([imatmul(A[n], B[n], abits, bbits, azp, bzp, "int64") for n in range(A.shape[0])])
        else:
            out = ibmm(A, B, abits, bbits, azp, bzp)
        return [self._acc_out(o, out)]

    def op_CONV2D(self, o, x, w, bias, _izp, _wzp):
        a = o.attrs
        izp, wzp = self._scalar(o.inputs[3]), self._scalar(o.inputs[4])
        pt, pb, pl, pr = a["Pad"]
        sy, sx = a["Stride"]
        dy, dx = a["Dilation"]
        N, IH, IW, IC = x.shape
        OC, KH, KW, _ = w.shape
        OH = (IH - 1 + pt + pb - (KH - 1) * dy) // sy + 1
        OW = (IW - 1 + pl + pr - (KW - 1) * dx) // sx + 1
        if KH == KW == 1 and sy == sx == 1 and not any(a["Pad"]):
            cols = x.reshape(N * IH * IW, IC)
        else:
            if any(a["Pad"]):  # pad with input_zp: (zp - zp) == 0 == skipped tap
                xp = torch.full((N, IH + pt + pb, IW + pl + pr, IC), izp, dtype=torch.int64, device=x.device)
                xp[:, pt:pt + IH, pl:pl + IW, :] = x
            else:
                xp = x
            taps = [xp[:, ky * dy: ky * dy + (OH - 1) * sy + 1: sy, kx * dx: kx * dx + (OW - 1) * sx + 1: sx, :]
                    for ky in range(KH) for kx in range(KW)]
            cols = torch.stack(taps, dim=3).reshape(N * OH * OW, KH * KW * IC)
        wm = w.reshape(OC, KH * KW * IC).t()
        acc = imatmul(cols, wm, BITS[self.g.dtype(o.inputs[0])], BITS[self.g.dtype(o.inputs[1])], izp, wzp,
                      self.mm_backend)
        acc = acc + bias.reshape(1, -1)
        return [self._acc_out(o, acc.reshape(N, OH, OW, OC))]

    def op_TRANSPOSE(self, o, x):
        return [x.permute(*o.attrs["Perms"])]

    def op_RESHAPE(self, o, x, _shape):
        return [x.reshape(tuple(int(v) for v in self._host(o.inputs[1])))]

    def op_SLICE(self, o, x, _start, _size):
        start, size = self._host(o.inputs[1]), self._host(o.inputs[2])
        return [x[tuple(slice(int(b), int(b) + int(n)) for b, n in zip(start, size))]]

    def op_CONCAT(self, o, *xs):
        return [torch.cat(xs, dim=o.attrs["Axis"])]

    def op_GATHER(self, o, values, idx):
        N, K, C = values.shape
        self._count_bad(o, idx, 0, K - 1, "GATHER index outside [0, K)")
        idx = idx.clamp(0, K - 1)
        return [torch.gather(values, 1, idx.unsqueeze(-1).expand(N, idx.shape[1], C))]
