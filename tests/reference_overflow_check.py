"""cu-tosa (CUDA) against the TOSA reference model on tiny INT32 overflow graphs: ADD / SUB / MUL, REDUCE_SUM running
sums and totals, and overflowing results that feed RESCALE, REDUCE_SUM and ADD; every output bit and the graph status.
usage: PYTHONPATH=. python tests/reference_overflow_check.py"""
import sys

import numpy as np
import tosa_reference_model as reference_model
import tosa_serializer as ts

from cutosa import Emulator, TosaGraphIR, graph_status

I31 = 1 << 31


class _Graph:
    """A TOSA 1.0 graph under construction; tensors are INT32 unless a dtype is given."""

    def __init__(self) -> None:
        self.s = ts.TosaSerializer("", targetMajor=1, targetMinor=0, targetPatch=0, targetDraft=False)
        self.block = self.s.currRegion.currBasicBlock

    def inp(self, name, shape, dtype=ts.DType.INT32):
        self.s.addInputTensor(ts.TosaSerializerTensor(name, list(shape), dtype, data=None))

    def tensor(self, name, shape, dtype=ts.DType.INT32):
        self.block.addTensor(name, list(shape), dtype)

    def const(self, name, shape, dtype, values):
        self.s.addConst(list(shape), dtype, values, name=name)

    def op(self, kind, inputs, outputs, attr_name, *attr_args, **attr_kw):
        attr = ts.TosaSerializerAttribute()
        getattr(attr, attr_name)(*attr_args, **attr_kw)
        self.s.addOperator(getattr(ts.Op, kind), inputs, outputs, attr)

    def out(self, name):
        self.s.addOutputTensor(self.block.tensors[name])


def _graph(build) -> bytes:
    g = _Graph()
    build(g)
    return g.s.serialize()


def _binary(kind):
    def build(g):
        g.inp("a", [4])
        g.inp("b", [4])
        g.tensor("c", [4])
        g.op(kind, ["a", "b"], ["c"], kind.title() + "Attribute")
        g.out("c")
    return build


def _mul(shift, dtype=ts.DType.INT32):
    def build(g):
        g.inp("a", [4], dtype)
        g.inp("b", [4], dtype)
        g.tensor("c", [4])
        g.const("shift", [1], ts.DType.INT8, [shift])
        g.op("MUL", ["a", "b", "shift"], ["c"], "MulAttribute")
        g.out("c")
    return build


def _reduce_sum(shape, axis):
    def build(g):
        g.inp("x", shape)
        g.tensor("s", [1 if i == axis else d for i, d in enumerate(shape)])
        g.op("REDUCE_SUM", ["x"], ["s"], "ReduceSumAttribute", axis)
        g.out("s")
    return build


def _add_then_rescale(g):
    """ADD (overflowing) -> RESCALE to INT16 at scale 2^-16: the RESCALE reads the wrapped sum."""
    g.inp("a", [4])
    g.inp("b", [4])
    g.tensor("c", [4])
    g.tensor("y", [4], ts.DType.INT16)
    g.op("ADD", ["a", "b"], ["c"], "AddAttribute")
    g.const("m", [1], ts.DType.INT32, [1 << 30])
    g.const("sh", [1], ts.DType.INT8, [46])
    g.const("izp", [1], ts.DType.INT32, [0])
    g.const("ozp", [1], ts.DType.INT16, [0])
    g.op("RESCALE", ["c", "m", "sh", "izp", "ozp"], ["y"], "RescaleAttribute", scale32=True,
         rounding_mode=ts.RoundingMode.SINGLE_ROUND, per_channel=False, input_unsigned=False, output_unsigned=False)
    g.out("y")


def _add_then_sum(g):
    """ADD (overflowing) -> REDUCE_SUM over the wrapped sums."""
    g.inp("a", [1, 4])
    g.inp("b", [1, 4])
    g.tensor("c", [1, 4])
    g.tensor("s", [1, 1])
    g.op("ADD", ["a", "b"], ["c"], "AddAttribute")
    g.op("REDUCE_SUM", ["c"], ["s"], "ReduceSumAttribute", 1)
    g.out("s")


def _mul0_then_add(g):
    """MUL with shift 0 (overflow kept as the low 32 bits, VALID) -> ADD 0: the ADD sees the wrapped product."""
    g.inp("a", [4])
    g.inp("b", [4])
    g.inp("z", [4])
    g.tensor("p", [4])
    g.tensor("c", [4])
    g.const("shift", [1], ts.DType.INT8, [0])
    g.op("MUL", ["a", "b", "shift"], ["p"], "MulAttribute")
    g.op("ADD", ["p", "z"], ["c"], "AddAttribute")
    g.out("c")


def _i32(*v):
    return np.array(v, dtype=np.int32)


CASES = {  # name -> (graph builder, inputs)
    "ADD": (_binary("ADD"), [_i32(I31 - 1, -I31, 5, 1 << 30), _i32(1, -1, 7, 1 << 30)]),
    "SUB": (_binary("SUB"), [_i32(-I31, I31 - 1, 3, -(1 << 30)), _i32(1, -1, 2, 1 << 30)]),
    "MUL shift 0": (_mul(0), [_i32(65537, -65537, 3, 1048577), _i32(65536, 65537, 4, 4097)]),
    "MUL shift 2": (_mul(2), [_i32(1048576, -1048576, 3, 7), _i32(1048577, 1048576, 5, -9)]),
    "MUL int16 shift 0": (_mul(0, ts.DType.INT16), [np.array([32767, -32768, 3, -7], np.int16),
                                                     np.array([32767, -32768, 4, 9], np.int16)]),
    "REDUCE_SUM running sum": (_reduce_sum([1, 3], 1), [_i32(I31 - 1, 1, -1).reshape(1, 3)]),
    "REDUCE_SUM total, axis 1": (_reduce_sum([2, 2], 1), [_i32(I31 - 1, I31 - 1, 5, 6).reshape(2, 2)]),
    "REDUCE_SUM total, axis 0": (_reduce_sum([2, 2], 0), [_i32(-I31, 3, -1, 4).reshape(2, 2)]),
    "ADD -> RESCALE": (_add_then_rescale, [_i32(I31 - 1, -I31, 100, 1 << 30), _i32(2, -2, 200, 1 << 30)]),
    "ADD -> REDUCE_SUM": (_add_then_sum, [_i32(I31 - 1, 1, 2, 3).reshape(1, 4), _i32(1, 1, 1, 1).reshape(1, 4)]),
    "MUL shift 0 -> ADD": (_mul0_then_add, [_i32(65537, -65537, 3, 1048577), _i32(65536, 65537, 4, 4097),
                                            _i32(0, 0, 0, 0)]),
}


def run() -> bool:
    """Prints one row per case; True iff every case agrees."""
    ok = True
    for name, (build, inputs) in CASES.items():
        binary = _graph(build)
        ref_outs, st = reference_model.run(binary, inputs, verbosity="NONE", initialize_variable_tensor_from_numpy=True)
        ref_status = st.name.removeprefix("TOSA_")
        emu = Emulator(TosaGraphIR(binary), device="cuda")
        emu_outs = emu.run(inputs)
        emu_status = graph_status(emu.violations)
        same = len(ref_outs) == len(emu_outs) and all(
            r.dtype == e.dtype and r.shape == e.shape and np.array_equal(r, e)
            for r, e in zip(ref_outs, emu_outs, strict=True))
        row_ok = same and ref_status == emu_status
        ok &= row_ok
        print(f"{'PASS' if row_ok else 'FAIL'}  {name:26s} status ref={ref_status:13s} emu={emu_status:13s} "
              f"bits {'equal' if same else 'DIFFER'}  ref={[o.tolist() for o in ref_outs]} "
              f"emu={[o.tolist() for o in emu_outs]}  checks={[v.msg for v in emu.violations]}", flush=True)
    return ok


if __name__ == "__main__":
    passed = run()
    print(f"RESULT: {'PASS' if passed else 'FAIL'} ({len(CASES)} cases)")
    sys.exit(0 if passed else 1)
