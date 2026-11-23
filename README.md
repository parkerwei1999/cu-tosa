# cu-tosa

Bit-exact GPU emulator of TOSA 1.0 integer graphs (the `.tosa` flatbuffers ExecuTorch's Arm backend lowers to).
Ground truth is Arm's `tosa_reference_model`; `cutosa.Int64Emulator` (a torch int64 emulator, CPU or CUDA) is the
in-repo oracle and the fallback for operators that have no kernel yet.

```python
from cutosa import Emulator, TosaGraphIR, graph_status
emu = Emulator(TosaGraphIR("graph.tosa"), device="cuda")
outs = emu.run([np_input0, ...])          # numpy arrays at the dtype of each graph output (INT48 -> int64)
status = graph_status(emu.violations)     # "VALID" or "UNPREDICTABLE", as the reference model reports
```

## How an operator runs

Tensors are stored at their TOSA type (INT8 / INT16 / INT32 natively, INT48 in int64, BOOL in int8).

| path | operators | what happens |
|---|---|---|
| kernel | RESCALE (single rounding, signed, scalar or per channel), MATMUL INT16 x INT16, CONV2D 1x1 (stride 1, no padding), ADD, SUB, MUL, TABLE, REDUCE_SUM, GREATER, GREATER_EQUAL, GATHER | one fused Triton kernel reads and writes the stored types |
| as stored | MINIMUM, MAXIMUM, SELECT, CLAMP, REDUCE_MAX, TRANSPOSE, RESHAPE, SLICE, CONCAT | the int64 emulator's method on the stored tensors (the result does not depend on the width) |
| fallback | other CONV2D, MATMUL of other types, and any attribute combination a kernel declines (double rounding, unsigned RESCALE, a TABLE whose slope REQUIRE can fail) | widen to int64, run the int64 emulator's method, narrow |

`Emulator.how` records the path of every operator of the last run. On a CPU device the whole run is the int64 emulator.

## Tests (CUDA)

```
PYTHONPATH=. python tests/kernel_check.py name=graph.tosa ...     # each kernel against the int64 emulator's operator
PYTHONPATH=. python tests/graph_equiv.py name=graph.tosa ...      # whole graphs, bit for bit, with timing
tests/run_ut_vs_reference.sh <tosa_emu_op_check.py> graph.tosa ...  # operator instances against tosa_reference_model
python proofs/limb_proof.py                                        # Z3: the limb identity and accumulator bounds of MATMUL
```

Needs torch (with triton), numpy and `tosa-tools`; `z3-solver` only for `proofs/`.

## `cutosa/int64_ref.py`

A verbatim copy of `fine_grained_transformer_u85/tosa_exec/tosa_emu.py` of the MLEK-Transformer kit at commit af62dd1.
Keep it byte-identical to its source until the kit imports it from here; `cmp` the two files to check.
