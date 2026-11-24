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

With fusion (the default; `Emulator(fuse=False)` turns it off) the pointwise operators (RESCALE, ADD, SUB, MUL,
TABLE, MINIMUM, MAXIMUM, GREATER, GREATER_EQUAL, SELECT) that read data do not run one by one: `cutosa/fusion.py`
groups them into units, each a tree of such operators (RESHAPE allowed between them) whose inner results are read
once and have the root's element count, or a single operator when nothing can join it. A unit runs as one generated
kernel (`cutosa/kernels/fused.py`, one kernel per unit structure): the members are computed in 64-bit registers,
every member's REQUIREs are still counted (a TABLE slope outside INT16 included), and only the root's result is
stored. Without fusion the table above applies to every operator.

Kernels are launched through `cutosa/kernels/_launch.py`, which keeps the compiled kernel per element types and calls
Triton's launcher directly; it follows the Triton versions listed there and uses Triton's own path with any other.

`Emulator.how` records the path of every operator of the last run (`fused` for the members of a unit). On a CPU
device the whole run is the int64 emulator.

## Limits

These stop a run with an exception; none gives a wrong result.

- Values that do not fit the signed stored type: the output of a RESCALE with `output_unsigned` above the signed
  maximum (0..255 in an INT8 tensor), or a graph input given as such codes. The int64 emulator carries them; here
  they raise `NotImplementedError`. The same bits passed as int8 work.
- More than 65535 samples in one `run(samples=n)` (the samples are an axis of the kernels' launch grid).
- A kernel's own shape limits (MATMUL / CONV2D: K <= 65280, batch <= 65535, fewer than 2^31 elements per operand)
  are not limits of the emulator: such an operator takes the fallback.

## Tests (CUDA)

```
PYTHONPATH=. python tests/kernel_check.py name=graph.tosa ...     # each kernel against the int64 emulator's operator
PYTHONPATH=. python tests/graph_equiv.py name=graph.tosa ...      # whole graphs, bit for bit, with timing
tests/run_ut_vs_reference.sh <tosa_emu_op_check.py> graph.tosa ...  # operator instances against tosa_reference_model
PYTHONPATH=. python tests/fusion_cases.py                         # fusion on synthetic and random graphs
PYTHONPATH=. python tests/fusion_stats.py name=graph.tosa ...     # what the fusion plan removes
PYTHONPATH=. python tests/big_index_check.py                      # flat indices beyond 2^31 (5 GiB of GPU memory)
PYTHONPATH=. python tests/profile_ops.py name=graph.tosa ...      # time per operator and path
python proofs/limb_proof.py                                        # Z3: the limb identity and accumulator bounds of MATMUL
```

Needs torch (with triton), numpy and `tosa-tools`; `z3-solver` only for `proofs/`.

## Running the kit on cu-tosa without editing it

`tests/kit_overlay/sitecustomize.py` replaces the `Emulator` of the kit's `tosa_exec.tosa_emu` by `cutosa.Emulator`
as that module loads, so `--backends tosa_torch` runs on cu-tosa (worker processes included):

```
export PYTHONPATH=<cu-tosa>/tests/kit_overlay:$PWD/ovl:$PWD CUTOSA_KIT_OVERLAY=1      # in fine_grained_transformer_u85
python3 tosa_exec/integration_test.py --tosa-torch-device cuda --swin                 # TOSA_CPU vs cu-tosa, bit for bit
```

The line `[cu-tosa overlay] ...` on stderr confirms the replacement. This is a test aid until the kit has its own
`tosa_cu` backend.

## `cutosa/int64_ref.py`

A verbatim copy of `fine_grained_transformer_u85/tosa_exec/tosa_emu.py` of the MLEK-Transformer kit at commit 47dd6e5.
Keep it byte-identical to its source until the kit imports it from here; `cmp` the two files to check.
