"""Stand-in for the kit's tosa_exec.tosa_emu so that a test written against it runs the cu-tosa emulator instead
(put tests/shim before the kit on PYTHONPATH)."""
from cutosa import RANGE, Emulator, TosaGraphIR, graph_status  # noqa: F401
