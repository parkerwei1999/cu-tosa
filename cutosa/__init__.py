"""cu-tosa: bit-exact GPU emulator of TOSA 1.0 integer graphs.

Emulator      narrow storage + fused kernels (cutosa.emulator)
Int64Emulator the int64 torch emulator it falls back to and is tested against (cutosa.int64_ref)
"""
from .emulator import STORE, Emulator
from .int64_ref import RANGE, UNPREDICTABLE, VALID, TosaGraphIR, graph_status
from .int64_ref import Emulator as Int64Emulator

__all__ = ["Emulator", "Int64Emulator", "TosaGraphIR", "graph_status", "RANGE", "STORE", "VALID", "UNPREDICTABLE"]
