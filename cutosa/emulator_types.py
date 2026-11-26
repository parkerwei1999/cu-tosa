from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple

import torch

from .fusion import Unit
from .int64_ref import OpInfo, Require, Violation


class How(StrEnum):
    KERNEL = "kernel"
    FUSED = "fused"
    METHOD = "method"  # the int64 emulator's op_<NAME> on the stored tensors: NO_WIDEN_OPS only move or pick values
    METHOD_I64 = "method_i64"  # the same on inputs widened to int64; the outputs are narrowed back


class PlanStep(NamedTuple):
    op: OpInfo
    unit: Unit | None
    reads_inputs: bool  # some input of op was computed from the graph inputs
    free_after: tuple[str, ...]


class RequireCounter(NamedTuple):
    op_idx: int
    op: str
    require: Require
    msg: str
    per_sample: bool  # False: one count for the whole tensor, which every sample's own run would compute

    def violation(self, failed: int | list[int]) -> Violation:
        return Violation(self.op_idx, self.op, self.require, self.msg, failed)


@dataclass
class CompiledUnit:
    prog: tuple
    kernel: Callable
    types: tuple[torch.dtype, ...]
    params: torch.Tensor
    leaves: list[tuple[str, tuple[int, ...] | None]]
    aux: list[torch.Tensor]
    shape: tuple[int, ...]
    numel: int
    store: torch.dtype
    checked_shapes: list[tuple[OpInfo, tuple[int, ...]]]
    counters_per_sample: list[RequireCounter]
    counters_whole: list[RequireCounter]
    cached_parts_by_samples: dict[int, int]
