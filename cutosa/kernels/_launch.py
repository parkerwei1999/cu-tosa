"""Kernel launch without Triton's per-call lookup.

`fn[grid](*args)` derives a specialization from the argument *values* on every call (an integer equal to 1 becomes a
constant, divisibility by 16, i32 or i64 by magnitude, pointer alignment), hashes it and looks the compiled kernel up:
about 15 us a launch, as much as the launch itself for the small tensors of a decoder step.

`jit` declares a kernel whose compiled form depends only on the element types of its tensor arguments and on its
constexprs: no parameter is specialized on its value, and every scalar parameter is annotated tl.int64. The
parameters come in the order: tensors (named *_ptr), scalars, constexprs. `Kernel.launch` keeps the compiled kernel
per (device, tensor element types, constexprs) and calls Triton's launcher directly. The direct call follows
triton/runtime/jit.py (JITFunction.run) of the versions in DIRECT_VERSIONS; with another version, with a launch hook
installed, or with CUTOSA_DIRECT_LAUNCH=0, every launch goes through `fn[grid]`.
"""
import inspect
import os

import triton
import triton.language as tl
from triton import knobs
from triton.runtime import driver

DIRECT_VERSIONS = ("3.7.0",)
_ENTER, _EXIT = knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook
DIRECT = triton.__version__ in DIRECT_VERSIONS and os.environ.get("CUTOSA_DIRECT_LAUNCH", "1") != "0"


class Kernel:
    def __init__(self, fn):
        params = list(inspect.signature(fn).parameters.values())
        kinds = ["c" if p.annotation is tl.constexpr else "p" if p.name.endswith("_ptr") else "s" for p in params]
        if kinds != sorted(kinds, key="psc".index):
            raise TypeError(f"{fn.__name__}: parameters must be tensors (*_ptr), then scalars, then constexprs")
        for p, k in zip(params, kinds):
            if (k == "s" and p.annotation is not tl.int64) or (k == "p" and p.annotation is not inspect.Parameter.empty):
                raise TypeError(f"{fn.__name__}: scalar parameter {p.name} must be annotated tl.int64 (and tensors not "
                                f"annotated), so that the compiled kernel does not depend on argument values")
        free = [p.name for p, k in zip(params, kinds) if k != "c"]
        self.fn = triton.jit(do_not_specialize=free, do_not_specialize_on_alignment=free)(fn)
        self.nptr = kinds.count("p")
        self._compiled = {}

    def launch(self, grid, args, consts=(), types=None, **options):
        """args: the tensors, then the scalars (python ints). consts: the constexprs, in order. types: the element
        types of the tensors when the caller already has them as a tuple (else they are read from args). options:
        Triton compile options (num_stages), used when the kernel is first compiled."""
        if types is None:
            types = tuple([a.dtype for a in args[:self.nptr]])
        device = driver.active.get_current_device()
        key = (device, types, consts)
        k = self._compiled.get(key)
        if k is None or _ENTER.calls or _EXIT.calls:  # a profiler's launch hooks: Triton's own path calls them
            c = self.fn[grid](*args, *consts, **options)
            if DIRECT:
                self._compiled[key] = (c.run, c.function, c.packed_metadata)
            return
        g = len(grid)
        k[0](grid[0], grid[1] if g > 1 else 1, grid[2] if g > 2 else 1, driver.active.get_current_stream(device),
             k[1], k[2], None, None, None, *args, *consts)


def jit(fn):
    return Kernel(fn)
