"""Launches Triton kernels without Triton's per-call argument inspection (~15 us a launch, as long as a small
kernel). Only for the Triton versions in DIRECT_VERSIONS; any other version, a profiler hook, or
CUTOSA_DIRECT_LAUNCH=0 uses Triton's normal launch."""
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
        """args: tensors, then python-int scalars. types: the tensors' dtypes when the caller has them. options:
        Triton compile options, used at the first compile."""
        if types is None:
            types = tuple([a.dtype for a in args[:self.nptr]])
        device = driver.active.get_current_device()  # pyright: ignore[reportAttributeAccessIssue]
        key = (device, types, consts)
        k = self._compiled.get(key)
        if k is None or _ENTER.calls or _EXIT.calls:  # profiler hooks only run on Triton's normal launch path
            c = self.fn[grid](*args, *consts, **options)
            if DIRECT:
                self._compiled[key] = (c.run, c.function, c.packed_metadata)
            return
        g = len(grid)
        k[0](grid[0], grid[1] if g > 1 else 1, grid[2] if g > 2 else 1, driver.active.get_current_stream(device),  # pyright: ignore[reportAttributeAccessIssue]
             k[1], k[2], None, None, None, *args, *consts)


def jit(fn):
    return Kernel(fn)
