"""Run the MLEK-Transformer kit's TOSA_TORCH backend on cu-tosa without editing the kit.

With this directory first on PYTHONPATH and CUTOSA_KIT_OVERLAY=1, the kit module `tosa_exec.tosa_emu` gets its
`Emulator` replaced by `cutosa.Emulator` as it loads, so `from .tosa_emu import Emulator` in tosa_backends.py binds
the cu-tosa class (in spawned worker processes too: they inherit the environment). The kit's own class stays
available as `tosa_exec.tosa_emu.Int64Emulator`. Without the variable this file does nothing.

Python imports one `sitecustomize` at startup; another one further down sys.path (Ubuntu ships one) is run from here.
"""
import importlib.abc
import importlib.machinery
import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        import cutosa
        module.Int64Emulator = module.Emulator
        module.Emulator = cutosa.Emulator
        print(f"[cu-tosa overlay] {module.__name__}.Emulator -> cutosa.Emulator ({cutosa.__file__})", file=sys.stderr)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != "tosa_exec.tosa_emu":
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader)
        return spec


if os.environ.get("CUTOSA_KIT_OVERLAY") == "1":
    sys.path.append(os.path.dirname(os.path.dirname(_HERE)))  # the repository root: `import cutosa`
    sys.meta_path.insert(0, _Finder())

for _d in sys.path:
    _f = os.path.join(_d, "sitecustomize.py")
    if _d and os.path.abspath(_d) != _HERE and os.path.isfile(_f):
        runpy.run_path(_f)
        break
