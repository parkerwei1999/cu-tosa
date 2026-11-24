"""Time of one cu-tosa run per operator and path, synchronising after each operator (total > unsynchronised).
usage: python tests/profile_ops.py name=graph.tosa ..."""
import sys
import time
from collections import defaultdict

import numpy as np
import torch

from cutosa import RANGE, Emulator, TosaGraphIR

for spec in sys.argv[1:]:
    name, path = spec.split("=", 1)
    g = TosaGraphIR(path)
    emu = Emulator(g, device="cuda")
    rng = np.random.default_rng(0)
    f = [rng.integers(RANGE[g.tensors[n].dtype][0], RANGE[g.tensors[n].dtype][1] + 1, g.tensors[n].shape,
                      dtype=np.int64) for n in g.inputs]
    emu.run(f)
    acc = defaultdict(lambda: [0, 0.0])
    last = [0.0]

    def hook(o, outs):
        torch.cuda.synchronize()
        now = time.perf_counter()
        k = (o.op, emu.how[o.idx])
        acc[k][0] += 1
        acc[k][1] += now - last[0]
        last[0] = now

    torch.cuda.synchronize()
    t0 = last[0] = time.perf_counter()
    emu.run(f, hook=hook)
    total = time.perf_counter() - t0
    print(f"== {name}: {total * 1e3:.0f} ms with a sync per operator")
    for (op, how), (n, t) in sorted(acc.items(), key=lambda kv: -kv[1][1])[:12]:
        print(f"   {100 * t / total:5.1f} %  {t * 1e3:8.1f} ms  {n:5d} x {1e6 * t / n:7.0f} us  {op} ({how})")
