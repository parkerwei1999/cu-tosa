"""Whole graphs, cu-tosa (fused and unfused) vs the int64 emulator, bit for bit over 7 input sets: outputs,
violations and a checksum of every stored result; prints each operator's path and the time of one run.
usage: python tests/graph_equiv.py [--trials 7] name=graph.tosa ..."""
import argparse
import statistics
import sys
import time
from collections import Counter

import numpy as np
import torch

from cutosa import RANGE, Emulator, Int64Emulator, TosaGraphIR


def feeds(g, trial, rng):
    out = []
    for n in g.inputs:
        t = g.tensors[n]
        lo, hi = RANGE[t.dtype]
        if trial == 0:
            x = np.full(t.shape, lo, np.int64)
        elif trial == 1:
            x = np.full(t.shape, hi, np.int64)
        elif trial == 2:
            x = np.where(rng.integers(0, 2, t.shape) == 1, hi, lo).astype(np.int64)
        elif trial == 3:
            x = rng.integers(-3, 4, t.shape, dtype=np.int64).clip(lo, hi)
        else:
            x = rng.integers(lo, hi + 1, t.shape, dtype=np.int64)
        out.append(x)
    return out


def checksum(t):
    if t.numel() == 0:
        return (0, 0, 0, 0)
    w = t.to(torch.int64)
    mn, mx = torch.aminmax(w)
    return (int(w.sum()), int((w * w).sum()), int(mn), int(mx))


def timed(emu, f, reps):
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        emu.run(f)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main(args):
    ok_all = True
    for spec in args.graphs:
        name, path = spec.split("=", 1)
        g = TosaGraphIR(path)
        ref = Int64Emulator(g, device="cuda")
        arms = {"fused": Emulator(g, device="cuda"),
                "unfused": Emulator(g, device="cuda", fuse=False)}
        rng = np.random.default_rng(args.seed)
        ok, first, nviol = True, None, 0
        for trial in range(args.trials):
            f = feeds(g, trial, rng)
            sums = {}
            want = ref.run(f, hook=lambda o, outs: sums.__setitem__(o.idx, [checksum(t) for t in outs]))
            want_v = list(ref.violations)

            for arm, cu in arms.items():
                def cmp(o, outs, trial=trial, arm=arm, cu=cu):
                    nonlocal first
                    if first is None and [checksum(t) for t in outs] != sums[o.idx]:
                        first = (arm, trial, o.idx, o.op)

                got = cu.run(f, hook=cmp)
                if first is None and want_v != cu.violations:
                    first = (arm, trial, "violations", [v for v in cu.violations if v not in want_v][:2],
                             [v for v in want_v if v not in cu.violations][:2])
                ok &= all(a.dtype == b.dtype and np.array_equal(a, b) for a, b in zip(want, got)) and first is None
            nviol += len(want_v)
        how = Counter(arms["fused"].how.values())
        f = feeds(g, 5, rng)
        t = {"int64": timed(ref, f, args.reps)}
        for arm in ("unfused", "fused"):
            fast = Emulator(g, device="cuda", fuse=arm == "fused")
            fast.run(f)
            t[arm] = timed(fast, f, args.reps)
            del fast
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {args.trials} input sets, {nviol} violation records equal, "
              f"ops {dict(how)}, int64 {t['int64'] * 1e3:.0f} ms, unfused {t['unfused'] * 1e3:.0f} ms, cu-tosa "
              f"{t['fused'] * 1e3:.0f} ms ({t['int64'] / t['fused']:.2f}x)"
              + (f", first difference (arm, input set, op, name): {first}" if first else ""), flush=True)
        ok_all &= ok
        del ref, arms
        torch.cuda.empty_cache()
    print("RESULT:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("graphs", nargs="+", help="name=path.tosa")
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    sys.exit(main(ap.parse_args()))
