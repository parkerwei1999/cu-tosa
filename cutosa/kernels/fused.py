"""Generated kernels for fusion units (cutosa/fusion.py): one pass over the root's flat index computes every member
operator in 64-bit registers, counts each operator's REQUIREs, and stores only the root's result.

A unit's kernel is generated from its structure, a hashable `prog`:
    prog = (leaf modes, ops)
    leaf modes: one of "f" (read at the flat index) / "i" (read at (flat // inner) % n: a broadcast operand) per leaf
    ops: per member, in graph order, (kind, flags, operand refs); a ref is "L<k>" (leaf k) or "V<j>" (member j)
        ("RESCALE", (scale32, per_channel), (x,))
        ("ADD" | "SUB" | "MUL", (bits of the output type,), (a, b))
        ("TABLE", (wide, slope_check), (x,))      slope_check: the table has a slope outside INT16 (a REQUIRE)
        ("MINIMUM" | "MAXIMUM" | "GREATER" | "GREATER_EQUAL", (), (a, b))
        ("SELECT", (), (condition, a, b))
The numbers of a unit (zero points, multipliers, shifts, bounds, broadcast strides) are not in the structure: they
are read from an int64 parameter tensor, laid out by params() in the order the generated code reads it:
    per "i" leaf: inner, n
    RESCALE with one multiplier: izp, multiplier, shift, ozp, olo, ohi;  per channel: izp, ozp, olo, ohi, C
    ADD / SUB: lo, hi        MUL: shift, lo, hi
Tensor arguments after the leaves: the tables of the TABLEs in member order, then (multipliers, shifts), int64, of
the per-channel RESCALEs in member order.
REQUIRE counters, in member order: RESCALE 2, ADD / SUB / MUL 1, TABLE with slope_check 1; counter k of sample s is
the element cbase + k * cstride + s of the counter buffer.

Every register is int64. A stored ADD / SUB / MUL result is the low bits of the exact value (the reference model's
wrap), so the register is truncated the same way before the next member reads it; RESCALE clamps and TABLE cannot
leave its output type.
"""
import hashlib
import linecache

import triton
import triton.language as tl

from ._launch import jit

BLOCK = 1024
N_PARAMS = {"RESCALE": lambda f: 5 if f[1] else 6, "ADD": lambda f: 2, "SUB": lambda f: 2, "MUL": lambda f: 3}
N_COUNTERS = {"RESCALE": lambda f: 2, "ADD": lambda f: 1, "SUB": lambda f: 1, "MUL": lambda f: 1,
              "TABLE": lambda f: int(f[1])}
_INT = {8: "tl.int8", 16: "tl.int16", 32: "tl.int32"}
_kernels = {}


def n_counters(prog):
    return sum(N_COUNTERS[kind](flags) for kind, flags, _ in prog[1] if kind in N_COUNTERS)


def n_params(prog):
    return 2 * sum(m == "i" for m in prog[0]) + sum(N_PARAMS[k](f) for k, f, _ in prog[1] if k in N_PARAMS)


def source(prog, name):
    modes, ops = prog
    aux = [f"t{j}_ptr" for j, (k, f, _) in enumerate(ops) if k == "TABLE"] \
        + [a for j, (k, f, _) in enumerate(ops) if k == "RESCALE" and f[1] for a in (f"m{j}_ptr", f"s{j}_ptr")]
    args = ["y_ptr", "c_ptr", "p_ptr"] + [f"x{k}_ptr" for k in range(len(modes))] + aux \
        + ["n_per: tl.int64", "cbase: tl.int64", "cstride: tl.int64"]
    L = ["@jit",
         f"def {name}({', '.join(args)}, BLOCK: tl.constexpr):",
         "    pid = tl.program_id(0)",
         "    sid = tl.program_id(1)",
         "    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)",
         "    mask = offs < n_per",
         "    g = sid.to(tl.int64) * n_per + offs",
         "    one = tl.full((BLOCK,), 1, tl.int64)",
         "    cptr = c_ptr + cbase + sid"]
    p = c = 0

    def par():
        nonlocal p
        p += 1
        return f"tl.load(p_ptr + {p - 1})"

    def count(bad):
        nonlocal c
        L.append(f"    n{c} = tl.sum((({bad}) & mask).to(tl.int64), axis=0)")
        L.append(f"    if n{c} > 0:")
        L.append(f"        tl.atomic_add(cptr + {c} * cstride, n{c})")
        c += 1

    for k, mode in enumerate(modes):
        idx = "g" if mode == "f" else f"(g // {par()}) % {par()}"
        L.append(f"    L{k} = tl.load(x{k}_ptr + {idx}, mask=mask, other=0).to(tl.int64)")
    for j, (kind, flags, refs) in enumerate(ops):
        v = f"V{j}"
        if kind == "RESCALE":
            scale32, per_channel = flags
            if per_channel:
                izp, ozp, olo, ohi, C = par(), par(), par(), par(), par()
                L.append(f"    ch{j} = g % {C}")
                L.append(f"    mm{j} = tl.load(m{j}_ptr + ch{j}, mask=mask, other=0).to(tl.int64)")
                L.append(f"    ss{j} = tl.load(s{j}_ptr + ch{j}, mask=mask, other=2).to(tl.int64)")
            else:
                izp, m, s, ozp, olo, ohi = par(), par(), par(), par(), par(), par()
                L.append(f"    mm{j} = one * {m}")
                L.append(f"    ss{j} = one * {s}")
            L.append(f"    x{j} = {refs[0]} - {izp}")
            L.append(f"    h{j} = one << (ss{j} - 1)")
            L.append(f"    {v} = (x{j} * mm{j} + h{j}) >> ss{j}")
            count(f"(x{j} < -h{j}) | (x{j} > h{j} - 1)" if scale32 else f"({v} < -2147483648) | ({v} > 2147483647)")
            L.append(f"    {v} = {v} + {ozp}")
            count(f"({v} < -2147483648) | ({v} > 2147483647)")
            L.append(f"    {v} = tl.minimum(tl.maximum({v}, {olo}), {ohi})")
        elif kind in ("ADD", "SUB", "MUL"):
            a, b = refs
            if kind == "MUL":
                L.append(f"    sh{j} = {par()}")
                L.append(f"    {v} = ({a} * {b} + ((one << sh{j}) >> 1)) >> sh{j}")
            else:
                L.append(f"    {v} = {a} {'+' if kind == 'ADD' else '-'} {b}")
            lo, hi = par(), par()
            count(f"({v} < {lo}) | ({v} > {hi})")
            L.append(f"    {v} = {v}.to({_INT[flags[0]]}).to(tl.int64)")
        elif kind == "TABLE":
            x = refs[0]
            if flags[0]:
                L.append(f"    i{j} = ({x} >> 7) + 256")
                L.append(f"    b{j} = tl.load(t{j}_ptr + i{j}, mask=mask, other=0).to(tl.int64)")
                L.append(f"    e{j} = tl.load(t{j}_ptr + i{j} + 1, mask=mask, other=0).to(tl.int64)")
                L.append(f"    {v} = b{j} * 128 + (e{j} - b{j}) * ({x} & 127)")
                if flags[1]:
                    count(f"(e{j} - b{j} < -32768) | (e{j} - b{j} > 32767)")
            else:
                L.append(f"    {v} = tl.load(t{j}_ptr + {x} + 128, mask=mask, other=0).to(tl.int64)")
        elif kind in ("MINIMUM", "MAXIMUM"):
            L.append(f"    {v} = tl.{kind.lower()}({refs[0]}, {refs[1]})")
        elif kind in ("GREATER", "GREATER_EQUAL"):
            L.append(f"    {v} = ({refs[0]} {'>' if kind == 'GREATER' else '>='} {refs[1]}).to(tl.int64)")
        elif kind == "SELECT":
            L.append(f"    {v} = tl.where({refs[0]} != 0, {refs[1]}, {refs[2]})")
        else:
            raise ValueError(kind)
    L.append(f"    tl.store(y_ptr + g, V{len(ops) - 1}.to(y_ptr.dtype.element_ty), mask=mask)")
    assert p == n_params(prog) and c == n_counters(prog)
    return "\n".join(L) + "\n"


def kernel(prog):
    """the compiled-on-first-use kernel of a structure (one per structure in the process; Triton caches the binary on
    disk by source)."""
    fn = _kernels.get(prog)
    if fn is None:
        name = "fused_" + hashlib.sha256(repr(prog).encode()).hexdigest()[:16]
        src = source(prog, name)
        filename = f"<cutosa {name}>"
        linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)  # for inspect.getsource
        ns = {"jit": jit, "tl": tl}
        exec(compile(src, filename, "exec"), ns)
        fn = _kernels[prog] = ns[name]
    return fn


def launch(k, types, y, cbuf, params, n_per, parts, cbase, cstride, leaves, aux):
    """k: kernel(prog). types: the element types of (y, cbuf, params, *leaves, *aux)."""
    k.launch((triton.cdiv(n_per, BLOCK), parts), (y, cbuf, params, *leaves, *aux, n_per, cbase, cstride), (BLOCK,), types)
