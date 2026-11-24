"""cutosa/int64_ref.py has the same code as the kit's tosa_emu.py (AST without docstrings; comments are not in it).
usage: python tests/ref_sync.py <kit>/fine_grained_transformer_u85/tosa_exec/tosa_emu.py"""
import ast
import pathlib
import sys


def code(path):
    tree = ast.parse(pathlib.Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and ast.get_docstring(node, clean=False) is not None:
            node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree)


same = code(pathlib.Path(__file__).parents[1] / "cutosa" / "int64_ref.py") == code(sys.argv[1])
print("RESULT:", "PASS (same code)" if same else "FAIL (code differs)")
sys.exit(0 if same else 1)
