"""Whether two Python files are the same program: equal syntax trees once docstrings are taken out (comments never are in)."""
import ast, sys

def strip(tree):
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                node.body = body[1:] or [ast.Pass()]
    return tree

a, b = sys.argv[1], sys.argv[2]
da = ast.dump(strip(ast.parse(open(a, encoding="utf-8").read())))
db = ast.dump(strip(ast.parse(open(b, encoding="utf-8").read())))
print(f"{a}\n{b}\n -> {'the same program (they differ at most in comments and docstrings)' if da == db else 'DIFFERENT programs'}")
sys.exit(0 if da == db else 1)
