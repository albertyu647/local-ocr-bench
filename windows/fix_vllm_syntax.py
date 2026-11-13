import sys, re, pathlib, ast
root = pathlib.Path(sys.argv[1])
for f in root.rglob("*.py"):
    s = f.read_text(encoding="utf-8")
    try:
        ast.parse(s)
    except SyntaxError as e:
        fixed = s.replace("\" && platform", "\" and platform")
        try:
            ast.parse(fixed); f.write_text(fixed, encoding="utf-8"); print("fixed", f, "line", e.lineno)
        except SyntaxError as e2:
            print("STILL BROKEN", f, e2.lineno, e2.msg)
