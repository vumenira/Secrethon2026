from pathlib import Path
p = Path("outputs/w13a/patch_reserve.py")
lines = p.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").split("\n")
i = next(n for n, l in enumerate(lines) if l.startswith("crlf = "))
j = next(n for n, l in enumerate(lines) if l.startswith("assert \"strait_weeks\""))
lines[i:j] = ['crlf = chr(13) + chr(10) in s', 's = s.replace(chr(13) + chr(10), chr(10))']
k = next(n for n, l in enumerate(lines) if l.startswith("p.write_bytes("))
m = next(n for n, l in enumerate(lines) if l.startswith("print(\"reserve.py:\""))
lines[k:m] = ['p.write_bytes((s.replace(chr(10), chr(13) + chr(10)) if crlf else s).encode("utf-8"))']
p.write_bytes("\n".join(lines).encode("utf-8"))
