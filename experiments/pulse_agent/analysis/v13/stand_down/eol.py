import sys
from pathlib import Path
for f in sys.argv[1:]:
    b = Path(f).read_bytes()
    print(f, "bytes", len(b), "CRLF", b.count(b"\r\n"), "LF", b.count(b"\n"))
