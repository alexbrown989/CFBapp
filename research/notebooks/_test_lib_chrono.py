"""Smoke test: _lib_chrono._chrono_key matches the inline 02c version byte-for-byte."""
import re
import sys

sys.path.insert(0, r"C:\Users\Alexander\Documents\CFB\CFBapp\research\notebooks")
from _lib_chrono import _chrono_key, CHRONO_KEY_SOURCE  # noqa: E402

print("=== Canonical source (from _lib_chrono) ===")
print(CHRONO_KEY_SOURCE)
print()

src = open(
    r"C:\Users\Alexander\Documents\CFB\CFBapp\research\notebooks\_build_02c.py",
    encoding="utf-8",
).read()

m = re.search(
    r"(def _chrono_key\(p: dict\) -> tuple\[int, int, int, int\]:\n"
    r".*?int\(p\.get\(\"playNumber\"\) or 0\),\n\s*\))",
    src,
    re.DOTALL,
)
print(f"02c inline match found: {m is not None}")
if m:
    inline = m.group(1)
    print(f"inline len:  {len(inline)}")
    print(f"canonical len: {len(CHRONO_KEY_SOURCE)}")
    print(f"byte-identical: {inline == CHRONO_KEY_SOURCE}")
    if inline != CHRONO_KEY_SOURCE:
        for i, (a, b) in enumerate(zip(inline, CHRONO_KEY_SOURCE)):
            if a != b:
                print(f"  first diff at index {i}: inline={a!r}, canonical={b!r}")
                break

p_sample = {"period": 1, "clock": {"minutes": 7, "seconds": 27}, "driveNumber": 1, "playNumber": 3}
print()
print(f"sample _chrono_key: {_chrono_key(p_sample)}")
