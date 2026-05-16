"""Validate the unexecuted 02b notebook without running cells.

Checks:
  - File parses as JSON and matches nbformat 4.5 schema.
  - Every code cell compiles (syntax-only via ast.parse).
  - No stray output (must be unexecuted).
  - Cell IDs are unique.
"""
from __future__ import annotations

import ast
import json
import pathlib
import sys

NB_PATH = pathlib.Path(__file__).resolve().parent / "02b_opening_drive_shock.ipynb"

raw = NB_PATH.read_text(encoding="utf-8")
nb = json.loads(raw)

assert nb["nbformat"] == 4, f"nbformat={nb['nbformat']}"
assert nb["nbformat_minor"] == 5, f"nbformat_minor={nb['nbformat_minor']}"

cells = nb["cells"]
print(f"cells: {len(cells)}")

cell_ids = [c["id"] for c in cells]
dups = [cid for cid in set(cell_ids) if cell_ids.count(cid) > 1]
assert not dups, f"duplicate cell ids: {dups}"
print(f"all {len(cell_ids)} cell ids unique")

n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
n_code = sum(1 for c in cells if c["cell_type"] == "code")
print(f"markdown cells: {n_md}, code cells: {n_code}")

bad: list[tuple[str, str]] = []
for c in cells:
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    assert c.get("execution_count") in (None, 0), (
        f"cell {c['id']} has execution_count={c['execution_count']!r} (should be None)"
    )
    assert c.get("outputs") == [], f"cell {c['id']} has outputs (should be [])"
    try:
        ast.parse(src)
    except SyntaxError as e:
        bad.append((c["id"], f"{e.msg} at line {e.lineno}"))

if bad:
    print("\nSYNTAX ERRORS:")
    for cid, msg in bad:
        print(f"  {cid}: {msg}")
    sys.exit(1)

print("\n[ok] all code cells parse cleanly; notebook is unexecuted; no duplicate ids.")
print(f"file size: {NB_PATH.stat().st_size:,} bytes")
