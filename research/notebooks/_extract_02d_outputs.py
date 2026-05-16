"""Extract all required pieces from the executed 02d notebook for the execution report."""
from __future__ import annotations
import json
import re
from pathlib import Path

REPO = Path(r"C:\Users\Alexander\Documents\CFB\CFBapp")
NB_PATH = REPO / "research" / "notebooks" / "02d_turnover_and_short_field.executed.ipynb"
CSV_PATH = REPO / "research" / "results" / "feature_validation.csv"
SIDECAR_PATH = REPO / "research" / "results" / "feature_validation.schema.md"

def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)
cells = nb["cells"]

print(f"executed notebook: {NB_PATH.name}")
print(f"  size: {NB_PATH.stat().st_size:,} bytes")
print(f"  cell count: {len(cells)}")

def stdout_of(cell):
    if cell.get("cell_type") != "code":
        return ""
    out = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream" and o.get("name") == "stdout":
            t = o.get("text", "")
            if isinstance(t, list):
                t = "".join(t)
            out.append(t)
    return "".join(out)

def stderr_of(cell):
    if cell.get("cell_type") != "code":
        return ""
    out = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream" and o.get("name") == "stderr":
            t = o.get("text", "")
            if isinstance(t, list):
                t = "".join(t)
            out.append(t)
    return "".join(out)

banner("FULL STDOUT BY CELL (numbered)")
for i, c in enumerate(cells):
    if c.get("cell_type") != "code":
        continue
    out = stdout_of(c).rstrip()
    if not out:
        continue
    print(f"\n--- cell {i} ---")
    print(out)

banner("ANY WARNINGS / ERRORS BY CELL")
warn_cells = []
for i, c in enumerate(cells):
    if c.get("cell_type") != "code":
        continue
    se = stderr_of(c).strip()
    if se:
        first_line = se.splitlines()[0]
        warn_cells.append((i, first_line, se))
    for o in c.get("outputs", []):
        if o.get("output_type") == "error":
            ename = o.get("ename", "<no-ename>")
            evalue = o.get("evalue", "<no-evalue>")
            warn_cells.append((i, f"ERROR: {ename}: {evalue}", ""))
            break
if not warn_cells:
    print("(none)")
else:
    for idx, line, full in warn_cells:
        print(f"  cell {idx}: {line}")
        if full and len(full.splitlines()) > 1:
            print("    --- full stderr ---")
            for ln in full.splitlines():
                print(f"    {ln}")
print(f"\ntotal warning-emitting cells: {len(warn_cells)}")

banner("CSV / SIDECAR DELIVERABLE SANITY")
import csv
print(f"feature_validation.csv: {CSV_PATH.stat().st_size:,} bytes")
with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
    rows = list(csv.reader(f))
print(f"  rows including header: {len(rows)}")
print(f"  data rows: {len(rows)-1}")
if rows:
    print(f"  header: {rows[0]}")
seen_versions: dict[str, int] = {}
for r in rows[1:]:
    if len(r) < 2:
        continue
    fsv = r[1] if len(r) > 1 else ""
    seen_versions[fsv] = seen_versions.get(fsv, 0) + 1
print("  rows by feature_set_version:")
for k, v in sorted(seen_versions.items()):
    print(f"    {k}: {v}")

print(f"\nfeature_validation.schema.md: {SIDECAR_PATH.stat().st_size:,} bytes")
with open(SIDECAR_PATH, "r", encoding="utf-8") as f:
    sidecar = f.read()
print(f"  contains '<!-- BEGIN: 02d': {'<!-- BEGIN: 02d' in sidecar}")
print(f"  contains '<!-- BEGIN: 02c': {'<!-- BEGIN: 02c' in sidecar}")
print(f"  contains '<!-- BEGIN: 02b': {'<!-- BEGIN: 02b' in sidecar}")
print(f"  contains '<!-- BEGIN: 02a': {'<!-- BEGIN: 02a' in sidecar}")

banner("END OF EXTRACTION")
