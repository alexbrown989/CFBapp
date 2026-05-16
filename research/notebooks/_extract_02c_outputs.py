"""Extract all required pieces from the executed 02c notebook for the execution report.

Sections (per user-asked report):
- Budget print verbatim
- Three deliverable file sizes
- Total scoring plays classified vs excluded
- Drive-1 trigger count (D4 reporting symmetry)
- Per-feature null counts
- Per-train-window imputation medians (D8)
- Per-feature stability verdict table (Brier / ECE / verdict)
- P3 trigger-logic disposition (which case from D11)
- Any warning cells (cell index + first line)
- Conditional-identity caveats discovered at execution time
- Sanity: CSV row count + schema sidecar 02c section presence
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path

REPO = Path(r"C:\Users\Alexander\Documents\CFB\CFBapp")
NB_PATH = REPO / "research" / "notebooks" / "02c_explosive_vs_sustained.executed.ipynb"
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

def all_text_of(cell):
    if cell.get("cell_type") != "code":
        return ""
    out = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            t = o.get("text", "")
            if isinstance(t, list):
                t = "".join(t)
            out.append(t)
        elif o.get("output_type") in ("error",):
            tb = o.get("traceback", [])
            if isinstance(tb, list):
                tb = "\n".join(tb)
            out.append(tb)
    return "".join(out)

def find_cells_containing(keyword: str):
    hits = []
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        src = c.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        out = all_text_of(c)
        if keyword in src or keyword in out:
            hits.append(i)
    return hits

banner("BUDGET PRINT (verbatim)")
for i, c in enumerate(cells):
    if c.get("cell_type") != "code":
        continue
    out = stdout_of(c)
    if "fresh CFBD calls" in out or "monthly free-tier" in out or ("budget" in out.lower() and "calls" in out.lower()):
        print(f"--- cell {i} ---")
        print(out.rstrip())

banner("SCORING-PLAYTYPE ENUMERATION (Phase 02c-c, classified vs excluded)")
for i, c in enumerate(cells):
    if c.get("cell_type") != "code":
        continue
    out = stdout_of(c)
    if "Scoring playType counts across cache" in out or "[ok] all" in out:
        print(f"--- cell {i} ---")
        print(out.rstrip())
        print()

banner("DRIVE-1 TRIGGER COUNT (D4 reporting symmetry)")
for i, c in enumerate(cells):
    if c.get("cell_type") != "code":
        continue
    out = stdout_of(c)
    if "drive_number_in_game" in out and ("trigger" in out.lower() or "trig" in out.lower()):
        print(f"--- cell {i} ---")
        print(out.rstrip())

banner("PER-FEATURE NULL COUNTS")
for i, c in enumerate(cells):
    if c.get("cell_type") != "code":
        continue
    out = stdout_of(c)
    if re.search(r"null\s*count", out, re.IGNORECASE) or "non-null" in out or re.search(r"NULL'd by", out):
        print(f"--- cell {i} ---")
        print(out.rstrip())

banner("D8 PER-TRAIN-WINDOW IMPUTATION MEDIANS")
for i, c in enumerate(cells):
    if c.get("cell_type") != "code":
        continue
    out = stdout_of(c)
    if "imputation" in out.lower() and ("median" in out.lower() or "_value" in out):
        print(f"--- cell {i} ---")
        print(out.rstrip())

banner("PER-FEATURE STABILITY VERDICT TABLE")
for i, c in enumerate(cells):
    if c.get("cell_type") != "code":
        continue
    out = stdout_of(c)
    if ("PASS" in out and "FAIL" in out) or ("brier" in out.lower() and "ece" in out.lower() and "verdict" in out.lower()):
        print(f"--- cell {i} ---")
        print(out.rstrip())

banner("P3 TRIGGER-LOGIC DISPOSITION (D11)")
for i, c in enumerate(cells):
    if c.get("cell_type") != "code":
        continue
    out = stdout_of(c)
    if "P3" in out or "D11" in out or "categorical" in out.lower() or "future_features" in out.lower():
        print(f"--- cell {i} ---")
        print(out.rstrip())

banner("ANY WARNINGS / ERRORS BY CELL (cell index + first stream-error line)")
warn_cells = []
for i, c in enumerate(cells):
    if c.get("cell_type") != "code":
        continue
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream" and o.get("name") == "stderr":
            t = o.get("text", "")
            if isinstance(t, list):
                t = "".join(t)
            first_line = t.strip().splitlines()[0] if t.strip() else "(empty stderr)"
            warn_cells.append((i, first_line))
            break
        if o.get("output_type") == "error":
            ename = o.get("ename", "<no-ename>")
            evalue = o.get("evalue", "<no-evalue>")
            warn_cells.append((i, f"ERROR: {ename}: {evalue}"))
            break
if not warn_cells:
    print("(none)")
else:
    for idx, line in warn_cells:
        print(f"  cell {idx}: {line}")
print(f"\ntotal warning-emitting cells: {len(warn_cells)}")

banner("CONDITIONAL-IDENTITY DISCOVERIES (any new ones from execution)")
for i, c in enumerate(cells):
    if c.get("cell_type") != "code":
        continue
    out = stdout_of(c)
    if "conditional identit" in out.lower() or "redundancy" in out.lower() or "redundant_with" in out.lower():
        print(f"--- cell {i} ---")
        print(out.rstrip())

banner("CSV / SIDECAR DELIVERABLE SANITY")
import csv
print(f"feature_validation.csv: {CSV_PATH.stat().st_size:,} bytes")
with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
    rows = list(csv.reader(f))
print(f"  rows including header: {len(rows)}")
print(f"  data rows: {len(rows)-1}")
if rows:
    print(f"  header: {rows[0]}")
seen_versions = {}
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
print(f"  contains '<!-- BEGIN: 02c': {'<!-- BEGIN: 02c' in sidecar}")
print(f"  contains '<!-- BEGIN: 02b': {'<!-- BEGIN: 02b' in sidecar}")
print(f"  contains '<!-- BEGIN: 02a': {'<!-- BEGIN: 02a' in sidecar}")

banner("END OF EXTRACTION")
