"""Extract relevant outputs from the executed 02a notebook for the report.

Reads research/notebooks/02a_baseline_features.executed.ipynb,
walks its cells, and prints:
  - any cell that produced a stderr/error stream (warnings, exceptions)
  - the full stdout of cells whose source contains a tag we care about
    (load-triggers, build-feature-matrix, eval-loop, write-csv, write-schema,
    summary, budget)
"""
from __future__ import annotations

import json
import pathlib

NB = pathlib.Path(__file__).resolve().parent / "02a_baseline_features.executed.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))

# Map cell IDs to short tags for readability.
TAG_BY_ID = {
    "c02a0001": "imports",
    "c02a0002": "helpers",
    "c02a0004": "config",
    "c02a0006": "load-triggers",
    "c02a0008": "cache-plays",
    "c02a000a": "assert-no-lookahead",
    "c02a000b": "feature-fns",
    "c02a000d": "build-matrix",
    "c02a000f": "ece-helper",
    "c02a0010": "eval-loop",
    "c02a0012": "write-csv",
    "c02a0014": "write-schema",
    "c02a0016": "summary",
    "c02a0017": "budget",
}

print("=" * 70)
print(f"EXECUTED NOTEBOOK: {NB.name}  ({NB.stat().st_size:,} bytes)")
print("=" * 70)

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
print(f"code cells: {len(code_cells)}")
print(f"code cells with execution_count: "
      f"{sum(1 for c in code_cells if c.get('execution_count') is not None)}")
print(f"code cells with outputs:         "
      f"{sum(1 for c in code_cells if c.get('outputs'))}")

# Any stderr/error outputs at all?
print("\n--- stderr / error scan (across all code cells) ---")
n_stderr = 0
n_error = 0
for c in code_cells:
    cid = c.get("id", "?")
    tag = TAG_BY_ID.get(cid, "?")
    for out in c.get("outputs", []):
        if out.get("output_type") == "stream" and out.get("name") == "stderr":
            n_stderr += 1
            text = "".join(out.get("text", []))
            print(f"\n[stderr] cell={cid} ({tag})")
            for line in text.splitlines():
                print(f"  | {line}")
        elif out.get("output_type") == "error":
            n_error += 1
            tb = out.get("traceback", [])
            print(f"\n[ERROR] cell={cid} ({tag}): {out.get('ename')}: {out.get('evalue')}")
            for line in tb:
                print(f"  | {line}")
if n_stderr == 0 and n_error == 0:
    print("  (none)")

# Full stdout for tagged cells.
TAG_PRINT = {"load-triggers", "cache-plays", "build-matrix", "eval-loop",
             "write-csv", "write-schema", "summary", "budget"}
for c in code_cells:
    cid = c.get("id", "?")
    tag = TAG_BY_ID.get(cid, "?")
    if tag not in TAG_PRINT:
        continue
    print(f"\n{'=' * 70}\nCELL {cid} ({tag})\n{'=' * 70}")
    for out in c.get("outputs", []):
        if out.get("output_type") == "stream" and out.get("name") == "stdout":
            text = "".join(out.get("text", []))
            print(text, end="")
        elif out.get("output_type") == "execute_result":
            # Rare; pandas head() or similar
            data = out.get("data", {})
            if "text/plain" in data:
                print("".join(data["text/plain"]))
