"""Validate the unexecuted 02d notebook without running cells.

Checks:
  - File parses as JSON and matches nbformat 4.5 schema.
  - Every code cell compiles (syntax-only via ast.parse).
  - No stray output (must be unexecuted).
  - Cell IDs are unique.
  - The shared `_lib_chrono.CHRONO_KEY_SOURCE` was embedded into the
    cache-reload cell (verifies _build_02d.py is keeping its import-then-
    splice contract).
"""
from __future__ import annotations

import ast
import json
import pathlib
import sys

NB_PATH = pathlib.Path(__file__).resolve().parent / "02d_turnover_and_short_field.ipynb"

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

# Verify the _chrono_key source was spliced into the cache-reload cell.
# The _lib_chrono module is the single source of truth across all
# _build_02X.py scripts; this check ensures the splice contract is intact.
all_code_src = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
assert "def _chrono_key(p: dict)" in all_code_src, (
    "Notebook does not contain the _chrono_key function body. The build "
    "script's import-then-splice contract from _lib_chrono.py is broken. "
    "Investigate research/notebooks/_lib_chrono.py and _build_02d.py."
)
print(f"[ok] _chrono_key function body present in the notebook (spliced via _lib_chrono)")

# Sanity: verify the 4 candidate features are referenced in the notebook.
expected_feats = {
    "fav_turnovers_so_far",
    "dog_points_off_turnovers",
    "dog_avg_starting_field_pos",
    "short_field_tds_allowed",
}
for feat in expected_feats:
    assert feat in all_code_src, (
        f"Candidate feature {feat!r} not referenced in any code cell. "
        f"_build_02d.py may have dropped it from CANDIDATE_FEATURES."
    )
print(f"[ok] all 4 candidate features referenced in code: {sorted(expected_feats)}")

# Sanity: verify the diff-vs-leaky verification path is present.
assert 'plays_before_filter="leaky_playnumber"' in all_code_src, (
    "Diff-vs-leaky verification cell missing the leaky-filter pass. The "
    "plan-approval addition 1 (D10) requires building the feature matrix "
    "under both chrono_key and leaky-playNumber filters."
)
assert 'plays_before_filter="chrono_key"' in all_code_src, (
    "Diff-vs-leaky verification cell missing the chrono_key pass."
)
print(f"[ok] diff-vs-leaky verification path present (both filter modes)")

# Sanity: verify the overlap diagnostic recomputes dog_points_from_returns.
assert "_recompute_dog_points_from_returns" in all_code_src, (
    "D11 overlap diagnostic missing the dog_points_from_returns recomputation."
)
assert "SCORING_PLAYTYPE_REGISTRY_FOR_DIAGNOSTIC" in all_code_src, (
    "D11 overlap diagnostic missing the duplicated 02c registry."
)
print(f"[ok] D11 overlap diagnostic path present (registry duplicate + recomputation)")

print("\n[ok] all code cells parse cleanly; notebook is unexecuted; no duplicate ids.")
print(f"file size: {NB_PATH.stat().st_size:,} bytes")
