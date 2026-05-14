"""Validate the unexecuted 02e notebook without running cells.

Checks:
  - File parses as JSON and matches nbformat 4.5 schema.
  - Every code cell compiles (syntax-only via ast.parse).
  - No stray output (must be unexecuted).
  - Cell IDs are unique.
  - The shared `_lib_chrono.CHRONO_KEY_SOURCE` was embedded into the
    cache-reload cell (verifies _build_02e.py is keeping its import-then-
    splice contract).
  - All 3 candidate features are referenced.
  - The diff-vs-leaky D10 path includes BOTH filter modes plus the
    magnitude-distribution diagnostic (per plan-approval addition 1).
  - The D7 two-bucket NULL breakdown path is present (per plan-approval
    addition 2).
  - The D11 red-zone conversion diagnostic path is present.
"""
from __future__ import annotations

import ast
import json
import pathlib
import sys

NB_PATH = pathlib.Path(__file__).resolve().parent / "02e_red_zone_failure.ipynb"

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
all_code_src = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
all_md_src = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
assert "def _chrono_key(p: dict)" in all_code_src, (
    "Notebook does not contain the _chrono_key function body. The build "
    "script's import-then-splice contract from _lib_chrono.py is broken. "
    "Investigate research/notebooks/_lib_chrono.py and _build_02e.py."
)
print(f"[ok] _chrono_key function body present in the notebook (spliced via _lib_chrono)")

# Sanity: verify the 3 candidate features are referenced in the notebook.
expected_feats = {
    "fav_red_zone_trips",
    "fav_red_zone_tds",
    "fav_yards_per_point",
}
for feat in expected_feats:
    assert feat in all_code_src, (
        f"Candidate feature {feat!r} not referenced in any code cell. "
        f"_build_02e.py may have dropped it from CANDIDATE_FEATURES."
    )
print(f"[ok] all 3 candidate features referenced in code: {sorted(expected_feats)}")

# Verify the diff-vs-leaky verification path is present (BOTH filter modes).
assert 'plays_before_filter="leaky_playnumber"' in all_code_src, (
    "Diff-vs-leaky verification cell missing the leaky-filter pass. The "
    "plan-approval addition 1 (D10) requires building the feature matrix "
    "under both chrono_key and leaky-playNumber filters."
)
assert 'plays_before_filter="chrono_key"' in all_code_src, (
    "Diff-vs-leaky verification cell missing the chrono_key pass."
)
print(f"[ok] diff-vs-leaky verification path present (both filter modes)")

# Verify the D10 magnitude-distribution diagnostic is present (bidirectional
# buckets per sign: +1/+2/+3+ when chrono > leaky, -1/-2/<=-3 when leaky > chrono).
combined_src = all_code_src + "\n" + all_md_src
for marker in [
    "magnitude-distribution",
    "chrono > leaky: +1",
    "chrono > leaky: +2",
    "chrono > leaky: +3+",
    "leaky > chrono: -1",
    "catB_distributions",
]:
    assert marker in combined_src, (
        f"D10 magnitude-distribution diagnostic missing marker {marker!r}. "
        "Plan-approval addition 1 requires magnitude bucketing of "
        "`diff = chrono - leaky` per trigger for Category B features."
    )
print(
    "[ok] D10 magnitude-distribution diagnostic present "
    "(bidirectional +/+ buckets and -/- buckets)"
)

# Verify the D7 two-bucket NULL breakdown path is present.
assert "_classify_yards_per_point_null_bucket" in all_code_src, (
    "D7 two-bucket NULL breakdown classifier missing. plan-approval "
    "addition 2 requires separate counting of bucket (a) no completed fav "
    "drives vs bucket (b) zero fav offensive points."
)
assert "ypp_null_bucket" in all_code_src, (
    "D7 two-bucket diagnostic column 'ypp_null_bucket' missing from build_feature_matrix."
)
print(f"[ok] D7 two-bucket NULL breakdown path present "
      f"(classifier + diagnostic column)")

# Verify the D11 red-zone conversion diagnostic is present.
for marker in ["zero_trips", "zero_conv", "perfect_conv", "partial_conv"]:
    assert marker in all_code_src, (
        f"D11 red-zone conversion diagnostic missing marker {marker!r}."
    )
print(f"[ok] D11 red-zone conversion diagnostic path present "
      f"(zero/partial/perfect bucket classification)")

# Verify the paired-indicator (D8 Mode B) for fav_yards_per_point is present.
assert "fav_yards_per_point_is_null" in all_code_src, (
    "Paired-indicator 'fav_yards_per_point_is_null' missing. D8 Mode B "
    "from 02c (carry-forward) requires the R16-safe paired indicator."
)
print(f"[ok] D8 Mode B paired-indicator present (fav_yards_per_point_is_null)")

# Verify the sentinel-spliced sidecar markers are present.
assert ("<!-- BEGIN: 02e red_zone_failure -->" in all_code_src
        or "<!-- BEGIN: 02e red_zone_failure -->" in all_md_src), (
    "Sidecar sentinel BEGIN marker missing."
)
assert ("<!-- END: 02e red_zone_failure -->" in all_code_src
        or "<!-- END: 02e red_zone_failure -->" in all_md_src), (
    "Sidecar sentinel END marker missing."
)
print(f"[ok] sentinel-spliced sidecar markers present")

print("\n[ok] all code cells parse cleanly; notebook is unexecuted; no duplicate ids.")
print(f"file size: {NB_PATH.stat().st_size:,} bytes")
