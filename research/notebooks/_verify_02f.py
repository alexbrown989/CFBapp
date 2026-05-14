"""Validate the unexecuted 02f notebook without running cells.

Checks:
  - Parses as nbformat 4.5 JSON; empty outputs / null execution counts.
  - Code cells compile (`ast.parse`).
  - Canonical `_chrono_key` body embedded (import-then-splice contract).
  - All four DDL candidate features referenced in code plus each
    `*_insufficient_sample` indicator column key.
  - Diff-vs-leaky path uses BOTH `plays_before_filter` modes and the
    D10 micro quantization stack (`MICRO_NAN_SENT`, `_rate_to_micro_series`
    naming in emitted notebook).
  - D11 pairwise early-vs-third correlation identifiers present (rho + n_pair).
  - Sentinel markers for `feature_validation.schema.md` splicing (**02f** block).
"""
from __future__ import annotations

import ast
import json
import pathlib
import sys

NB_PATH = pathlib.Path(__file__).resolve().parent / "02f_down_distance_efficiency.ipynb"

raw = NB_PATH.read_text(encoding="utf-8")
nb = json.loads(raw)

assert nb["nbformat"] == 4, f"nbformat={nb['nbformat']}"
assert nb["nbformat_minor"] == 5, f"nbformat_minor={nb['nbformat_minor']}"

cells = nb["cells"]
cell_ids = [c["id"] for c in cells]
dups = [cid for cid in set(cell_ids) if cell_ids.count(cid) > 1]
assert not dups, f"duplicate cell ids: {dups}"

bad: list[tuple[str, str]] = []
for c in cells:
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    assert c.get("execution_count") in (None, 0), (
        f"cell {c['id']} has execution_count={c['execution_count']!r}"
    )
    assert c.get("outputs") == [], f"cell {c['id']} has outputs"
    try:
        ast.parse(src)
    except SyntaxError as e:
        bad.append((c["id"], f"{e.msg} at line {e.lineno}"))
if bad:
    for cid, msg in bad:
        print(f"  {cid}: {msg}")
    sys.exit(1)

all_code_src = "\n".join(
    "".join(c["source"]) for c in cells if c["cell_type"] == "code"
)
all_md_src = "\n".join(
    "".join(c["source"]) for c in cells if c["cell_type"] == "markdown"
)
combined = all_code_src + "\n" + all_md_src

assert "def _chrono_key(p: dict)" in all_code_src, (
    "_chrono_key splice missing; check research/notebooks/_lib_chrono.py linkage."
)

expected_feats = {
    "fav_early_down_success_rate",
    "fav_third_down_success_rate",
    "dog_early_down_success_rate",
    "dog_third_down_success_rate",
}
for feat in expected_feats:
    assert feat in all_code_src, (
        f"Candidate {feat!r} missing from code cells "
        "(CANDIDATE_FEATURES mismatch?)."
    )

insuff_suffix = "_insufficient_sample"
for feat in sorted(expected_feats):
    assert feat + insuff_suffix in combined, (
        f"Paired insufficient-sample column for {feat!r} missing from notebook "
        "(INSUFFICIENT_SAMPLE_COLS)."
    )

for mode in ('plays_before_filter="leaky_playnumber"', 'plays_before_filter="chrono_key"'):
    assert mode in all_code_src, (
        f"Notebook must reference {mode} for diff-vs-leaky / canonical passes."
    )

for marker in (
    "catB_distributions",
    "MICRO_NAN_SENT",
    "_rate_to_micro_series",
    "chrono > leaky: +1",
    "leaky > chrono: -1",
):
    assert marker in combined, f"D10 stack marker missing: {marker!r}"

for marker in ("rho_fav_early_third", "n_pair_fav_early_third", "rho_dog_early_third"):
    assert marker in all_code_src, f"D11 diagnostics missing marker: {marker!r}"

legacy_02e = ("fav_red_zone_trips", "fav_yards_per_point", "_classify_yards_per_point_null")
for lm in legacy_02e:
    assert lm not in all_code_src, f"Stale 02e symbol leaked into 02f code: {lm!r}"

for sent in (
    "<!-- BEGIN: 02f down_distance_efficiency -->",
    "<!-- END: 02f down_distance_efficiency -->",
):
    assert sent in combined, f"Schema sentinel missing: {sent!r}"

print(f"[ok] 02f notebook structure valid (unexecuted). cells={len(cells)}")
print(f"file: {NB_PATH} ({NB_PATH.stat().st_size:,} bytes)")
