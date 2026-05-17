"""Validate the unexecuted 02g notebook without running cells.

Checks:
  - nbformat 4.5 JSON; empty outputs / null execution counts.
  - code cells compile.
  - canonical _chrono_key body is embedded from _lib_chrono.
  - reduced five-feature candidate set is present.
  - weather candidates and HTTP helpers are absent from code.
  - cached /games neutralSite path is present.
  - D10 references both filter labels and asserts byte-identical mismatches.
  - 02g schema sentinel markers are present.
"""
from __future__ import annotations

import ast
import json
import pathlib
import sys

NB_PATH = pathlib.Path(__file__).resolve().parent / "02g_context_week_home_neutral.ipynb"

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
    "_chrono_key splice missing; check _lib_chrono.py linkage."
)

expected_feats = {
    "season_phase_mid",
    "season_phase_late",
    "season_phase_bowl",
    "fav_is_home",
    "is_neutral_site",
}
for feat in expected_feats:
    assert feat in all_code_src, f"Candidate {feat!r} missing from code cells."

for stale in ("wind_mph", "temp_f", "is_dome"):
    assert f'"{stale}",' not in all_code_src, (
        f"Deferred weather/venue symbol leaked into a list-like code path: {stale!r}"
    )
    assert f"'{stale}'," not in all_code_src, (
        f"Deferred weather/venue symbol leaked into a list-like code path: {stale!r}"
    )

for forbidden in ("def cfbd_get", "def om_get", "httpx.get", "OPEN_METEO_BASE", "CFBD_BASE"):
    assert forbidden not in all_code_src, f"External helper/call path should be absent: {forbidden!r}"

for marker in ("cfbd__games__*.json", "neutralSite", "games_meta_by_id"):
    assert marker in all_code_src, f"cached /games neutral-site marker missing: {marker!r}"

for mode in ('plays_before_filter="chrono_key"', 'plays_before_filter="leaky_playnumber"'):
    assert mode in all_code_src, f"D10 mode marker missing: {mode!r}"

for marker in ("catA_mismatches", "byte-identical", "mismatches == 0"):
    assert marker in combined, f"D10 byte-identity marker missing: {marker!r}"

for sent in (
    "<!-- BEGIN: 02g context_week_home_neutral -->",
    "<!-- END: 02g context_week_home_neutral -->",
):
    assert sent in combined, f"Schema sentinel missing: {sent!r}"

print(f"[ok] 02g notebook structure valid (unexecuted). cells={len(cells)}")
print(f"file: {NB_PATH} ({NB_PATH.stat().st_size:,} bytes)")
