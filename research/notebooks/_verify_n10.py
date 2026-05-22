"""Validate the N10 notebook scaffold and, when present, deliverables."""

from __future__ import annotations

import ast
import json
import pathlib
import sys

import pandas as pd

NB_PATH = pathlib.Path(__file__).resolve().parent / "10_fluke_deficit_conditional_analysis.ipynb"
REPO_ROOT = NB_PATH.parents[2]
RESULTS_DIR = REPO_ROOT / "research" / "results"

raw = NB_PATH.read_text(encoding="utf-8")
nb = json.loads(raw)

assert nb["nbformat"] == 4
assert nb["nbformat_minor"] == 5

cells = nb["cells"]
ids = [cell["id"] for cell in cells]
dups = [cid for cid in set(ids) if ids.count(cid) > 1]
assert not dups, f"duplicate cell ids: {dups}"

bad: list[tuple[str, str]] = []
for cell in cells:
    if cell["cell_type"] != "code":
        continue
    assert cell.get("execution_count") in (None, 0)
    assert cell.get("outputs") == []
    src = "".join(cell["source"])
    try:
        ast.parse(src)
    except SyntaxError as exc:
        bad.append((cell["id"], f"{exc.msg} at line {exc.lineno}"))
if bad:
    for cid, msg in bad:
        print(f"{cid}: {msg}")
    sys.exit(1)

all_text = raw
all_code = "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")

for marker in [
    "fluke_composite",
    "fluke_bucket",
    "clear_fluky_lead",
    "spread_bucket",
    "candidate_live_watch",
    "favorite_final_win",
    "deficit_erased",
    "pregame_no_vig_implied_prob",
    "decimal_odds_best_available",
    "real_moneyline_only",
    "synthetic_fallback",
    "n10_conditional_rates.parquet",
    "n10_conditional_analysis.json",
    "n10_summary_report.md",
]:
    assert marker in all_text, f"N10 marker missing: {marker!r}"

for forbidden in [
    "httpx.get",
    "requests.get",
    "LogisticRegression(",
    "RandomForest",
    "XGBClassifier",
    "force_refresh=True",
]:
    assert forbidden not in all_code, f"forbidden N10 code path present: {forbidden!r}"

for required in [
    "market_prices_for_favorite",
    "bootstrap_cluster_mean",
    "bootstrap_cluster_roi",
    "subset_direct_hypothesis",
    "subset_inverse",
    "clear_fluky_lead",
    "fluky_lead",
    "sustained_lead",
    "huge_favorite",
    "big_favorite",
    "inverse_methodology_warning",
]:
    assert required in all_code, f"N10 code marker missing: {required!r}"

rates_path = RESULTS_DIR / "n10_conditional_rates.parquet"
analysis_path = RESULTS_DIR / "n10_conditional_analysis.json"
summary_path = RESULTS_DIR / "n10_summary_report.md"

if any(p.exists() for p in [rates_path, analysis_path, summary_path]):
    for path in [rates_path, analysis_path, summary_path]:
        assert path.exists(), f"missing N10 deliverable: {path}"

    rates = pd.read_parquet(rates_path)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    summary = summary_path.read_text(encoding="utf-8")

    assert len(rates) == 11412, f"expected 11,412 N10 rows, got {len(rates):,}"
    assert rates[["game_id", "fav_deficit", "trigger_sequence"]].duplicated().sum() == 0
    required_cols = [
        "game_id",
        "season",
        "fav_deficit",
        "period",
        "time_bucket",
        "fluke_composite",
        "fluke_bucket",
        "clear_fluky_lead",
        "dog_drive_yards_per_point_diagnostic",
        "spread_bucket",
        "favorite_final_win",
        "deficit_erased",
        "pregame_raw_implied_prob",
        "pregame_no_vig_implied_prob",
        "decimal_odds_best_available",
        "is_synthetic_fallback_price",
    ]
    for col in required_cols:
        assert col in rates.columns, f"missing N10 rates column: {col}"
    assert set(rates["time_bucket"]).issubset({"Q1", "Q2", "Q3", "Q4"})
    assert set(rates["fluke_bucket"]).issubset({
        "fluky_lead", "mixed_lead", "sustained_lead", "no_dog_points", "attribution_unclear"
    })
    assert rates["pregame_no_vig_implied_prob"].between(0, 1).all()
    assert rates["decimal_odds_best_available"].gt(1).all()

    for key in [
        "sanity_check_diagnostics",
        "tier4_single_dimension_splits",
        "tier25_two_way_matrices",
        "tier1_fluke_deficit_time",
        "tier2_fluke_spread_time",
        "tier3_dashboard_only",
        "direct_hypothesis_test",
        "inverse_hypothesis_sanity_check",
        "candidate_live_watch_cells",
        "interpretation_class",
    ]:
        assert key in analysis, f"missing N10 analysis key: {key}"
    assert len(analysis["tier1_fluke_deficit_time"]) > 0
    assert "Direct answer" in summary
    assert "live-watch" in summary
    print(f"[ok] N10 deliverables verified rows={len(rates):,}")
else:
    print("[ok] N10 notebook scaffold verified; deliverables not present yet")

print(f"[ok] N10 notebook structure valid. cells={len(cells)}")
print(f"file: {NB_PATH} ({NB_PATH.stat().st_size:,} bytes)")
