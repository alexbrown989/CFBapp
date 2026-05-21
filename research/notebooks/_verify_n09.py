"""Validate the N09 notebook scaffold and, when present, deliverables."""

from __future__ import annotations

import ast
import json
import pathlib
import sys

import pandas as pd

NB_PATH = pathlib.Path(__file__).resolve().parent / "09_trigger_state_analysis.ipynb"
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
    "predictive edge",
    "structural edge",
    "market edge",
    "betting edge",
    "baseline_C",
    "favorite_final_win",
    "deficit_erased",
    "turnover_composition_bucket",
    "fav_momentum_bucket",
    "n09_trigger_state_stratifications.parquet",
    "n09_baseline_analysis.json",
    "n09_betting_simulations.parquet",
    "n09_betting_summary.json",
    "n09_summary_report.md",
]:
    assert marker in all_text, f"N09 marker missing: {marker!r}"

for forbidden in [
    "httpx.get",
    "requests.get",
    "force_refresh=True",
    "to_csv(FEATURE_VALIDATION_CSV",
    "FEATURE_VALIDATION_CSV.write_text",
]:
    assert forbidden not in all_code, f"forbidden N09 code path present: {forbidden!r}"

for required in [
    "cluster_bootstrap_rate",
    "cluster_bootstrap_mean",
    "thin_flag",
    "selected_simulation_rows",
    "B_final_win_model_edge",
    "B_deficit_erased_heuristic",
    "model_prob_final_win",
    "price_subset",
    "real_moneyline_only",
    "synthetic_fallback_only",
    "eighth_kelly",
    "quarter_kelly",
    "small_sample_warnings",
    "baseline_C_deficit_erased",
    "baseline_C_favorite_final_win",
]:
    assert required in all_code, f"N09 code marker missing: {required!r}"

strat_path = RESULTS_DIR / "n09_trigger_state_stratifications.parquet"
baseline_path = RESULTS_DIR / "n09_baseline_analysis.json"
bets_path = RESULTS_DIR / "n09_betting_simulations.parquet"
betting_path = RESULTS_DIR / "n09_betting_summary.json"
summary_path = RESULTS_DIR / "n09_summary_report.md"

if any(p.exists() for p in [strat_path, baseline_path, bets_path, betting_path, summary_path]):
    for path in [strat_path, baseline_path, bets_path, betting_path, summary_path]:
        assert path.exists(), f"missing N09 deliverable: {path}"

    strat = pd.read_parquet(strat_path)
    bets = pd.read_parquet(bets_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    betting = json.loads(betting_path.read_text(encoding="utf-8"))
    summary = summary_path.read_text(encoding="utf-8")

    assert len(strat) == 11412, f"expected 11,412 strat rows, got {len(strat):,}"
    key = ["game_id", "fav_deficit", "trigger_sequence"]
    assert strat[key].duplicated().sum() == 0
    for col in [
        "turnover_composition_bucket",
        "short_field_composition_bucket",
        "explosive_composition_bucket",
        "epa_differential_bucket",
        "pace_bucket",
        "possessions_remaining_bucket",
        "favorite_tempo_bucket",
        "favorite_pass_rate_bucket",
        "fav_momentum_bucket",
        "dog_momentum_bucket",
    ]:
        assert col in strat.columns, f"missing stratification column {col}"

    assert "section1_baseline_C" in baseline
    assert "section2_stratifications" in baseline
    assert len(baseline["section1_baseline_C"]["twenty_cell_rate_table"]) == 40
    assert "summary" in betting
    assert "small_sample_warnings" in betting
    if len(bets):
        assert bets["stake"].ge(0).all()
        assert bets["decimal_odds"].gt(1).all()
        assert set(bets["staking_rule"]).issubset({"flat_1u", "eighth_kelly", "quarter_kelly"})
        for col in [
            "model_prob_final_win",
            "model_prob_deficit_erased",
            "selection_prob",
            "market_raw_break_even_prob",
            "edge_at_entry_final_win",
            "edge_at_entry_deficit_erased",
            "price_source_type",
        ]:
            assert col in bets.columns, f"missing N09 betting column {col}"
        assert set(bets["price_source_type"]).issubset({"real_moneyline", "synthetic_fallback"})
        assert {"B_final_win_model_edge", "B_deficit_erased_heuristic"}.issubset(set(bets["simulation"]))
    summary_rows = betting["summary"]
    assert any(
        row["simulation"] == "B_final_win_model_edge"
        and row["staking_rule"] == "flat_1u"
        and row["price_subset"] == "real_moneyline_only"
        for row in summary_rows
    ), "missing same-label real-moneyline Sim B summary"
    assert any(
        row["simulation"] == "B_deficit_erased_heuristic"
        and row["staking_rule"] == "flat_1u"
        and row["price_subset"] == "synthetic_fallback_only"
        for row in summary_rows
    ), "missing heuristic synthetic-fallback Sim B summary"
    summary_lower = summary.lower()
    assert "structural edge" in summary_lower
    assert "betting edge" in summary_lower
    print(f"[ok] N09 deliverables verified strat_rows={len(strat):,}, bet_rows={len(bets):,}")
else:
    print("[ok] N09 notebook scaffold verified; deliverables not present yet")

print(f"[ok] N09 notebook structure valid. cells={len(cells)}")
print(f"file: {NB_PATH} ({NB_PATH.stat().st_size:,} bytes)")
