"""Validate the N04 notebook scaffold and, when present, its deliverables."""

from __future__ import annotations

import ast
import json
import pathlib
import sys

import pandas as pd

NB_PATH = pathlib.Path(__file__).resolve().parent / "04_model_vs_market_validation.ipynb"
REPO_ROOT = NB_PATH.parents[2]
RESULTS_DIR = REPO_ROOT / "research" / "results"

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

all_code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
all_text = raw

for marker in [
    "N04 validates whether N03's calibrated trigger-state probabilities",
    "n04_validation_results.parquet",
    "n04_summary_report.md",
    "n04_spec.json",
]:
    assert marker in all_text, f"N04 marker missing: {marker!r}"

for marker in [
    "N_BOOTSTRAPS = 10_000",
    "BOOTSTRAP_SEED = 42",
    "cluster",
    "game_id",
    "SPORTSBOOKS = [\"Bovada\", \"DraftKings\", \"ESPN Bet\", \"Caesars\", \"William Hill (New Jersey)\"]",
    "ALGORITHMIC_PROVIDERS = [\"teamrankings\", \"numberfire\"]",
    "moneyline_side_conflict",
    "spread_conversion_moneyline_side_conflict",
]:
    assert marker in all_code, f"N04 code marker missing: {marker!r}"

for forbidden in [
    "httpx.get",
    "requests.get",
    "force_refresh=True",
]:
    assert forbidden not in all_code, f"forbidden N04 code path present: {forbidden!r}"

results_path = RESULTS_DIR / "n04_validation_results.parquet"
summary_path = RESULTS_DIR / "n04_summary_report.md"
spec_path = RESULTS_DIR / "n04_spec.json"

if results_path.exists() or summary_path.exists() or spec_path.exists():
    assert results_path.exists(), f"missing {results_path}"
    assert summary_path.exists(), f"missing {summary_path}"
    assert spec_path.exists(), f"missing {spec_path}"

    results = pd.read_parquet(results_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    summary = summary_path.read_text(encoding="utf-8")

    required_cols = {
        "game_id",
        "trigger_play_id",
        "scheme",
        "fold",
        "split_role",
        "fav_deficit",
        "trigger_sequence",
        "actual",
        "p_model",
        "p_market",
        "edge",
        "brier_model",
        "brier_market",
        "brier_improvement",
        "log_loss_model",
        "log_loss_market",
        "log_loss_improvement",
        "market_status",
        "market_provider_used",
        "fallback_reason",
        "decimal_odds",
    }
    missing = required_cols - set(results.columns)
    assert not missing, f"missing result columns: {sorted(missing)}"

    key_cols = ["game_id", "fav_deficit", "trigger_sequence", "scheme", "fold"]
    dup_count = int(results.duplicated(key_cols).sum())
    assert dup_count == 0, f"N04 result key duplicates on {key_cols}: {dup_count}"

    assert results["p_model"].between(0, 1, inclusive="both").all()
    assert results["p_market"].between(0, 1, inclusive="both").all()
    assert set(results["actual"].unique()).issubset({0, 1})
    assert results["market_status"].isin(["moneyline", "spread_conversion"]).all()
    assert results["market_probability"].isna().sum() == 0 if "market_probability" in results.columns else True

    fallback_reasons = set(results["fallback_reason"].fillna("").unique())
    allowed_fallback_reasons = {"", "no_moneyline_available", "moneyline_side_conflict"}
    assert fallback_reasons <= allowed_fallback_reasons, fallback_reasons

    assert spec["spread_conversion_model"]["coefficient"] < 0
    assert spec["bootstrap"]["cluster"] == "game_id"
    assert spec["bootstrap"]["n_resamples"] == 10000
    assert spec["line_coverage"]["market_probability_missing_rows"] == 0
    assert "fallback_unique_games_moneyline_side_conflict" in spec["line_coverage"]
    assert set(spec["result_status"]) == {"U", "W2"}
    assert len(spec["primary_metrics"]) >= 8
    assert spec["data_provenance"]["provider_policy"]["excluded_algorithmic_providers"] == [
        "teamrankings",
        "numberfire",
    ]
    assert "Primary finding" in summary
    assert "pre-game market" in summary

    print(f"[ok] N04 deliverables verified rows={len(results):,}")
else:
    print("[ok] N04 notebook scaffold verified; deliverables not present yet")
