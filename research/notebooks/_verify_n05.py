"""Validate the N05 notebook scaffold and, when present, deliverables."""

from __future__ import annotations

import ast
import json
import pathlib
import sys

import pandas as pd

NB_PATH = pathlib.Path(__file__).resolve().parent / "05_comeback_rate_validation.ipynb"
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
for cell in cells:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    assert cell.get("execution_count") in (None, 0), (
        f"cell {cell['id']} has execution_count={cell['execution_count']!r}"
    )
    assert cell.get("outputs") == [], f"cell {cell['id']} has outputs"
    try:
        ast.parse(src)
    except SyntaxError as exc:
        bad.append((cell["id"], f"{exc.msg} at line {exc.lineno}"))
if bad:
    for cid, msg in bad:
        print(f"  {cid}: {msg}")
    sys.exit(1)

all_text = raw
all_code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")

for marker in [
    "favorite_final_win",
    "deficit_erased",
    "deficit_erased_chrono_key",
    "BOOTSTRAP_RESAMPLES = 10_000",
    "baseline_C_policy",
    "02b_opening_drive_shock.ipynb",
    "n05_descriptive_rates.parquet",
    "n05_analysis_results.json",
    "n05_summary_report.md",
    "period_seconds_elapsed=900-clock_seconds_in_period_total",
]:
    assert marker in all_text, f"N05 marker missing: {marker!r}"

for forbidden in [
    "httpx.get",
    "requests.get",
    "force_refresh=True",
]:
    assert forbidden not in all_code, f"forbidden N05 code path present: {forbidden!r}"

for required_code in [
    "def _chrono_key",
    "readonly_cfbd_get",
    "bootstrap_cluster_mean_ci",
    "brier_improvement_baseline_C_minus_model",
    "game_id",
]:
    assert required_code in all_code, f"N05 code marker missing: {required_code!r}"

parquet_path = RESULTS_DIR / "n05_descriptive_rates.parquet"
json_path = RESULTS_DIR / "n05_analysis_results.json"
summary_path = RESULTS_DIR / "n05_summary_report.md"

if parquet_path.exists() or json_path.exists() or summary_path.exists():
    assert parquet_path.exists(), f"missing {parquet_path}"
    assert json_path.exists(), f"missing {json_path}"
    assert summary_path.exists(), f"missing {summary_path}"

    desc = pd.read_parquet(parquet_path)
    analysis = json.loads(json_path.read_text(encoding="utf-8"))
    summary = summary_path.read_text(encoding="utf-8")

    required_cols = {
        "game_id",
        "trigger_play_id",
        "fav_deficit",
        "trigger_sequence",
        "quarter",
        "period_seconds_elapsed",
        "time_bucket",
        "fav_team",
        "dog_team",
        "season",
        "favorite_final_win",
        "deficit_erased",
        "deficit_erased_chrono_key",
    }
    missing = required_cols - set(desc.columns)
    assert not missing, f"missing descriptive columns: {sorted(missing)}"
    assert len(desc) == 11416, f"expected 11,416 descriptive rows, got {len(desc):,}"
    assert desc[["game_id", "fav_deficit", "trigger_sequence"]].duplicated().sum() == 0
    assert set(desc["time_bucket"].dropna().unique()) <= {"Q1", "Q2-first-half", "Q3", "Q4"}
    assert desc["favorite_final_win"].notna().all()

    assert analysis["config"]["bootstrap_resamples"] == 10000
    assert analysis["data_quality"]["n_descriptive_rows"] == 11416
    assert analysis["data_quality"]["n_model_validation_rows_scheme_U"] == 3857
    for label in ["favorite_final_win", "deficit_erased"]:
        assert label in analysis["descriptive"]
        assert label in analysis["model_validation"]
        assert "overall_brier_vs_baseline_C" in analysis["model_validation"][label]
        assert "threshold_analysis" in analysis["model_validation"][label]
        assert "quintile_analysis" in analysis["model_validation"][label]
        assert "decile_analysis" in analysis["model_validation"][label]
        assert "per_deficit_analysis" in analysis["model_validation"][label]
    assert "Primary finding" in summary
    assert "deficit x time" in summary
    assert "favorite_final_win" in summary and "deficit_erased" in summary

    print(f"[ok] N05 deliverables verified rows={len(desc):,}")
else:
    print("[ok] N05 notebook scaffold verified; deliverables not present yet")
