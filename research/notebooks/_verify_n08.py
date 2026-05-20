"""Validate the N08 notebook scaffold and, when present, deliverables."""

from __future__ import annotations

import ast
import json
import pathlib
import sys

import pandas as pd

NB_PATH = pathlib.Path(__file__).resolve().parent / "08_stern_winston_conformal_diagnostic.ipynb"
REPO_ROOT = NB_PATH.parents[2]
RESULTS_DIR = REPO_ROOT / "research" / "results"

raw = NB_PATH.read_text(encoding="utf-8")
nb = json.loads(raw)

assert nb["nbformat"] == 4, f"nbformat={nb['nbformat']}"
assert nb["nbformat_minor"] == 5, f"nbformat_minor={nb['nbformat_minor']}"

cells = nb["cells"]
cell_ids = [cell["id"] for cell in cells]
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
all_code = "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")

for marker in [
    "Stern-Winston",
    "conformal",
    "M1_N06",
    "M2_N07_EXP",
    "M3_N06_CONFORMAL",
    "baseline_sw_pfr_prob",
    "baseline_sw_cfb_prob",
    "baseline_C_deficit_erased",
    "validation_slice_fits_rematerialized_for_conformal_only",
    "n08_diagnostic_predictions.parquet",
    "n08_comparison_results.json",
    "n08_summary_report.md",
    "n08_price_conversion_spec.json",
]:
    assert marker in all_text, f"N08 marker missing: {marker!r}"

for forbidden in [
    "httpx.get",
    "requests.get",
    "force_refresh=True",
    "FEATURE_VALIDATION_CSV.write_text",
    "to_csv(FEATURE_VALIDATION_CSV",
    "C_VALUES",
    "xgboost",
]:
    assert forbidden not in all_code, f"forbidden N08 code path present: {forbidden!r}"

for required_code in [
    "readonly_cfbd_get",
    "run_phase0_matrix_notebook",
    "stern_winston_prob",
    "bootstrap_cluster_mean_ci",
    "fit_all_folds",
    "n06_max_abs_diff <= 1e-10",
    "n07_max_abs_diff <= 1e-10",
    "Conformal validation coverage materially below 95%",
]:
    assert required_code in all_code, f"N08 code marker missing: {required_code!r}"

pred_path = RESULTS_DIR / "n08_diagnostic_predictions.parquet"
comparison_path = RESULTS_DIR / "n08_comparison_results.json"
summary_path = RESULTS_DIR / "n08_summary_report.md"
price_path = RESULTS_DIR / "n08_price_conversion_spec.json"

if pred_path.exists() or comparison_path.exists() or summary_path.exists() or price_path.exists():
    for path in [pred_path, comparison_path, summary_path, price_path]:
        assert path.exists(), f"missing N08 deliverable: {path}"

    pred = pd.read_parquet(pred_path)
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    price = json.loads(price_path.read_text(encoding="utf-8"))
    summary = summary_path.read_text(encoding="utf-8")

    assert len(pred) == 3854, f"expected 3,854 held-out rows, got {len(pred):,}"
    key = ["game_id", "fav_deficit", "trigger_sequence", "fold"]
    assert pred[key].duplicated().sum() == 0
    for col in [
        "n06_prob",
        "n07_prob",
        "baseline_c_prob",
        "baseline_sw_pfr_prob",
        "baseline_sw_cfb_prob",
        "conformal_lower",
        "conformal_upper",
    ]:
        assert col in pred.columns, f"N08 predictions missing {col}"
        assert pred[col].between(0, 1).all(), f"{col} outside [0,1]"
    assert (pred["conformal_lower"] <= pred["conformal_upper"]).all()

    matrix = comparison["comparison_matrix"]
    assert len(matrix) == 18, f"expected 18 comparison rows, got {len(matrix)}"
    assert comparison["design_resolution"]["no_new_models_trained"] is True
    assert comparison["deployment_recommendation"]["recommended_model"] in {
        "M1_N06",
        "M2_N07_EXP",
        "M3_N06_CONFORMAL",
    }
    assert price["function_name"] == "stern_winston_favorite_win_probability_v1"
    assert price["formula"]["pregame_spread_coefficient"] == 0.0
    assert "Deployment recommendation" in summary
    assert "Comparison Matrix: deficit_erased" in summary

    print(f"[ok] N08 deliverables verified rows={len(pred):,}, matrix_rows={len(matrix)}")
else:
    print("[ok] N08 notebook scaffold verified; deliverables not present yet")

print(f"[ok] N08 notebook structure valid. cells={len(cells)}")
print(f"file: {NB_PATH} ({NB_PATH.stat().st_size:,} bytes)")
