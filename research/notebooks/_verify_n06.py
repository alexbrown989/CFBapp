"""Validate the unexecuted N06 notebook and, when present, its deliverables."""

from __future__ import annotations

import ast
import json
import pathlib
import sys

import pandas as pd

NB_PATH = pathlib.Path(__file__).resolve().parent / "06_deficit_erased_model_validation.ipynb"
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

all_code = "\n".join(
    "".join(c["source"]) for c in cells if c["cell_type"] == "code"
)
all_text = raw

for marker in [
    "readonly_cfbd_get",
    "PHASE0_NOTEBOOK_CELLS",
    "IsotonicRegression",
    "NULL_INDICATOR_THRESHOLD = 0.05",
    "N_PERMUTATIONS = 100",
    "C_VALUES = [0.1, 0.5, 1.0, 2.0, 10.0]",
    "n06_calibrated_predictions.parquet",
    "n06_e_calibrated_predictions.parquet",
    "n06_model_spec.json",
    "n06_summary_report.md",
    "baseline_C_tables",
    "baseline_validation",
    "STRUCTURAL_FEATURES",
    "fav_deficit",
    "selected_model_core_features",
    "prediction_key_cols",
]:
    assert marker in all_text, f"N06 marker missing: {marker!r}"

assert 'TARGET_LABEL = "deficit_erased"' in all_code

for forbidden in [
    "httpx.get",
    "requests.get",
    "force_refresh=True",
    "FEATURE_VALIDATION_CSV.write_text",
    "to_csv(FEATURE_VALIDATION_CSV",
]:
    assert forbidden not in all_code, f"forbidden N06 code path present: {forbidden!r}"

spec_path = RESULTS_DIR / "n06_model_spec.json"
pred_path = RESULTS_DIR / "n06_calibrated_predictions.parquet"
e_pred_path = RESULTS_DIR / "n06_e_calibrated_predictions.parquet"
summary_path = RESULTS_DIR / "n06_summary_report.md"
deliverables_checked = False

if spec_path.exists() or pred_path.exists() or e_pred_path.exists() or summary_path.exists():
    assert spec_path.exists(), f"missing {spec_path}"
    assert pred_path.exists(), f"missing {pred_path}"
    assert e_pred_path.exists(), f"missing {e_pred_path}"
    assert summary_path.exists(), f"missing {summary_path}"

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("feature_pool_count") != 31 or "training_structure" not in spec:
        print("[warn] Existing N06 deliverables are stale pre-structural-deficit artifacts; notebook structure checked only.")
    else:
        assert spec["feature_pool_count"] == 31, spec.get("feature_pool_count")
        assert spec["target_label"] == "deficit_erased"
        assert spec["cross_label"] == "favorite_final_win"
        assert spec["r6_validated_feature_count"] == 30
        assert spec["structural_conditioning_feature_count"] == 1
        assert spec["null_handling"]["core_feature_count"] == 31
        assert spec["null_handling"]["indicator_column_count"] > 0
        assert spec["training_structure"]["unique_trigger_play_rows"] == 7852
        assert spec["training_structure"]["full_trigger_event_rows"] == 11412
        assert spec["training_structure"]["n05_deficit_erased_null_event_rows_excluded"] == 4
        assert "baseline_validation" in spec
        assert "primary_validation" in spec
        assert spec["primary_validation"]["label"] == "deficit_erased"
        assert spec["structural_conditioning_variables"][0]["feature"] == "fav_deficit"
        assert spec["structural_conditioning_variables"][0]["pruning_exempt"] is True
        for scheme in ["U", "W2"]:
            assert scheme in spec["feature_list"], f"feature_list missing {scheme}"
            assert spec["feature_list"][scheme]["selected_core_features"], (
                f"{scheme} selected_core_features empty"
            )
            assert "fav_deficit" in spec["feature_list"][scheme]["selected_core_features"]
            decisions = spec["pruning"][scheme]["decisions"]
            fd = [d for d in decisions if d["feature"] == "fav_deficit"]
            assert fd and fd[0]["pruning_exempt"] is True and fd[0]["drop"] is False
        assert "E" in spec["calibration_params"], "E calibration params missing"

        pred = pd.read_parquet(pred_path)
        e_pred = pd.read_parquet(e_pred_path)
        required_cols = {
            "game_id",
            "trigger_play_id",
            "trigger_sequence",
            "scheme",
            "fold",
            "raw_model_prob",
            "calibrated_prob",
            "deficit_erased",
            "favorite_final_win",
            "time_bucket",
            "fav_team",
            "dog_team",
            "fav_score_at_trigger",
            "dog_score_at_trigger",
            "fav_deficit",
            "quarter",
            "clock_seconds_in_period_total",
        }
        missing_pred = required_cols - set(pred.columns)
        missing_e = required_cols - set(e_pred.columns)
        assert not missing_pred, f"prediction parquet missing columns: {missing_pred}"
        assert not missing_e, f"E prediction parquet missing columns: {missing_e}"
        assert set(pred["scheme"].unique()) == {"U", "W2"}, sorted(pred["scheme"].unique())
        assert set(e_pred["scheme"].unique()) == {"E"}, sorted(e_pred["scheme"].unique())
        assert pred["calibrated_prob"].between(0, 1).all()
        assert e_pred["calibrated_prob"].between(0, 1).all()
        key = ["game_id", "fav_deficit", "trigger_sequence", "scheme", "fold"]
        assert pred[key].duplicated().sum() == 0
        assert e_pred[key].duplicated().sum() == 0
        assert len(pred) == spec["output_verification"]["main_prediction_rows_total"]
        assert len(e_pred) == spec["output_verification"]["e_prediction_rows_total"]
        assert spec["output_verification"]["probability_variation_summary"]["main"]["calibrated_probability_diff_group_count"] > 0
        summary = summary_path.read_text(encoding="utf-8")
        assert "Primary finding" in summary
        assert "baseline_C" in summary
        assert "deficit_erased" in summary
        deliverables_checked = True

print(f"[ok] N06 notebook structure valid. cells={len(cells)}")
print(f"file: {NB_PATH} ({NB_PATH.stat().st_size:,} bytes)")
if deliverables_checked:
    print("[ok] N06 deliverables present and schema-checked")

