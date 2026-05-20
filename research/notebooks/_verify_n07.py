"""Validate the N07 notebook scaffold and, when present, deliverables."""

from __future__ import annotations

import ast
import json
import pathlib
import sys

import pandas as pd

NB_PATH = pathlib.Path(__file__).resolve().parent / "07_feature_pool_expansion.ipynb"
REPO_ROOT = NB_PATH.parents[2]
RESULTS_DIR = REPO_ROOT / "research" / "results"

N07_FEATURES = [
    "estimated_possessions_remaining",
    "deficit_per_remaining_possession",
    "possessions_needed_to_tie",
    "clock_pressure_index",
    "dog_points_from_turnovers_pct",
    "dog_points_from_returns_pct",
    "dog_points_from_explosives_pct",
    "dog_offensive_points_pct",
    "fav_yards_per_point_ratio",
    "epa_per_play_gap",
    "success_rate_gap",
    "third_down_gap",
    "explosive_rate_gap",
    "drive_yards_gap",
]

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
    "PRE_REGISTERED_FEATURES",
    "BONFERRONI_ALPHA",
    "baseline_C",
    "alpha_predictions",
    "N07_FEATURE_NAMES",
    "n07_descriptive_features.parquet",
    "n07_stability_results.json",
    "n07_summary_report.md",
    "n07_expanded_model_predictions.parquet",
    "n07_expanded_model_spec.json",
    "stable_seed_offset",
    "seconds_remaining_in_regulation",
    "play_number",
]:
    assert marker in all_text, f"N07 marker missing: {marker!r}"

for feature in N07_FEATURES:
    assert feature in all_text, f"pre-registered feature missing: {feature!r}"

for forbidden in [
    "httpx.get",
    "requests.get",
    "force_refresh=True",
    "FEATURE_VALIDATION_CSV.write_text",
    "to_csv(FEATURE_VALIDATION_CSV",
]:
    assert forbidden not in all_code, f"forbidden N07 code path present: {forbidden!r}"

for required_code in [
    "readonly_cfbd_get",
    "run_phase0_matrix_notebook",
    "TRIGGER_OUTCOMES_CSV",
    "assert len(base_df) == 11412",
    "bootstrap_cluster_mean_ci",
    "baseline_C_bonferroni",
]:
    assert required_code in all_code, f"N07 code marker missing: {required_code!r}"

features_path = RESULTS_DIR / "n07_descriptive_features.parquet"
stability_path = RESULTS_DIR / "n07_stability_results.json"
summary_path = RESULTS_DIR / "n07_summary_report.md"
expanded_pred_path = RESULTS_DIR / "n07_expanded_model_predictions.parquet"
expanded_spec_path = RESULTS_DIR / "n07_expanded_model_spec.json"

if features_path.exists() or stability_path.exists() or summary_path.exists():
    assert features_path.exists(), f"missing {features_path}"
    assert stability_path.exists(), f"missing {stability_path}"
    assert summary_path.exists(), f"missing {summary_path}"

    features = pd.read_parquet(features_path)
    stability = json.loads(stability_path.read_text(encoding="utf-8"))
    summary = summary_path.read_text(encoding="utf-8")

    assert len(features) == 11412, f"expected 11,412 N07 feature rows, got {len(features):,}"
    assert features[["game_id", "fav_deficit", "trigger_sequence"]].duplicated().sum() == 0
    missing = [feature for feature in N07_FEATURES if feature not in features.columns]
    assert not missing, f"N07 feature parquet missing columns: {missing}"
    assert features["deficit_erased"].notna().all(), "N07 rows should exclude deficit_erased NaNs"

    assert len(stability["pre_registered_features"]) == 14
    assert len(stability["feature_results"]) == 14
    assert set(stability["category_summary"]) == {"A", "B", "C"}
    assert stability["data_quality"]["n_feature_rows"] == 11412
    assert stability["data_quality"]["edge_case_handling"]["dog_points_from_explosives_pct_gt_1_set_to_null"] == 5
    assert stability["data_quality"]["edge_case_handling"]["fav_yards_per_point_ratio_lt_0_set_to_null"] == 5
    assert "Primary finding" in summary
    assert "baseline_C" in summary

    passing = stability.get("passing_features", [])
    if passing:
        assert expanded_pred_path.exists(), f"missing expanded predictions despite passing features: {passing}"
        assert expanded_spec_path.exists(), f"missing expanded spec despite passing features: {passing}"
        expanded = pd.read_parquet(expanded_pred_path)
        expanded_spec = json.loads(expanded_spec_path.read_text(encoding="utf-8"))
        assert set(expanded["scheme"].unique()) == {"U", "W2"}
        key = ["game_id", "fav_deficit", "trigger_sequence", "scheme", "fold"]
        assert expanded[key].duplicated().sum() == 0
        assert expanded["calibrated_prob"].between(0, 1).all()
        assert expanded_spec["expanded_features"] == passing
    else:
        assert not expanded_pred_path.exists(), "expanded predictions should not exist when no features pass"
        assert not expanded_spec_path.exists(), "expanded spec should not exist when no features pass"

    print(f"[ok] N07 deliverables verified rows={len(features):,}, passing={len(passing)}")
else:
    print("[ok] N07 notebook scaffold verified; deliverables not present yet")

print(f"[ok] N07 notebook structure valid. cells={len(cells)}")
print(f"file: {NB_PATH} ({NB_PATH.stat().st_size:,} bytes)")
