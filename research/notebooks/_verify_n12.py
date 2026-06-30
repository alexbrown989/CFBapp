"""Validate N12 notebook scaffold and deliverables."""

from __future__ import annotations

import ast
import importlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

NB_PATH = pathlib.Path(__file__).resolve().parent / "12_probability_lookup_layer.ipynb"
REPO_ROOT = NB_PATH.parents[2]
RESULTS_DIR = REPO_ROOT / "research" / "results"
NOTEBOOK_DIR = REPO_ROOT / "research" / "notebooks"

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
    "n12_probability_lookup.parquet",
    "n12_live_scoring_spec.json",
    "n12_lookup_key_schema.md",
    "n12_summary_report.md",
    "score_live_trigger",
    "baseline_c_rate",
    "conditional_rate_full",
    "ranking_rate",
    "stern_winston_state_price",
]:
    assert marker in all_text, f"N12 marker missing: {marker!r}"

for forbidden in [
    "requests.get",
    "httpx.get",
    "LogisticRegression(",
    "StandardScaler()",
    "IsotonicRegression(",
    "fit(",
    "force_refresh=True",
]:
    if forbidden == "fit(":
        assert ".fit(" not in all_code, "forbidden N12 fit call present"
    else:
        assert forbidden not in all_code, f"forbidden N12 code path present: {forbidden!r}"

lookup_path = RESULTS_DIR / "n12_probability_lookup.parquet"
spec_path = RESULTS_DIR / "n12_live_scoring_spec.json"
schema_path = RESULTS_DIR / "n12_lookup_key_schema.md"
summary_path = RESULTS_DIR / "n12_summary_report.md"
lib_path = NOTEBOOK_DIR / "_lib_lookup.py"

if any(p.exists() for p in [lookup_path, spec_path, schema_path, summary_path]):
    for path in [lookup_path, spec_path, schema_path, summary_path, lib_path]:
        assert path.exists(), f"missing N12 deliverable: {path}"

    lookup = pd.read_parquet(lookup_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    schema = schema_path.read_text(encoding="utf-8")
    summary = summary_path.read_text(encoding="utf-8")

    expected_cols = [
        "lookup_key",
        "dimension_set",
        "metric_name",
        "label",
        "value",
        "ci_lower",
        "ci_upper",
        "n_events",
        "n_games",
        "n_seasons",
        "reliability_flag",
        "source_notebook",
        "source_artifact",
    ]
    assert list(lookup.columns) == expected_cols
    required_metrics = {
        "baseline_c_rate",
        "n06_calibrated_prob",
        "conformal_lower",
        "conformal_upper",
        "stern_winston_state_price",
        "conditional_rate_full",
        "ranking_rate",
        "market_no_vig_historical",
    }
    present = set(lookup["metric_name"])
    missing = required_metrics - present
    assert not missing, f"missing N12 metrics: {missing}"
    counts = lookup.groupby("metric_name").size().to_dict()
    for metric in required_metrics:
        assert counts[metric] > 0, f"metric has zero rows: {metric}"
    assert lookup[["lookup_key", "dimension_set", "metric_name", "label", "source_notebook", "source_artifact"]].notna().all().all()
    assert lookup["value"].notna().all()
    assert set(lookup["label"]).issubset({"favorite_final_win", "deficit_erased"})
    assert set(lookup["reliability_flag"]).issubset({"reliable", "thin", "unreliable", "n_a"})

    n09 = json.loads((RESULTS_DIR / "n09_baseline_analysis.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(NOTEBOOK_DIR))
    import _lib_lookup

    importlib.reload(_lib_lookup)
    _lib_lookup.load_lookup.cache_clear()
    _lib_lookup.load_scoring_spec.cache_clear()

    for rec in n09["section1_baseline_C"]["twenty_cell_rate_table"]:
        key = _lib_lookup.build_lookup_key("baseline_c_rate", deficit=rec["fav_deficit"], time_bucket=rec["time_bucket"])
        row = lookup[
            (lookup["lookup_key"] == key)
            & (lookup["metric_name"] == "baseline_c_rate")
            & (lookup["label"] == rec["label"])
        ]
        assert len(row) == 1
        assert abs(float(row["value"].iloc[0]) - float(rec["rate"])) < 1e-15

    assert spec["deployment_choice"]["scheme"] == "E"
    assert spec["deployment_choice"]["train_window"] == "2015-2023"
    assert spec["n12_reproduction_check"]["passed"] is True
    assert spec["n12_reproduction_check"]["calibrated_prob_max_abs_diff"] < 1e-6

    n06 = pd.read_parquet(RESULTS_DIR / "n06_calibrated_predictions.parquet")
    fit_features = {
        (fit["scheme"], int(fit["fold"])): fit["core_features"]
        for fit in spec["n06_fitted_state"]["fits"]
    }
    raw_diffs: list[float] = []
    cal_diffs: list[float] = []
    for rec in n06.to_dict(orient="records"):
        features = {feat: rec.get(feat) for feat in fit_features[(rec["scheme"], int(rec["fold"]))]}
        scored = _lib_lookup.score_live_trigger(features, scheme=rec["scheme"], fold=int(rec["fold"]), spec_path=spec_path)
        raw_diffs.append(abs(float(scored["raw_model_prob"]) - float(rec["raw_model_prob"])))
        cal_diffs.append(abs(float(scored["calibrated_prob"]) - float(rec["calibrated_prob"])))
    assert max(cal_diffs) < 1e-6
    assert max(raw_diffs) < 1e-6

    assert "lookup_key" in schema
    assert "score_live_trigger" in schema
    assert "No New Estimates" in summary
    print(f"[ok] N12 deliverables verified rows={len(lookup):,}, metrics={counts}")
    print(f"[ok] score_live_trigger max raw diff={max(raw_diffs):.12g}, max cal diff={max(cal_diffs):.12g}")
else:
    print("[ok] N12 notebook scaffold verified; deliverables not present yet")

print(f"[ok] N12 notebook structure valid. cells={len(cells)}")
print(f"file: {NB_PATH} ({NB_PATH.stat().st_size:,} bytes)")
