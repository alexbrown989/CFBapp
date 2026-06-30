"""
Deterministic builder for research/notebooks/06_deficit_erased_model_validation.ipynb.

N06 consumes the Phase 0 R6-PASS feature pool, rebuilds feature matrices
from the committed 02a-02g notebook extractors in read-only/cache-only mode,
fits calibrated walk-forward models on the N05 `deficit_erased` label, and
writes comeback-detection validation artifacts.
"""

from __future__ import annotations

import json
import pathlib
import textwrap

OUT = pathlib.Path(__file__).resolve().parent / "06_deficit_erased_model_validation.ipynb"

CELLS: list[tuple[str, str, str]] = []


def add(cell_type: str, cell_id: str, src: str) -> None:
    CELLS.append((cell_type, cell_id, textwrap.dedent(src).lstrip("\n")))


add("markdown", "m06_0000", """
# Notebook 06 -- Deficit-erased model validation

N06 fits the full Phase 0 R6-PASS feature pool in calibrated walk-forward
models with one intentional change from N03: the training label is
`deficit_erased` instead of `favorite_final_win`.

Locked design decisions:

- all **30** R6-PASS Phase 0 feature names enter the candidate pool;
- `fav_deficit` enters as a protected structural conditioning variable,
  yielding 31 core model features before missingness indicators;
- primary model is L1 logistic regression with `C=1.0`, `solver='liblinear'`,
  `random_state=42`, and `StandardScaler`;
- C sensitivity sweep uses `C in {0.1, 0.5, 1.0, 2.0, 10.0}`;
- calibration is `IsotonicRegression(out_of_bounds='clip')` fit on the
  validation slice only;
- U and W2 use the same three locked walk-forward folds and differ only in
  aggregation/pruning weights;
- E is the deployment-proximate single window: train 2015-2023, validate on
  2024, no held-out test;
- pruning drops a core feature only when L1, permutation, and ablation agree.
  `fav_deficit` is exempt from pruning because it defines the threshold-
  conditioned trigger state rather than a Phase 0 hypothesis feature.

N06 null policy, explicitly locked after the planning halt:

- features with full-corpus null rate `> 5%` receive a paired missingness
  indicator;
- existing Phase 0 indicators are reused rather than duplicated;
- all NaNs are imputed using train-fold-only medians, then applied to
  train/validation/test without leakage;
- indicators are preprocessing structure, not new semantic candidate features.

N06 training/evaluation is play-level: each unique trigger play is represented
once using its lowest qualifying deficit threshold. Prediction output is
event-level: held-out test plays are replicated across all qualifying deficit
thresholds and rescored with varying `fav_deficit`.

Primary validation is Brier improvement over the strict N05 baseline_C:
training-years-only `fav_deficit x time_bucket` rate for `deficit_erased`.
""")


add("code", "c06_0001", """
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import pathlib
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
except Exception as exc:  # noqa: BLE001
    xgb = None
    XGBOOST_IMPORT_ERROR = repr(exc)
else:
    XGBOOST_IMPORT_ERROR = ""

NOTEBOOK_DIR = pathlib.Path(".").resolve()
RESEARCH_DIR = (NOTEBOOK_DIR / "..").resolve()
REPO_ROOT = (RESEARCH_DIR / "..").resolve()
DATA_DIR = RESEARCH_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = RESEARCH_DIR / "results"
CALL_LOG = CACHE_DIR / "cfbd_call_log.csv"

TRIGGER_EVENTS_CSV = RESULTS_DIR / "trigger_events.csv"
TRIGGER_OUTCOMES_CSV = RESULTS_DIR / "trigger_outcomes.csv"
FEATURE_VALIDATION_CSV = RESULTS_DIR / "feature_validation.csv"
FULL_CORR_CSV = RESULTS_DIR / "_02g_full_correlation_matrix.csv"
N05_DESCRIPTIVE_RATES_PARQUET = RESULTS_DIR / "n05_descriptive_rates.parquet"
N05_ANALYSIS_JSON = RESULTS_DIR / "n05_analysis_results.json"

N06_PREDICTIONS_PARQUET = RESULTS_DIR / "n06_calibrated_predictions.parquet"
N06_E_PREDICTIONS_PARQUET = RESULTS_DIR / "n06_e_calibrated_predictions.parquet"
N06_MODEL_SPEC_JSON = RESULTS_DIR / "n06_model_spec.json"
N06_SUMMARY_REPORT_MD = RESULTS_DIR / "n06_summary_report.md"
N06_FULL_FITTED_STATE_JSON = RESULTS_DIR / "n06_full_fitted_state.json"
N06_STATE_EXPORT_ONLY = os.environ.get("N06_STATE_EXPORT_ONLY", "").strip().lower() in {"1", "true", "yes"}

assert RESEARCH_DIR.name == "research", (
    f"Expected to run inside research/notebooks; got {NOTEBOOK_DIR}"
)
for path in [TRIGGER_EVENTS_CSV, TRIGGER_OUTCOMES_CSV, FEATURE_VALIDATION_CSV, FULL_CORR_CSV, N05_DESCRIPTIVE_RATES_PARQUET, N05_ANALYSIS_JSON]:
    assert path.exists(), f"Missing required Phase 0 artifact: {path}"
assert CACHE_DIR.exists(), f"Missing cache dir: {CACHE_DIR}"

RANDOM_STATE = 42
NULL_INDICATOR_THRESHOLD = 0.05
N_PERMUTATIONS = 100
C_VALUES = [0.1, 0.5, 1.0, 2.0, 10.0]

WALK_FORWARD_WINDOWS = [
    {"fold": 2022, "train_seasons": list(range(2015, 2021)), "val_season": 2021, "test_season": 2022, "train_window_label": "2015-2020"},
    {"fold": 2023, "train_seasons": list(range(2015, 2022)), "val_season": 2022, "test_season": 2023, "train_window_label": "2015-2021"},
    {"fold": 2024, "train_seasons": list(range(2015, 2023)), "val_season": 2023, "test_season": 2024, "train_window_label": "2015-2022"},
]
DEPLOYMENT_WINDOW_E = {
    "fold": 2024,
    "train_seasons": list(range(2015, 2024)),
    "val_season": 2024,
    "test_season": None,
    "train_window_label": "2015-2023",
}

SCHEME_WEIGHTS = {
    "U": {2022: 1 / 3, 2023: 1 / 3, 2024: 1 / 3},
    "W2": {2022: 0.25, 2023: 0.25, 2024: 0.50},
}

TARGET_LABEL = "deficit_erased"
CROSS_LABEL = "favorite_final_win"
LABELS_FOR_VALIDATION = [TARGET_LABEL, CROSS_LABEL]
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 42
EDGE_THRESHOLDS = [0.00, 0.03, 0.05, 0.08, 0.10]

PHASE0_NOTEBOOK_CELLS = {
    "02a_baseline_features.ipynb": [4, 6, 8, 10, 11, 13],
    "02b_opening_drive_shock.ipynb": [4, 6, 8, 10, 11, 13],
    "02c_explosive_vs_sustained.ipynb": [4, 6, 8, 10, 12, 13, 15],
    "02d_turnover_and_short_field.ipynb": [4, 6, 8, 10, 11, 13],
    "02e_red_zone_failure.ipynb": [4, 6, 8, 10, 11, 13],
    "02f_down_distance_efficiency.ipynb": [4, 6, 8, 10, 11, 13],
    "02g_context_week_home_neutral.ipynb": [4, 6, 8, 10],
}

PHASE0_INDICATOR_MAP = {
    "seconds_since_last_dog_explosive_play": "seconds_since_last_dog_explosive_play_is_null",
    "fav_yards_per_point": "fav_yards_per_point_is_null",
    "fav_early_down_success_rate": "fav_early_down_success_rate_insufficient_sample",
    "fav_third_down_success_rate": "fav_third_down_success_rate_insufficient_sample",
    "dog_early_down_success_rate": "dog_early_down_success_rate_insufficient_sample",
    "dog_third_down_success_rate": "dog_third_down_success_rate_insufficient_sample",
}

STRUCTURAL_FEATURES = ["fav_deficit"]
PROTECTED_FEATURES = set(STRUCTURAL_FEATURES)

CONTEXT_COLUMNS = [
    "fav_team",
    "dog_team",
    "fav_score_at_trigger",
    "dog_score_at_trigger",
    "quarter",
    "clock_seconds_in_period_total",
]

N06_CONTEXT_COLUMNS = [
    "fav_team",
    "dog_team",
    "fav_score_at_trigger",
    "dog_score_at_trigger",
    "fav_deficit",
    "quarter",
    "clock_seconds_in_period_total",
]

N06_OUTPUT_COLUMNS = [
    "game_id",
    "trigger_play_id",
    "trigger_sequence",
    "season",
    "time_bucket",
    "deficit_erased",
    "favorite_final_win",
    *N06_CONTEXT_COLUMNS,
]

print(f"[ok] N06 paths resolved at {NOTEBOOK_DIR}")
print(f"[ok] xgboost available: {xgb is not None} {XGBOOST_IMPORT_ERROR}")
""")


add("code", "c06_0002", """
def _params_hash(params: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def _cache_key(endpoint: str, params: dict[str, Any]) -> pathlib.Path:
    endpoint_key = endpoint.strip("/").replace("/", "_")
    return CACHE_DIR / f"cfbd__{endpoint_key}__{_params_hash(params)}.json"


def readonly_cfbd_get(endpoint: str, force_refresh: bool = False, **params: Any) -> Any:
    if force_refresh:
        raise AssertionError("N06 forbids force_refresh; cache-only extraction is required")
    key = _cache_key(endpoint, params)
    if not key.exists():
        raise AssertionError(
            f"N06 missing local cache for {endpoint} {params}; halt before any external fetch."
        )
    return json.loads(key.read_text(encoding="utf-8"))


def _base_phase0_namespace() -> dict[str, Any]:
    return {
        "__name__": "_n06_phase0_cell_exec",
        "Any": Any,
        "Callable": Callable,
        "contextlib": contextlib,
        "hashlib": hashlib,
        "io": io,
        "json": json,
        "math": math,
        "np": np,
        "os": os,
        "pathlib": pathlib,
        "pd": pd,
        "subprocess": subprocess,
        "time": time,
        "NOTEBOOK_DIR": NOTEBOOK_DIR,
        "RESEARCH_DIR": RESEARCH_DIR,
        "DATA_DIR": DATA_DIR,
        "RESULTS_DIR": RESULTS_DIR,
        "CACHE_DIR": CACHE_DIR,
        "CALL_LOG": CALL_LOG,
        "REPO_ROOT": REPO_ROOT,
        "TRIGGER_EVENTS_CSV": TRIGGER_EVENTS_CSV,
        "TRIGGER_OUTCOMES_CSV": TRIGGER_OUTCOMES_CSV,
        "FEATURE_VALIDATION_CSV": FEATURE_VALIDATION_CSV,
        "FEATURE_VALIDATION_SCHEMA": RESULTS_DIR / "feature_validation.schema.md",
        "cfbd_get": readonly_cfbd_get,
    }


def run_phase0_matrix_notebook(nb_name: str, cell_indexes: list[int]) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    nb_path = NOTEBOOK_DIR / nb_name
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    ns = _base_phase0_namespace()
    stdout = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(stdout):
        for idx in cell_indexes:
            src = "".join(nb["cells"][idx]["source"])
            exec(compile(src, f"{nb_name}:cell{idx}", "exec"), ns)
            # Keep Phase 0 extractor calls cache-only even if a selected cell
            # redefines cfbd_get in a future notebook revision.
            ns["cfbd_get"] = readonly_cfbd_get
    elapsed = time.perf_counter() - t0
    fm = ns.get("feature_matrix_df")
    if fm is None:
        raise AssertionError(f"{nb_name} did not build feature_matrix_df")
    candidates = list(ns.get("CANDIDATE_FEATURES", []))
    return fm.copy(), candidates, {
        "notebook": nb_name,
        "cells": cell_indexes,
        "rows": int(len(fm)),
        "cols": int(fm.shape[1]),
        "candidate_count": int(len(candidates)),
        "elapsed_sec": elapsed,
        "stdout_tail": stdout.getvalue()[-2000:],
    }


def load_pass_feature_pool() -> tuple[list[str], dict[str, dict[str, Any]]]:
    fv = pd.read_csv(FEATURE_VALIDATION_CSV, keep_default_na=False)
    required = {"feature", "feature_set_version", "passed_stability", "redundant_with"}
    missing = required - set(fv.columns)
    assert not missing, f"feature_validation.csv missing columns: {missing}"

    feature_order: list[str] = []
    feature_meta: dict[str, dict[str, Any]] = {}
    for feat in fv["feature"].tolist():
        if feat in feature_order:
            continue
        sub = fv[fv["feature"] == feat].copy()
        if len(sub) == 0:
            continue
        passed_all_rows = bool(sub["passed_stability"].astype(str).str.lower().eq("true").all())
        if not passed_all_rows:
            continue
        feature_order.append(str(feat))
        feature_meta[str(feat)] = {
            "feature_set_version": str(sub["feature_set_version"].iloc[0]),
            "redundant_with": str(sub["redundant_with"].iloc[0]),
            "phase0_brier_improvements": [float(x) for x in sub.sort_values("test_season")["brier_improvement"].tolist()],
            "phase0_ece_improvements": [float(x) for x in sub.sort_values("test_season")["calibration_improvement"].tolist()],
        }
    assert len(feature_order) == 30, (
        f"Expected 30 R6-PASS feature groups from Phase 0, got {len(feature_order)}: {feature_order}"
    )
    return feature_order, feature_meta


print("[ok] cache-only Phase 0 notebook execution helpers defined")
""")


add("code", "c06_0003", """
t_extract0 = time.perf_counter()
pass_features, feature_meta = load_pass_feature_pool()

print("R6-PASS feature pool (30 semantic features):")
for i, feat in enumerate(pass_features, start=1):
    red = feature_meta[feat]["redundant_with"]
    suffix = f"  redundant_with={red}" if red else ""
    print(f"  {i:02d}. {feat}{suffix}")

phase0_matrices: dict[str, pd.DataFrame] = {}
phase0_candidates: dict[str, list[str]] = {}
phase0_extract_summary: list[dict[str, Any]] = []

for nb_name, cell_indexes in PHASE0_NOTEBOOK_CELLS.items():
    fm, candidates, summary = run_phase0_matrix_notebook(nb_name, cell_indexes)
    phase0_matrices[nb_name] = fm
    phase0_candidates[nb_name] = candidates
    phase0_extract_summary.append(summary)
    print(
        f"[ok] {nb_name}: rows={summary['rows']:,}, cols={summary['cols']:,}, "
        f"candidates={summary['candidate_count']}, elapsed={summary['elapsed_sec']:.1f}s"
    )

key_cols = ["game_id", "fav_deficit", "trigger_sequence"]
assert len(set(key_cols + CONTEXT_COLUMNS)) == len(key_cols + CONTEXT_COLUMNS), (
    "Duplicate column in key_cols + CONTEXT_COLUMNS"
)
n05_df = pd.read_parquet(N05_DESCRIPTIVE_RATES_PARQUET)
required_n05_cols = {
    "game_id",
    "fav_deficit",
    "trigger_sequence",
    "trigger_play_id",
    "season",
    "time_bucket",
    "favorite_final_win",
    "deficit_erased",
    "fav_team",
    "dog_team",
    "quarter",
    "clock_seconds_in_period_total",
}
missing_n05_cols = required_n05_cols - set(n05_df.columns)
assert not missing_n05_cols, f"N05 descriptive rates missing columns: {missing_n05_cols}"

trigger_context_df = pd.read_csv(TRIGGER_EVENTS_CSV)
trigger_context_cols = key_cols + [
    "fav_score_at_trigger",
    "dog_score_at_trigger",
]
missing_trigger_cols = set(trigger_context_cols) - set(trigger_context_df.columns)
assert not missing_trigger_cols, f"trigger_events.csv missing context columns: {missing_trigger_cols}"

base_df = n05_df[n05_df[TARGET_LABEL].notna()].copy()
excluded_deficit_erased_null_event_rows = int(n05_df[TARGET_LABEL].isna().sum())
assert excluded_deficit_erased_null_event_rows == 4, (
    f"Expected 4 N05 null deficit_erased event rows, got {excluded_deficit_erased_null_event_rows}"
)
base_df = base_df.merge(
    trigger_context_df[trigger_context_cols],
    on=key_cols,
    how="inner",
    validate="one_to_one",
)
assert base_df.shape[0] == 11412, (
    f"Expected 11,412 rows after N05+trigger_events merge, got {base_df.shape[0]}"
)
base_df[TARGET_LABEL] = base_df[TARGET_LABEL].astype(bool).astype(int)
base_df[CROSS_LABEL] = base_df[CROSS_LABEL].astype(bool).astype(int)

wide_df = base_df[
    key_cols
    + ["season", "trigger_play_id", "time_bucket", TARGET_LABEL, CROSS_LABEL]
    + CONTEXT_COLUMNS
].copy()

indicator_source_columns: set[str] = set(PHASE0_INDICATOR_MAP.values())
for nb_name, fm in phase0_matrices.items():
    cols: list[str] = []
    for col in pass_features:
        if col in fm.columns:
            cols.append(col)
    for col in indicator_source_columns:
        if col in fm.columns:
            cols.append(col)
    cols = list(dict.fromkeys(cols))
    if not cols:
        continue
    merge_cols = key_cols + cols
    before_cols = set(wide_df.columns)
    wide_df = wide_df.merge(fm[merge_cols], on=key_cols, how="left", validate="one_to_one")
    new_cols = [c for c in wide_df.columns if c not in before_cols]
    print(f"[ok] merged {nb_name}: +{len(new_cols)} columns")

missing_features = [f for f in pass_features if f not in wide_df.columns]
assert not missing_features, f"PASS feature(s) missing from rebuilt matrix: {missing_features}"

assert len(wide_df) == len(base_df), f"wide matrix row mismatch: {len(wide_df)} vs {len(base_df)}"
assert wide_df[key_cols].duplicated().sum() == 0, "duplicate trigger keys in N06 wide matrix"

event_df = wide_df.sort_values(
    ["season", "game_id", "trigger_play_id", "fav_deficit", "trigger_sequence"]
).reset_index(drop=True)
assert int(event_df[key_cols].duplicated().sum()) == 0, "duplicate trigger-event rows"

play_key_cols = ["game_id", "trigger_play_id"]
threshold_counts_by_play = event_df.groupby(play_key_cols).size()
threshold_count_distribution = {
    int(k): int(v) for k, v in threshold_counts_by_play.value_counts().sort_index().items()
}
n_multi_threshold_plays_full = int((threshold_counts_by_play > 1).sum())

constant_cols = [c for c in [*pass_features, TARGET_LABEL, CROSS_LABEL, "time_bucket", *CONTEXT_COLUMNS, "season"] if c in event_df.columns]
nonconstant_feature_groups = 0
for _, grp in event_df[event_df.duplicated(play_key_cols, keep=False)].groupby(play_key_cols):
    for col in constant_cols:
        if grp[col].nunique(dropna=False) > 1:
            nonconstant_feature_groups += 1
            break
assert nonconstant_feature_groups == 0, (
    f"{nonconstant_feature_groups} multi-threshold play group(s) vary on non-deficit feature/context columns"
)

canonical_idx = event_df.groupby(play_key_cols)["fav_deficit"].idxmin()
wide_df = event_df.loc[canonical_idx].sort_values(
    ["season", "game_id", "trigger_play_id"]
).reset_index(drop=True)
assert len(wide_df) == int(threshold_counts_by_play.shape[0]), (
    f"play-level matrix row mismatch: {len(wide_df)} vs {threshold_counts_by_play.shape[0]}"
)
assert len(wide_df) == 7852, f"expected 7,852 unique trigger plays after N05 null exclusions, got {len(wide_df):,}"
assert int(wide_df[play_key_cols].duplicated().sum()) == 0, "duplicate trigger plays after dedup"
assert int(wide_df["fav_deficit"].isna().sum()) == 0, "fav_deficit unexpectedly null"
assert wide_df["fav_deficit"].nunique() > 1, "fav_deficit has no variance after play-level dedup"

model_core_features = [*pass_features, *STRUCTURAL_FEATURES]

extract_elapsed = time.perf_counter() - t_extract0
print(f"[ok] full N06 event matrix: {len(event_df):,} rows x {event_df.shape[1]:,} columns")
print(f"[ok] play-level training/evaluation matrix: {len(wide_df):,} rows x {wide_df.shape[1]:,} columns")
print(f"[ok] excluded N05 null {TARGET_LABEL} event rows: {excluded_deficit_erased_null_event_rows}")
print(f"[ok] unique trigger plays: {len(wide_df):,}; multi-threshold plays: {n_multi_threshold_plays_full:,}")
print(f"[ok] thresholds per play distribution: {threshold_count_distribution}")
print(f"[ok] model core features: {len(model_core_features)} = {len(pass_features)} R6-PASS + {len(STRUCTURAL_FEATURES)} structural")
print(f"[ok] extraction elapsed: {extract_elapsed:.1f}s")
""")


add("code", "c06_0004", """
null_rows: list[dict[str, Any]] = []
indicator_meta: dict[str, dict[str, Any]] = {}
model_indicator_cols: list[str] = []

for feat in pass_features:
    null_count = int(event_df[feat].isna().sum())
    null_rate = null_count / len(event_df)
    needs_indicator = null_rate > NULL_INDICATOR_THRESHOLD
    indicator_col = ""
    provenance = ""
    if needs_indicator:
        existing = PHASE0_INDICATOR_MAP.get(feat, "")
        if existing and existing in wide_df.columns:
            indicator_col = existing
            provenance = "phase0_existing"
            # Normalize the source indicator to 0/1 and cross-check against
            # the actual null mask before using it in the model matrix.
            event_df[indicator_col] = event_df[indicator_col].fillna(0).astype(int)
            wide_df[indicator_col] = wide_df[indicator_col].fillna(0).astype(int)
            mismatch = int((event_df[indicator_col].astype(int) != event_df[feat].isna().astype(int)).sum())
            assert mismatch == 0, (
                f"Existing Phase 0 indicator {indicator_col} mismatches null mask for {feat}: {mismatch}"
            )
        else:
            indicator_col = f"{feat}_is_null"
            provenance = "n06_created"
            event_df[indicator_col] = event_df[feat].isna().astype(int)
            wide_df[indicator_col] = wide_df[feat].isna().astype(int)
        model_indicator_cols.append(indicator_col)
        indicator_meta[indicator_col] = {
            "core_feature": feat,
            "provenance": provenance,
            "null_count": null_count,
            "null_rate": null_rate,
        }
    null_rows.append({
        "feature": feat,
        "null_count": null_count,
        "null_rate": null_rate,
        "indicator_col": indicator_col,
        "indicator_provenance": provenance,
    })

model_indicator_cols = list(dict.fromkeys(model_indicator_cols))
null_policy_df = pd.DataFrame(null_rows)

print("N06 null policy summary:")
print(null_policy_df.to_string(index=False, formatters={"null_rate": lambda x: f"{x:.3%}"}))
print(f"Core semantic feature count: {len(pass_features)}")
print(f"Structural conditioning feature count: {len(STRUCTURAL_FEATURES)}")
print(f"Indicator preprocessing column count: {len(model_indicator_cols)}")
print(f"Post-imputation model column count: {len(model_core_features) + len(model_indicator_cols)}")
""")


add("code", "c06_0005", """
def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(y_prob[mask].mean()) - float(y_true[mask].mean()))
    return float(ece)


def metric_bundle(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    out = {
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["auc"] = float("nan")
    return out


def reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict[str, Any]]:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin": i, "lo": lo, "hi": hi, "n": 0, "mean_prob": None, "actual_rate": None, "gap": None})
            continue
        mean_prob = float(y_prob[mask].mean())
        actual = float(y_true[mask].mean())
        rows.append({"bin": i, "lo": lo, "hi": hi, "n": n, "mean_prob": mean_prob, "actual_rate": actual, "gap": actual - mean_prob})
    return rows


def weighted_metric(rows: list[dict[str, Any]], metric: str, weights: dict[int, float]) -> float:
    return float(sum(float(r[metric]) * weights[int(r["fold"])] for r in rows))


def _make_l1(c: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=float(c),
            random_state=RANDOM_STATE,
            max_iter=1000,
        )),
    ])


def feature_group_columns(feature: str, indicator_cols: list[str]) -> list[str]:
    cols = [feature]
    for ind in indicator_cols:
        if indicator_meta[ind]["core_feature"] == feature:
            cols.append(ind)
    return cols


def model_columns_for_core(core_features: list[str], indicator_cols: list[str]) -> list[str]:
    cols = list(core_features)
    core_set = set(core_features)
    for ind in indicator_cols:
        if indicator_meta[ind]["core_feature"] in core_set:
            cols.append(ind)
    return cols


def fit_preprocessor(train_df: pd.DataFrame, core_features: list[str], indicator_cols: list[str]) -> dict[str, Any]:
    medians: dict[str, float] = {}
    all_null_features: list[str] = []
    for feat in core_features:
        med = train_df[feat].median(skipna=True)
        if pd.isna(med):
            med = 0.0
            all_null_features.append(feat)
        medians[feat] = float(med)
    return {
        "core_features": list(core_features),
        "indicator_cols": list(indicator_cols),
        "model_columns": model_columns_for_core(core_features, indicator_cols),
        "medians": medians,
        "all_null_features": all_null_features,
    }


def transform_with_preprocessor(df: pd.DataFrame, prep: dict[str, Any]) -> pd.DataFrame:
    cols = prep["model_columns"]
    out = pd.DataFrame(index=df.index)
    for feat in prep["core_features"]:
        out[feat] = df[feat].astype(float).fillna(prep["medians"][feat])
    for ind in prep["indicator_cols"]:
        if indicator_meta[ind]["core_feature"] in prep["core_features"]:
            out[ind] = df[ind].astype(float).fillna(0.0)
    return out[cols]


def check_calibration_health(label: str, probs: np.ndarray) -> dict[str, Any]:
    probs = np.asarray(probs, dtype=float)
    n_unique = int(len(np.unique(np.round(probs, 8))))
    std = float(np.std(probs))
    out = {"label": label, "n_unique_rounded8": n_unique, "std": std, "warning": ""}
    if n_unique <= 1 or std < 1e-8:
        raise AssertionError(f"Calibration failure for {label}: constant calibrated probabilities")
    if n_unique <= 3 or std < 0.005:
        out["warning"] = "near_constant_calibration_check"
    return out


@dataclass
class FitResult:
    fold: int
    train_window_label: str
    val_season: int
    test_season: int | None
    core_features: list[str]
    indicator_cols: list[str]
    model_columns: list[str]
    prep: dict[str, Any]
    estimator: Pipeline
    calibrator: IsotonicRegression
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame | None
    x_val: pd.DataFrame
    x_test: pd.DataFrame | None
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray | None
    raw_val: np.ndarray
    cal_val: np.ndarray
    raw_test: np.ndarray | None
    cal_test: np.ndarray | None
    coef_by_col: dict[str, float]
    metrics_val: dict[str, float]
    metrics_test: dict[str, float] | None
    constant_metrics_test: dict[str, float] | None
    calibration_health: dict[str, Any]


def fit_l1_window(
    window: dict[str, Any],
    core_features: list[str],
    indicator_cols: list[str],
    *,
    c: float = 1.0,
    test_mode: bool = True,
) -> FitResult:
    train_df = wide_df[wide_df["season"].isin(window["train_seasons"])].copy()
    val_df = wide_df[wide_df["season"] == window["val_season"]].copy()
    test_df = None
    if test_mode:
        assert window["test_season"] is not None
        test_df = wide_df[wide_df["season"] == window["test_season"]].copy()

    prep = fit_preprocessor(train_df, core_features, indicator_cols)
    x_train = transform_with_preprocessor(train_df, prep)
    x_val = transform_with_preprocessor(val_df, prep)
    x_test = transform_with_preprocessor(test_df, prep) if test_df is not None else None
    assert not x_train.isna().any().any(), "NaN survived train preprocessing"
    assert not x_val.isna().any().any(), "NaN survived validation preprocessing"
    if x_test is not None:
        assert not x_test.isna().any().any(), "NaN survived test preprocessing"

    y_train = train_df[TARGET_LABEL].astype(int).to_numpy()
    y_val = val_df[TARGET_LABEL].astype(int).to_numpy()
    y_test = test_df[TARGET_LABEL].astype(int).to_numpy() if test_df is not None else None

    estimator = _make_l1(c)
    estimator.fit(x_train, y_train)
    raw_val = estimator.predict_proba(x_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_val, y_val)
    cal_val = calibrator.predict(raw_val)

    raw_test = None
    cal_test = None
    metrics_test = None
    constant_metrics_test = None
    if x_test is not None and y_test is not None:
        raw_test = estimator.predict_proba(x_test)[:, 1]
        cal_test = calibrator.predict(raw_test)
        metrics_test = metric_bundle(y_test, cal_test)
        metrics_test["raw_brier"] = metric_bundle(y_test, raw_test)["brier"]
        metrics_test["raw_ece"] = metric_bundle(y_test, raw_test)["ece"]
        p_const = np.repeat(float(np.mean(y_train)), len(y_test))
        constant_metrics_test = metric_bundle(y_test, p_const)

    coef_arr = estimator.named_steps["logreg"].coef_[0]
    coef_by_col = {col: float(coef_arr[i]) for i, col in enumerate(prep["model_columns"])}
    health_target = cal_test if cal_test is not None else cal_val
    health = check_calibration_health(
        f"fold={window['fold']} C={c} test_mode={test_mode}",
        np.asarray(health_target),
    )

    return FitResult(
        fold=int(window["fold"]),
        train_window_label=str(window["train_window_label"]),
        val_season=int(window["val_season"]),
        test_season=int(window["test_season"]) if window["test_season"] is not None else None,
        core_features=list(core_features),
        indicator_cols=list(indicator_cols),
        model_columns=list(prep["model_columns"]),
        prep=prep,
        estimator=estimator,
        calibrator=calibrator,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        raw_val=raw_val,
        cal_val=cal_val,
        raw_test=raw_test,
        cal_test=cal_test,
        coef_by_col=coef_by_col,
        metrics_val=metric_bundle(y_val, cal_val),
        metrics_test=metrics_test,
        constant_metrics_test=constant_metrics_test,
        calibration_health=health,
    )


def fit_l1_windows(core_features: list[str], indicator_cols: list[str], *, c: float = 1.0) -> list[FitResult]:
    return [
        fit_l1_window(w, core_features, indicator_cols, c=c, test_mode=True)
        for w in WALK_FORWARD_WINDOWS
    ]


def fold_metric_rows(fits: list[FitResult], label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fit in fits:
        assert fit.metrics_test is not None
        row = {
            "label": label,
            "fold": fit.fold,
            "train_window": fit.train_window_label,
            "val_season": fit.val_season,
            "test_season": fit.test_season,
            "n_train": int(len(fit.train_df)),
            "n_val": int(len(fit.val_df)),
            "n_test": int(len(fit.test_df) if fit.test_df is not None else 0),
            **fit.metrics_test,
            "constant_brier": float(fit.constant_metrics_test["brier"]),
            "constant_ece": float(fit.constant_metrics_test["ece"]),
            "constant_auc": float(fit.constant_metrics_test["auc"]),
        }
        row["brier_vs_constant"] = row["brier"] - row["constant_brier"]
        row["ece_vs_constant"] = row["ece"] - row["constant_ece"]
        rows.append(row)
    return rows


print("[ok] modeling helpers defined")
""")


add("code", "c06_0006", """
def summarize_coef_group(fits: list[FitResult], weights: dict[int, float], feature: str) -> dict[str, Any]:
    group_cols = feature_group_columns(feature, model_indicator_cols)
    per_fold: dict[int, dict[str, float]] = {}
    max_abs = 0.0
    weighted_abs_core = 0.0
    weighted_abs_group = 0.0
    for fit in fits:
        w = weights[fit.fold]
        fold_vals = {col: float(fit.coef_by_col.get(col, 0.0)) for col in group_cols}
        per_fold[fit.fold] = fold_vals
        core_abs = abs(fold_vals.get(feature, 0.0))
        group_abs = max(abs(v) for v in fold_vals.values()) if fold_vals else 0.0
        weighted_abs_core += w * core_abs
        weighted_abs_group += w * group_abs
        max_abs = max(max_abs, group_abs)
    return {
        "feature": feature,
        "group_columns": group_cols,
        "per_fold_coefficients": per_fold,
        "weighted_abs_core_coef": weighted_abs_core,
        "weighted_abs_group_coef": weighted_abs_group,
        "max_abs_group_coef": max_abs,
        "l1_zero_group": bool(max_abs < 1e-6),
    }


def permutation_importance_for_groups(
    fits: list[FitResult],
    weights: dict[int, float],
    groups: dict[str, list[str]],
    *,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = RANDOM_STATE,
) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, Any]] = {}
    for group_name, cols in groups.items():
        fold_rows: list[dict[str, Any]] = []
        weighted_delta = 0.0
        weighted_p = 0.0
        for fit in fits:
            base_ece = expected_calibration_error(fit.y_val, fit.cal_val)
            deltas: list[float] = []
            for _ in range(n_permutations):
                x_perm = fit.x_val.copy()
                perm_idx = rng.permutation(len(x_perm))
                for col in cols:
                    if col in x_perm.columns:
                        x_perm[col] = x_perm[col].to_numpy()[perm_idx]
                raw = fit.estimator.predict_proba(x_perm[fit.model_columns])[:, 1]
                cal = fit.calibrator.predict(raw)
                deltas.append(expected_calibration_error(fit.y_val, cal) - base_ece)
            mean_delta = float(np.mean(deltas))
            p_value = float((1 + sum(d <= 0 for d in deltas)) / (len(deltas) + 1))
            weighted_delta += weights[fit.fold] * mean_delta
            weighted_p += weights[fit.fold] * p_value
            fold_rows.append({
                "fold": fit.fold,
                "base_val_ece": base_ece,
                "mean_ece_delta": mean_delta,
                "p_value_degradation": p_value,
            })
        out[group_name] = {
            "group": group_name,
            "columns": cols,
            "weighted_mean_ece_delta": weighted_delta,
            "weighted_p_value_degradation": weighted_p,
            "folds": fold_rows,
        }

    ordered = sorted(out, key=lambda k: out[k]["weighted_mean_ece_delta"])
    for rank_from_bottom, name in enumerate(ordered, start=1):
        out[name]["rank_from_bottom"] = rank_from_bottom
        out[name]["bottom_20pct"] = bool(rank_from_bottom <= max(1, math.ceil(0.20 * len(ordered))))
    return out


def ablation_decisions(
    full_fits: list[FitResult],
    weights: dict[int, float],
    candidate_features: list[str],
    base_core_features: list[str],
    indicator_cols: list[str],
    *,
    c: float = 1.0,
) -> dict[str, dict[str, Any]]:
    full_rows = fold_metric_rows(full_fits, "full")
    full_weighted_ece = weighted_metric(full_rows, "ece", weights)
    out: dict[str, dict[str, Any]] = {}
    for feat in candidate_features:
        ablated_core = [f for f in base_core_features if f != feat]
        ablated_indicators = [
            ind for ind in indicator_cols
            if indicator_meta[ind]["core_feature"] in set(ablated_core)
        ]
        fits = fit_l1_windows(ablated_core, ablated_indicators, c=c)
        rows = fold_metric_rows(fits, f"drop_{feat}")
        weighted_ece_without = weighted_metric(rows, "ece", weights)
        effect = weighted_ece_without - full_weighted_ece
        out[feat] = {
            "feature": feat,
            "weighted_ece_full": full_weighted_ece,
            "weighted_ece_without_feature": weighted_ece_without,
            "ablation_effect_ece": effect,
            "ablation_safe_to_drop": bool(effect <= 0.001),
            "fold_metrics_without_feature": rows,
        }
    return out


def run_pruning_for_scheme(
    scheme: str,
    full_fits: list[FitResult],
    candidate_features: list[str],
    base_core_features: list[str],
    indicator_cols: list[str],
) -> tuple[list[str], dict[str, Any]]:
    weights = SCHEME_WEIGHTS[scheme]
    coef_summary = {
        feat: summarize_coef_group(full_fits, weights, feat)
        for feat in candidate_features
    }
    groups = {feat: feature_group_columns(feat, indicator_cols) for feat in candidate_features}
    perm = permutation_importance_for_groups(
        full_fits,
        weights,
        groups,
        n_permutations=N_PERMUTATIONS,
        seed=RANDOM_STATE + (0 if scheme == "U" else 1000),
    )
    ablation = ablation_decisions(full_fits, weights, candidate_features, base_core_features, indicator_cols, c=1.0)

    decisions: list[dict[str, Any]] = []
    dropped: list[str] = []
    for feat in candidate_features:
        l1_zero = bool(coef_summary[feat]["l1_zero_group"])
        perm_unimportant = bool(
            perm[feat]["bottom_20pct"]
            and perm[feat]["weighted_p_value_degradation"] > 0.10
        )
        abl_safe = bool(ablation[feat]["ablation_safe_to_drop"])
        drop = bool(l1_zero and perm_unimportant and abl_safe)
        if drop:
            dropped.append(feat)
        decisions.append({
            "feature": feat,
            "feature_set_version": feature_meta[feat]["feature_set_version"],
            "redundant_with": feature_meta[feat]["redundant_with"],
            **coef_summary[feat],
            "permutation_weighted_mean_ece_delta": perm[feat]["weighted_mean_ece_delta"],
            "permutation_weighted_p_value_degradation": perm[feat]["weighted_p_value_degradation"],
            "permutation_rank_from_bottom": perm[feat]["rank_from_bottom"],
            "permutation_bottom_20pct": perm[feat]["bottom_20pct"],
            "permutation_unimportant": perm_unimportant,
            "ablation_effect_ece": ablation[feat]["ablation_effect_ece"],
            "ablation_safe_to_drop": abl_safe,
            "drop": drop,
            "pruning_exempt": False,
        })

    for feat in STRUCTURAL_FEATURES:
        decisions.append({
            "feature": feat,
            "feature_set_version": "structural_conditioning_variable",
            "redundant_with": "",
            "structural_conditioning_variable": True,
            "pruning_exempt": True,
            "drop": False,
            "reason": "fav_deficit defines trigger-threshold conditioning and is protected from pruning",
            **summarize_coef_group(full_fits, weights, feat),
        })

    selected = [f for f in candidate_features if f not in set(dropped)]
    return selected, {
        "scheme": scheme,
        "weights": weights,
        "selected_features": selected,
        "protected_features": list(STRUCTURAL_FEATURES),
        "dropped_features": dropped,
        "decisions": decisions,
        "permutation_detail": perm,
        "ablation_detail": ablation,
    }


def indicator_diagnostics(fits: list[FitResult], weights: dict[int, float]) -> list[dict[str, Any]]:
    groups = {ind: [ind] for ind in model_indicator_cols}
    perm = permutation_importance_for_groups(
        fits,
        weights,
        groups,
        n_permutations=N_PERMUTATIONS,
        seed=RANDOM_STATE + 2000,
    )
    rows: list[dict[str, Any]] = []
    for ind in model_indicator_cols:
        per_fold_coef = {fit.fold: float(fit.coef_by_col.get(ind, 0.0)) for fit in fits}
        max_abs = max(abs(v) for v in per_fold_coef.values()) if per_fold_coef else 0.0
        rows.append({
            "indicator": ind,
            **indicator_meta[ind],
            "per_fold_coefficients": per_fold_coef,
            "max_abs_coef": max_abs,
            "l1_zero_indicator": bool(max_abs < 1e-6),
            "permutation_weighted_mean_ece_delta": perm[ind]["weighted_mean_ece_delta"],
            "permutation_weighted_p_value_degradation": perm[ind]["weighted_p_value_degradation"],
            "permutation_rank_from_bottom": perm[ind]["rank_from_bottom"],
        })
    return rows


print("[ok] pruning helpers defined")
""")


add("code", "c06_0007", """
t_model0 = time.perf_counter()

full_indicator_cols = list(model_indicator_cols)
full_model_columns = model_columns_for_core(model_core_features, full_indicator_cols)
print(f"Fitting primary full L1 model with {len(model_core_features)} core features + {len(full_indicator_cols)} indicators = {len(full_model_columns)} columns")

full_l1_fits = fit_l1_windows(model_core_features, full_indicator_cols, c=1.0)
full_l1_rows = fold_metric_rows(full_l1_fits, "L1_full_C1")
for r in full_l1_rows:
    print(
        f"[full L1] fold={r['fold']} brier={r['brier']:.5f} ece={r['ece']:.5f} "
        f"auc={r['auc']:.4f} vs_const_brier={r['brier_vs_constant']:+.5f}"
    )

pruning_by_scheme: dict[str, Any] = {}
selected_features_by_scheme: dict[str, list[str]] = {}
for scheme in ["U", "W2"]:
    selected, pruning = run_pruning_for_scheme(scheme, full_l1_fits, pass_features, model_core_features, full_indicator_cols)
    selected_features_by_scheme[scheme] = selected
    pruning_by_scheme[scheme] = pruning
    print(f"[pruning {scheme}] selected={len(selected)} dropped={len(pruning['dropped_features'])}: {pruning['dropped_features']}")

for fit in full_l1_fits:
    coef = abs(float(fit.coef_by_col.get("fav_deficit", 0.0)))
    assert coef >= 1e-6, f"fav_deficit coefficient is zero in full L1 fold {fit.fold}"

indicator_diag_by_scheme = {
    scheme: indicator_diagnostics(full_l1_fits, SCHEME_WEIGHTS[scheme])
    for scheme in ["U", "W2"]
}

print(f"[ok] pruning complete in {time.perf_counter() - t_model0:.1f}s")
""")


add("code", "c06_0008", """
def selected_indicator_cols(selected_core: list[str]) -> list[str]:
    selected_set = set(selected_core)
    return [
        ind for ind in model_indicator_cols
        if indicator_meta[ind]["core_feature"] in selected_set
    ]


def selected_model_core_features(selected_r6_features: list[str]) -> list[str]:
    return [*selected_r6_features, *STRUCTURAL_FEATURES]


def score_event_rows(fit: FitResult, events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x_event = transform_with_preprocessor(events, fit.prep)
    assert not x_event.isna().any().any(), "NaN survived event-level scoring preprocessing"
    raw = fit.estimator.predict_proba(x_event[fit.model_columns])[:, 1]
    cal = fit.calibrator.predict(raw)
    return raw, cal


final_fits_by_scheme: dict[str, list[FitResult]] = {}
final_metric_rows: list[dict[str, Any]] = []
prediction_frames: list[pd.DataFrame] = []

for scheme in ["U", "W2"]:
    selected_core = selected_features_by_scheme[scheme]
    selected_inds = selected_indicator_cols(selected_core)
    fits = fit_l1_windows(selected_model_core_features(selected_core), selected_inds, c=1.0)
    final_fits_by_scheme[scheme] = fits
    rows = fold_metric_rows(fits, f"L1_final_{scheme}")
    for r in rows:
        r["scheme"] = scheme
    final_metric_rows.extend(rows)

    for fit in fits:
        assert fit.test_df is not None
        event_test_df = event_df[event_df["season"] == fit.test_season].copy()
        raw_event, cal_event = score_event_rows(fit, event_test_df)
        pred = event_test_df[
            [*N06_OUTPUT_COLUMNS, *pass_features]
        ].copy()
        pred.insert(2, "fold", fit.fold)
        pred.insert(2, "scheme", scheme)
        pred["raw_model_prob"] = raw_event
        pred["calibrated_prob"] = cal_event
        pred["split_role"] = "test"
        prediction_frames.append(pred)

predictions_df = pd.concat(prediction_frames, ignore_index=True)
if N06_STATE_EXPORT_ONLY:
    print(f"[state-export-only] skipped rewrite of {N06_PREDICTIONS_PARQUET.relative_to(REPO_ROOT)}")
else:
    predictions_df.to_parquet(N06_PREDICTIONS_PARQUET, index=False)
    print(f"[ok] wrote {N06_PREDICTIONS_PARQUET.relative_to(REPO_ROOT)} rows={len(predictions_df):,} cols={predictions_df.shape[1]}")

# Deployment-proximate E: train 2015-2023, calibrate/evaluate on 2024 validation.
# Use the W2 selected feature set as the recent-weighted production reference.
e_core = selected_features_by_scheme["W2"]
e_inds = selected_indicator_cols(e_core)
fit_e = fit_l1_window(DEPLOYMENT_WINDOW_E, selected_model_core_features(e_core), e_inds, c=1.0, test_mode=False)
e_event_df = event_df[event_df["season"] == DEPLOYMENT_WINDOW_E["val_season"]].copy()
e_raw_event, e_cal_event = score_event_rows(fit_e, e_event_df)
e_pred = e_event_df[
    [*N06_OUTPUT_COLUMNS, *pass_features]
].copy()
e_pred.insert(2, "fold", fit_e.fold)
e_pred.insert(2, "scheme", "E")
e_pred["raw_model_prob"] = e_raw_event
e_pred["calibrated_prob"] = e_cal_event
e_pred["split_role"] = "validation"
if N06_STATE_EXPORT_ONLY:
    print(f"[state-export-only] skipped rewrite of {N06_E_PREDICTIONS_PARQUET.relative_to(REPO_ROOT)}")
else:
    e_pred.to_parquet(N06_E_PREDICTIONS_PARQUET, index=False)
    print(f"[ok] wrote {N06_E_PREDICTIONS_PARQUET.relative_to(REPO_ROOT)} rows={len(e_pred):,} cols={e_pred.shape[1]}")

fav_deficit_coefficients: dict[str, dict[int, float]] = {
    scheme: {fit.fold: float(fit.coef_by_col.get("fav_deficit", 0.0)) for fit in fits}
    for scheme, fits in final_fits_by_scheme.items()
}
fav_deficit_coefficients["E"] = {fit_e.fold: float(fit_e.coef_by_col.get("fav_deficit", 0.0))}
for scheme, fold_map in fav_deficit_coefficients.items():
    for fold, coef in fold_map.items():
        assert abs(coef) >= 1e-6, f"fav_deficit coefficient is zero for scheme={scheme} fold={fold}"

prediction_key_cols = ["game_id", "fav_deficit", "trigger_sequence", "scheme", "fold"]
assert int(predictions_df.duplicated(prediction_key_cols, keep=False).sum()) == 0, (
    "Main prediction parquet is not unique on trigger-event key + scheme + fold"
)
assert int(e_pred.duplicated(prediction_key_cols, keep=False).sum()) == 0, (
    "E prediction parquet is not unique on trigger-event key + scheme + fold"
)

expected_test_event_rows_per_scheme = int(sum(
    len(event_df[event_df["season"] == int(w["test_season"])])
    for w in WALK_FORWARD_WINDOWS
))
assert len(predictions_df) == expected_test_event_rows_per_scheme * 2, (
    f"main prediction rows {len(predictions_df):,} != "
    f"{expected_test_event_rows_per_scheme:,} test events x 2 schemes"
)
assert len(event_df) == 11412, f"full event matrix expected 11,412 rows after N05 null exclusions, got {len(event_df):,}"


def _probability_variation_summary(df: pd.DataFrame) -> dict[str, Any]:
    grp_cols = ["scheme", "fold", "game_id", "trigger_play_id"]
    multi = df[df.duplicated(grp_cols, keep=False)].copy()
    n_groups = int(multi.groupby(grp_cols).ngroups) if len(multi) else 0
    raw_diff = 0
    cal_diff = 0
    for _, grp in multi.groupby(grp_cols):
        if grp["raw_model_prob"].nunique(dropna=False) > 1:
            raw_diff += 1
        if grp["calibrated_prob"].nunique(dropna=False) > 1:
            cal_diff += 1
    return {
        "multi_threshold_group_count": n_groups,
        "raw_probability_diff_group_count": raw_diff,
        "calibrated_probability_diff_group_count": cal_diff,
    }


probability_variation_summary = {
    "main": _probability_variation_summary(predictions_df),
    "E": _probability_variation_summary(e_pred),
}
assert probability_variation_summary["main"]["raw_probability_diff_group_count"] > 0, (
    "fav_deficit scoring did not change raw probabilities for any held-out multi-threshold play"
)
assert probability_variation_summary["main"]["calibrated_probability_diff_group_count"] > 0, (
    "fav_deficit scoring did not change calibrated probabilities for any held-out multi-threshold play"
)

sample_rows: list[dict[str, Any]] = []
sample_source = predictions_df[predictions_df["scheme"] == "U"].copy()
for _, grp in sample_source[sample_source.duplicated(["fold", "game_id", "trigger_play_id"], keep=False)].groupby(["fold", "game_id", "trigger_play_id"]):
    if grp["calibrated_prob"].nunique(dropna=False) <= 1:
        continue
    keep = grp.sort_values("fav_deficit")[
        ["fold", "game_id", "trigger_play_id", "fav_deficit", "trigger_sequence", "raw_model_prob", "calibrated_prob"]
    ]
    sample_rows.extend(keep.to_dict(orient="records"))
    if len({(r["fold"], r["game_id"], r["trigger_play_id"]) for r in sample_rows}) >= 10:
        break
probability_variation_sample = sample_rows
print(f"[ok] fav_deficit coefficients: {fav_deficit_coefficients}")
print(f"[ok] prediction key uniqueness passed on {prediction_key_cols}")
print(f"[ok] probability variation summary: {probability_variation_summary}")
""")


add("code", "c06_0008b", """
def _state_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _state_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_state_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_state_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return None if math.isnan(val) or math.isinf(val) else val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def fitted_state_for_fit(scheme: str, fit: FitResult, *, deployment_model: bool = False) -> dict[str, Any]:
    scaler = fit.estimator.named_steps["scaler"]
    logreg = fit.estimator.named_steps["logreg"]
    return {
        "scheme": scheme,
        "fold": fit.fold,
        "train_window": fit.train_window_label,
        "val_season": fit.val_season,
        "test_season": fit.test_season,
        "deployment_model": deployment_model,
        "target_label": TARGET_LABEL,
        "model_class": {
            "pipeline": "StandardScaler -> LogisticRegression",
            "penalty": "l1",
            "solver": "liblinear",
            "C": 1.0,
            "random_state": RANDOM_STATE,
        },
        "core_features": fit.core_features,
        "indicator_columns": fit.indicator_cols,
        "model_columns": fit.model_columns,
        "imputation_medians": fit.prep["medians"],
        "all_null_features": fit.prep["all_null_features"],
        "standard_scaler": {
            "mean": {col: float(scaler.mean_[i]) for i, col in enumerate(fit.model_columns)},
            "scale": {col: float(scaler.scale_[i]) for i, col in enumerate(fit.model_columns)},
            "var": {col: float(scaler.var_[i]) for i, col in enumerate(fit.model_columns)},
            "n_features_in": int(scaler.n_features_in_),
        },
        "logistic_regression": {
            "classes": [int(x) for x in logreg.classes_.tolist()],
            "intercept": float(logreg.intercept_[0]),
            "coefficients": {col: float(logreg.coef_[0][i]) for i, col in enumerate(fit.model_columns)},
            "n_iter": [int(x) for x in np.ravel(logreg.n_iter_).tolist()],
        },
        "isotonic_calibration": {
            "out_of_bounds": "clip",
            "x_thresholds": [float(x) for x in fit.calibrator.X_thresholds_.tolist()],
            "y_thresholds": [float(y) for y in fit.calibrator.y_thresholds_.tolist()],
            "health": fit.calibration_health,
        },
    }


def compare_predictions_to_committed(new_df: pd.DataFrame, committed_path: pathlib.Path, label: str) -> dict[str, Any]:
    committed = pd.read_parquet(committed_path)
    key = prediction_key_cols
    assert len(new_df) == len(committed), (
        f"{label} row count mismatch: regenerated={len(new_df):,} committed={len(committed):,}"
    )
    new_keyed = new_df[key + ["raw_model_prob", "calibrated_prob"]].sort_values(key).reset_index(drop=True)
    old_keyed = committed[key + ["raw_model_prob", "calibrated_prob"]].sort_values(key).reset_index(drop=True)
    key_mismatch = int((new_keyed[key] != old_keyed[key]).any(axis=1).sum())
    assert key_mismatch == 0, f"{label} key mismatch rows: {key_mismatch}"
    raw_diff = np.abs(new_keyed["raw_model_prob"].to_numpy(float) - old_keyed["raw_model_prob"].to_numpy(float))
    cal_diff = np.abs(new_keyed["calibrated_prob"].to_numpy(float) - old_keyed["calibrated_prob"].to_numpy(float))
    return {
        "label": label,
        "committed_artifact": str(committed_path.relative_to(REPO_ROOT)),
        "row_count": int(len(new_keyed)),
        "raw_model_prob_max_abs_diff": float(raw_diff.max()) if len(raw_diff) else 0.0,
        "calibrated_prob_max_abs_diff": float(cal_diff.max()) if len(cal_diff) else 0.0,
        "calibrated_prob_mean_abs_diff": float(cal_diff.mean()) if len(cal_diff) else 0.0,
    }


main_reproduction_gate = compare_predictions_to_committed(
    predictions_df,
    N06_PREDICTIONS_PARQUET,
    "main_heldout_U_W2",
)
e_reproduction_gate = compare_predictions_to_committed(
    e_pred,
    N06_E_PREDICTIONS_PARQUET,
    "scheme_E_validation",
)
assert main_reproduction_gate["calibrated_prob_max_abs_diff"] < 1e-9, main_reproduction_gate
assert e_reproduction_gate["calibrated_prob_max_abs_diff"] < 1e-9, e_reproduction_gate

full_fitted_state = {
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
    "artifact_purpose": (
        "Additive fitted-state provenance export for committed N06. This file is not a new model fit "
        "or new estimate; it records the full fitted preprocessing/model/calibration state needed for live scoring."
    ),
    "reproduction_gate": {
        "required_max_abs_diff": 1e-9,
        "main_heldout_U_W2": main_reproduction_gate,
        "scheme_E_validation": e_reproduction_gate,
        "passed": True,
    },
    "environment": {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
    },
    "deployment_choice": {
        "scheme": "E",
        "fold": int(fit_e.fold),
        "train_window": fit_e.train_window_label,
        "validation_season": int(fit_e.val_season),
        "rationale": "Scheme E is trained on 2015-2023 and validated on 2024, using the most historical data available before live deployment.",
    },
    "indicator_meta": indicator_meta,
    "fits": [
        fitted_state_for_fit(scheme, fit)
        for scheme, fits in final_fits_by_scheme.items()
        for fit in fits
    ] + [
        fitted_state_for_fit("E", fit_e, deployment_model=True)
    ],
}

N06_FULL_FITTED_STATE_JSON.write_text(json.dumps(_state_jsonable(full_fitted_state), indent=2), encoding="utf-8")
print(f"[ok] wrote {N06_FULL_FITTED_STATE_JSON.relative_to(REPO_ROOT)} size={N06_FULL_FITTED_STATE_JSON.stat().st_size:,} bytes")
print(f"[ok] N06 reproduction gate main calibrated max abs diff={main_reproduction_gate['calibrated_prob_max_abs_diff']:.12g}")
print(f"[ok] N06 reproduction gate E calibrated max abs diff={e_reproduction_gate['calibrated_prob_max_abs_diff']:.12g}")
print(f"[info] versions: sklearn={sklearn.__version__}, numpy={np.__version__}, scipy={scipy.__version__}")
""")


add("code", "c06_0009", """
def run_c_sweep() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for c in C_VALUES:
        fits = fit_l1_windows(model_core_features, full_indicator_cols, c=c)
        metric_rows = fold_metric_rows(fits, f"L1_C{c}")
        coef_abs_weighted: dict[str, float] = {}
        nonzero_features_by_fold: dict[int, list[str]] = {}
        coef_reporting_features = [*pass_features, *STRUCTURAL_FEATURES]
        for scheme, weights in SCHEME_WEIGHTS.items():
            for feat in coef_reporting_features:
                vals = []
                for fit in fits:
                    group_cols = feature_group_columns(feat, full_indicator_cols)
                    vals.append(weights[fit.fold] * max(abs(fit.coef_by_col.get(col, 0.0)) for col in group_cols))
                coef_abs_weighted[f"{scheme}:{feat}"] = float(sum(vals))

        for fit in fits:
            nz = []
            for feat in coef_reporting_features:
                group_cols = feature_group_columns(feat, full_indicator_cols)
                if max(abs(fit.coef_by_col.get(col, 0.0)) for col in group_cols) >= 1e-6:
                    nz.append(feat)
            nonzero_features_by_fold[fit.fold] = nz

        row = {
            "C": c,
            "fold_metrics": metric_rows,
            "mean_brier_U": weighted_metric(metric_rows, "brier", SCHEME_WEIGHTS["U"]),
            "mean_ece_U": weighted_metric(metric_rows, "ece", SCHEME_WEIGHTS["U"]),
            "mean_auc_U": weighted_metric(metric_rows, "auc", SCHEME_WEIGHTS["U"]),
            "weighted_brier_W2": weighted_metric(metric_rows, "brier", SCHEME_WEIGHTS["W2"]),
            "weighted_ece_W2": weighted_metric(metric_rows, "ece", SCHEME_WEIGHTS["W2"]),
            "weighted_auc_W2": weighted_metric(metric_rows, "auc", SCHEME_WEIGHTS["W2"]),
            "nonzero_features_by_fold": nonzero_features_by_fold,
            "nonzero_feature_union": sorted(set().union(*[set(v) for v in nonzero_features_by_fold.values()])),
        }
        for scheme in ["U", "W2"]:
            top = sorted(
                ((feat, coef_abs_weighted[f"{scheme}:{feat}"]) for feat in coef_reporting_features),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            row[f"top_weighted_abs_coef_{scheme}"] = [{"feature": f, "weighted_abs_group_coef": v} for f, v in top]
        rows.append(row)
        print(f"[C sweep] C={c} U_ECE={row['mean_ece_U']:.5f} W2_ECE={row['weighted_ece_W2']:.5f} nonzero_union={len(row['nonzero_feature_union'])}")
    return rows


def fit_xgb_window(window: dict[str, Any], core_features: list[str], indicator_cols: list[str]) -> dict[str, Any]:
    if xgb is None:
        return {"fold": int(window["fold"]), "error": XGBOOST_IMPORT_ERROR}
    train_df = wide_df[wide_df["season"].isin(window["train_seasons"])].copy()
    val_df = wide_df[wide_df["season"] == window["val_season"]].copy()
    test_df = wide_df[wide_df["season"] == window["test_season"]].copy()
    prep = fit_preprocessor(train_df, core_features, indicator_cols)
    x_train = transform_with_preprocessor(train_df, prep)
    x_val = transform_with_preprocessor(val_df, prep)
    x_test = transform_with_preprocessor(test_df, prep)
    y_train = train_df[TARGET_LABEL].astype(int).to_numpy()
    y_val = val_df[TARGET_LABEL].astype(int).to_numpy()
    y_test = test_df[TARGET_LABEL].astype(int).to_numpy()
    clf = xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0, n_jobs=1)
    clf.fit(x_train, y_train)
    raw_val = clf.predict_proba(x_val)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_val, y_val)
    raw_test = clf.predict_proba(x_test)[:, 1]
    cal_test = iso.predict(raw_test)
    health = check_calibration_health(f"xgboost_fold={window['fold']}", cal_test)
    row = {
        "fold": int(window["fold"]),
        "train_window": str(window["train_window_label"]),
        "val_season": int(window["val_season"]),
        "test_season": int(window["test_season"]),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        **metric_bundle(y_test, cal_test),
        "raw_brier": metric_bundle(y_test, raw_test)["brier"],
        "raw_ece": metric_bundle(y_test, raw_test)["ece"],
        "calibration_health": health,
    }
    return row


def run_xgb_comparison() -> dict[str, Any]:
    rows = [
        fit_xgb_window(w, model_core_features, full_indicator_cols)
        for w in WALK_FORWARD_WINDOWS
    ]
    if any("error" in r for r in rows):
        return {"available": False, "rows": rows, "import_error": XGBOOST_IMPORT_ERROR}
    l1_full_rows = fold_metric_rows(full_l1_fits, "L1_full_C1")
    comp = {
        "available": True,
        "rows": rows,
        "weighted_ece_U": weighted_metric(rows, "ece", SCHEME_WEIGHTS["U"]),
        "weighted_ece_W2": weighted_metric(rows, "ece", SCHEME_WEIGHTS["W2"]),
        "l1_full_weighted_ece_U": weighted_metric(l1_full_rows, "ece", SCHEME_WEIGHTS["U"]),
        "l1_full_weighted_ece_W2": weighted_metric(l1_full_rows, "ece", SCHEME_WEIGHTS["W2"]),
    }
    comp["xgb_ece_improves_over_l1_by_gt_10pct_U"] = bool(
        comp["weighted_ece_U"] < 0.90 * comp["l1_full_weighted_ece_U"]
    )
    comp["xgb_ece_improves_over_l1_by_gt_10pct_W2"] = bool(
        comp["weighted_ece_W2"] < 0.90 * comp["l1_full_weighted_ece_W2"]
    )
    for r in rows:
        print(f"[XGB] fold={r['fold']} brier={r['brier']:.5f} ece={r['ece']:.5f} auc={r['auc']:.4f}")
    return comp


sensitivity_sweep_summary = run_c_sweep()
xgboost_comparison = {
    "available": False,
    "not_run_reason": "N06 locked scope is the apples-to-apples L1 label-change test; no XGBoost diagnostic was run.",
}
""")


add("code", "c06_0009b", """
def brier(y_true: pd.Series | np.ndarray, y_prob: pd.Series | np.ndarray) -> float:
    return float(np.mean((np.asarray(y_prob, dtype=float) - np.asarray(y_true, dtype=float)) ** 2))


def bootstrap_cluster_mean_ci(
    df: pd.DataFrame,
    value_col: str,
    *,
    seed: int,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, float | None]:
    if len(df) == 0:
        return {"2.5": None, "25": None, "50": None, "75": None, "97.5": None}
    grouped = df.groupby("game_id", sort=False)[value_col].agg(["sum", "count"]).reset_index(drop=True)
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(grouped), size=(n_resamples, len(grouped)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    qs = np.percentile(means, [2.5, 25, 50, 75, 97.5])
    return {"2.5": float(qs[0]), "25": float(qs[1]), "50": float(qs[2]), "75": float(qs[3]), "97.5": float(qs[4])}


def fit_baseline_c_tables() -> dict[str, list[dict[str, Any]]]:
    n05_all = pd.read_parquet(N05_DESCRIPTIVE_RATES_PARQUET)
    tables: dict[str, list[dict[str, Any]]] = {}
    for label in LABELS_FOR_VALIDATION:
        train = n05_all[(n05_all["season"].between(2015, 2021)) & n05_all[label].notna()].copy()
        train[label] = train[label].astype(bool).astype(int)
        tbl = (
            train.groupby(["fav_deficit", "time_bucket"])[label]
            .agg(["mean", "sum", "count"])
            .reset_index()
            .rename(columns={"mean": "baseline_C_rate", "sum": "successes", "count": "n"})
        )
        tables[label] = tbl.to_dict(orient="records")
    return tables


baseline_C_tables = fit_baseline_c_tables()


def attach_baseline_c(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for label, rows in baseline_C_tables.items():
        tbl = pd.DataFrame(rows)
        out = out.merge(
            tbl[["fav_deficit", "time_bucket", "baseline_C_rate"]].rename(
                columns={"baseline_C_rate": f"baseline_C_{label}"}
            ),
            on=["fav_deficit", "time_bucket"],
            how="left",
            validate="many_to_one",
        )
        assert out[f"baseline_C_{label}"].notna().all(), f"missing baseline_C cells for {label}"
    return out


predictions_eval_df = attach_baseline_c(predictions_df)
e_predictions_eval_df = attach_baseline_c(e_pred)


def _rate_summary(df: pd.DataFrame, label: str) -> dict[str, Any]:
    if len(df) == 0:
        return {"n": 0, "successes": 0, "rate": None}
    y = df[label].astype(int)
    return {"n": int(len(df)), "successes": int(y.sum()), "rate": float(y.mean())}


def threshold_analysis(df: pd.DataFrame, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_col = f"baseline_C_{label}"
    for threshold in EDGE_THRESHOLDS:
        sel = df[df["calibrated_prob"] > df[baseline_col] + threshold].copy()
        rec = {"threshold": threshold, **_rate_summary(sel, label)}
        if len(sel):
            rec.update({
                "mean_model_prob": float(sel["calibrated_prob"].mean()),
                "mean_baseline_C": float(sel[baseline_col].mean()),
                "actual_minus_mean_model_prob": float(sel[label].astype(int).mean() - sel["calibrated_prob"].mean()),
                "actual_minus_mean_baseline_C": float(sel[label].astype(int).mean() - sel[baseline_col].mean()),
            })
            tmp = sel.assign(_label_value=sel[label].astype(int))
            rec["bootstrap_rate_ci"] = bootstrap_cluster_mean_ci(
                tmp, "_label_value", seed=BOOTSTRAP_SEED + 100 + int(threshold * 1000) + (0 if label == TARGET_LABEL else 1000)
            )
        else:
            rec.update({
                "mean_model_prob": None,
                "mean_baseline_C": None,
                "actual_minus_mean_model_prob": None,
                "actual_minus_mean_baseline_C": None,
                "bootstrap_rate_ci": {"2.5": None, "25": None, "50": None, "75": None, "97.5": None},
            })
        rows.append(rec)
    return rows


def quintile_analysis(df: pd.DataFrame, label: str) -> dict[str, Any]:
    clean = df.copy()
    clean["model_quintile"] = pd.qcut(clean["calibrated_prob"], q=5, labels=False, duplicates="drop") + 1
    baseline_col = f"baseline_C_{label}"
    rows: list[dict[str, Any]] = []
    for q, grp in clean.groupby("model_quintile", sort=True):
        y = grp[label].astype(int)
        rows.append({
            "quintile": int(q),
            "n": int(len(grp)),
            "model_prob_min": float(grp["calibrated_prob"].min()),
            "model_prob_max": float(grp["calibrated_prob"].max()),
            "mean_model_prob": float(grp["calibrated_prob"].mean()),
            "mean_baseline_C": float(grp[baseline_col].mean()),
            "rate": float(y.mean()),
            "actual_minus_mean_model_prob": float(y.mean() - grp["calibrated_prob"].mean()),
            "actual_minus_mean_baseline_C": float(y.mean() - grp[baseline_col].mean()),
        })
    spearman = float(pd.Series(clean["calibrated_prob"]).corr(pd.Series(clean[label].astype(int)), method="spearman"))
    return {"spearman_model_prob_vs_actual": spearman, "rows": rows}


def decile_analysis(df: pd.DataFrame, label: str) -> list[dict[str, Any]]:
    clean = df.copy()
    bins = np.linspace(0.0, 1.0, 11)
    clean["model_decile"] = pd.cut(clean["calibrated_prob"], bins=bins, include_lowest=True, right=False, labels=False)
    clean.loc[clean["calibrated_prob"].eq(1.0), "model_decile"] = 9
    baseline_col = f"baseline_C_{label}"
    rows: list[dict[str, Any]] = []
    for decile in range(10):
        grp = clean[clean["model_decile"].eq(decile)].copy()
        rec = {"decile": decile, "lo": float(bins[decile]), "hi": float(bins[decile + 1]), **_rate_summary(grp, label)}
        if len(grp):
            y = grp[label].astype(int)
            rec.update({
                "mean_model_prob": float(grp["calibrated_prob"].mean()),
                "mean_baseline_C": float(grp[baseline_col].mean()),
                "actual_minus_mean_model_prob": float(y.mean() - grp["calibrated_prob"].mean()),
                "actual_minus_mean_baseline_C": float(y.mean() - grp[baseline_col].mean()),
            })
            tmp = grp.assign(_label_value=y)
            rec["bootstrap_rate_ci"] = bootstrap_cluster_mean_ci(
                tmp, "_label_value", seed=BOOTSTRAP_SEED + 200 + decile + (0 if label == TARGET_LABEL else 1000)
            )
        else:
            rec.update({
                "mean_model_prob": None,
                "mean_baseline_C": None,
                "actual_minus_mean_model_prob": None,
                "actual_minus_mean_baseline_C": None,
                "bootstrap_rate_ci": {"2.5": None, "25": None, "50": None, "75": None, "97.5": None},
            })
        rows.append(rec)
    return rows


def per_deficit_analysis(df: pd.DataFrame, label: str) -> list[dict[str, Any]]:
    baseline_col = f"baseline_C_{label}"
    clean = df.copy()
    clean[f"brier_model_{label}"] = (clean["calibrated_prob"] - clean[label].astype(float)) ** 2
    clean[f"brier_baseline_C_{label}"] = (clean[baseline_col] - clean[label].astype(float)) ** 2
    clean[f"brier_improvement_{label}"] = clean[f"brier_baseline_C_{label}"] - clean[f"brier_model_{label}"]
    rows: list[dict[str, Any]] = []
    for deficit, grp in clean.groupby("fav_deficit", sort=True):
        y = grp[label].astype(int)
        rows.append({
            "fav_deficit": int(deficit),
            "n": int(len(grp)),
            "actual_rate": float(y.mean()),
            "mean_model_prob": float(grp["calibrated_prob"].mean()),
            "mean_baseline_C": float(grp[baseline_col].mean()),
            "brier_model": float(grp[f"brier_model_{label}"].mean()),
            "brier_baseline_C": float(grp[f"brier_baseline_C_{label}"].mean()),
            "brier_improvement_baseline_C_minus_model": float(grp[f"brier_improvement_{label}"].mean()),
            "brier_improvement_bootstrap_ci": bootstrap_cluster_mean_ci(
                grp, f"brier_improvement_{label}", seed=BOOTSTRAP_SEED + 300 + int(deficit) + (0 if label == TARGET_LABEL else 1000)
            ),
        })
    return rows


def overall_brier_summary(df: pd.DataFrame, label: str) -> dict[str, Any]:
    baseline_col = f"baseline_C_{label}"
    clean = df.copy()
    clean[f"brier_model_{label}"] = (clean["calibrated_prob"] - clean[label].astype(float)) ** 2
    clean[f"brier_baseline_C_{label}"] = (clean[baseline_col] - clean[label].astype(float)) ** 2
    clean[f"brier_improvement_{label}"] = clean[f"brier_baseline_C_{label}"] - clean[f"brier_model_{label}"]
    y = clean[label].astype(int)
    return {
        "label": label,
        "n": int(len(clean)),
        "n_games": int(clean["game_id"].nunique()),
        "actual_rate": float(y.mean()),
        "mean_model_prob": float(clean["calibrated_prob"].mean()),
        "mean_baseline_C": float(clean[baseline_col].mean()),
        "brier_model": brier(y, clean["calibrated_prob"]),
        "brier_baseline_C": brier(y, clean[baseline_col]),
        "brier_improvement_baseline_C_minus_model": float(clean[f"brier_improvement_{label}"].mean()),
        "brier_improvement_bootstrap_ci": bootstrap_cluster_mean_ci(
            clean, f"brier_improvement_{label}", seed=BOOTSTRAP_SEED + 400 + (0 if label == TARGET_LABEL else 1000)
        ),
        "ece_model": expected_calibration_error(y.to_numpy(), clean["calibrated_prob"].to_numpy()),
        "ece_baseline_C": expected_calibration_error(y.to_numpy(), clean[baseline_col].to_numpy()),
        "auc_model": metric_bundle(y.to_numpy(), clean["calibrated_prob"].to_numpy())["auc"],
        "auc_baseline_C": metric_bundle(y.to_numpy(), clean[baseline_col].to_numpy())["auc"],
    }


def fold_brier_summary(df: pd.DataFrame, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold, grp in df.groupby("fold", sort=True):
        rec = overall_brier_summary(grp, label)
        rec["fold"] = int(fold)
        rows.append(rec)
    return rows


def classify_primary(summary: dict[str, Any]) -> str:
    imp = summary["brier_improvement_baseline_C_minus_model"]
    ci = summary["brier_improvement_bootstrap_ci"]
    if imp > 0 and ci["2.5"] is not None and ci["2.5"] > 0:
        return "yes_materially"
    if imp > 0:
        return "yes_marginally"
    return "no"


def calibration_gap_summary(deciles: list[dict[str, Any]]) -> dict[str, Any]:
    populated = [r for r in deciles if r["n"] and r["actual_minus_mean_model_prob"] is not None]
    if not populated:
        return {"weighted_mean_abs_gap": None, "max_abs_gap": None}
    total = sum(r["n"] for r in populated)
    weighted = sum(r["n"] * abs(r["actual_minus_mean_model_prob"]) for r in populated) / total
    return {
        "weighted_mean_abs_gap": float(weighted),
        "max_abs_gap": float(max(abs(r["actual_minus_mean_model_prob"]) for r in populated)),
    }


def validation_analysis_for_scheme(df: pd.DataFrame, scheme: str) -> dict[str, Any]:
    scheme_df = df[df["scheme"] == scheme].copy()
    out: dict[str, Any] = {}
    for label in LABELS_FOR_VALIDATION:
        overall = overall_brier_summary(scheme_df, label)
        deciles = decile_analysis(scheme_df, label)
        out[label] = {
            "overall_brier_vs_baseline_C": overall,
            "classification": classify_primary(overall),
            "fold_brier_vs_baseline_C": fold_brier_summary(scheme_df, label),
            "threshold_analysis": threshold_analysis(scheme_df, label),
            "quintile_analysis": quintile_analysis(scheme_df, label),
            "decile_analysis": deciles,
            "calibration_gap_summary": calibration_gap_summary(deciles),
            "per_deficit_analysis": per_deficit_analysis(scheme_df, label),
        }
    return out


baseline_validation = {
    scheme: validation_analysis_for_scheme(predictions_eval_df, scheme)
    for scheme in ["U", "W2"]
}

n05_reference = json.loads(N05_ANALYSIS_JSON.read_text(encoding="utf-8"))
n03_on_deficit_erased_reference = n05_reference["model_validation"][TARGET_LABEL]["overall_brier_vs_baseline_C"]
n03_on_favorite_final_win_reference = n05_reference["model_validation"][CROSS_LABEL]["overall_brier_vs_baseline_C"]

primary_summary = baseline_validation["U"][TARGET_LABEL]["overall_brier_vs_baseline_C"]
primary_classification = baseline_validation["U"][TARGET_LABEL]["classification"]
target_gap = baseline_validation["U"][TARGET_LABEL]["calibration_gap_summary"]

print("[ok] N06 baseline_C validation complete")
print("Primary U deficit_erased:", primary_summary)
print("N03 reference on deficit_erased:", n03_on_deficit_erased_reference)
""")


add("code", "c06_000a", """
def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if math.isnan(val) or math.isinf(val) else val
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, (np.ndarray,)):
        return [_jsonable(v) for v in obj.tolist()]
    return obj


def calibration_params_for_fit(fit: FitResult) -> dict[str, Any]:
    return {
        "fold": fit.fold,
        "train_window": fit.train_window_label,
        "val_season": fit.val_season,
        "test_season": fit.test_season,
        "x_thresholds": fit.calibrator.X_thresholds_.tolist(),
        "y_thresholds": fit.calibrator.y_thresholds_.tolist(),
        "health": fit.calibration_health,
    }


def coefficients_for_fit(fit: FitResult) -> dict[str, Any]:
    return {
        "fold": fit.fold,
        "train_window": fit.train_window_label,
        "coefficients": fit.coef_by_col,
    }


scheme_comparison: list[dict[str, Any]] = []
for scheme in ["U", "W2"]:
    rows = [r for r in final_metric_rows if r["scheme"] == scheme]
    weights = SCHEME_WEIGHTS[scheme]
    scheme_comparison.append({
        "scheme": scheme,
        "selected_r6_feature_count": len(selected_features_by_scheme[scheme]),
        "selected_core_feature_count": len(selected_model_core_features(selected_features_by_scheme[scheme])),
        "selected_indicator_count": len(selected_indicator_cols(selected_features_by_scheme[scheme])),
        "dropped_features": pruning_by_scheme[scheme]["dropped_features"],
        "fold_metrics": rows,
        "weighted_brier": weighted_metric(rows, "brier", weights),
        "weighted_ece": weighted_metric(rows, "ece", weights),
        "weighted_auc": weighted_metric(rows, "auc", weights),
        "weighted_brier_vs_constant": weighted_metric(rows, "brier_vs_constant", weights),
    })

e_metrics = metric_bundle(fit_e.y_val, fit_e.cal_val)
e_raw_metrics = metric_bundle(fit_e.y_val, fit_e.raw_val)
scheme_e_summary = {
    "scheme": "E",
    "structure": "train_2015_2023_validate_2024_no_heldout_test",
    "selected_feature_source": "W2",
    "n_train": int(len(fit_e.train_df)),
    "n_val": int(len(fit_e.val_df)),
    "metrics_on_2024_validation_calibrated_on_same_slice": e_metrics,
    "raw_metrics_on_2024_validation": e_raw_metrics,
    "calibration_health": fit_e.calibration_health,
}

reliability_summary: dict[str, Any] = {}
for scheme, fits in final_fits_by_scheme.items():
    reliability_summary[scheme] = {
        fit.fold: reliability_bins(fit.y_test, fit.cal_test)
        for fit in fits
    }
reliability_summary["E"] = {2024: reliability_bins(fit_e.y_val, fit_e.cal_val)}

spec = {
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
    "target_label": TARGET_LABEL,
    "cross_label": CROSS_LABEL,
    "methodology_integrity": (
        "N06 intentionally changes one variable from N03: the training label is deficit_erased "
        "instead of favorite_final_win. Feature pool, L1 model class, null handling, play-level "
        "deduplication, walk-forward windows, and isotonic calibration remain aligned with N03."
    ),
    "feature_pool_source": str(FEATURE_VALIDATION_CSV.relative_to(REPO_ROOT)),
    "feature_pool_count": len(model_core_features),
    "r6_validated_feature_count": len(pass_features),
    "structural_conditioning_feature_count": len(STRUCTURAL_FEATURES),
    "feature_pool": [
        {"feature": feat, **feature_meta[feat]}
        for feat in pass_features
    ],
    "structural_conditioning_variables": [
        {
            "feature": "fav_deficit",
            "structural_conditioning_variable": True,
            "pruning_exempt": True,
            "note": (
                "Defines the trigger deficit threshold. It was not stability-tested in Phase 0 "
                "because it conditions the event definition rather than measuring a new game-state hypothesis."
            ),
        }
    ],
    "training_structure": {
        "policy": "deduplicate_to_unique_trigger_plays_using_lowest_qualifying_deficit",
        "full_trigger_event_rows": int(len(event_df)),
        "unique_trigger_play_rows": int(len(wide_df)),
        "n05_deficit_erased_null_event_rows_excluded": excluded_deficit_erased_null_event_rows,
        "unique_trigger_play_key": play_key_cols,
        "trigger_event_key": ["game_id", "fav_deficit", "trigger_sequence"],
        "threshold_count_distribution": threshold_count_distribution,
        "multi_threshold_play_count_full_corpus": n_multi_threshold_plays_full,
        "nonconstant_multi_threshold_feature_groups": nonconstant_feature_groups,
    },
    "null_handling": {
        "policy": "train_fold_median_imputation_plus_missingness_indicators_for_full_corpus_null_rate_gt_5pct",
        "threshold": NULL_INDICATOR_THRESHOLD,
        "null_policy_table": null_policy_df.to_dict(orient="records"),
        "indicator_meta": indicator_meta,
        "core_feature_count": len(model_core_features),
        "r6_validated_core_feature_count": len(pass_features),
        "structural_core_feature_count": len(STRUCTURAL_FEATURES),
        "indicator_column_count": len(model_indicator_cols),
        "post_imputation_model_column_count_full": len(model_core_features) + len(model_indicator_cols),
    },
    "walk_forward_windows": WALK_FORWARD_WINDOWS,
    "deployment_window_E": DEPLOYMENT_WINDOW_E,
    "feature_list": {
        scheme: {
            "selected_r6_features": selected_features_by_scheme[scheme],
            "selected_core_features": selected_model_core_features(selected_features_by_scheme[scheme]),
            "selected_model_columns": model_columns_for_core(
                selected_model_core_features(selected_features_by_scheme[scheme]),
                selected_indicator_cols(selected_features_by_scheme[scheme]),
            ),
            "dropped_core_features": pruning_by_scheme[scheme]["dropped_features"],
            "protected_features": list(STRUCTURAL_FEATURES),
        }
        for scheme in ["U", "W2"]
    },
    "coefficients": {
        scheme: [coefficients_for_fit(fit) for fit in fits]
        for scheme, fits in final_fits_by_scheme.items()
    } | {"E": [coefficients_for_fit(fit_e)]},
    "calibration_params": {
        scheme: [calibration_params_for_fit(fit) for fit in fits]
        for scheme, fits in final_fits_by_scheme.items()
    } | {"E": [calibration_params_for_fit(fit_e)]},
    "sensitivity_sweep_summary": sensitivity_sweep_summary,
    "xgboost_comparison": xgboost_comparison,
    "scheme_comparison": scheme_comparison,
    "scheme_E_summary": scheme_e_summary,
    "baseline_C_tables": baseline_C_tables,
    "baseline_validation": baseline_validation,
    "primary_validation": {
        "scheme": "U",
        "label": TARGET_LABEL,
        "result": primary_summary,
        "classification": primary_classification,
        "calibration_gap_summary": target_gap,
        "n03_reference_on_deficit_erased": n03_on_deficit_erased_reference,
        "n03_reference_on_favorite_final_win": n03_on_favorite_final_win_reference,
    },
    "pruning": pruning_by_scheme,
    "indicator_diagnostics": indicator_diag_by_scheme,
    "reliability_summary": reliability_summary,
    "phase0_extract_summary": phase0_extract_summary,
    "output_verification": {
        "prediction_key_cols": prediction_key_cols,
        "expected_test_event_rows_per_scheme": expected_test_event_rows_per_scheme,
        "main_prediction_rows_total": int(len(predictions_df)),
        "e_prediction_rows_total": int(len(e_pred)),
        "fav_deficit_coefficients": fav_deficit_coefficients,
        "probability_variation_summary": probability_variation_summary,
        "probability_variation_sample": probability_variation_sample,
    },
}

if N06_STATE_EXPORT_ONLY:
    print(f"[state-export-only] skipped rewrite of {N06_MODEL_SPEC_JSON.relative_to(REPO_ROOT)}")
else:
    N06_MODEL_SPEC_JSON.write_text(json.dumps(_jsonable(spec), indent=2), encoding="utf-8")
    print(f"[ok] wrote {N06_MODEL_SPEC_JSON.relative_to(REPO_ROOT)} size={N06_MODEL_SPEC_JSON.stat().st_size:,} bytes")
""")


add("code", "c06_000ab", """
def fmt(x: Any, digits: int = 5, *, signed: bool = False) -> str:
    if x is None:
        return "NA"
    val = float(x)
    if math.isnan(val):
        return "NA"
    return f"{val:+.{digits}f}" if signed else f"{val:.{digits}f}"


def pct(x: Any, digits: int = 1) -> str:
    if x is None:
        return "NA"
    return f"{float(x) * 100:.{digits}f}%"


def ci_text(ci: dict[str, Any], digits: int = 5, *, signed: bool = False) -> str:
    return f"[{fmt(ci['2.5'], digits, signed=signed)}, {fmt(ci['97.5'], digits, signed=signed)}]"


primary_ci = primary_summary["brier_improvement_bootstrap_ci"]
cal_gap = target_gap["weighted_mean_abs_gap"]
if primary_classification == "yes_materially":
    structural_finding = (
        "N06 produces statistically supported comeback-detection edge over baseline_C on `deficit_erased`."
    )
elif cal_gap is not None and cal_gap <= 0.05:
    structural_finding = (
        "N06 calibrates the `deficit_erased` label much better than N03 but does not beat baseline_C on Brier."
    )
else:
    structural_finding = (
        "N06 does not produce a clean comeback-detection edge over baseline_C, and calibration gaps remain material."
    )

lines: list[str] = []
lines.append("# N06 deficit-erased model validation")
lines.append("")
lines.append(
    "**Primary finding:** Fitting directly on `deficit_erased` dramatically repaired the N03 label-calibration problem, "
    "but it did **not** produce edge over the deficit x time-bucket baseline_C. N03 under-predicted `deficit_erased` "
    "by roughly +15-30 percentage points across the middle probability deciles; N06 reduces that to a weighted mean "
    f"absolute decile gap of **{fmt(target_gap['weighted_mean_abs_gap'], 3)}** with max gap **{fmt(target_gap['max_abs_gap'], 3)}**. "
    f"Even so, Scheme U Brier improvement (`baseline_C - model`) is **{fmt(primary_summary['brier_improvement_baseline_C_minus_model'], signed=True)}** "
    f"with 95% cluster-bootstrap CI **{ci_text(primary_ci, signed=True)}**: calibrated, but flat against baseline."
)
lines.append("")
lines.append(
    f"**Mechanistic interpretation:** The model AUC (**{fmt(primary_summary['auc_model'], 4)}**) is essentially tied with "
    f"baseline_C AUC (**{fmt(primary_summary['auc_baseline_C'], 4)}**). The 30 engineered features plus protected "
    "`fav_deficit` add no ranking improvement over a 20-cell deficit x time lookup table. Whatever signal the engineered "
    "features carry is being absorbed by their correlation with deficit and time; they do not carry independent "
    "comeback-erasure signal beyond what the structural variables encode."
)
lines.append("")
cross_summary = baseline_validation["U"][CROSS_LABEL]["overall_brier_vs_baseline_C"]
lines.append(
    f"**Cross-label confirmation:** N06 on `favorite_final_win` is materially worse than baseline_C: "
    f"improvement **{fmt(cross_summary['brier_improvement_baseline_C_minus_model'], signed=True)}** with CI "
    f"**{ci_text(cross_summary['brier_improvement_bootstrap_ci'], signed=True)}**. That confirms the experiment was a clean "
    "A/B test: each model performs best on its trained label and badly on the other, and neither beats baseline_C on its own label."
)
lines.append("")
lines.append(
    "**Per-deficit pattern:** N06 shows no supported positive per-deficit edge against baseline_C. D=3 is significantly worse, "
    "D=7/D=10/D=14 are near-zero with CIs crossing zero, and D=21 is a tiny positive estimate with a CI crossing zero. "
    "N04's monotonic per-deficit improvement against pre-game market does not replicate against baseline_C, confirming N05's "
    "interpretation that N04's pattern was about market staleness at deeper deficits, not model deep-deficit insight."
)
lines.append("")
lines.append(
    "**Project conclusion:** The validated feature pool is exhausted relative to baseline_C for both labels. Future research "
    "requires either feature expansion (possession-adjusted deficit, trajectory features, fluke-score decomposition) or a "
    "different validation target, especially live market comparison once data is available."
)
lines.append("")
lines.append("**Methodology integrity:** N06 changes one variable from N03: the training label is `deficit_erased` instead of `favorite_final_win`. The 30 R6-validated features, protected `fav_deficit` structural variable, play-level deduplication, null handling, L1 model class, walk-forward windows, and isotonic calibration structure remain aligned with N03.")
lines.append("")
lines.append("## Primary validation versus baseline_C")
lines.append("")
lines.append("| Scheme | Label | N | Model Brier | Baseline C Brier | Improvement | 95% CI | Model ECE | Baseline ECE | Model AUC | Baseline AUC | Classification |")
lines.append("|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|")
for scheme in ["U", "W2"]:
    for label in LABELS_FOR_VALIDATION:
        s = baseline_validation[scheme][label]["overall_brier_vs_baseline_C"]
        lines.append(
            f"| {scheme} | `{label}` | {s['n']} | {fmt(s['brier_model'])} | {fmt(s['brier_baseline_C'])} | "
            f"{fmt(s['brier_improvement_baseline_C_minus_model'], signed=True)} | "
            f"{ci_text(s['brier_improvement_bootstrap_ci'], signed=True)} | "
            f"{fmt(s['ece_model'])} | {fmt(s['ece_baseline_C'])} | {fmt(s['auc_model'], 4)} | {fmt(s['auc_baseline_C'], 4)} | "
            f"{baseline_validation[scheme][label]['classification']} |"
        )

lines.append("")
lines.append("## Per-fold target-label metrics")
lines.append("")
lines.append("| Scheme | Fold | N | Model Brier | Baseline C Brier | Improvement | 95% CI | Model ECE | Model AUC |")
lines.append("|---|---:|---:|---:|---:|---:|---|---:|---:|")
for scheme in ["U", "W2"]:
    for r in baseline_validation[scheme][TARGET_LABEL]["fold_brier_vs_baseline_C"]:
        lines.append(
            f"| {scheme} | {r['fold']} | {r['n']} | {fmt(r['brier_model'])} | {fmt(r['brier_baseline_C'])} | "
            f"{fmt(r['brier_improvement_baseline_C_minus_model'], signed=True)} | "
            f"{ci_text(r['brier_improvement_bootstrap_ci'], signed=True)} | {fmt(r['ece_model'])} | {fmt(r['auc_model'], 4)} |"
        )

lines.append("")
lines.append("## Per-deficit target-label pattern")
lines.append("")
lines.append("| Deficit | N | Actual rate | Mean model prob | Mean baseline C | Model Brier | Baseline C Brier | Improvement | 95% CI |")
lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|")
for r in baseline_validation["U"][TARGET_LABEL]["per_deficit_analysis"]:
    lines.append(
        f"| D={r['fav_deficit']} | {r['n']} | {pct(r['actual_rate'])} | {pct(r['mean_model_prob'])} | "
        f"{pct(r['mean_baseline_C'])} | {fmt(r['brier_model'])} | {fmt(r['brier_baseline_C'])} | "
        f"{fmt(r['brier_improvement_baseline_C_minus_model'], signed=True)} | {ci_text(r['brier_improvement_bootstrap_ci'], signed=True)} |"
    )

lines.append("")
lines.append("## Threshold analysis")
lines.append("")
lines.append("| X | N | Actual rate | Mean model prob | Mean baseline C | Actual - model | Actual - baseline C |")
lines.append("|---:|---:|---:|---:|---:|---:|---:|")
for r in baseline_validation["U"][TARGET_LABEL]["threshold_analysis"]:
    lines.append(
        f"| {fmt(r['threshold'], 2)} | {r['n']} | {pct(r['rate'])} | {pct(r['mean_model_prob'])} | "
        f"{pct(r['mean_baseline_C'])} | {fmt(r['actual_minus_mean_model_prob'], signed=True)} | "
        f"{fmt(r['actual_minus_mean_baseline_C'], signed=True)} |"
    )

lines.append("")
lines.append("## Quintiles and calibration deciles")
lines.append("")
lines.append(f"Spearman(`calibrated_prob`, `deficit_erased`) = **{fmt(baseline_validation['U'][TARGET_LABEL]['quintile_analysis']['spearman_model_prob_vs_actual'], 4)}**.")
lines.append("")
lines.append("| Quintile | N | Prob range | Mean model prob | Mean baseline C | Actual rate | Actual - model | Actual - baseline C |")
lines.append("|---:|---:|---|---:|---:|---:|---:|---:|")
for r in baseline_validation["U"][TARGET_LABEL]["quintile_analysis"]["rows"]:
    lines.append(
        f"| {r['quintile']} | {r['n']} | [{fmt(r['model_prob_min'])}, {fmt(r['model_prob_max'])}] | "
        f"{pct(r['mean_model_prob'])} | {pct(r['mean_baseline_C'])} | {pct(r['rate'])} | "
        f"{fmt(r['actual_minus_mean_model_prob'], signed=True)} | {fmt(r['actual_minus_mean_baseline_C'], signed=True)} |"
    )

lines.append("")
lines.append("| Decile | N | Mean model prob | Mean baseline C | Actual rate | Actual - model | Actual - baseline C |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")
for r in baseline_validation["U"][TARGET_LABEL]["decile_analysis"]:
    lines.append(
        f"| {r['decile']} ({fmt(r['lo'], 1)}-{fmt(r['hi'], 1)}) | {r['n']} | {pct(r['mean_model_prob'])} | "
        f"{pct(r['mean_baseline_C'])} | {pct(r['rate'])} | {fmt(r['actual_minus_mean_model_prob'], signed=True)} | "
        f"{fmt(r['actual_minus_mean_baseline_C'], signed=True)} |"
    )

lines.append("")
lines.append("## Cross-label comparison")
lines.append("")
lines.append("N06 is trained on `deficit_erased`. This table checks how the same probabilities behave against `favorite_final_win` and keeps the label distinction explicit.")
lines.append("")
lines.append("| Model/use | Label | Brier improvement vs baseline_C | 95% CI | Model Brier | Baseline C Brier |")
lines.append("|---|---|---:|---|---:|---:|")
for model_name, ref in [
    ("N06 Scheme U", baseline_validation["U"][CROSS_LABEL]["overall_brier_vs_baseline_C"]),
    ("N03 reference", n03_on_favorite_final_win_reference),
]:
    lines.append(
        f"| {model_name} | `{CROSS_LABEL}` | {fmt(ref['brier_improvement_baseline_C_minus_model'], signed=True)} | "
        f"{ci_text(ref['brier_improvement_bootstrap_ci'], signed=True)} | {fmt(ref['brier_model'])} | {fmt(ref['brier_baseline_C'])} |"
    )
for model_name, ref in [
    ("N06 Scheme U", baseline_validation["U"][TARGET_LABEL]["overall_brier_vs_baseline_C"]),
    ("N03 reference", n03_on_deficit_erased_reference),
]:
    lines.append(
        f"| {model_name} | `{TARGET_LABEL}` | {fmt(ref['brier_improvement_baseline_C_minus_model'], signed=True)} | "
        f"{ci_text(ref['brier_improvement_bootstrap_ci'], signed=True)} | {fmt(ref['brier_model'])} | {fmt(ref['brier_baseline_C'])} |"
    )

lines.append("")
lines.append("## Feature selection and sensitivity")
lines.append("")
for row in scheme_comparison:
    dropped = ", ".join(row["dropped_features"]) if row["dropped_features"] else "none"
    lines.append(
        f"- {row['scheme']}: selected {row['selected_r6_feature_count']} R6 features + {len(STRUCTURAL_FEATURES)} protected structural feature; indicators={row['selected_indicator_count']}; dropped={dropped}."
    )
lines.append("")
lines.append("| C | U weighted Brier | U weighted ECE | U weighted AUC | Nonzero feature union |")
lines.append("|---:|---:|---:|---:|---:|")
for row in sensitivity_sweep_summary:
    lines.append(
        f"| {row['C']} | {fmt(row['mean_brier_U'])} | {fmt(row['mean_ece_U'])} | {fmt(row['mean_auc_U'], 4)} | {len(row['nonzero_feature_union'])} |"
    )

lines.append("")
lines.append("## Data and outputs")
lines.append("")
lines.append(f"- N05 non-null `deficit_erased` event rows used: {len(event_df):,}; excluded null rows: {excluded_deficit_erased_null_event_rows}.")
lines.append(f"- Play-level model rows after deduplication: {len(wide_df):,}.")
lines.append(f"- Held-out prediction rows per scheme: {expected_test_event_rows_per_scheme:,}; main parquet rows across U/W2: {len(predictions_df):,}.")
lines.append(f"- Scheme E 2024 validation rows: {len(e_pred):,}.")
lines.append("")
lines.append("## Honest interpretation")
lines.append("")
if primary_classification == "yes_materially":
    lines.append("N06 supports the hypothesis that the existing engineered feature pool carries comeback-erasure signal beyond a simple deficit/time lookup once the model is fit on the correct label. The next question would be whether this edge is stable enough for deployment-style decision rules.")
elif cal_gap is not None and cal_gap <= 0.05:
    lines.append("N06 appears to fix the label-calibration problem but still does not add Brier value beyond baseline_C. That would mean the right label matters for probability scale, but the current feature pool is mostly exhausted relative to the deficit/time baseline.")
else:
    lines.append("N06 does not rescue the comeback-detection question. Even after training directly on `deficit_erased`, the model does not show a statistically supported Brier edge over baseline_C and calibration gaps remain material enough to treat the feature pool as insufficient for this target.")

if N06_STATE_EXPORT_ONLY:
    print(f"[state-export-only] skipped rewrite of {N06_SUMMARY_REPORT_MD.relative_to(REPO_ROOT)}")
else:
    N06_SUMMARY_REPORT_MD.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
    print(f"[ok] wrote {N06_SUMMARY_REPORT_MD.relative_to(REPO_ROOT)} size={N06_SUMMARY_REPORT_MD.stat().st_size:,} bytes")
""")


add("code", "c06_000b", """
print("=" * 80)
print("N06 summary")
print("=" * 80)
print(f"Full trigger-event rows: {len(event_df):,}")
print(f"Play-level training/evaluation rows: {len(wide_df):,}")
print(f"Semantic Phase 0 features: {len(pass_features)}")
print(f"Structural conditioning features: {len(STRUCTURAL_FEATURES)}")
print(f"Missingness indicators: {len(model_indicator_cols)}")
print(f"Full post-imputation model columns: {len(model_core_features) + len(model_indicator_cols)}")

print("\\nFinal scheme comparison:")
for row in scheme_comparison:
    print(
        f"  {row['scheme']}: selected={row['selected_r6_feature_count']} R6 + {len(STRUCTURAL_FEATURES)} structural, "
        f"indicators={row['selected_indicator_count']}, "
        f"weighted_brier={row['weighted_brier']:.5f}, "
        f"weighted_ece={row['weighted_ece']:.5f}, "
        f"weighted_auc={row['weighted_auc']:.4f}, "
        f"weighted_brier_vs_constant={row['weighted_brier_vs_constant']:+.5f}"
    )
    if row["dropped_features"]:
        print(f"    dropped: {', '.join(row['dropped_features'])}")
    else:
        print("    dropped: none")

print("\\nFold metrics:")
for r in final_metric_rows:
    print(
        f"  {r['scheme']} fold={r['fold']} "
        f"brier={r['brier']:.5f} ece={r['ece']:.5f} auc={r['auc']:.4f} "
        f"brier_vs_constant={r['brier_vs_constant']:+.5f}"
    )

print("\\nScheme E deployment-proximate:")
print(
    f"  n_train={scheme_e_summary['n_train']:,} n_val_2024={scheme_e_summary['n_val']:,} "
    f"brier={scheme_e_summary['metrics_on_2024_validation_calibrated_on_same_slice']['brier']:.5f} "
    f"ece={scheme_e_summary['metrics_on_2024_validation_calibrated_on_same_slice']['ece']:.5f} "
    f"auc={scheme_e_summary['metrics_on_2024_validation_calibrated_on_same_slice']['auc']:.4f}"
)

print("\\nPrimary baseline_C validation:")
print(
    f"  U {TARGET_LABEL}: improvement={primary_summary['brier_improvement_baseline_C_minus_model']:+.5f} "
    f"CI=[{primary_summary['brier_improvement_bootstrap_ci']['2.5']:+.5f}, "
    f"{primary_summary['brier_improvement_bootstrap_ci']['97.5']:+.5f}] "
    f"classification={primary_classification}"
)
print(
    f"  calibration weighted mean abs decile gap={target_gap['weighted_mean_abs_gap']:.5f} "
    f"max abs gap={target_gap['max_abs_gap']:.5f}"
)

print("\\nDeliverables:")
for path in [N06_PREDICTIONS_PARQUET, N06_E_PREDICTIONS_PARQUET, N06_MODEL_SPEC_JSON, N06_SUMMARY_REPORT_MD, N06_FULL_FITTED_STATE_JSON]:
    print(f"  {path.relative_to(REPO_ROOT)}  {path.stat().st_size:,} bytes")

print("\\nOutput verification:")
print(f"  full event matrix rows: {len(event_df):,}")
print(f"  expected held-out test events per scheme: {expected_test_event_rows_per_scheme:,}")
print(f"  main prediction rows total: {len(predictions_df):,}")
print(f"  E prediction rows total: {len(e_pred):,}")
print(f"  fav_deficit coefficients: {fav_deficit_coefficients}")
print(f"  probability variation summary: {probability_variation_summary}")

cal_warnings = []
for scheme, fits in final_fits_by_scheme.items():
    for fit in fits:
        if fit.calibration_health.get("warning"):
            cal_warnings.append((scheme, fit.fold, fit.calibration_health))
if fit_e.calibration_health.get("warning"):
    cal_warnings.append(("E", fit_e.fold, fit_e.calibration_health))
print("\\nCalibration health warnings:")
if cal_warnings:
    for row in cal_warnings:
        print(f"  {row}")
else:
    print("  none")

print("\\n[ok] N06 complete -- halt for diagnostics/report; no commit performed.")
""")


def _to_lines(s: str) -> list[str]:
    lines = s.split("\n")
    out = [ln + "\n" for ln in lines[:-1]]
    if lines[-1] != "":
        out.append(lines[-1])
    return out


def _cell_dict(cell_type: str, cell_id: str, src: str) -> dict[str, Any]:
    d: dict[str, Any] = {
        "cell_type": cell_type,
        "id": cell_id,
        "metadata": {},
        "source": _to_lines(src),
    }
    if cell_type == "code":
        d["execution_count"] = None
        d["outputs"] = []
    return d


nb = {
    "cells": [_cell_dict(t, cid, s) for (t, cid, s) in CELLS],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"[ok] wrote {OUT} ({OUT.stat().st_size:,} bytes, {len(CELLS)} cells)")



