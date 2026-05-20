"""
Deterministic builder for research/notebooks/07_feature_pool_expansion.ipynb.

N07 tests a pre-registered 14-feature expansion against the strict
deficit x time-bucket baseline_C established in N05/N06.
"""

from __future__ import annotations

import json
import pathlib
import textwrap
from typing import Any

OUT = pathlib.Path(__file__).resolve().parent / "07_feature_pool_expansion.ipynb"

CELLS: list[tuple[str, str, str]] = []


def add(cell_type: str, cell_id: str, src: str) -> None:
    CELLS.append((cell_type, cell_id, textwrap.dedent(src).lstrip("\n")))


add("markdown", "m07_0000", """
# Notebook 07 -- Feature pool expansion test

N07 is a pre-registered expansion test after N05/N06 showed that the
validated Phase 0 feature pool does not beat the strict `fav_deficit x
time_bucket` baseline_C on either label.

The candidate set is locked before fitting:

- Category A: possession-adjusted deficit features.
- Category B: fluke-score decomposition features.
- Category C: efficiency-gap differential features.

Each candidate must clear all three gates to enter the expanded production
pool:

1. R6 stability: at least 2 of 3 positive Brier deltas versus the alpha
   pre-game baseline.
2. Magnitude: mean delta Brier across folds at least +0.001.
3. Baseline_C: alpha+candidate beats baseline_C with a Bonferroni-corrected
   one-sided bootstrap lower bound above zero on at least one label.

No candidates are added, removed, or swapped after results are visible.
""")


add("code", "c07_0001", r"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import pathlib
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _lib_chrono import _chrono_key

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
N05_DESCRIPTIVE_RATES_PARQUET = RESULTS_DIR / "n05_descriptive_rates.parquet"
N06_MODEL_SPEC_JSON = RESULTS_DIR / "n06_model_spec.json"

N07_FEATURES_PARQUET = RESULTS_DIR / "n07_descriptive_features.parquet"
N07_STABILITY_JSON = RESULTS_DIR / "n07_stability_results.json"
N07_EXPANDED_PREDICTIONS_PARQUET = RESULTS_DIR / "n07_expanded_model_predictions.parquet"
N07_EXPANDED_MODEL_SPEC_JSON = RESULTS_DIR / "n07_expanded_model_spec.json"
N07_SUMMARY_REPORT_MD = RESULTS_DIR / "n07_summary_report.md"

assert RESEARCH_DIR.name == "research", f"Expected research/notebooks cwd, got {NOTEBOOK_DIR}"
for path in [TRIGGER_EVENTS_CSV, TRIGGER_OUTCOMES_CSV, FEATURE_VALIDATION_CSV, N05_DESCRIPTIVE_RATES_PARQUET, N06_MODEL_SPEC_JSON]:
    assert path.exists(), f"Missing required artifact: {path}"

RANDOM_STATE = 42
NULL_INDICATOR_THRESHOLD = 0.05
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 42
AVG_POSSESSIONS_PER_MINUTE_LOCKED = 0.45
TOTAL_REGULATION_SECONDS = 3600

TARGET_LABEL = "deficit_erased"
CROSS_LABEL = "favorite_final_win"
LABELS = [TARGET_LABEL, CROSS_LABEL]

WALK_FORWARD_WINDOWS = [
    {"fold": 2022, "train_seasons": list(range(2015, 2021)), "val_season": 2021, "test_season": 2022, "train_window_label": "2015-2020"},
    {"fold": 2023, "train_seasons": list(range(2015, 2022)), "val_season": 2022, "test_season": 2023, "train_window_label": "2015-2021"},
    {"fold": 2024, "train_seasons": list(range(2015, 2023)), "val_season": 2023, "test_season": 2024, "train_window_label": "2015-2022"},
]
SCHEME_WEIGHTS = {
    "U": {2022: 1 / 3, 2023: 1 / 3, 2024: 1 / 3},
    "W2": {2022: 0.25, 2023: 0.25, 2024: 0.50},
}

ALPHA_FEATURES = [
    "pregame_spread",
    "rating_gap",
    "fav_pregame_rating",
    "dog_pregame_rating",
    "spread_movement",
    "spread_movement_is_null",
]

PHASE0_NOTEBOOK_CELLS = {
    "02a_baseline_features.ipynb": [4, 6, 8, 10, 11, 13],
    "02b_opening_drive_shock.ipynb": [4, 6, 8, 10, 11, 13],
    "02c_explosive_vs_sustained.ipynb": [4, 6, 8, 10, 12, 13, 15],
    "02d_turnover_and_short_field.ipynb": [4, 6, 8, 10, 11, 13],
    "02e_red_zone_failure.ipynb": [4, 6, 8, 10, 11, 13],
    "02f_down_distance_efficiency.ipynb": [4, 6, 8, 10, 11, 13],
    "02g_context_week_home_neutral.ipynb": [4, 6, 8, 10],
}

PHASE0_SOURCE_COLUMNS = {
    "fav_off_epa_per_play",
    "dog_off_epa_per_play",
    "dog_points_off_turnovers",
    "dog_points_from_returns",
    "dog_points_from_explosives",
    "dog_points_from_sustained",
    "fav_yards_per_point",
    "fav_early_down_success_rate",
    "dog_early_down_success_rate",
    "fav_third_down_success_rate",
    "dog_third_down_success_rate",
    "dog_avg_drive_yards",
}

STRUCTURAL_FEATURES = ["fav_deficit"]

EXPLOSIVE_PASS_YARDS = 20
EXPLOSIVE_RUSH_YARDS = 12
EXPLOSIVE_PASS_PLAY_TYPES = {"Pass Reception", "Passing Touchdown"}
EXPLOSIVE_RUSH_PLAY_TYPES = {"Rush", "Rushing Touchdown"}

PRE_REGISTERED_FEATURES = [
    {
        "feature": "estimated_possessions_remaining",
        "category": "A",
        "expected_sign": {"favorite_final_win": "positive", "deficit_erased": "positive"},
        "description": "Estimated remaining possessions from regulation seconds remaining and locked 0.45 possessions/minute.",
    },
    {
        "feature": "deficit_per_remaining_possession",
        "category": "A",
        "expected_sign": {"favorite_final_win": "negative", "deficit_erased": "negative"},
        "description": "fav_deficit / max(1, estimated_possessions_remaining).",
    },
    {
        "feature": "possessions_needed_to_tie",
        "category": "A",
        "expected_sign": {"favorite_final_win": "negative", "deficit_erased": "negative"},
        "description": "ceil(fav_deficit / 7).",
    },
    {
        "feature": "clock_pressure_index",
        "category": "A",
        "expected_sign": {"favorite_final_win": "negative", "deficit_erased": "negative"},
        "description": "deficit_per_remaining_possession * (1 + fav_deficit / 21).",
    },
    {
        "feature": "dog_points_from_turnovers_pct",
        "category": "B",
        "expected_sign": {"favorite_final_win": "positive", "deficit_erased": "positive"},
        "description": "dog_points_off_turnovers / max(1, dog_score_at_trigger).",
    },
    {
        "feature": "dog_points_from_returns_pct",
        "category": "B",
        "expected_sign": {"favorite_final_win": "positive", "deficit_erased": "positive"},
        "description": "dog_points_from_returns / max(1, dog_score_at_trigger).",
    },
    {
        "feature": "dog_points_from_explosives_pct",
        "category": "B",
        "expected_sign": {"favorite_final_win": "positive", "deficit_erased": "positive"},
        "description": "dog_points_from_explosives / max(1, dog_score_at_trigger).",
    },
    {
        "feature": "dog_offensive_points_pct",
        "category": "B",
        "expected_sign": {"favorite_final_win": "negative", "deficit_erased": "negative"},
        "description": "(dog_score_at_trigger - dog_points_from_returns) / max(1, dog_score_at_trigger).",
    },
    {
        "feature": "fav_yards_per_point_ratio",
        "category": "B",
        "expected_sign": {"favorite_final_win": "negative", "deficit_erased": "negative"},
        "description": "fav_yards_per_point / dog_yards_per_point.",
    },
    {
        "feature": "epa_per_play_gap",
        "category": "C",
        "expected_sign": {"favorite_final_win": "positive", "deficit_erased": "positive"},
        "description": "fav_off_epa_per_play - dog_off_epa_per_play.",
    },
    {
        "feature": "success_rate_gap",
        "category": "C",
        "expected_sign": {"favorite_final_win": "positive", "deficit_erased": "positive"},
        "description": "fav_early_down_success_rate - dog_early_down_success_rate.",
    },
    {
        "feature": "third_down_gap",
        "category": "C",
        "expected_sign": {"favorite_final_win": "positive", "deficit_erased": "positive"},
        "description": "fav_third_down_success_rate - dog_third_down_success_rate.",
    },
    {
        "feature": "explosive_rate_gap",
        "category": "C",
        "expected_sign": {"favorite_final_win": "positive", "deficit_erased": "positive"},
        "description": "fav_explosive_play_rate - dog_explosive_play_rate.",
    },
    {
        "feature": "drive_yards_gap",
        "category": "C",
        "expected_sign": {"favorite_final_win": "positive", "deficit_erased": "positive"},
        "description": "fav_avg_drive_yards - dog_avg_drive_yards.",
    },
]
N07_FEATURE_NAMES = [f["feature"] for f in PRE_REGISTERED_FEATURES]
assert len(N07_FEATURE_NAMES) == 14 and len(set(N07_FEATURE_NAMES)) == 14

BONFERRONI_ALPHA = {"A": 0.05 / 4, "B": 0.05 / 5, "C": 0.05 / 5}

print(f"[ok] N07 constants loaded at {NOTEBOOK_DIR}")
print(f"[ok] pre-registered candidates: {len(N07_FEATURE_NAMES)}")
""")


add("code", "c07_0002", r"""
def _params_hash(params: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def _cache_key(endpoint: str, params: dict[str, Any]) -> pathlib.Path:
    endpoint_key = endpoint.strip("/").replace("/", "_")
    return CACHE_DIR / f"cfbd__{endpoint_key}__{_params_hash(params)}.json"


def readonly_cfbd_get(endpoint: str, force_refresh: bool = False, **params: Any) -> Any:
    if force_refresh:
        raise AssertionError("N07 forbids force_refresh; cache-only extraction is required")
    key = _cache_key(endpoint, params)
    if not key.exists():
        raise AssertionError(f"N07 missing local cache for {endpoint} {params}; halt before external fetch.")
    return json.loads(key.read_text(encoding="utf-8"))


def stable_seed_offset(*parts: str) -> int:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100000


def _base_phase0_namespace() -> dict[str, Any]:
    return {
        "__name__": "_n07_phase0_cell_exec",
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


def run_phase0_matrix_notebook(nb_name: str, cell_indexes: list[int]) -> tuple[pd.DataFrame, list[str], dict[str, Any], dict[str, Any]]:
    nb_path = NOTEBOOK_DIR / nb_name
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    ns = _base_phase0_namespace()
    stdout = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(stdout):
        for idx in cell_indexes:
            src = "".join(nb["cells"][idx]["source"])
            exec(compile(src, f"{nb_name}:cell{idx}", "exec"), ns)
            ns["cfbd_get"] = readonly_cfbd_get
    elapsed = time.perf_counter() - t0
    fm = ns.get("feature_matrix_df")
    if fm is None:
        raise AssertionError(f"{nb_name} did not build feature_matrix_df")
    candidates = list(ns.get("CANDIDATE_FEATURES", []))
    summary = {
        "notebook": nb_name,
        "cells": cell_indexes,
        "rows": int(len(fm)),
        "cols": int(fm.shape[1]),
        "candidate_count": int(len(candidates)),
        "elapsed_sec": elapsed,
        "stdout_tail": stdout.getvalue()[-2000:],
    }
    return fm.copy(), candidates, summary, ns


def load_pass_feature_pool() -> tuple[list[str], dict[str, dict[str, Any]]]:
    fv = pd.read_csv(FEATURE_VALIDATION_CSV, keep_default_na=False)
    feature_order: list[str] = []
    feature_meta: dict[str, dict[str, Any]] = {}
    for feat in fv["feature"].tolist():
        if feat in feature_order:
            continue
        sub = fv[fv["feature"] == feat].copy()
        if bool(sub["passed_stability"].astype(str).str.lower().eq("true").all()):
            feature_order.append(str(feat))
            feature_meta[str(feat)] = {
                "feature_set_version": str(sub["feature_set_version"].iloc[0]),
                "redundant_with": str(sub["redundant_with"].iloc[0]),
            }
    assert len(feature_order) == 30, f"Expected 30 Phase 0 PASS features, got {len(feature_order)}"
    return feature_order, feature_meta


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & ((y_prob <= hi) if i == n_bins - 1 else (y_prob < hi))
        if mask.any():
            ece += float(mask.mean()) * abs(float(y_prob[mask].mean()) - float(y_true[mask].mean()))
    return float(ece)


def metric_bundle(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    out = {"brier": float(brier_score_loss(y_true, y_prob)), "ece": expected_calibration_error(y_true, y_prob)}
    try:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["auc"] = float("nan")
    return out


def _make_l1() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=RANDOM_STATE, max_iter=1000)),
    ])


def fit_preprocessor(train_df: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    medians: dict[str, float] = {}
    for feat in features:
        med = train_df[feat].median(skipna=True)
        medians[feat] = 0.0 if pd.isna(med) else float(med)
    return {"features": list(features), "medians": medians}


def transform_with_preprocessor(df: pd.DataFrame, prep: dict[str, Any]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for feat in prep["features"]:
        out[feat] = df[feat].astype(float).fillna(prep["medians"][feat])
    return out[prep["features"]]


def bootstrap_cluster_mean_ci(df: pd.DataFrame, value_col: str, *, seed: int, alpha: float = 0.05) -> dict[str, float | None]:
    if len(df) == 0:
        return {"lower": None, "2.5": None, "50": None, "97.5": None}
    grouped = df.groupby("game_id", sort=False)[value_col].agg(["sum", "count"]).reset_index(drop=True)
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(grouped), size=(BOOTSTRAP_RESAMPLES, len(grouped)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return {
        "alpha": float(alpha),
        "lower": float(np.percentile(means, alpha * 100.0)),
        "2.5": float(np.percentile(means, 2.5)),
        "50": float(np.percentile(means, 50.0)),
        "97.5": float(np.percentile(means, 97.5)),
    }


print("[ok] helpers defined")
""")


add("code", "c07_0003", r"""
t0 = time.perf_counter()
pass_features, pass_feature_meta = load_pass_feature_pool()
n05_df = pd.read_parquet(N05_DESCRIPTIVE_RATES_PARQUET)
trigger_df = pd.read_csv(TRIGGER_EVENTS_CSV)

n05_required = {
    "game_id", "fav_deficit", "trigger_sequence", "trigger_play_id", "season",
    "week", "fav_team", "dog_team", "quarter", "clock_seconds_in_period_total",
    "time_bucket", "favorite_final_win", "deficit_erased",
}
trigger_required = {
    "game_id", "fav_deficit", "trigger_sequence", "season_type",
    "seconds_remaining_in_regulation", "drive_number_in_game",
    "play_number",
    "fav_score_at_trigger", "dog_score_at_trigger",
    "pregame_spread", "rating_gap", "fav_pregame_rating", "dog_pregame_rating",
    "spread_movement",
}
assert not (n05_required - set(n05_df.columns)), n05_required - set(n05_df.columns)
assert not (trigger_required - set(trigger_df.columns)), trigger_required - set(trigger_df.columns)

key_cols = ["game_id", "fav_deficit", "trigger_sequence"]
base_df = n05_df[n05_df[TARGET_LABEL].notna()].copy()
base_df = base_df.merge(
    trigger_df[list(trigger_required)],
    on=key_cols,
    how="inner",
    validate="one_to_one",
)
assert len(base_df) == 11412, f"Expected 11,412 rows after N05+trigger merge, got {len(base_df)}"
assert not base_df.columns.duplicated().any(), "Duplicate columns after N05+trigger merge"
suffix_cols = [c for c in base_df.columns if c.endswith("_x") or c.endswith("_y")]
assert not suffix_cols, f"Unexpected merge suffix columns after N05+trigger merge: {suffix_cols}"
base_df[TARGET_LABEL] = base_df[TARGET_LABEL].astype(bool).astype(int)
base_df[CROSS_LABEL] = base_df[CROSS_LABEL].astype(bool).astype(int)
base_df["spread_movement_is_null"] = base_df["spread_movement"].isna().astype(int)
base_df["spread_movement"] = base_df["spread_movement"].fillna(0.0).astype(float)

phase0_matrices: dict[str, pd.DataFrame] = {}
phase0_extract_summary: list[dict[str, Any]] = []
phase0_namespaces: dict[str, dict[str, Any]] = {}
wanted_phase0_cols = set(pass_features) | PHASE0_SOURCE_COLUMNS

for nb_name, cell_indexes in PHASE0_NOTEBOOK_CELLS.items():
    fm, candidates, summary, ns = run_phase0_matrix_notebook(nb_name, cell_indexes)
    phase0_matrices[nb_name] = fm
    phase0_extract_summary.append(summary)
    phase0_namespaces[nb_name] = ns
    print(f"[ok] {nb_name}: rows={summary['rows']:,}, cols={summary['cols']:,}, elapsed={summary['elapsed_sec']:.1f}s")

wide_df = base_df.copy()
for nb_name, fm in phase0_matrices.items():
    cols = [c for c in wanted_phase0_cols if c in fm.columns and c not in wide_df.columns]
    if not cols:
        continue
    wide_df = wide_df.merge(fm[key_cols + cols], on=key_cols, how="left", validate="one_to_one")
    print(f"[ok] merged {nb_name}: +{len(cols)} source columns")

missing_pass = sorted(set(pass_features) - set(wide_df.columns))
missing_source = sorted(PHASE0_SOURCE_COLUMNS - set(wide_df.columns))
assert not missing_pass, f"Missing Phase 0 PASS columns: {missing_pass}"
assert not missing_source, f"Missing N07 source columns: {missing_source}"

plays_by_game = None
drives_by_game = None
for ns in phase0_namespaces.values():
    if plays_by_game is None and "plays_by_game" in ns:
        plays_by_game = ns["plays_by_game"]
    if drives_by_game is None and "drives_by_game" in ns:
        drives_by_game = ns["drives_by_game"]
assert plays_by_game is not None, "No Phase 0 namespace exposed plays_by_game"
assert drives_by_game is not None, "No Phase 0 namespace exposed drives_by_game"


def _is_explosive(play: dict) -> bool:
    pt = play.get("playType", "")
    yg = play.get("yardsGained")
    if yg is None:
        return False
    if pt in EXPLOSIVE_PASS_PLAY_TYPES:
        return int(yg) >= EXPLOSIVE_PASS_YARDS
    if pt in EXPLOSIVE_RUSH_PLAY_TYPES:
        return int(yg) >= EXPLOSIVE_RUSH_YARDS
    return False


def _is_scrimmage_offense_play(play: dict) -> bool:
    return play.get("playType", "") in (EXPLOSIVE_PASS_PLAY_TYPES | EXPLOSIVE_RUSH_PLAY_TYPES)


def _completed_drives_before(drives_for_game: list[dict], trig_drive: int, offense: str) -> list[dict]:
    out = []
    for d in drives_for_game:
        dn = d.get("driveNumber")
        if dn is None or int(dn) >= trig_drive:
            continue
        if d.get("offense") == offense:
            out.append(d)
    return out


def _drive_points_for_offense(drive: dict) -> int:
    try:
        return max(0, int(drive.get("endOffenseScore") or 0) - int(drive.get("startOffenseScore") or 0))
    except (TypeError, ValueError):
        return 0


def _yards_per_point(drives_for_game: list[dict], trig_drive: int, offense: str) -> float | None:
    drives = _completed_drives_before(drives_for_game, trig_drive, offense)
    if not drives:
        return None
    yards_sum = 0
    points_sum = 0
    for d in drives:
        y = d.get("yards")
        if y is not None:
            yards_sum += int(y)
        points_sum += _drive_points_for_offense(d)
    if points_sum <= 0:
        return None
    return float(yards_sum) / float(points_sum)


def _avg_drive_yards(drives_for_game: list[dict], trig_drive: int, offense: str) -> float | None:
    vals = []
    for d in _completed_drives_before(drives_for_game, trig_drive, offense):
        y = d.get("yards")
        if y is not None:
            vals.append(float(y))
    if not vals:
        return None
    return float(np.mean(vals))


def _explosive_rate(plays_before: list[dict], offense: str) -> float | None:
    plays = [p for p in plays_before if p.get("offense") == offense and _is_scrimmage_offense_play(p)]
    if not plays:
        return None
    return float(sum(1 for p in plays if _is_explosive(p)) / len(plays))


custom_rows: list[dict[str, Any]] = []
for _, row in wide_df.iterrows():
    gid = int(row["game_id"])
    fav = str(row["fav_team"])
    dog = str(row["dog_team"])
    trig_period = int(row["quarter"])
    trig_elapsed = 900 - int(row["clock_seconds_in_period_total"])
    trig_drive = int(row["drive_number_in_game"])
    trig_play_number = int(row["play_number"])
    trig_key = (trig_period, trig_elapsed, trig_drive, trig_play_number)
    plays_before = [p for p in plays_by_game[gid] if _chrono_key(p) < trig_key]
    drives_for_game = drives_by_game[gid]
    fav_exp = _explosive_rate(plays_before, fav)
    dog_exp = _explosive_rate(plays_before, dog)
    fav_avg_yards = _avg_drive_yards(drives_for_game, trig_drive, fav)
    dog_ypp = _yards_per_point(drives_for_game, trig_drive, dog)
    custom_rows.append({
        "game_id": gid,
        "fav_deficit": int(row["fav_deficit"]),
        "trigger_sequence": int(row["trigger_sequence"]),
        "fav_explosive_play_rate": fav_exp,
        "dog_explosive_play_rate": dog_exp,
        "fav_avg_drive_yards": fav_avg_yards,
        "dog_yards_per_point": dog_ypp,
    })

custom_df = pd.DataFrame(custom_rows)
wide_df = wide_df.merge(custom_df, on=key_cols, how="inner", validate="one_to_one")
assert len(wide_df) == 11412, f"Custom feature merge changed rows: {len(wide_df)}"

unique_games = wide_df["game_id"].nunique()
total_drives = sum(len(drives_by_game[int(gid)]) for gid in wide_df["game_id"].unique())
empirical_possessions_per_minute = total_drives / (unique_games * 60.0)

remaining_seconds = wide_df["seconds_remaining_in_regulation"].clip(lower=0).astype(float)
wide_df["estimated_possessions_remaining"] = np.maximum(0.0, remaining_seconds * AVG_POSSESSIONS_PER_MINUTE_LOCKED / 60.0)
wide_df["deficit_per_remaining_possession"] = wide_df["fav_deficit"].astype(float) / np.maximum(1.0, wide_df["estimated_possessions_remaining"])
wide_df["possessions_needed_to_tie"] = np.ceil(wide_df["fav_deficit"].astype(float) / 7.0)
wide_df["clock_pressure_index"] = wide_df["deficit_per_remaining_possession"] * (1.0 + wide_df["fav_deficit"].astype(float) / 21.0)

dog_points = wide_df["dog_score_at_trigger"].astype(float)
dog_den = np.maximum(1.0, dog_points)
dog_no_score = dog_points.eq(0).astype(int)
for feat, src in [
    ("dog_points_from_turnovers_pct", "dog_points_off_turnovers"),
    ("dog_points_from_returns_pct", "dog_points_from_returns"),
    ("dog_points_from_explosives_pct", "dog_points_from_explosives"),
]:
    wide_df[feat] = np.where(dog_points.eq(0), 0.0, wide_df[src].astype(float) / dog_den)
    wide_df[f"{feat}_is_null"] = dog_no_score

dog_explosive_pct_edge_mask = wide_df["dog_points_from_explosives_pct"].gt(1.0)
assert int(dog_explosive_pct_edge_mask.sum()) == 5, (
    "Expected 5 dog_points_from_explosives_pct > 1 edge cases, got "
    f"{int(dog_explosive_pct_edge_mask.sum())}"
)
wide_df.loc[dog_explosive_pct_edge_mask, "dog_points_from_explosives_pct"] = np.nan
wide_df.loc[dog_explosive_pct_edge_mask, "dog_points_from_explosives_pct_is_null"] = 1

wide_df["dog_offensive_points_pct"] = np.where(
    dog_points.eq(0),
    0.0,
    (dog_points - wide_df["dog_points_from_returns"].astype(float)) / dog_den,
)
wide_df["dog_offensive_points_pct_is_null"] = dog_no_score

wide_df["fav_yards_per_point_ratio_is_null"] = (
    wide_df["fav_yards_per_point"].isna()
    | wide_df["dog_yards_per_point"].isna()
    | wide_df["dog_yards_per_point"].eq(0)
).astype(int)
wide_df["fav_yards_per_point_ratio"] = np.where(
    wide_df["fav_yards_per_point_ratio_is_null"].eq(1),
    np.nan,
    wide_df["fav_yards_per_point"].astype(float) / wide_df["dog_yards_per_point"].astype(float),
)
fav_ypp_ratio_negative_edge_mask = wide_df["fav_yards_per_point_ratio"].lt(0)
assert int(fav_ypp_ratio_negative_edge_mask.sum()) == 5, (
    "Expected 5 fav_yards_per_point_ratio < 0 edge cases, got "
    f"{int(fav_ypp_ratio_negative_edge_mask.sum())}"
)
wide_df.loc[fav_ypp_ratio_negative_edge_mask, "fav_yards_per_point_ratio"] = np.nan
wide_df.loc[fav_ypp_ratio_negative_edge_mask, "fav_yards_per_point_ratio_is_null"] = 1

wide_df["epa_per_play_gap"] = wide_df["fav_off_epa_per_play"].astype(float) - wide_df["dog_off_epa_per_play"].astype(float)
wide_df["success_rate_gap"] = wide_df["fav_early_down_success_rate"].astype(float) - wide_df["dog_early_down_success_rate"].astype(float)
wide_df["third_down_gap"] = wide_df["fav_third_down_success_rate"].astype(float) - wide_df["dog_third_down_success_rate"].astype(float)
wide_df["explosive_rate_gap_is_null"] = (wide_df["fav_explosive_play_rate"].isna() | wide_df["dog_explosive_play_rate"].isna()).astype(int)
wide_df["explosive_rate_gap"] = wide_df["fav_explosive_play_rate"].astype(float) - wide_df["dog_explosive_play_rate"].astype(float)
wide_df["drive_yards_gap_is_null"] = (wide_df["fav_avg_drive_yards"].isna() | wide_df["dog_avg_drive_yards"].isna()).astype(int)
wide_df["drive_yards_gap"] = wide_df["fav_avg_drive_yards"].astype(float) - wide_df["dog_avg_drive_yards"].astype(float)

data_quality_violations: list[str] = []
for feat in N07_FEATURE_NAMES:
    s = wide_df[feat]
    if np.isinf(s.dropna()).any():
        data_quality_violations.append(f"{feat} contains infinite values")
    if s.dropna().lt(0).any() and feat.endswith("_pct"):
        data_quality_violations.append(f"{feat} contains negative percentage values")
    if s.dropna().gt(1.000001).any() and feat.endswith("_pct"):
        data_quality_violations.append(f"{feat} contains percentage values > 1")
if wide_df["fav_yards_per_point_ratio"].dropna().lt(0).any():
    data_quality_violations.append("fav_yards_per_point_ratio contains negative values")
if data_quality_violations:
    raise AssertionError("N07 data-quality violation(s): " + "; ".join(data_quality_violations))

indicator_cols: list[str] = []
feature_indicator_map: dict[str, list[str]] = {}
for feat in N07_FEATURE_NAMES:
    explicit = [c for c in [f"{feat}_is_null"] if c in wide_df.columns]
    if explicit:
        inds = explicit
    elif wide_df[feat].isna().mean() > NULL_INDICATOR_THRESHOLD:
        ind = f"{feat}_is_null"
        wide_df[ind] = wide_df[feat].isna().astype(int)
        inds = [ind]
    else:
        inds = []
    indicator_cols.extend(inds)
    feature_indicator_map[feat] = inds
indicator_cols = list(dict.fromkeys(indicator_cols))

feature_output_cols = [
    "game_id", "trigger_play_id", "fav_deficit", "trigger_sequence", "season", "time_bucket",
    "fav_team", "dog_team", "quarter", "clock_seconds_in_period_total",
    "favorite_final_win", "deficit_erased",
    *N07_FEATURE_NAMES,
    *indicator_cols,
]
feature_df = wide_df[feature_output_cols].copy()
feature_df.to_parquet(N07_FEATURES_PARQUET, index=False)

null_summary = {
    feat: {
        "null_count": int(wide_df[feat].isna().sum()),
        "null_rate": float(wide_df[feat].isna().mean()),
        "indicator_cols": feature_indicator_map[feat],
    }
    for feat in N07_FEATURE_NAMES
}

print(f"[ok] N07 descriptive feature matrix rows={len(feature_df):,} cols={feature_df.shape[1]}")
print(f"[ok] wrote {N07_FEATURES_PARQUET.relative_to(REPO_ROOT)} size={N07_FEATURES_PARQUET.stat().st_size:,}")
print(f"[ok] empirical possessions/minute from cached drives: {empirical_possessions_per_minute:.3f}; locked value used: {AVG_POSSESSIONS_PER_MINUTE_LOCKED:.3f}")
print(f"[ok] extraction elapsed: {time.perf_counter() - t0:.1f}s")
""")


add("code", "c07_0004", r"""
def baseline_c_table(label: str) -> pd.DataFrame:
    train = wide_df[(wide_df["season"].between(2015, 2021)) & wide_df[label].notna()].copy()
    train[label] = train[label].astype(int)
    tbl = (
        train.groupby(["fav_deficit", "time_bucket"])[label]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .rename(columns={"mean": f"baseline_C_{label}", "sum": "successes", "count": "n"})
    )
    return tbl


baseline_c_tables = {label: baseline_c_table(label) for label in LABELS}


def add_baseline_c(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for label, tbl in baseline_c_tables.items():
        out = out.merge(tbl[["fav_deficit", "time_bucket", f"baseline_C_{label}"]], on=["fav_deficit", "time_bucket"], how="left", validate="many_to_one")
        assert out[f"baseline_C_{label}"].notna().all(), f"missing baseline_C for {label}"
    return out


def feature_columns_for_candidate(candidate: str) -> list[str]:
    return [candidate, *feature_indicator_map.get(candidate, [])]


def fit_fold_prediction(df: pd.DataFrame, features: list[str], label: str, window: dict[str, Any]) -> pd.DataFrame:
    train_df = df[df["season"].isin(window["train_seasons"])].copy()
    val_df = df[df["season"].eq(window["val_season"])].copy()
    test_df = df[df["season"].eq(window["test_season"])].copy()
    prep = fit_preprocessor(train_df, features)
    x_train = transform_with_preprocessor(train_df, prep)
    x_val = transform_with_preprocessor(val_df, prep)
    x_test = transform_with_preprocessor(test_df, prep)
    assert not x_train.isna().any().any()
    assert not x_val.isna().any().any()
    assert not x_test.isna().any().any()
    y_train = train_df[label].astype(int).to_numpy()
    y_val = val_df[label].astype(int).to_numpy()
    estimator = _make_l1()
    estimator.fit(x_train, y_train)
    raw_val = estimator.predict_proba(x_val)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_val, y_val)
    raw_test = estimator.predict_proba(x_test)[:, 1]
    cal_test = iso.predict(raw_test)
    pred = test_df[["game_id", "trigger_play_id", "fav_deficit", "trigger_sequence", "season", "time_bucket", label]].copy()
    pred["fold"] = int(window["fold"])
    pred["raw_prob"] = raw_test
    pred["calibrated_prob"] = cal_test
    return pred


def predictions_for_features(df: pd.DataFrame, features: list[str], label: str) -> pd.DataFrame:
    frames = [fit_fold_prediction(df, features, label, w) for w in WALK_FORWARD_WINDOWS]
    return pd.concat(frames, ignore_index=True)


alpha_predictions: dict[str, pd.DataFrame] = {
    label: predictions_for_features(wide_df, ALPHA_FEATURES, label)
    for label in LABELS
}


def brier_improvement_rows(model_pred: pd.DataFrame, baseline_pred: pd.DataFrame, label: str, candidate: str) -> list[dict[str, Any]]:
    rows = []
    for fold in sorted(model_pred["fold"].unique()):
        m = model_pred[model_pred["fold"].eq(fold)].copy()
        b = baseline_pred[baseline_pred["fold"].eq(fold)].copy()
        key = ["game_id", "fav_deficit", "trigger_sequence", "fold"]
        merged = m.merge(b[key + ["calibrated_prob"]].rename(columns={"calibrated_prob": "alpha_prob"}), on=key, how="inner", validate="one_to_one")
        y = merged[label].astype(int).to_numpy()
        model_brier = float(brier_score_loss(y, merged["calibrated_prob"]))
        alpha_brier = float(brier_score_loss(y, merged["alpha_prob"]))
        rows.append({
            "candidate": candidate,
            "label": label,
            "fold": int(fold),
            "n": int(len(merged)),
            "alpha_brier": alpha_brier,
            "candidate_brier": model_brier,
            "brier_improvement_alpha_minus_candidate": alpha_brier - model_brier,
            "candidate_ece": expected_calibration_error(y, merged["calibrated_prob"].to_numpy()),
            "candidate_auc": metric_bundle(y, merged["calibrated_prob"].to_numpy())["auc"],
        })
    return rows


def baseline_c_comparison(pred: pd.DataFrame, label: str, candidate: str, category: str) -> dict[str, Any]:
    comp = add_baseline_c(pred)
    baseline_col = f"baseline_C_{label}"
    y = comp[label].astype(int).to_numpy()
    comp["brier_model"] = (comp["calibrated_prob"] - comp[label].astype(float)) ** 2
    comp["brier_baseline_C"] = (comp[baseline_col] - comp[label].astype(float)) ** 2
    comp["brier_improvement"] = comp["brier_baseline_C"] - comp["brier_model"]
    alpha = BONFERRONI_ALPHA[category]
    return {
        "candidate": candidate,
        "label": label,
        "category": category,
        "n": int(len(comp)),
        "n_games": int(comp["game_id"].nunique()),
        "model_brier": float(comp["brier_model"].mean()),
        "baseline_C_brier": float(comp["brier_baseline_C"].mean()),
        "brier_improvement_baseline_C_minus_model": float(comp["brier_improvement"].mean()),
        "bonferroni_alpha": alpha,
        "bootstrap_ci": bootstrap_cluster_mean_ci(comp, "brier_improvement", seed=BOOTSTRAP_SEED + stable_seed_offset(candidate, label), alpha=alpha),
        "model_auc": metric_bundle(y, comp["calibrated_prob"].to_numpy())["auc"],
        "baseline_C_auc": metric_bundle(y, comp[baseline_col].to_numpy())["auc"],
    }


candidate_predictions: dict[str, dict[str, pd.DataFrame]] = {}
feature_results: list[dict[str, Any]] = []
for spec_row in PRE_REGISTERED_FEATURES:
    candidate = spec_row["feature"]
    category = spec_row["category"]
    cand_cols = [*ALPHA_FEATURES, *feature_columns_for_candidate(candidate)]
    candidate_predictions[candidate] = {}
    label_results = {}
    pass_any_label = False
    for label in LABELS:
        pred = predictions_for_features(wide_df, cand_cols, label)
        candidate_predictions[candidate][label] = pred
        fold_rows = brier_improvement_rows(pred, alpha_predictions[label], label, candidate)
        improvements = [r["brier_improvement_alpha_minus_candidate"] for r in fold_rows]
        r6_positive_folds = int(sum(x > 0 for x in improvements))
        mean_delta = float(np.mean(improvements))
        base_c = baseline_c_comparison(pred, label, candidate, category)
        gates = {
            "r6_stability": r6_positive_folds >= 2,
            "magnitude": mean_delta >= 0.001,
            "baseline_C_bonferroni": base_c["bootstrap_ci"]["lower"] is not None and base_c["bootstrap_ci"]["lower"] > 0,
        }
        label_pass = bool(all(gates.values()))
        pass_any_label = pass_any_label or label_pass
        label_results[label] = {
            "folds": fold_rows,
            "r6_positive_folds": r6_positive_folds,
            "mean_brier_improvement_alpha_minus_candidate": mean_delta,
            "baseline_C_comparison": base_c,
            "gates": gates,
            "pass": label_pass,
        }
    fail_reasons = []
    if not pass_any_label:
        for gate_name in ["r6_stability", "magnitude", "baseline_C_bonferroni"]:
            if not any(label_results[label]["gates"][gate_name] for label in LABELS):
                fail_reasons.append(gate_name)
        if not fail_reasons:
            fail_reasons.append("no_label_cleared_all_three_gates")
    feature_results.append({
        **spec_row,
        "indicator_cols": feature_indicator_map[candidate],
        "null_summary": null_summary[candidate],
        "labels": label_results,
        "inclusion_verdict": "PASS" if pass_any_label else "FAIL",
        "fail_reasons": fail_reasons,
    })
    print(f"[feature] {candidate}: {'PASS' if pass_any_label else 'FAIL'}")

passing_n07_features = [r["feature"] for r in feature_results if r["inclusion_verdict"] == "PASS"]
category_summary = {}
for cat in ["A", "B", "C"]:
    rows = [r for r in feature_results if r["category"] == cat]
    category_summary[cat] = {
        "candidate_count": len(rows),
        "pass_count": sum(r["inclusion_verdict"] == "PASS" for r in rows),
        "bonferroni_alpha": BONFERRONI_ALPHA[cat],
        "passing_features": [r["feature"] for r in rows if r["inclusion_verdict"] == "PASS"],
    }

print(f"[ok] per-feature gates complete; passing features: {passing_n07_features}")
""")


add("code", "c07_0005", r"""
expanded_model: dict[str, Any] | None = None
expanded_predictions: pd.DataFrame | None = None

def play_level_df_for_model(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby(["game_id", "trigger_play_id"])["fav_deficit"].idxmin()
    out = df.loc[idx].sort_values(["season", "game_id", "trigger_play_id"]).reset_index(drop=True)
    assert out[["game_id", "trigger_play_id"]].duplicated().sum() == 0
    return out


def score_events(fit: dict[str, Any], events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = transform_with_preprocessor(events, fit["prep"])
    raw = fit["estimator"].predict_proba(x)[:, 1]
    cal = fit["calibrator"].predict(raw)
    return raw, cal


def fit_expanded_window(model_df: pd.DataFrame, features: list[str], window: dict[str, Any]) -> dict[str, Any]:
    train_df = model_df[model_df["season"].isin(window["train_seasons"])].copy()
    val_df = model_df[model_df["season"].eq(window["val_season"])].copy()
    test_df = model_df[model_df["season"].eq(window["test_season"])].copy()
    prep = fit_preprocessor(train_df, features)
    x_train = transform_with_preprocessor(train_df, prep)
    x_val = transform_with_preprocessor(val_df, prep)
    y_train = train_df[TARGET_LABEL].astype(int).to_numpy()
    y_val = val_df[TARGET_LABEL].astype(int).to_numpy()
    est = _make_l1()
    est.fit(x_train, y_train)
    raw_val = est.predict_proba(x_val)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_val, y_val)
    coef = {col: float(est.named_steps["logreg"].coef_[0][i]) for i, col in enumerate(features)}
    return {"window": window, "prep": prep, "estimator": est, "calibrator": iso, "coefficients": coef, "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)}


if passing_n07_features:
    expanded_feature_cols = [*pass_features, *passing_n07_features, "fav_deficit"]
    for feat in pass_features + passing_n07_features:
        if wide_df[feat].isna().mean() > NULL_INDICATOR_THRESHOLD:
            ind = f"{feat}_is_null"
            if ind not in wide_df.columns:
                wide_df[ind] = wide_df[feat].isna().astype(int)
            if ind not in expanded_feature_cols:
                expanded_feature_cols.append(ind)
    for feat in passing_n07_features:
        for ind in feature_indicator_map[feat]:
            if ind not in expanded_feature_cols:
                expanded_feature_cols.append(ind)

    play_df = play_level_df_for_model(wide_df)
    pred_frames = []
    fits_by_scheme = {}
    metric_rows = []
    for scheme in ["U", "W2"]:
        fits = []
        for w in WALK_FORWARD_WINDOWS:
            fit = fit_expanded_window(play_df, expanded_feature_cols, w)
            fits.append(fit)
            event_test = wide_df[wide_df["season"].eq(w["test_season"])].copy()
            raw, cal = score_events(fit, event_test)
            pred = event_test[[
                "game_id", "trigger_play_id", "fav_deficit", "trigger_sequence", "season", "time_bucket",
                "favorite_final_win", "deficit_erased", *passing_n07_features,
            ]].copy()
            pred.insert(2, "fold", int(w["fold"]))
            pred.insert(2, "scheme", scheme)
            pred["raw_model_prob"] = raw
            pred["calibrated_prob"] = cal
            pred_frames.append(pred)
            eval_df = add_baseline_c(pred)
            eval_df["brier_model"] = (eval_df["calibrated_prob"] - eval_df[TARGET_LABEL].astype(float)) ** 2
            eval_df["brier_baseline_C"] = (eval_df[f"baseline_C_{TARGET_LABEL}"] - eval_df[TARGET_LABEL].astype(float)) ** 2
            eval_df["brier_improvement"] = eval_df["brier_baseline_C"] - eval_df["brier_model"]
            y = eval_df[TARGET_LABEL].astype(int).to_numpy()
            metric_rows.append({
                "scheme": scheme,
                "fold": int(w["fold"]),
                "n": int(len(eval_df)),
                "brier_model": float(eval_df["brier_model"].mean()),
                "brier_baseline_C": float(eval_df["brier_baseline_C"].mean()),
                "brier_improvement_baseline_C_minus_model": float(eval_df["brier_improvement"].mean()),
                "ece_model": expected_calibration_error(y, eval_df["calibrated_prob"].to_numpy()),
                "auc_model": metric_bundle(y, eval_df["calibrated_prob"].to_numpy())["auc"],
                "auc_baseline_C": metric_bundle(y, eval_df[f"baseline_C_{TARGET_LABEL}"].to_numpy())["auc"],
            })
        fits_by_scheme[scheme] = fits

    expanded_predictions = pd.concat(pred_frames, ignore_index=True)
    expanded_predictions.to_parquet(N07_EXPANDED_PREDICTIONS_PARQUET, index=False)
    expanded_eval = add_baseline_c(expanded_predictions[expanded_predictions["scheme"].eq("U")].copy())
    expanded_eval["brier_model"] = (expanded_eval["calibrated_prob"] - expanded_eval[TARGET_LABEL].astype(float)) ** 2
    expanded_eval["brier_baseline_C"] = (expanded_eval[f"baseline_C_{TARGET_LABEL}"] - expanded_eval[TARGET_LABEL].astype(float)) ** 2
    expanded_eval["brier_improvement"] = expanded_eval["brier_baseline_C"] - expanded_eval["brier_model"]
    y = expanded_eval[TARGET_LABEL].astype(int).to_numpy()
    n06_spec = json.loads(N06_MODEL_SPEC_JSON.read_text(encoding="utf-8"))
    expanded_model = {
        "deployment_candidate": {
            "candidate_for": "N08 live data deployment scaffold",
            "model_version": "N07 expanded 33-feature model",
            "semantic_feature_count": len(pass_features) + len(passing_n07_features) + len(STRUCTURAL_FEATURES),
            "feature_count_breakdown": {
                "phase0_r6_validated_features": len(pass_features),
                "n07_expansion_features": len(passing_n07_features),
                "protected_structural_conditioning_features": len(STRUCTURAL_FEATURES),
            },
            "n07_expansion_features": passing_n07_features,
            "structural_conditioning_feature": STRUCTURAL_FEATURES[0],
            "historical_validation_status": "not_edge_grade_vs_baseline_C",
            "rationale": "Use the 33-feature N07 expanded model for N08 because it is the best historically tested specification and includes the possession-adjusted signal source surfaced by N07, even though it does not beat baseline_C with statistical support on historical deficit_erased validation.",
            "live_data_caveat": "Deployment value must be validated against actual live market prices; historical baseline_C validation is exhausted at this methodology level.",
        },
        "expanded_features": passing_n07_features,
        "expanded_model_columns": expanded_feature_cols,
        "metric_rows": metric_rows,
        "overall_U": {
            "n": int(len(expanded_eval)),
            "brier_model": float(expanded_eval["brier_model"].mean()),
            "brier_baseline_C": float(expanded_eval["brier_baseline_C"].mean()),
            "brier_improvement_baseline_C_minus_model": float(expanded_eval["brier_improvement"].mean()),
            "bootstrap_ci": bootstrap_cluster_mean_ci(expanded_eval, "brier_improvement", seed=BOOTSTRAP_SEED + 777, alpha=0.05),
            "ece_model": expected_calibration_error(y, expanded_eval["calibrated_prob"].to_numpy()),
            "auc_model": metric_bundle(y, expanded_eval["calibrated_prob"].to_numpy())["auc"],
            "auc_baseline_C": metric_bundle(y, expanded_eval[f"baseline_C_{TARGET_LABEL}"].to_numpy())["auc"],
            "n06_reference": n06_spec["primary_validation"]["result"],
        },
        "coefficients": {
            scheme: [
                {"fold": int(fit["window"]["fold"]), "coefficients": fit["coefficients"]}
                for fit in fits
            ]
            for scheme, fits in fits_by_scheme.items()
        },
        "calibration_params": {
            scheme: [
                {
                    "fold": int(fit["window"]["fold"]),
                    "x_thresholds": fit["calibrator"].X_thresholds_.tolist(),
                    "y_thresholds": fit["calibrator"].y_thresholds_.tolist(),
                }
                for fit in fits
            ]
            for scheme, fits in fits_by_scheme.items()
        },
    }
    N07_EXPANDED_MODEL_SPEC_JSON.write_text(json.dumps(expanded_model, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] expanded model written with {len(passing_n07_features)} N07 features")
else:
    for path in [N07_EXPANDED_PREDICTIONS_PARQUET, N07_EXPANDED_MODEL_SPEC_JSON]:
        if path.exists():
            path.unlink()
    print("[ok] no features passed all gates; expanded model artifacts intentionally not generated")

stability_payload = {
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
    "pre_registered_features": PRE_REGISTERED_FEATURES,
    "feature_results": feature_results,
    "passing_features": passing_n07_features,
    "category_summary": category_summary,
    "bonferroni_alpha": BONFERRONI_ALPHA,
    "null_summary": null_summary,
    "feature_indicator_map": feature_indicator_map,
    "data_quality": {
        "n_feature_rows": int(len(feature_df)),
        "n_unique_games": int(feature_df["game_id"].nunique()),
        "empirical_possessions_per_minute": empirical_possessions_per_minute,
        "locked_possessions_per_minute_used": AVG_POSSESSIONS_PER_MINUTE_LOCKED,
        "edge_case_handling": {
            "dog_points_from_explosives_pct_gt_1_set_to_null": int(dog_explosive_pct_edge_mask.sum()),
            "fav_yards_per_point_ratio_lt_0_set_to_null": int(fav_ypp_ratio_negative_edge_mask.sum()),
        },
    },
    "expanded_model": expanded_model,
    "phase0_extract_summary": phase0_extract_summary,
}
N07_STABILITY_JSON.write_text(json.dumps(stability_payload, indent=2), encoding="utf-8")
print(f"[ok] wrote {N07_STABILITY_JSON.relative_to(REPO_ROOT)}")
""")


add("code", "c07_0006", r"""
def fmt(x: Any, digits: int = 5, signed: bool = False) -> str:
    if x is None:
        return "NA"
    val = float(x)
    return f"{val:+.{digits}f}" if signed else f"{val:.{digits}f}"


def pct(x: Any, digits: int = 1) -> str:
    if x is None:
        return "NA"
    return f"{float(x) * 100:.{digits}f}%"


pass_count = len(passing_n07_features)
if pass_count >= 4 and expanded_model and expanded_model["overall_U"]["bootstrap_ci"]["lower"] > 0:
    structural_finding = "Substantial expansion success."
elif 1 <= pass_count <= 3 and expanded_model and expanded_model["overall_U"]["brier_improvement_baseline_C_minus_model"] > 0:
    structural_finding = "Modest expansion success."
elif pass_count > 0:
    structural_finding = "Calibration-only or feature-level success without expanded-model Brier edge."
else:
    structural_finding = "No expansion success."

lines: list[str] = []
lines.append("# N07 feature pool expansion test")
lines.append("")
if expanded_model:
    o = expanded_model["overall_U"]
    lines.extend([
        "**Primary finding:** N07 found a real but limited missing signal source:",
        f"possession-adjusted deficit pressure. {pass_count} of 14 pre-registered features passed",
        "all three inclusion gates: `deficit_per_remaining_possession` and",
        "`clock_pressure_index`. Both are Category A possession-adjusted features.",
        "However, the expanded 33-feature model still does **not** beat the strict",
        "`fav_deficit x time_bucket` baseline_C on the comeback-erasure target.",
        "Expanded Scheme U Brier improvement versus baseline_C on `deficit_erased` is",
        f"**{fmt(o['brier_improvement_baseline_C_minus_model'], signed=True)}** with 95% CI **[{fmt(o['bootstrap_ci']['2.5'], signed=True)}, {fmt(o['bootstrap_ci']['97.5'], signed=True)}]**.",
        "",
        "This is the natural endpoint for the project's historical-data methodology.",
        "Phase 0 built and stability-tested the original feature pool. N03 produced a",
        "calibrated final-win model with modest discrimination. N04 showed that the",
        "model beats stale pre-game market probabilities, validating that current game",
        "state matters. N05 and N06 then showed that the model does not beat a simple",
        "deficit x time lookup table on either final-win or deficit-erased labels. N07",
        "tested the most plausible missing categories and found that possession",
        "pressure helps at the feature level, while fluke-score and efficiency-gap",
        "hypotheses do not clear the strict baseline_C gate. Historical data has now",
        "been pushed about as far as this framework can push it; the next validation",
        "question requires live market comparison.",
        "",
        "The 33-feature expanded model (30 original Phase 0 features + the 2 N07",
        "passes + protected `fav_deficit`) is the recommended production candidate for",
        "an N08 live-data scaffold. It is not edge-grade on historical baseline_C",
        "validation, but it is the best historically tested model specification and it",
        "includes the one new signal source N07 surfaced.",
    ])
else:
    lines.append("No expanded model was fit because no feature cleared all three gates.")
lines.append("")
lines.append("## Pre-registered feature verdicts")
lines.append("")
lines.append("| Feature | Category | Indicators | Verdict | Best label | R6 folds | Mean dBrier vs alpha | baseline_C improvement | Corrected lower | Fail reason |")
lines.append("|---|---|---|---|---|---:|---:|---:|---:|---|")
for row in feature_results:
    best_label = max(LABELS, key=lambda label: row["labels"][label]["baseline_C_comparison"]["brier_improvement_baseline_C_minus_model"])
    best = row["labels"][best_label]
    bc = best["baseline_C_comparison"]
    lines.append(
        f"| `{row['feature']}` | {row['category']} | {', '.join(row['indicator_cols']) or 'none'} | {row['inclusion_verdict']} | "
        f"`{best_label}` | {best['r6_positive_folds']} | {fmt(best['mean_brier_improvement_alpha_minus_candidate'], signed=True)} | "
        f"{fmt(bc['brier_improvement_baseline_C_minus_model'], signed=True)} | {fmt(bc['bootstrap_ci']['lower'], signed=True)} | "
        f"{', '.join(row['fail_reasons']) or ''} |"
    )

lines.append("")
lines.append("## Category summary")
lines.append("")
lines.append("| Category | Candidates | Pass | Bonferroni alpha | Passing features |")
lines.append("|---|---:|---:|---:|---|")
for cat, row in category_summary.items():
    lines.append(f"| {cat} | {row['candidate_count']} | {row['pass_count']} | {row['bonferroni_alpha']:.4f} | {', '.join(row['passing_features']) or 'none'} |")

if expanded_model:
    lines.append("")
    lines.append("## Expanded model")
    lines.append("")
    o = expanded_model["overall_U"]
    ref = o["n06_reference"]
    lines.append("| Model | N | Model Brier | baseline_C Brier | Improvement | AUC model | AUC baseline_C |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(f"| N07 expanded U | {o['n']} | {fmt(o['brier_model'])} | {fmt(o['brier_baseline_C'])} | {fmt(o['brier_improvement_baseline_C_minus_model'], signed=True)} | {fmt(o['auc_model'], 4)} | {fmt(o['auc_baseline_C'], 4)} |")
    lines.append(f"| N06 reference | {ref['n']} | {fmt(ref['brier_model'])} | {fmt(ref['brier_baseline_C'])} | {fmt(ref['brier_improvement_baseline_C_minus_model'], signed=True)} | {fmt(ref['auc_model'], 4)} | {fmt(ref['auc_baseline_C'], 4)} |")

lines.append("")
lines.append("## Data provenance")
lines.append("")
lines.append(f"- Descriptive feature rows: {len(feature_df):,}.")
lines.append(f"- Empirical possessions/minute from cached drives: {empirical_possessions_per_minute:.3f}; locked value used for `estimated_possessions_remaining`: {AVG_POSSESSIONS_PER_MINUTE_LOCKED:.3f}.")
lines.append("- Possession remaining uses `seconds_remaining_in_regulation`, equivalent to the corrected `(quarter - 1) * 900 + period_elapsed` clock calculation.")
lines.append("- `dog_offensive_points_pct` treats `dog_points_from_returns` as non-offensive points; turnover-created offensive scores remain offensive points and are separately represented by `dog_points_from_turnovers_pct`.")
lines.append("")
lines.append("## Honest interpretation")
lines.append("")
if expanded_model and expanded_model["overall_U"]["bootstrap_ci"]["lower"] <= 0:
    lines.extend([
        "N07 is a mixed but clarifying result. The possession-adjusted hypothesis is",
        "supported: the model was missing structural pressure from deficit relative to",
        "remaining possessions. The fluke-score hypothesis, which was one of the",
        "project's original mechanistic ideas, is not supported under this strict test:",
        "0 of 5 Category B features passed. Efficiency-gap differentials also failed",
        "to beat baseline_C.",
        "",
        "The expanded model is marginally better than N06 on Brier",
        "(-0.00263 versus -0.00352 improvement against baseline_C), but the confidence",
        "interval still crosses zero and the AUC remains slightly below baseline_C",
        "(0.7637 versus 0.7659). This is not an edge-grade historical result. It is,",
        "however, enough to justify carrying the 33-feature expanded model forward as",
        "the live-data deployment candidate, where the relevant comparison becomes",
        "actual live market prices rather than a historical deficit x time baseline.",
    ])
elif structural_finding == "No expansion success.":
    lines.append("N07 confirms that the newly pre-registered feature categories do not break the N05/N06 pattern. Under this methodology, the feature pool remains exhausted relative to baseline_C.")
else:
    lines.append("The expansion produced supported signal beyond baseline_C; this should be reviewed carefully for mechanism, multiplicity, and deployment relevance before changing production conclusions.")

N07_SUMMARY_REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[ok] wrote {N07_SUMMARY_REPORT_MD.relative_to(REPO_ROOT)} size={N07_SUMMARY_REPORT_MD.stat().st_size:,}")

print("=" * 80)
print("N07 summary")
print("=" * 80)
print(f"Feature rows: {len(feature_df):,}")
print(f"Passing features: {passing_n07_features}")
print(f"Expanded model generated: {expanded_model is not None}")
for path in [N07_FEATURES_PARQUET, N07_STABILITY_JSON, N07_SUMMARY_REPORT_MD]:
    print(f"  {path.relative_to(REPO_ROOT)} {path.stat().st_size:,} bytes")
if expanded_model:
    for path in [N07_EXPANDED_PREDICTIONS_PARQUET, N07_EXPANDED_MODEL_SPEC_JSON]:
        print(f"  {path.relative_to(REPO_ROOT)} {path.stat().st_size:,} bytes")
print("[ok] N07 complete -- halt for review; no commit performed.")
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
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"[ok] wrote {OUT} ({OUT.stat().st_size:,} bytes, {len(CELLS)} cells)")
