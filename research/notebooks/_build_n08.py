"""
Deterministic builder for research/notebooks/08_stern_winston_conformal_diagnostic.ipynb.

N08 is diagnostic only. It compares the existing N06 and N07-expanded
deficit_erased models against Stern-Winston analytical baselines and adds
split-conformal intervals around the locked N06 point predictions.
"""

from __future__ import annotations

import json
import pathlib
import textwrap

OUT = pathlib.Path(__file__).resolve().parent / "08_stern_winston_conformal_diagnostic.ipynb"

CELLS: list[tuple[str, str, str]] = []


def add(cell_type: str, cell_id: str, src: str) -> None:
    CELLS.append((cell_type, cell_id, textwrap.dedent(src).lstrip("\n")))


add("markdown", "m08_0000", """
# Notebook 08 -- Stern-Winston baseline and conformal diagnostic

N08 is a diagnostic notebook. It does not introduce a new production model,
feature pool, or hyperparameter search.

Locked interpretation after the design halt:

- **M1:** N06 calibrated probability, trained on `deficit_erased`.
- **M2:** N07 expanded calibrated probability, also trained on
  `deficit_erased`, with the N07 possession-adjusted feature expansion.
- **M3:** N06 point probability plus split-conformal prediction intervals.

The trained-label evaluation is `deficit_erased`. `favorite_final_win` is a
cross-label diagnostic. N08 re-materializes the exact locked N06/N07 fold fits
only to recover validation-slice predictions needed for conformal calibration;
it verifies rebuilt held-out predictions against the committed artifacts before
using them.
""")


add("code", "c08_0001", r"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import pathlib
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
N06_PREDICTIONS_PARQUET = RESULTS_DIR / "n06_calibrated_predictions.parquet"
N06_MODEL_SPEC_JSON = RESULTS_DIR / "n06_model_spec.json"
N07_FEATURES_PARQUET = RESULTS_DIR / "n07_descriptive_features.parquet"
N07_PREDICTIONS_PARQUET = RESULTS_DIR / "n07_expanded_model_predictions.parquet"
N07_MODEL_SPEC_JSON = RESULTS_DIR / "n07_expanded_model_spec.json"

N08_DIAGNOSTIC_PREDICTIONS_PARQUET = RESULTS_DIR / "n08_diagnostic_predictions.parquet"
N08_COMPARISON_RESULTS_JSON = RESULTS_DIR / "n08_comparison_results.json"
N08_SUMMARY_REPORT_MD = RESULTS_DIR / "n08_summary_report.md"
N08_PRICE_CONVERSION_SPEC_JSON = RESULTS_DIR / "n08_price_conversion_spec.json"

assert RESEARCH_DIR.name == "research", f"Expected research/notebooks cwd, got {NOTEBOOK_DIR}"
for path in [
    TRIGGER_EVENTS_CSV,
    TRIGGER_OUTCOMES_CSV,
    FEATURE_VALIDATION_CSV,
    N05_DESCRIPTIVE_RATES_PARQUET,
    N06_PREDICTIONS_PARQUET,
    N06_MODEL_SPEC_JSON,
    N07_FEATURES_PARQUET,
    N07_PREDICTIONS_PARQUET,
    N07_MODEL_SPEC_JSON,
]:
    assert path.exists(), f"Missing required N08 input artifact: {path}"

RANDOM_STATE = 42
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 42
ALPHA = 0.05
PFR_STD = 13.45
TOTAL_GAME_SECONDS = 3600.0
TARGET_LABEL = "deficit_erased"
CROSS_LABEL = "favorite_final_win"
LABELS = [TARGET_LABEL, CROSS_LABEL]
MODEL_LABELS = {
    "M1_N06": "N06 calibrated probability",
    "M2_N07_EXP": "N07 expanded calibrated probability",
    "M3_N06_CONFORMAL": "N06 + conformal interval",
}
BASELINE_LABELS = {
    "B_C": "deficit x time-bucket baseline_C",
    "B_SW_PFR": "Stern-Winston PFR std=13.45",
    "B_SW_CFB": "Stern-Winston empirical CFB std",
}
WALK_FORWARD_WINDOWS = [
    {"fold": 2022, "train_seasons": list(range(2015, 2021)), "val_season": 2021, "test_season": 2022, "train_window_label": "2015-2020"},
    {"fold": 2023, "train_seasons": list(range(2015, 2022)), "val_season": 2022, "test_season": 2023, "train_window_label": "2015-2021"},
    {"fold": 2024, "train_seasons": list(range(2015, 2023)), "val_season": 2023, "test_season": 2024, "train_window_label": "2015-2022"},
]
FOLD_WEIGHTS = {2022: 0.25, 2023: 0.25, 2024: 0.50}

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
KEY_COLS = ["game_id", "fav_deficit", "trigger_sequence"]
PLAY_KEY_COLS = ["game_id", "trigger_play_id"]
N07_PASS_FEATURES = ["deficit_per_remaining_possession", "clock_pressure_index"]

print(f"[ok] N08 setup at {NOTEBOOK_DIR}")
""")


add("code", "c08_0002", r"""
def _params_hash(params: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def _cache_key(endpoint: str, params: dict[str, Any]) -> pathlib.Path:
    endpoint_key = endpoint.strip("/").replace("/", "_")
    return CACHE_DIR / f"cfbd__{endpoint_key}__{_params_hash(params)}.json"


def readonly_cfbd_get(endpoint: str, force_refresh: bool = False, **params: Any) -> Any:
    if force_refresh:
        raise AssertionError("N08 forbids force_refresh; cache-only extraction is required")
    key = _cache_key(endpoint, params)
    if not key.exists():
        raise AssertionError(f"N08 missing local cache for {endpoint} {params}")
    return json.loads(key.read_text(encoding="utf-8"))


def _base_phase0_namespace() -> dict[str, Any]:
    return {
        "__name__": "_n08_phase0_cell_exec",
        "Any": Any,
        "contextlib": contextlib,
        "hashlib": hashlib,
        "io": io,
        "json": json,
        "math": math,
        "np": np,
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


def run_phase0_matrix_notebook(nb_name: str, cell_indexes: list[int]) -> pd.DataFrame:
    nb_path = NOTEBOOK_DIR / nb_name
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    ns = _base_phase0_namespace()
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        for idx in cell_indexes:
            src = "".join(nb["cells"][idx]["source"])
            exec(compile(src, f"{nb_name}:cell{idx}", "exec"), ns)
            ns["cfbd_get"] = readonly_cfbd_get
    fm = ns.get("feature_matrix_df")
    if fm is None:
        raise AssertionError(f"{nb_name} did not build feature_matrix_df")
    return fm.copy()


def load_pass_features() -> list[str]:
    fv = pd.read_csv(FEATURE_VALIDATION_CSV, keep_default_na=False)
    features: list[str] = []
    for feat in fv["feature"].tolist():
        if feat in features:
            continue
        sub = fv[fv["feature"].eq(feat)]
        if sub["passed_stability"].astype(str).str.lower().eq("true").all():
            features.append(str(feat))
    assert len(features) == 30, f"Expected 30 R6-pass features, got {len(features)}"
    return features


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi) if i == n_bins - 1 else (y_prob >= lo) & (y_prob < hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(y_prob[mask].mean()) - float(y_true[mask].mean()))
    return float(ece)


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(np.asarray(y_true).astype(int), np.asarray(y_prob).astype(float)))
    except ValueError:
        return float("nan")


def metric_bundle(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    return {
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "auc": safe_auc(y_true, y_prob),
    }


def bootstrap_cluster_mean_ci(
    df: pd.DataFrame,
    value_col: str,
    *,
    cluster_col: str = "game_id",
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    clusters = np.array(sorted(df[cluster_col].dropna().unique()))
    grouped = {gid: df.loc[df[cluster_col].eq(gid), value_col].to_numpy(dtype=float) for gid in clusters}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        values = np.concatenate([grouped[gid] for gid in sampled])
        draws[i] = float(np.mean(values))
    return {
        "n_resamples": int(n_resamples),
        "n_clusters": int(len(clusters)),
        "mean": float(df[value_col].mean()),
        "lower": float(np.quantile(draws, alpha / 2.0)),
        "p25": float(np.quantile(draws, 0.25)),
        "median": float(np.quantile(draws, 0.50)),
        "p75": float(np.quantile(draws, 0.75)),
        "upper": float(np.quantile(draws, 1.0 - alpha / 2.0)),
    }


def _make_l1() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=1.0,
            random_state=RANDOM_STATE,
            max_iter=1000,
        )),
    ])


def fit_simple_preprocessor(train_df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    medians: dict[str, float] = {}
    for col in columns:
        med = train_df[col].median(skipna=True)
        medians[col] = 0.0 if pd.isna(med) else float(med)
    return {"columns": list(columns), "medians": medians}


def transform_simple(df: pd.DataFrame, prep: dict[str, Any]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in prep["columns"]:
        out[col] = df[col].astype(float).fillna(prep["medians"][col])
    return out[prep["columns"]]


def normal_cdf(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=float)
    out = 0.5 * (1.0 + np.vectorize(math.erf)(arr / math.sqrt(2.0)))
    return float(out) if np.ndim(x) == 0 else out


def stern_winston_prob(current_margin: np.ndarray, remaining_seconds: np.ndarray, std: float) -> np.ndarray:
    current_margin = np.asarray(current_margin, dtype=float)
    remaining_seconds = np.asarray(remaining_seconds, dtype=float)
    frac_remaining = np.clip(remaining_seconds / TOTAL_GAME_SECONDS, 0.0, 1.0)
    sd = float(std) * np.sqrt(frac_remaining)
    probs = np.empty(len(current_margin), dtype=float)
    tiny = sd <= 1e-12
    probs[tiny] = (current_margin[tiny] >= 0.5).astype(float)
    z = (0.5 - current_margin[~tiny]) / sd[~tiny]
    probs[~tiny] = 1.0 - np.asarray(normal_cdf(z), dtype=float)
    return np.clip(probs, 0.0, 1.0)


print("[ok] helper functions defined")
""")


add("code", "c08_0003", r"""
t0 = time.perf_counter()
pass_features = load_pass_features()
n06_spec = json.loads(N06_MODEL_SPEC_JSON.read_text(encoding="utf-8"))
n07_spec = json.loads(N07_MODEL_SPEC_JSON.read_text(encoding="utf-8"))
n06_model_columns = list(n06_spec["feature_list"]["U"]["selected_model_columns"])
n06_core_features = list(n06_spec["feature_list"]["U"]["selected_core_features"])
n07_model_columns = list(n07_spec["expanded_model_columns"])

assert len(n06_core_features) == 31, f"N06 expected 31 core features, got {len(n06_core_features)}"
assert len(n06_model_columns) == 50, f"N06 expected 50 model columns, got {len(n06_model_columns)}"
assert len(n07_model_columns) == 52, f"N07 expected 52 expanded model columns, got {len(n07_model_columns)}"
assert n07_spec["deployment_candidate"]["model_version"] == "N07 expanded 33-feature model"

n05_df = pd.read_parquet(N05_DESCRIPTIVE_RATES_PARQUET)
trigger_df = pd.read_csv(TRIGGER_EVENTS_CSV)
outcomes_df = pd.read_csv(TRIGGER_OUTCOMES_CSV)
n07_feature_df = pd.read_parquet(N07_FEATURES_PARQUET)

base_cols = [
    "game_id",
    "trigger_play_id",
    "fav_deficit",
    "trigger_sequence",
    "quarter",
    "clock_seconds_in_period_total",
    "time_bucket",
    "fav_team",
    "dog_team",
    "season",
    "favorite_final_win",
    "deficit_erased",
]
base_df = n05_df[n05_df[TARGET_LABEL].notna()][base_cols].copy()
base_df[TARGET_LABEL] = base_df[TARGET_LABEL].astype(bool).astype(int)
base_df[CROSS_LABEL] = base_df[CROSS_LABEL].astype(bool).astype(int)
assert len(base_df) == 11412, f"Expected 11,412 non-null N05 rows, got {len(base_df)}"

trigger_cols = KEY_COLS + [
    "fav_score_at_trigger",
    "dog_score_at_trigger",
    "seconds_remaining_in_regulation",
    "pregame_spread",
]
base_df = base_df.merge(
    trigger_df[trigger_cols],
    on=KEY_COLS,
    how="inner",
    validate="one_to_one",
)
assert len(base_df) == 11412, f"Expected 11,412 after trigger merge, got {len(base_df)}"

n07_merge_cols = KEY_COLS + N07_PASS_FEATURES
base_df = base_df.merge(
    n07_feature_df[n07_merge_cols],
    on=KEY_COLS,
    how="inner",
    validate="one_to_one",
)
assert len(base_df) == 11412, f"Expected 11,412 after N07 feature merge, got {len(base_df)}"

wide_df = base_df.copy()
indicator_sources = set(PHASE0_INDICATOR_MAP.values())
for nb_name, cell_indexes in PHASE0_NOTEBOOK_CELLS.items():
    fm = run_phase0_matrix_notebook(nb_name, cell_indexes)
    merge_cols = KEY_COLS + [
        c for c in list(pass_features) + list(indicator_sources)
        if c in fm.columns and c not in KEY_COLS
    ]
    merge_cols = list(dict.fromkeys(merge_cols))
    if len(merge_cols) <= len(KEY_COLS):
        continue
    before = set(wide_df.columns)
    wide_df = wide_df.merge(fm[merge_cols], on=KEY_COLS, how="left", validate="one_to_one")
    print(f"[ok] merged {nb_name}: +{len(set(wide_df.columns) - before)} columns")

missing_pass = [c for c in pass_features if c not in wide_df.columns]
assert not missing_pass, f"missing Phase 0 pass features: {missing_pass}"

for core, indicator in PHASE0_INDICATOR_MAP.items():
    if core in wide_df.columns:
        if indicator not in wide_df.columns:
            wide_df[indicator] = wide_df[core].isna().astype(int)
        else:
            wide_df[indicator] = wide_df[indicator].fillna(0).astype(int)

for col in n06_model_columns:
    if col.endswith("_is_null") and col not in wide_df.columns:
        core = col.removesuffix("_is_null")
        if core in wide_df.columns:
            wide_df[col] = wide_df[core].isna().astype(int)
for col in n07_model_columns:
    if col.endswith("_is_null") and col not in wide_df.columns:
        core = col.removesuffix("_is_null")
        if core in wide_df.columns:
            wide_df[col] = wide_df[core].isna().astype(int)

missing_n06_cols = [c for c in n06_model_columns if c not in wide_df.columns]
missing_n07_cols = [c for c in n07_model_columns if c not in wide_df.columns]
assert not missing_n06_cols, f"missing N06 model columns after reconstruction: {missing_n06_cols}"
assert not missing_n07_cols, f"missing N07 model columns after reconstruction: {missing_n07_cols}"

event_df = wide_df.sort_values(["season", "game_id", "trigger_play_id", "fav_deficit", "trigger_sequence"]).reset_index(drop=True)
assert event_df[KEY_COLS].duplicated().sum() == 0, "duplicate trigger event keys"

canonical_idx = event_df.groupby(PLAY_KEY_COLS)["fav_deficit"].idxmin()
play_df = event_df.loc[canonical_idx].sort_values(["season", "game_id", "trigger_play_id"]).reset_index(drop=True)
assert len(play_df) == 7852, f"Expected 7,852 unique trigger plays, got {len(play_df)}"
assert play_df[PLAY_KEY_COLS].duplicated().sum() == 0, "duplicate play-level rows"

print(f"[ok] reconstructed event matrix: {len(event_df):,} rows x {event_df.shape[1]:,} columns")
print(f"[ok] reconstructed play matrix: {len(play_df):,} rows x {play_df.shape[1]:,} columns")
print(f"[ok] reconstruction elapsed: {time.perf_counter() - t0:.1f}s")
""")


add("code", "c08_0004", r"""
@dataclass
class FoldFit:
    fold: int
    train_seasons: list[int]
    val_season: int
    test_season: int
    model_columns: list[str]
    prep: dict[str, Any]
    estimator: Pipeline
    calibrator: IsotonicRegression
    val_predictions: pd.DataFrame
    test_predictions: pd.DataFrame


def fit_fold(model_name: str, model_columns: list[str], window: dict[str, Any]) -> FoldFit:
    train_df = play_df[play_df["season"].isin(window["train_seasons"])].copy()
    val_df = play_df[play_df["season"].eq(window["val_season"])].copy()
    test_events = event_df[event_df["season"].eq(window["test_season"])].copy()

    prep = fit_simple_preprocessor(train_df, model_columns)
    x_train = transform_simple(train_df, prep)
    x_val = transform_simple(val_df, prep)
    x_test = transform_simple(test_events, prep)
    assert not x_train.isna().any().any()
    assert not x_val.isna().any().any()
    assert not x_test.isna().any().any()

    y_train = train_df[TARGET_LABEL].astype(int).to_numpy()
    y_val = val_df[TARGET_LABEL].astype(int).to_numpy()
    estimator = _make_l1()
    estimator.fit(x_train, y_train)
    raw_val = estimator.predict_proba(x_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_val, y_val)
    cal_val = calibrator.predict(raw_val)

    raw_test = estimator.predict_proba(x_test)[:, 1]
    cal_test = calibrator.predict(raw_test)

    val_pred = val_df[[
        "game_id", "trigger_play_id", "fav_deficit", "trigger_sequence",
        "season", "time_bucket", TARGET_LABEL, CROSS_LABEL,
    ]].copy()
    val_pred["fold"] = int(window["fold"])
    val_pred["split_role"] = "validation"
    val_pred["model"] = model_name
    val_pred["raw_model_prob"] = raw_val
    val_pred["calibrated_prob"] = cal_val

    test_pred = test_events[[
        "game_id", "trigger_play_id", "fav_deficit", "trigger_sequence",
        "season", "time_bucket", TARGET_LABEL, CROSS_LABEL,
    ]].copy()
    test_pred["fold"] = int(window["fold"])
    test_pred["split_role"] = "test"
    test_pred["model"] = model_name
    test_pred["raw_model_prob"] = raw_test
    test_pred["calibrated_prob"] = cal_test

    return FoldFit(
        fold=int(window["fold"]),
        train_seasons=list(window["train_seasons"]),
        val_season=int(window["val_season"]),
        test_season=int(window["test_season"]),
        model_columns=list(model_columns),
        prep=prep,
        estimator=estimator,
        calibrator=calibrator,
        val_predictions=val_pred,
        test_predictions=test_pred,
    )


def fit_all_folds(model_name: str, model_columns: list[str]) -> list[FoldFit]:
    return [fit_fold(model_name, model_columns, window) for window in WALK_FORWARD_WINDOWS]


n06_fits = fit_all_folds("M1_N06", n06_model_columns)
n07_fits = fit_all_folds("M2_N07_EXP", n07_model_columns)
n06_val_predictions = pd.concat([fit.val_predictions for fit in n06_fits], ignore_index=True)
n07_val_predictions = pd.concat([fit.val_predictions for fit in n07_fits], ignore_index=True)
n06_rebuilt_test = pd.concat([fit.test_predictions for fit in n06_fits], ignore_index=True)
n07_rebuilt_test = pd.concat([fit.test_predictions for fit in n07_fits], ignore_index=True)

n06_committed = pd.read_parquet(N06_PREDICTIONS_PARQUET)
n06_committed = n06_committed[n06_committed["scheme"].eq("U")].copy()
n07_committed = pd.read_parquet(N07_PREDICTIONS_PARQUET)
n07_committed = n07_committed[n07_committed["scheme"].eq("U")].copy()

cmp_key = ["game_id", "fav_deficit", "trigger_sequence", "fold"]
n06_cmp = n06_rebuilt_test.merge(
    n06_committed[cmp_key + ["calibrated_prob"]].rename(columns={"calibrated_prob": "committed_prob"}),
    on=cmp_key,
    how="inner",
    validate="one_to_one",
)
n07_cmp = n07_rebuilt_test.merge(
    n07_committed[cmp_key + ["calibrated_prob"]].rename(columns={"calibrated_prob": "committed_prob"}),
    on=cmp_key,
    how="inner",
    validate="one_to_one",
)
assert len(n06_cmp) == len(n06_committed) == 3854, f"N06 comparison row mismatch: {len(n06_cmp)}"
assert len(n07_cmp) == len(n07_committed) == 3854, f"N07 comparison row mismatch: {len(n07_cmp)}"
n06_max_abs_diff = float((n06_cmp["calibrated_prob"] - n06_cmp["committed_prob"]).abs().max())
n07_max_abs_diff = float((n07_cmp["calibrated_prob"] - n07_cmp["committed_prob"]).abs().max())
assert n06_max_abs_diff <= 1e-10, f"rebuilt N06 held-out probabilities differ from committed artifact: {n06_max_abs_diff}"
assert n07_max_abs_diff <= 1e-10, f"rebuilt N07 held-out probabilities differ from committed artifact: {n07_max_abs_diff}"

print(f"[ok] rebuilt N06 fits match committed held-out probabilities; max abs diff={n06_max_abs_diff:.3g}")
print(f"[ok] rebuilt N07 fits match committed held-out probabilities; max abs diff={n07_max_abs_diff:.3g}")
print(f"[ok] validation predictions materialized: N06={len(n06_val_predictions):,}, N07={len(n07_val_predictions):,}")
""")


add("code", "c08_0005", r"""
def baseline_c_tables(label: str) -> pd.DataFrame:
    train = event_df[(event_df["season"].between(2015, 2021)) & event_df[label].notna()].copy()
    train[label] = train[label].astype(int)
    tbl = (
        train.groupby(["fav_deficit", "time_bucket"])[label]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .rename(columns={"mean": f"baseline_C_{label}", "sum": "successes", "count": "n"})
    )
    return tbl


baseline_c_by_label = {label: baseline_c_tables(label) for label in LABELS}
for label, tbl in baseline_c_by_label.items():
    assert len(tbl) == 20, f"expected 20 baseline_C cells for {label}, got {len(tbl)}"

outcomes_full = trigger_df[["game_id", "fav_deficit", "season"]].merge(
    outcomes_df,
    on=["game_id", "fav_deficit"],
    how="inner",
    validate="many_to_one",
)
margin_train = outcomes_full[outcomes_full["season"].between(2015, 2021)].drop_duplicates("game_id").copy()
margin_train["final_margin"] = margin_train["final_fav_score"].astype(float) - margin_train["final_dog_score"].astype(float)
cfb_margin_variance = float(margin_train["final_margin"].var(ddof=1))
cfb_margin_std = float(math.sqrt(cfb_margin_variance))
margin_distribution = {
    "n_games": int(len(margin_train)),
    "mean": float(margin_train["final_margin"].mean()),
    "median": float(margin_train["final_margin"].median()),
    "variance": cfb_margin_variance,
    "std": cfb_margin_std,
    "skew": float(margin_train["final_margin"].skew()),
    "excess_kurtosis": float(margin_train["final_margin"].kurt()),
}

test_base = event_df[event_df["season"].isin([2022, 2023, 2024])].copy()
test_base["current_margin"] = test_base["fav_score_at_trigger"].astype(float) - test_base["dog_score_at_trigger"].astype(float)
test_base["baseline_sw_pfr_prob"] = stern_winston_prob(
    test_base["current_margin"].to_numpy(),
    test_base["seconds_remaining_in_regulation"].to_numpy(),
    PFR_STD,
)
test_base["baseline_sw_cfb_prob"] = stern_winston_prob(
    test_base["current_margin"].to_numpy(),
    test_base["seconds_remaining_in_regulation"].to_numpy(),
    cfb_margin_std,
)
for label, tbl in baseline_c_by_label.items():
    test_base = test_base.merge(
        tbl[["fav_deficit", "time_bucket", f"baseline_C_{label}"]],
        on=["fav_deficit", "time_bucket"],
        how="left",
        validate="many_to_one",
    )
    assert test_base[f"baseline_C_{label}"].notna().all(), f"missing baseline_C for {label}"

print("[ok] baseline_C tables and Stern-Winston probabilities built")
print(f"[ok] empirical CFB final-margin std={cfb_margin_std:.3f}; PFR std={PFR_STD:.3f}")
""")


add("code", "c08_0006", r"""
diagnostic = test_base[[
    "game_id", "trigger_play_id", "fav_deficit", "trigger_sequence",
    "season", "time_bucket", "fav_team", "dog_team",
    "fav_score_at_trigger", "dog_score_at_trigger", "seconds_remaining_in_regulation",
    "pregame_spread", "favorite_final_win", "deficit_erased",
    "baseline_C_deficit_erased", "baseline_C_favorite_final_win",
    "baseline_sw_pfr_prob", "baseline_sw_cfb_prob",
]].copy()

prob_key = ["game_id", "fav_deficit", "trigger_sequence"]
n06_probs = n06_committed[prob_key + ["fold", "calibrated_prob"]].rename(
    columns={"calibrated_prob": "n06_prob"}
)
n07_probs = n07_committed[prob_key + ["calibrated_prob"]].rename(
    columns={"calibrated_prob": "n07_prob"}
)
diagnostic = diagnostic.merge(n06_probs, on=prob_key, how="inner", validate="one_to_one")
diagnostic = diagnostic.merge(n07_probs, on=prob_key, how="inner", validate="one_to_one")
assert len(diagnostic) == 3854, f"Expected 3,854 held-out trigger events, got {len(diagnostic)}"
assert diagnostic[["game_id", "fav_deficit", "trigger_sequence", "fold"]].duplicated().sum() == 0

conformal_rows: list[pd.DataFrame] = []
conformal_by_fold: dict[int, dict[str, Any]] = {}
for fit in n06_fits:
    val = fit.val_predictions.copy()
    val_scores = (val[TARGET_LABEL].astype(float) - val["calibrated_prob"].astype(float)).abs().to_numpy()
    n = len(val_scores)
    # Split-conformal finite-sample quantile: ceil((n+1)*(1-alpha))/n with
    # upper clipping to the maximum observed conformity score.
    rank = min(n, int(math.ceil((n + 1) * (1.0 - ALPHA))))
    q_hat = float(np.sort(val_scores)[rank - 1])
    val_lower = np.clip(val["calibrated_prob"].to_numpy() - q_hat, 0.0, 1.0)
    val_upper = np.clip(val["calibrated_prob"].to_numpy() + q_hat, 0.0, 1.0)
    val_coverage = float(((val[TARGET_LABEL].to_numpy() >= val_lower) & (val[TARGET_LABEL].to_numpy() <= val_upper)).mean())
    if val_coverage < 0.93:
        raise AssertionError(
            f"Conformal validation coverage materially below 95% for fold {fit.fold}: {val_coverage:.3f}"
        )
    fold_rows = diagnostic[diagnostic["fold"].eq(fit.fold)].copy()
    fold_rows["conformal_q_hat"] = q_hat
    fold_rows["conformal_lower"] = np.clip(fold_rows["n06_prob"].to_numpy() - q_hat, 0.0, 1.0)
    fold_rows["conformal_upper"] = np.clip(fold_rows["n06_prob"].to_numpy() + q_hat, 0.0, 1.0)
    fold_rows["conformal_width"] = fold_rows["conformal_upper"] - fold_rows["conformal_lower"]
    fold_rows["conformal_covered"] = (
        (fold_rows[TARGET_LABEL].to_numpy() >= fold_rows["conformal_lower"].to_numpy())
        & (fold_rows[TARGET_LABEL].to_numpy() <= fold_rows["conformal_upper"].to_numpy())
    )
    conformal_rows.append(fold_rows)
    conformal_by_fold[fit.fold] = {
        "fold": fit.fold,
        "val_season": fit.val_season,
        "test_season": fit.test_season,
        "n_validation_play_rows": int(n),
        "q_hat": q_hat,
        "validation_coverage": val_coverage,
        "test_event_coverage": float(fold_rows["conformal_covered"].mean()),
        "test_average_width": float(fold_rows["conformal_width"].mean()),
    }

diagnostic = pd.concat(conformal_rows, ignore_index=True).sort_values(
    ["season", "game_id", "trigger_play_id", "fav_deficit", "trigger_sequence"]
).reset_index(drop=True)
diagnostic["baseline_c_prob"] = diagnostic["baseline_C_deficit_erased"]
diagnostic.to_parquet(N08_DIAGNOSTIC_PREDICTIONS_PARQUET, index=False)

print(f"[ok] wrote diagnostic predictions rows={len(diagnostic):,} cols={diagnostic.shape[1]}")
print("[ok] conformal by fold:")
for row in conformal_by_fold.values():
    print(row)
""")


add("code", "c08_0007", r"""
def model_prob_column(model: str) -> str:
    if model in {"M1_N06", "M3_N06_CONFORMAL"}:
        return "n06_prob"
    if model == "M2_N07_EXP":
        return "n07_prob"
    raise KeyError(model)


def baseline_prob_column(baseline: str, label: str) -> str:
    if baseline == "B_C":
        return f"baseline_C_{label}"
    if baseline == "B_SW_PFR":
        return "baseline_sw_pfr_prob"
    if baseline == "B_SW_CFB":
        return "baseline_sw_cfb_prob"
    raise KeyError(baseline)


comparison_rows: list[dict[str, Any]] = []
per_fold_rows: list[dict[str, Any]] = []
per_deficit_rows: list[dict[str, Any]] = []

for label in LABELS:
    y = diagnostic[label].astype(int).to_numpy()
    for model in MODEL_LABELS:
        p_model_col = model_prob_column(model)
        p_model = diagnostic[p_model_col].to_numpy(dtype=float)
        model_metrics = metric_bundle(y, p_model)
        weighted_model_ece = 0.0
        for fold, weight in FOLD_WEIGHTS.items():
            sub = diagnostic[diagnostic["fold"].eq(fold)]
            weighted_model_ece += weight * expected_calibration_error(
                sub[label].astype(int).to_numpy(),
                sub[p_model_col].to_numpy(dtype=float),
            )
        for baseline in BASELINE_LABELS:
            p_base_col = baseline_prob_column(baseline, label)
            p_base = diagnostic[p_base_col].to_numpy(dtype=float)
            tmp = diagnostic[["game_id", "fold", "fav_deficit", label, p_model_col, p_base_col]].copy()
            tmp["brier_improvement"] = (
                (tmp[p_base_col].astype(float) - tmp[label].astype(float)) ** 2
                - (tmp[p_model_col].astype(float) - tmp[label].astype(float)) ** 2
            )
            ci = bootstrap_cluster_mean_ci(tmp, "brier_improvement", seed=BOOTSTRAP_SEED + len(comparison_rows))
            base_metrics = metric_bundle(y, p_base)
            comparison_rows.append({
                "label": label,
                "model": model,
                "model_description": MODEL_LABELS[model],
                "baseline": baseline,
                "baseline_description": BASELINE_LABELS[baseline],
                "n": int(len(tmp)),
                "n_games": int(tmp["game_id"].nunique()),
                "model_brier": model_metrics["brier"],
                "baseline_brier": base_metrics["brier"],
                "brier_improvement_baseline_minus_model": float(tmp["brier_improvement"].mean()),
                "bootstrap_ci": ci,
                "model_ece": model_metrics["ece"],
                "baseline_ece": base_metrics["ece"],
                "weighted_model_ece": float(weighted_model_ece),
                "model_auc": model_metrics["auc"],
                "baseline_auc": base_metrics["auc"],
                "statistically_supported": bool(ci["lower"] > 0),
            })
            for fold in sorted(diagnostic["fold"].unique()):
                fsub = tmp[tmp["fold"].eq(fold)].copy()
                fy = fsub[label].astype(int).to_numpy()
                per_fold_rows.append({
                    "label": label,
                    "model": model,
                    "baseline": baseline,
                    "fold": int(fold),
                    "n": int(len(fsub)),
                    "brier_improvement_baseline_minus_model": float(fsub["brier_improvement"].mean()),
                    "model_brier": float(brier_score_loss(fy, fsub[p_model_col].to_numpy(dtype=float))),
                    "baseline_brier": float(brier_score_loss(fy, fsub[p_base_col].to_numpy(dtype=float))),
                    "model_ece": expected_calibration_error(fy, fsub[p_model_col].to_numpy(dtype=float)),
                    "baseline_ece": expected_calibration_error(fy, fsub[p_base_col].to_numpy(dtype=float)),
                    "model_auc": safe_auc(fy, fsub[p_model_col].to_numpy(dtype=float)),
                    "baseline_auc": safe_auc(fy, fsub[p_base_col].to_numpy(dtype=float)),
                })
            for deficit in sorted(diagnostic["fav_deficit"].unique()):
                dsub = tmp[tmp["fav_deficit"].eq(deficit)].copy()
                per_deficit_rows.append({
                    "label": label,
                    "model": model,
                    "baseline": baseline,
                    "fav_deficit": int(deficit),
                    "n": int(len(dsub)),
                    "n_games": int(dsub["game_id"].nunique()),
                    "brier_improvement_baseline_minus_model": float(dsub["brier_improvement"].mean()),
                    "bootstrap_ci": bootstrap_cluster_mean_ci(
                        dsub,
                        "brier_improvement",
                        seed=BOOTSTRAP_SEED + 10000 + len(per_deficit_rows),
                    ),
                })

comparison_df = pd.DataFrame(comparison_rows)
per_fold_df = pd.DataFrame(per_fold_rows)
per_deficit_df = pd.DataFrame(per_deficit_rows)

weighted_ece_by_model_label = (
    comparison_df[["label", "model", "weighted_model_ece"]]
    .drop_duplicates()
    .sort_values(["label", "weighted_model_ece", "model"])
    .to_dict(orient="records")
)
supported = comparison_df[comparison_df["statistically_supported"]].copy()
model_selection_df = (
    comparison_df[comparison_df["label"].eq(TARGET_LABEL)]
    [["model", "weighted_model_ece"]]
    .drop_duplicates()
    .copy()
)
supported_models = set(supported["model"].unique())
model_selection_df["beats_any_baseline"] = model_selection_df["model"].isin(supported_models)
model_selection_df["conformal_average_width"] = np.where(
    model_selection_df["model"].eq("M3_N06_CONFORMAL"),
    float(diagnostic["conformal_width"].mean()),
    float("inf"),
)
stability_rows = []
for model in MODEL_LABELS:
    sub = per_deficit_df[
        per_deficit_df["label"].eq(TARGET_LABEL)
        & per_deficit_df["model"].eq(model)
        & per_deficit_df["baseline"].eq("B_C")
    ].copy()
    stability_rows.append({
        "model": model,
        "per_deficit_pattern_stability_std": float(sub["brier_improvement_baseline_minus_model"].std(ddof=0)),
    })
stability_df = pd.DataFrame(stability_rows)
model_selection_df = model_selection_df.merge(stability_df, on="model", how="left", validate="one_to_one")

selection_path: list[dict[str, Any]] = []
eps = 1e-12
candidates = model_selection_df.copy()
if candidates["beats_any_baseline"].any():
    candidates = candidates[candidates["beats_any_baseline"]].copy()
    recommendation_basis = "supported_brier_edge_then_weighted_ece_then_conformal_width"
    selection_path.append({"criterion": "a_supported_brier_edge", "remaining_models": candidates["model"].tolist()})
else:
    recommendation_basis = "calibration_quality_no_supported_brier_edge"
    selection_path.append({"criterion": "a_supported_brier_edge", "remaining_models": candidates["model"].tolist()})

min_ece = float(candidates["weighted_model_ece"].min())
candidates = candidates[candidates["weighted_model_ece"] <= min_ece + eps].copy()
selection_path.append({
    "criterion": "b_lowest_weighted_ece",
    "value": min_ece,
    "remaining_models": candidates["model"].tolist(),
})

finite_widths = candidates[np.isfinite(candidates["conformal_average_width"])].copy()
if len(candidates) > 1 and len(finite_widths):
    min_width = float(finite_widths["conformal_average_width"].min())
    narrowed = candidates[candidates["conformal_average_width"] <= min_width + eps].copy()
    if len(narrowed):
        candidates = narrowed
        selection_path.append({
            "criterion": "c_narrowest_conformal_interval",
            "value": min_width,
            "remaining_models": candidates["model"].tolist(),
        })

if len(candidates) > 1:
    min_stability = float(candidates["per_deficit_pattern_stability_std"].min())
    candidates = candidates[candidates["per_deficit_pattern_stability_std"] <= min_stability + eps].copy()
    selection_path.append({
        "criterion": "d_per_deficit_pattern_stability",
        "value": min_stability,
        "remaining_models": candidates["model"].tolist(),
    })

best_selection = candidates.sort_values("model").iloc[0].to_dict()
best = (
    comparison_df[
        comparison_df["label"].eq(TARGET_LABEL)
        & comparison_df["model"].eq(best_selection["model"])
    ]
    .sort_values("baseline")
    .iloc[0]
    .to_dict()
)

deployment_recommendation = {
    "recommended_model": best_selection["model"],
    "recommended_model_description": MODEL_LABELS[best_selection["model"]],
    "basis": recommendation_basis,
    "selection_path": selection_path,
    "label_context": TARGET_LABEL,
    "beats_any_baseline": bool(best_selection["beats_any_baseline"]),
    "weighted_ece": float(best_selection["weighted_model_ece"]),
    "conformal_average_width": (
        None
        if not np.isfinite(float(best_selection["conformal_average_width"]))
        else float(best_selection["conformal_average_width"])
    ),
    "per_deficit_pattern_stability_std": float(best_selection["per_deficit_pattern_stability_std"]),
    "caveat": (
        "Recommendation is diagnostic. N08 does not train a new model; it selects among existing N06/N07 point predictions "
        "and the N06 conformal layer for future live-data scaffolding."
    ),
}

conformal_summary = {
    "alpha": ALPHA,
    "coverage_level": 1.0 - ALPHA,
    "by_fold": conformal_by_fold,
    "overall_test_event_coverage": float(diagnostic["conformal_covered"].mean()),
    "overall_average_width": float(diagnostic["conformal_width"].mean()),
    "width_by_deficit": (
        diagnostic.groupby("fav_deficit")["conformal_width"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .to_dict(orient="records")
    ),
    "width_by_time_bucket": (
        diagnostic.groupby("time_bucket")["conformal_width"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .to_dict(orient="records")
    ),
}

print("[ok] comparison matrices computed")
print("Deployment recommendation:", deployment_recommendation)
""")


add("code", "c08_0008", r"""
comparison_payload = {
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
    "design_resolution": {
        "M1": "N06 calibrated probability trained on deficit_erased",
        "M2": "N07 expanded calibrated probability trained on deficit_erased",
        "M3": "N06 point probability plus split-conformal intervals",
        "trained_label": TARGET_LABEL,
        "cross_label": CROSS_LABEL,
        "no_new_models_trained": True,
        "validation_slice_fits_rematerialized_for_conformal_only": True,
        "held_out_probability_rebuild_max_abs_diff": {
            "n06": n06_max_abs_diff,
            "n07": n07_max_abs_diff,
        },
    },
    "stern_winston": {
        "pfr_std": PFR_STD,
        "empirical_cfb_margin_distribution_2015_2021": margin_distribution,
        "deficit_erased_approximation": (
            "Stern-Winston probabilities estimate P(final margin >= 0). For deficit_erased, "
            "N08 uses this as an approximation to P(the favorite ties or retakes the lead before game end)."
        ),
    },
    "comparison_matrix": comparison_df.to_dict(orient="records"),
    "per_fold": per_fold_df.to_dict(orient="records"),
    "per_deficit": per_deficit_df.to_dict(orient="records"),
    "weighted_ece_by_model_label": weighted_ece_by_model_label,
    "conformal": conformal_summary,
    "deployment_recommendation": deployment_recommendation,
}
N08_COMPARISON_RESULTS_JSON.write_text(json.dumps(comparison_payload, indent=2) + "\n", encoding="utf-8")

price_conversion_spec = {
    "created_at": comparison_payload["created_at"],
    "source_commit": comparison_payload["source_commit"],
    "function_name": "stern_winston_favorite_win_probability_v1",
    "interface": {
        "inputs": {
            "current_score_diff": "favorite score minus underdog score at evaluation time",
            "time_remaining_seconds": "seconds remaining in regulation, clipped to [0, 3600]",
            "pregame_spread": "accepted for future compatibility; coefficient is 0.0 in N08 v1",
        },
        "output": "favorite implied win probability",
    },
    "formula": {
        "epa_adjustment": 0.0,
        "pregame_spread_coefficient": 0.0,
        "mean": "current_score_diff",
        "std": f"{cfb_margin_std:.12f} * sqrt(time_remaining_seconds / 3600)",
        "probability": "1 - normal_cdf((0.5 - mean) / std)",
        "clip": "[0, 1]",
    },
    "parameters": {
        "pfr_reference_std": PFR_STD,
        "empirical_cfb_std": cfb_margin_std,
        "empirical_cfb_variance": cfb_margin_variance,
        "training_years": [2015, 2016, 2017, 2018, 2019, 2020, 2021],
        "n_training_games": margin_distribution["n_games"],
    },
    "limitations": [
        "EPA adjustment is held at zero in N08 v1.",
        "pregame_spread is accepted by the interface but not used in v1.",
        "For deficit_erased, the final-win probability is only an approximation to ever-erased probability.",
    ],
}
N08_PRICE_CONVERSION_SPEC_JSON.write_text(json.dumps(price_conversion_spec, indent=2) + "\n", encoding="utf-8")


def fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "NA"
    try:
        if math.isnan(float(x)):
            return "NA"
    except Exception:
        pass
    return f"{float(x):.{digits}f}"


def matrix_markdown(label: str) -> list[str]:
    rows = ["| Model | Baseline | Brier improvement | 95% CI | Model ECE | Baseline ECE | Model AUC | Baseline AUC |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    sub = comparison_df[comparison_df["label"].eq(label)].copy()
    for _, r in sub.iterrows():
        ci = r["bootstrap_ci"]
        rows.append(
            f"| {r['model']} | {r['baseline']} | {fmt(r['brier_improvement_baseline_minus_model'])} "
            f"| [{fmt(ci['lower'])}, {fmt(ci['upper'])}] | {fmt(r['model_ece'])} | {fmt(r['baseline_ece'])} "
            f"| {fmt(r['model_auc'])} | {fmt(r['baseline_auc'])} |"
        )
    return rows


trained_label_supported = comparison_df[
    comparison_df["label"].eq(TARGET_LABEL) & comparison_df["statistically_supported"]
].copy()
any_supported = comparison_df[comparison_df["statistically_supported"]].copy()
m1_ece = float(comparison_df[(comparison_df["label"].eq(TARGET_LABEL)) & (comparison_df["model"].eq("M1_N06"))]["weighted_model_ece"].iloc[0])
m2_ece = float(comparison_df[(comparison_df["label"].eq(TARGET_LABEL)) & (comparison_df["model"].eq("M2_N07_EXP"))]["weighted_model_ece"].iloc[0])
m3_ece = float(comparison_df[(comparison_df["label"].eq(TARGET_LABEL)) & (comparison_df["model"].eq("M3_N06_CONFORMAL"))]["weighted_model_ece"].iloc[0])
m3_sw_cfb_row = comparison_df[
    comparison_df["label"].eq(TARGET_LABEL)
    & comparison_df["model"].eq("M3_N06_CONFORMAL")
    & comparison_df["baseline"].eq("B_SW_CFB")
].iloc[0]
cfb_variance_lift_pct = 100.0 * ((cfb_margin_variance / (PFR_STD ** 2)) - 1.0)

lines: list[str] = []
lines.append("# N08 -- Stern-Winston baseline and conformal diagnostic")
lines.append("")
lines.append(f"**Deployment recommendation:** `{deployment_recommendation['recommended_model']}` ({deployment_recommendation['recommended_model_description']}). Basis: `{deployment_recommendation['basis']}`; weighted ECE on `{deployment_recommendation['label_context']}` = **{fmt(deployment_recommendation['weighted_ece'])}**.")
lines.append("")
lines.append(f"**Primary finding:** deploy `M3_N06_CONFORMAL`: the N06 calibrated point predictions with split-conformal intervals added as a descriptive uncertainty layer. Locked Decision 4's ordering is decisive: all three variants clear the first criterion by beating at least one baseline, M1 and M3 tie on weighted ECE (**{fmt(m1_ece)}**) ahead of M2 (**{fmt(m2_ece)}**), and the conformal-interval criterion breaks the M1/M3 tie in favor of M3.")
lines.append("")
lines.append(f"Important methodological context: no model beats baseline_C on `deficit_erased`. The supported Brier edge is against Stern-Winston baselines, which is large but mostly reflects that Stern-Winston is poorly calibrated for the path-dependent deficit-erasure label: B_SW_CFB ECE is **{fmt(m3_sw_cfb_row['baseline_ece'])}** versus model ECE **{fmt(m3_sw_cfb_row['model_ece'])}**. This is informative about Stern-Winston as an evaluation baseline, but it is not new evidence of comeback-detection edge beyond baseline_C.")
lines.append("")
lines.append(f"For deployment, use N06 calibrated point predictions and display conformal intervals as descriptive uncertainty. The average conformal interval width is **{fmt(conformal_summary['overall_average_width'])}**, so individual trigger predictions carry substantial uncertainty beyond the point probability. For N09 bet sizing, that width argues against narrow-confidence Kelly assumptions; eighth-Kelly or flat staking should be considered as primary simulation strategies.")
lines.append("")
lines.append("N08 compares the locked N06 point model, the locked N07 expanded model, and the N06 conformal layer against baseline_C and two Stern-Winston analytical baselines. M1 and M2 are both trained on `deficit_erased`; `favorite_final_win` is reported only as a cross-label diagnostic.")
lines.append("")
lines.append("## Stern-Winston variance")
lines.append("")
lines.append(f"The PFR/NFL reference standard deviation is **{PFR_STD:.2f}**. The empirical CFB 2015-2021 favorite final-margin standard deviation is **{cfb_margin_std:.2f}** (variance **{cfb_margin_variance:.2f}**, n={margin_distribution['n_games']:,} games). CFB variance is approximately **{cfb_variance_lift_pct:.1f}%** higher than the NFL/PFR reference. Headline Stern-Winston comparisons and the exported N09 price-conversion function use the empirical CFB standard deviation.")
lines.append("")
lines.append("The `deficit_erased` Stern-Winston comparison is an approximation: the analytical model estimates final favorite win probability, not the path-dependent probability of tying or retaking the lead before game end.")
lines.append("")
lines.append("## Comparison Matrix: deficit_erased")
lines.extend(matrix_markdown(TARGET_LABEL))
lines.append("")
lines.append("## Comparison Matrix: favorite_final_win")
lines.extend(matrix_markdown(CROSS_LABEL))
lines.append("")
lines.append("## Conformal intervals")
lines.append("")
lines.append(f"N06 split-conformal intervals use validation-slice conformity scores with alpha={ALPHA:.2f}. Overall held-out event coverage is **{fmt(conformal_summary['overall_test_event_coverage'])}** and average interval width is **{fmt(conformal_summary['overall_average_width'])}**.")
lines.append("")
lines.append("| Fold | q_hat | Validation coverage | Test coverage | Avg width |")
lines.append("|---:|---:|---:|---:|---:|")
for fold, row in conformal_by_fold.items():
    lines.append(f"| {fold} | {fmt(row['q_hat'])} | {fmt(row['validation_coverage'])} | {fmt(row['test_event_coverage'])} | {fmt(row['test_average_width'])} |")
lines.append("")
lines.append("## Per-deficit trained-label pattern")
lines.append("")
lines.append("| Model | Baseline | Deficit | n | Brier improvement | 95% CI |")
lines.append("|---|---|---:|---:|---:|---:|")
td = per_deficit_df[per_deficit_df["label"].eq(TARGET_LABEL)].copy()
for _, r in td.iterrows():
    ci = r["bootstrap_ci"]
    lines.append(f"| {r['model']} | {r['baseline']} | {int(r['fav_deficit'])} | {int(r['n'])} | {fmt(r['brier_improvement_baseline_minus_model'])} | [{fmt(ci['lower'])}, {fmt(ci['upper'])}] |")
lines.append("")
lines.append("## Verification")
lines.append("")
lines.append(f"- Rebuilt N06 held-out probabilities matched committed artifact with max absolute difference `{n06_max_abs_diff:.3g}`.")
lines.append(f"- Rebuilt N07 held-out probabilities matched committed artifact with max absolute difference `{n07_max_abs_diff:.3g}`.")
lines.append(f"- Diagnostic prediction rows: {len(diagnostic):,}.")
lines.append(f"- Price conversion spec: `stern_winston_favorite_win_probability_v1`, using empirical CFB std {cfb_margin_std:.4f} and pregame-spread coefficient 0.0 in v1.")
lines.append("")
lines.append("## Honest interpretation")
lines.append("")
lines.append("N08 sharpens the deployment choice without changing the research conclusion. M3 is defensible because it preserves the better-calibrated N06 point predictions and adds uncertainty intervals, not because it discovers new comeback-detection signal. Baseline_C remains unbeaten on the trained label. The Stern-Winston result says more about the limitations of final-margin analytical baselines for path-dependent `deficit_erased` than about historical edge. The conformal layer is useful precisely because the intervals are wide: it makes visible the uncertainty that a single calibrated probability can hide.")
lines.append("")

N08_SUMMARY_REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

print(f"[ok] wrote {N08_DIAGNOSTIC_PREDICTIONS_PARQUET.relative_to(REPO_ROOT)} size={N08_DIAGNOSTIC_PREDICTIONS_PARQUET.stat().st_size:,}")
print(f"[ok] wrote {N08_COMPARISON_RESULTS_JSON.relative_to(REPO_ROOT)} size={N08_COMPARISON_RESULTS_JSON.stat().st_size:,}")
print(f"[ok] wrote {N08_PRICE_CONVERSION_SPEC_JSON.relative_to(REPO_ROOT)} size={N08_PRICE_CONVERSION_SPEC_JSON.stat().st_size:,}")
print(f"[ok] wrote {N08_SUMMARY_REPORT_MD.relative_to(REPO_ROOT)} size={N08_SUMMARY_REPORT_MD.stat().st_size:,}")
""")


def _format_source(src: str) -> list[str]:
    return [line + "\n" for line in src.splitlines()]


nb = {
    "cells": [
        {
            "cell_type": cell_type,
            "id": cell_id,
            "metadata": {},
            "source": _format_source(src),
            **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
        }
        for cell_type, cell_id, src in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")
