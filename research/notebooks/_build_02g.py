"""
Deterministic builder for research/notebooks/02g_context_week_home_neutral.ipynb.

Reduced 02g scope after the 2026-05-17 planning halt:
  - season_phase_mid
  - season_phase_late
  - season_phase_bowl
  - fav_is_home
  - is_neutral_site

Weather / venue candidates (`is_dome`, `wind_mph`, `temp_f`) are deferred to
future_features.md and tech_debt.md. This builder performs no external API
calls and defines no CFBD/Open-Meteo HTTP helper; it reads cached /games JSON
directly from research/data/cache only to attach neutralSite.
"""

from __future__ import annotations

import json
import pathlib
import sys
import textwrap

OUT = pathlib.Path(__file__).resolve().parent / "02g_context_week_home_neutral.ipynb"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _lib_chrono import CHRONO_KEY_SOURCE  # noqa: E402

CELLS: list[tuple[str, str, str]] = []


def add(cell_type: str, cell_id: str, src: str) -> None:
    CELLS.append((cell_type, cell_id, textwrap.dedent(src).lstrip("\n")))


add("markdown", "m02g0000", """
# Phase 0 -- Notebook 02g: Context (week phase + home/away/neutral)

Reduced 02g candidate set after the weather-plumbing halt:

1. `season_phase_mid`
2. `season_phase_late`
3. `season_phase_bowl`
4. `fav_is_home`
5. `is_neutral_site`

Weather / venue candidates (`is_dome`, `wind_mph`, `temp_f`) are deferred
pending a full Open-Meteo cache backfill plus an authoritative venue-coordinate
source. That deferral is documented in `research/future_features.md` and
`research/tech_debt.md`.

## Reduced-scope D1-D12 decisions

- **D1 source map:** `season_phase_*` and `fav_is_home` come directly from
  `trigger_events.csv`; `is_neutral_site` joins cached `/games` metadata by
  `game_id`.
- **D2 fetch policy:** no external calls. No CFBD helper and no Open-Meteo
  helper are defined in this notebook.
- **D3 season phase:** regular weeks `1-4` = early reference, `5-9` = mid,
  `10-16` = late; all postseason rows = bowl.
- **D4 phase encoding:** one-hot with early regular season as the omitted
  reference; no ordinal and no cyclical encoding.
- **D5 home/away/neutral encoding:** `fav_is_home` and `is_neutral_site`
  binary flags; away favorite is the reference state.
- **D6 Category classification:** all five features are Category A. None
  iterate plays or drives.
- **D7 null policy:** no nulls allowed after the cached `/games` join. Halt
  if any candidate null appears.
- **D8 imputation:** none.
- **D9 baseline:** same locked pre-game baseline as 02a-02f.
- **D10 diff-vs-leaky:** because all candidates are pre-game/game-metadata
  fields, chrono and leaky builds must be byte-identical.
- **D11 correlation audit:** run candidate-vs-candidate in-notebook; run the
  full cross-notebook matrix after execution for N03 prep.
- **D12 rollup:** append five features x three folds to
  `feature_validation.csv`, splice a 02g sidecar section, and report the
  cumulative validated set.

## Four-audit pattern

**Candidate-vs-candidate.** Expected strongest overlap:
`season_phase_bowl` with `is_neutral_site`, because postseason games are often
neutral-site. This is correlation risk, not identity.

**Candidate-vs-validated.** Expected low to moderate correlations. Watch
`season_phase_bowl` vs opening-drive / red-zone / drive-volume features and
`fav_is_home` vs `dog_received_opening_kickoff`, but no pre-execution
`|rho| >= 0.6` prediction against prior validated columns.

**Candidate-vs-trigger-fields.** Four features are directly trigger-row
derivable: `season_phase_mid`, `season_phase_late`, `season_phase_bowl`,
`fav_is_home`. `is_neutral_site` requires cached `/games.neutralSite`; the
cache covers all 4,311 unique trigger games.

**Candidate-vs-extractor-structure.** All five are Category A. D10 should be
trivially byte-identical because the extractor ignores any play filter mode.

Plan-time prediction: **2-4 PASS** features, mostly small magnitudes. Rho
corridor: most cross-notebook `|rho| < 0.3`; candidate-candidate
`season_phase_bowl` vs `is_neutral_site` may exceed `0.6`.
""")


add("code", "c02g0001", '''
"""
Notebook 02g -- imports, paths, and fail-fast checks.

This reduced-scope notebook is intentionally local-cache only. It does not
define cfbd_get(), om_get(), or any HTTP client helper.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NOTEBOOK_DIR = pathlib.Path(".").resolve()
RESEARCH_DIR = (NOTEBOOK_DIR / "..").resolve()
DATA_DIR = (RESEARCH_DIR / "data").resolve()
RESULTS_DIR = (RESEARCH_DIR / "results").resolve()
CACHE_DIR = DATA_DIR / "cache"
CALL_LOG = CACHE_DIR / "cfbd_call_log.csv"
REPO_ROOT = (RESEARCH_DIR / "..").resolve()

assert RESEARCH_DIR.name == "research", (
    f"Expected to run inside research/notebooks/. Got NOTEBOOK_DIR={NOTEBOOK_DIR}."
)
assert CACHE_DIR.exists(), f"Expected cache directory at {CACHE_DIR}"

TRIGGER_EVENTS_CSV = RESULTS_DIR / "trigger_events.csv"
TRIGGER_OUTCOMES_CSV = RESULTS_DIR / "trigger_outcomes.csv"
FEATURE_VALIDATION_CSV = RESULTS_DIR / "feature_validation.csv"
FEATURE_VALIDATION_SCHEMA = RESULTS_DIR / "feature_validation.schema.md"

assert TRIGGER_EVENTS_CSV.exists(), f"Missing {TRIGGER_EVENTS_CSV}"
assert TRIGGER_OUTCOMES_CSV.exists(), f"Missing {TRIGGER_OUTCOMES_CSV}"
assert CALL_LOG.exists(), f"Missing call log at {CALL_LOG}"

print(f"[ok] paths resolved relative to {NOTEBOOK_DIR}")
print(f"[ok] trigger artifacts present")
print(f"[ok] cache dir: {CACHE_DIR}")
''')


add("code", "c02g0002", CHRONO_KEY_SOURCE + '''

print("[ok] canonical _chrono_key body spliced from _lib_chrono")
print("[info] 02g candidates are Category A; _chrono_key is present for the shared notebook contract.")
''')


add("markdown", "m02g0003", """
## Configuration

The walk-forward windows and baseline pre-game columns are carried verbatim
from 02a-02f. `FEATURE_SET_VERSION = "v1_context_week_home_neutral"`.

All candidates are non-null binary indicators. No per-window imputation values
are written for 02g.
""")


add("code", "c02g0004", '''
SEASONS: list[int] = list(range(2015, 2025))
FEATURE_SET_VERSION: str = "v1_context_week_home_neutral"

WALK_FORWARD_WINDOWS: list[dict] = [
    {"train_seasons": list(range(2015, 2021)), "val_season": 2021,
     "test_season": 2022, "train_window_label": "2015-2020"},
    {"train_seasons": list(range(2015, 2022)), "val_season": 2022,
     "test_season": 2023, "train_window_label": "2015-2021"},
    {"train_seasons": list(range(2015, 2023)), "val_season": 2023,
     "test_season": 2024, "train_window_label": "2015-2022"},
]

ALWAYS_PRESENT_PREGAME_COLS: list[str] = [
    "pregame_spread",
    "rating_gap",
    "fav_pregame_rating",
    "dog_pregame_rating",
]
BASELINE_PREGAME_FEATURES: list[str] = [
    *ALWAYS_PRESENT_PREGAME_COLS,
    "spread_movement",
    "spread_movement_is_null",
]

CANDIDATE_FEATURES: list[str] = [
    "season_phase_mid",
    "season_phase_late",
    "season_phase_bowl",
    "fav_is_home",
    "is_neutral_site",
]

EXTRACTOR_CATEGORY: dict[str, str] = {f: "A" for f in CANDIDATE_FEATURES}
REDUNDANT_WITH: dict[str, str] = {}
RANDOM_STATE: int = 42

print(f"feature_set_version: {FEATURE_SET_VERSION}")
print(f"baseline pre-game ({len(BASELINE_PREGAME_FEATURES)}): {BASELINE_PREGAME_FEATURES}")
print(f"candidates ({len(CANDIDATE_FEATURES)}) all Category A:")
for f in CANDIDATE_FEATURES:
    print(f"  - {f}")
''')


add("markdown", "m02g0005", """
## Phase 02g-a -- Load trigger artifacts

Same inner join as earlier 02 notebooks, followed by the standard
`final_fav_won.notna()` filter. The label is only used by the evaluator.
""")


add("code", "c02g0006", '''
triggers_df = pd.read_csv(TRIGGER_EVENTS_CSV)
outcomes_df = pd.read_csv(TRIGGER_OUTCOMES_CSV)
print(f"trigger_events.csv:    {len(triggers_df):>6,} rows x {triggers_df.shape[1]} cols")
print(f"trigger_outcomes.csv:  {len(outcomes_df):>6,} rows x {outcomes_df.shape[1]} cols")

trigger_full_df = triggers_df.merge(
    outcomes_df,
    on=["game_id", "fav_deficit"],
    how="inner",
    validate="one_to_one",
)
assert len(trigger_full_df) == len(triggers_df), (
    f"inner join lost rows: {len(triggers_df)} -> {len(trigger_full_df)}"
)

n_pre_drop = len(trigger_full_df)
trigger_full_df = trigger_full_df[trigger_full_df["final_fav_won"].notna()].copy()
trigger_full_df["final_fav_won"] = trigger_full_df["final_fav_won"].astype(bool)
n_dropped_tie = n_pre_drop - len(trigger_full_df)

for col in ["season_type", "week", "home_is_fav"]:
    assert col in trigger_full_df.columns, f"Required trigger column missing: {col}"
    assert int(trigger_full_df[col].isna().sum()) == 0, f"Unexpected nulls in {col}"

for col in ALWAYS_PRESENT_PREGAME_COLS:
    n_null = int(trigger_full_df[col].isna().sum())
    assert n_null == 0, f"Always-present pre-game column {col!r} has {n_null} nulls"

print(f"merged:                {n_pre_drop:>6,} rows")
print(f"dropped final_fav_won NaN rows: {n_dropped_tie:,}")
print(f"in-scope rows for 02g: {len(trigger_full_df):,}")
print("[ok] direct trigger columns available: season_type, week, home_is_fav")
''')


add("markdown", "m02g0007", """
## Phase 02g-b -- Load cached `/games` metadata directly from disk

`is_neutral_site` is the only feature that is not already in
`trigger_events.csv`. It joins cached `/games.neutralSite` by `game_id`.

No HTTP helper is defined here. The cell scans `research/data/cache` for
`cfbd__games__*.json` files and fails if any trigger game is missing.
""")


add("code", "c02g0008", '''
n_call_log_before = sum(1 for _ in CALL_LOG.open("r", encoding="utf-8")) - 1

games_rows: list[dict[str, Any]] = []
for path in sorted(CACHE_DIR.glob("cfbd__games__*.json")):
    rows = json.loads(path.read_text(encoding="utf-8"))
    for g in rows:
        if g.get("id") is not None:
            games_rows.append(g)

assert games_rows, f"No cached cfbd__games__*.json files found in {CACHE_DIR}"
games_df = pd.DataFrame(games_rows).drop_duplicates(subset=["id"], keep="last")
required_game_cols = ["id", "neutralSite", "season", "seasonType", "week", "homeTeam", "awayTeam"]
missing_game_cols = [c for c in required_game_cols if c not in games_df.columns]
assert not missing_game_cols, f"Cached /games metadata missing columns: {missing_game_cols}"

unique_trigger_games = pd.DataFrame({"game_id": sorted(trigger_full_df["game_id"].unique())})
game_join_df = unique_trigger_games.merge(
    games_df[required_game_cols],
    left_on="game_id",
    right_on="id",
    how="left",
    validate="one_to_one",
)
n_missing_games = int(game_join_df["id"].isna().sum())
n_missing_neutral = int(game_join_df["neutralSite"].isna().sum())
assert n_missing_games == 0, f"Cached /games metadata missing {n_missing_games} trigger games"
assert n_missing_neutral == 0, f"Cached /games neutralSite missing for {n_missing_neutral} trigger games"

games_meta_by_id: dict[int, dict[str, Any]] = {
    int(r["game_id"]): {
        "neutralSite": bool(r["neutralSite"]),
        "season": int(r["season"]),
        "seasonType": str(r["seasonType"]),
        "week": int(r["week"]),
        "homeTeam": str(r["homeTeam"]),
        "awayTeam": str(r["awayTeam"]),
    }
    for _, r in game_join_df.iterrows()
}

print(f"cached /games rows loaded: {len(games_df):,}")
print(f"unique trigger games:      {len(unique_trigger_games):,}")
print(f"cached join coverage:      {len(games_meta_by_id):,}/{len(unique_trigger_games):,}")
print(f"neutral-site trigger games: {int(game_join_df['neutralSite'].astype(bool).sum()):,}")
print("[ok] no external fetch needed for reduced 02g feature set")
''')


add("markdown", "m02g0009", """
## Phase 02g-c -- Build Category-A feature matrix

`plays_before_filter` is accepted only to keep the D10 call signature aligned
with earlier notebooks. The reduced 02g extractor does not inspect plays or
drives, so `chrono_key` and `leaky_playnumber` builds must match exactly.
""")


add("code", "c02g000a", '''
def _bool_int(x: Any) -> int:
    if isinstance(x, (bool, np.bool_)):
        return int(bool(x))
    if isinstance(x, (int, np.integer)):
        return int(x != 0)
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return 1
    if s in {"false", "0", "no", "n"}:
        return 0
    raise ValueError(f"Cannot coerce to bool: {x!r}")


def _season_phase(season_type: str, week: int) -> str:
    if str(season_type).lower() != "regular":
        return "bowl"
    w = int(week)
    if 1 <= w <= 4:
        return "early"
    if 5 <= w <= 9:
        return "mid"
    if 10 <= w <= 16:
        return "late"
    raise ValueError(f"Unexpected regular-season week: {week!r}")


def build_feature_matrix(
    source_df: pd.DataFrame,
    games_meta: dict[int, dict[str, Any]],
    *,
    plays_before_filter: str = "chrono_key",
) -> pd.DataFrame:
    if plays_before_filter not in {"chrono_key", "leaky_playnumber"}:
        raise ValueError(f"unsupported plays_before_filter={plays_before_filter!r}")

    out = source_df.copy()
    phases = [
        _season_phase(str(r["season_type"]), int(r["week"]))
        for _, r in out.iterrows()
    ]
    out["season_phase"] = phases
    out["season_phase_mid"] = [int(p == "mid") for p in phases]
    out["season_phase_late"] = [int(p == "late") for p in phases]
    out["season_phase_bowl"] = [int(p == "bowl") for p in phases]
    out["fav_is_home"] = out["home_is_fav"].map(_bool_int).astype(int)
    out["is_neutral_site"] = [
        int(bool(games_meta[int(gid)]["neutralSite"]))
        for gid in out["game_id"]
    ]

    out["spread_movement_is_null"] = out["spread_movement"].isna().astype(int)
    out["spread_movement"] = out["spread_movement"].fillna(0.0)

    for feat in CANDIDATE_FEATURES:
        n_null = int(out[feat].isna().sum())
        assert n_null == 0, f"02g candidate {feat!r} has {n_null} null rows"
        uniq = sorted(out[feat].dropna().unique().tolist())
        assert set(uniq).issubset({0, 1}), f"{feat!r} expected binary values, got {uniq}"

    return out


feature_matrix_df = build_feature_matrix(
    trigger_full_df,
    games_meta_by_id,
    plays_before_filter="chrono_key",
)

print(f"feature_matrix_df: {len(feature_matrix_df):,} rows x {feature_matrix_df.shape[1]} cols")
print("Candidate prevalence:")
for feat in CANDIDATE_FEATURES:
    n = int(feature_matrix_df[feat].sum())
    pct = n / len(feature_matrix_df) * 100
    print(f"  {feat:<24} {n:>6,} ({pct:5.2f}%)")
''')


add("markdown", "m02g000b", """
## Phase 02g-d -- D10 diff-vs-leaky verification

All five features are Category A. The candidate matrix must be byte-identical
under the canonical and leaky filter labels because no play sequence is read.
""")


add("code", "c02g000c", '''
feature_matrix_leaky_df = build_feature_matrix(
    trigger_full_df,
    games_meta_by_id,
    plays_before_filter="leaky_playnumber",
)

catA_mismatches: dict[str, int] = {}
for feat in CANDIDATE_FEATURES:
    a = feature_matrix_df[feat].astype("int8").to_numpy()
    b = feature_matrix_leaky_df[feat].astype("int8").to_numpy()
    mismatches = int((a != b).sum())
    catA_mismatches[feat] = mismatches
    assert mismatches == 0, f"D10 Category-A byte-identity failed for {feat}: {mismatches} mismatches"

print("D10 diff-vs-leaky byte-identical confirmation:")
for feat, n_mis in catA_mismatches.items():
    print(f"  {feat:<24} mismatches={n_mis}")
print("[ok] all 02g Category-A features are byte-identical under chrono vs leaky labels")
''')


add("markdown", "m02g000d", """
## Phase 02g-e -- Candidate-vs-candidate correlation

This is the in-notebook redundancy audit for the reduced 02g feature set.
The full cross-notebook matrix is run after notebook execution, because it
reuses the heavier 02f feature-matrix harness for N03 prep.
""")


add("code", "c02g000e", '''
candidate_corr_df = feature_matrix_df[CANDIDATE_FEATURES].astype(float).corr()
print("Candidate-vs-candidate Pearson correlation matrix:")
print(candidate_corr_df.to_string(float_format=lambda x: f"{x:+.3f}"))

candidate_high_pairs: list[dict[str, Any]] = []
for i, a in enumerate(CANDIDATE_FEATURES):
    for b in CANDIDATE_FEATURES[i + 1:]:
        rho = float(candidate_corr_df.loc[a, b])
        if abs(rho) >= 0.6:
            candidate_high_pairs.append({"feature_a": a, "feature_b": b, "pearson_rho": rho})

if candidate_high_pairs:
    print("\\nCandidate-candidate |rho| >= 0.6:")
    for r in candidate_high_pairs:
        print(f"  {r['feature_a']} <-> {r['feature_b']} rho={r['pearson_rho']:+.3f}")
else:
    print("\\nNo candidate-candidate |rho| >= 0.6 pairs.")
''')


add("markdown", "m02g000f", """
## Phase 02g-f -- Walk-forward evaluation

Same single-feature isolation test as 02a-02f: baseline pre-game model vs
baseline plus one candidate feature. Stability is R6: at least two of three
test seasons with positive held-out Brier improvement.
""")


add("code", "c02g000g", '''
def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (float(mask.mean()) * abs(acc - conf))
    return float(ece)


def _make_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            penalty="l1",
            C=1.0,
            solver="liblinear",
            random_state=RANDOM_STATE,
            max_iter=1000,
        )),
    ])


def _fit_calibrated_probs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    model = _make_model()
    y_train = train_df["final_fav_won"].astype(int).to_numpy()
    y_val = val_df["final_fav_won"].astype(int).to_numpy()

    model.fit(train_df[feature_cols], y_train)
    cal = CalibratedClassifierCV(estimator=model, method="isotonic", cv="prefit")
    cal.fit(val_df[feature_cols], y_val)
    return cal.predict_proba(test_df[feature_cols])[:, 1]


eval_rows: list[dict[str, Any]] = []
for window in WALK_FORWARD_WINDOWS:
    train_seasons = set(window["train_seasons"])
    val_season = int(window["val_season"])
    test_season = int(window["test_season"])
    win_label = str(window["train_window_label"])

    train_df = feature_matrix_df[feature_matrix_df["season"].isin(train_seasons)].copy()
    val_df = feature_matrix_df[feature_matrix_df["season"] == val_season].copy()
    test_df = feature_matrix_df[feature_matrix_df["season"] == test_season].copy()
    assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0, (
        f"empty split for window {win_label}"
    )

    y_test = test_df["final_fav_won"].astype(int).to_numpy()

    for feat in CANDIDATE_FEATURES:
        base_cols = BASELINE_PREGAME_FEATURES
        cand_cols = [*BASELINE_PREGAME_FEATURES, feat]

        p_base = _fit_calibrated_probs(train_df, val_df, test_df, base_cols)
        p_cand = _fit_calibrated_probs(train_df, val_df, test_df, cand_cols)

        brier_base = float(brier_score_loss(y_test, p_base))
        brier_cand = float(brier_score_loss(y_test, p_cand))
        ece_base = expected_calibration_error(y_test, p_base, n_bins=10)
        ece_cand = expected_calibration_error(y_test, p_cand, n_bins=10)

        eval_rows.append({
            "feature": feat,
            "feature_set_version": FEATURE_SET_VERSION,
            "train_window": win_label,
            "val_season": val_season,
            "test_season": test_season,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "brier_test_baseline": brier_base,
            "brier_test_candidate": brier_cand,
            "brier_improvement": brier_base - brier_cand,
            "ece_test_baseline": ece_base,
            "ece_test_candidate": ece_cand,
            "calibration_improvement": ece_base - ece_cand,
            "redundant_with": REDUNDANT_WITH.get(feat, ""),
            "imputation_value": "",
        })

eval_df = pd.DataFrame(eval_rows)
stability_decision: dict[str, bool] = {}
for feat in CANDIDATE_FEATURES:
    n_positive = int((eval_df[eval_df["feature"] == feat]["brier_improvement"] > 0).sum())
    stability_decision[feat] = n_positive >= 2
eval_df["passed_stability"] = eval_df["feature"].map(stability_decision)

CSV_COLUMNS = [
    "feature", "feature_set_version", "train_window", "val_season", "test_season",
    "n_train", "n_val", "n_test",
    "brier_test_baseline", "brier_test_candidate", "brier_improvement",
    "ece_test_baseline", "ece_test_candidate", "calibration_improvement",
    "passed_stability",
    "redundant_with",
    "imputation_value",
]
eval_df = eval_df[CSV_COLUMNS]

print("Per-feature stability verdicts:")
for feat in CANDIDATE_FEATURES:
    sub = eval_df[eval_df["feature"] == feat]
    n_pos = int((sub["brier_improvement"] > 0).sum())
    n_ece_pos = int((sub["calibration_improvement"] > 0).sum())
    verdict = "PASS" if stability_decision[feat] else "FAIL"
    mean_brier = float(sub["brier_improvement"].mean())
    print(f"  {feat:<24} {verdict:<5} ({n_pos}/3 dBrier+, {n_ece_pos}/3 dECE+) mean_dBrier={mean_brier:+.5f}")

print("\\nFold detail:")
print(
    eval_df[["feature", "train_window", "test_season", "brier_improvement", "calibration_improvement", "passed_stability"]]
    .to_string(index=False, float_format=lambda x: f"{x:+.5f}")
)
''')


add("markdown", "m02g000h", """
## Phase 02g-g -- Write `feature_validation.csv`

Defensive append by `(feature, train_window, test_season)`, preserving rows
from 02a-02f.
""")


add("code", "c02g000i", '''
NEW_KEYS = set(zip(
    eval_df["feature"],
    eval_df["train_window"],
    eval_df["test_season"].astype(int),
))

if FEATURE_VALIDATION_CSV.exists():
    existing_df = pd.read_csv(FEATURE_VALIDATION_CSV, keep_default_na=False)
    existing_keys = list(zip(
        existing_df["feature"],
        existing_df["train_window"],
        existing_df["test_season"].astype(int),
    ))
    mask_keep = [k not in NEW_KEYS for k in existing_keys]
    n_displaced = len(existing_df) - sum(mask_keep)
    existing_df = existing_df[mask_keep].reset_index(drop=True)
    combined_df = pd.concat([existing_df, eval_df], ignore_index=True)
    print(f"existing feature_validation.csv rows: {len(existing_keys):,}")
    print(f"displaced existing 02g-key rows: {n_displaced:,}")
else:
    combined_df = eval_df.copy()
    print("feature_validation.csv absent; creating new file")

if "imputation_value" not in combined_df.columns:
    combined_df["imputation_value"] = ""

combined_df = combined_df.sort_values(
    ["feature_set_version", "feature", "train_window", "test_season"]
).reset_index(drop=True)

dups = combined_df.duplicated(subset=["feature", "train_window", "test_season"], keep=False)
assert not dups.any(), (
    "natural-key duplicate after append:\\n"
    f"{combined_df[dups][['feature', 'train_window', 'test_season', 'feature_set_version']]}"
)

combined_df.to_csv(FEATURE_VALIDATION_CSV, index=False)
print(f"[ok] wrote feature_validation.csv: {len(combined_df):,} rows ({len(eval_df)} from 02g)")
''')


add("markdown", "m02g000j", """
## Phase 02g-h -- Splice schema sidecar

Adds a sentinel-delimited 02g section with reduced-scope decisions, D10,
candidate-candidate correlations, verdicts, and cumulative PASS rollup.
""")


add("code", "c02g000k", '''
def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
        ).strip()
    except Exception as e:  # noqa: BLE001
        return f"<unavailable: {e}>"


def _fmt_delta(x: float) -> str:
    return f"{x:+.5f}"


now_text = time.strftime("%Y-%m-%d %H:%M:%S %Z").strip() or time.strftime("%Y-%m-%d %H:%M:%S")
commit_hash = _git_commit()

prevalence_rows = []
for feat in CANDIDATE_FEATURES:
    n = int(feature_matrix_df[feat].sum())
    prevalence_rows.append(f"| `{feat}` | {n:,} | {n / len(feature_matrix_df) * 100:.2f}% |")

d10_rows = []
for feat in CANDIDATE_FEATURES:
    d10_rows.append(f"| `{feat}` | {catA_mismatches[feat]:,} | Category A byte-identical |")

cand_corr_rows = []
for i, a in enumerate(CANDIDATE_FEATURES):
    for b in CANDIDATE_FEATURES[i + 1:]:
        rho = float(candidate_corr_df.loc[a, b])
        flag = " **HIGH**" if abs(rho) >= 0.6 else ""
        cand_corr_rows.append(f"| `{a}` | `{b}` | {rho:+.4f}{flag} |")

verdict_rows = []
for feat in CANDIDATE_FEATURES:
    feat_rows = eval_df[eval_df["feature"] == feat].sort_values("test_season")
    for _, r in feat_rows.iterrows():
        verdict_rows.append(
            f"| `{feat}` | {r['train_window']} -> test {int(r['test_season'])} | "
            f"{_fmt_delta(float(r['brier_improvement']))} | "
            f"{_fmt_delta(float(r['calibration_improvement']))} | "
            f"{'**PASS**' if bool(r['passed_stability']) else 'FAIL'} |"
        )

fv_after = pd.read_csv(FEATURE_VALIDATION_CSV, keep_default_na=False)
fv_after["brier_improvement"] = fv_after["brier_improvement"].astype(float)
fv_after["calibration_improvement"] = fv_after["calibration_improvement"].astype(float)
cumulative_validated: list[tuple[str, str, int, int]] = []
for (fsv, feat), grp in fv_after.groupby(["feature_set_version", "feature"]):
    n_pos = int((grp["brier_improvement"] > 0).sum())
    n_ece = int((grp["calibration_improvement"] > 0).sum())
    if n_pos >= 2:
        cumulative_validated.append((str(fsv), str(feat), n_pos, n_ece))
cumulative_validated.sort()
cumul_rows = [
    f"| `{feat}` | {fsv} | {n_b}/3 | {n_e}/3 |"
    for fsv, feat, n_b, n_e in cumulative_validated
]

corr_summary = (
    "The full cross-notebook 02g correlation matrix is generated after "
    "execution by `research/notebooks/_diag_02g_correlations.py` and written "
    "to `research/results/_02g_correlations.csv` for N03 prep."
)
corr_path = RESULTS_DIR / "_02g_correlations.csv"
full_corr_path = RESULTS_DIR / "_02g_full_correlation_matrix.csv"
if corr_path.exists():
    corr_diag_df = pd.read_csv(corr_path)
    corr_diag_df["abs_rho"] = corr_diag_df["abs_rho"].astype(float)
    corr_diag_df["pearson_rho"] = corr_diag_df["pearson_rho"].astype(float)
    high_diag_df = corr_diag_df[corr_diag_df["abs_rho"] >= 0.6].copy()
    top_diag_rows = []
    for _, rr in corr_diag_df.sort_values("abs_rho", ascending=False).head(10).iterrows():
        top_diag_rows.append(
            f"| `{rr['new_feature']}` | `{rr['validated_feature']}` | "
            f"{int(rr['n_pair_nonnull']):,} | {float(rr['pearson_rho']):+.4f} |"
        )
    high_lines = []
    if high_diag_df.empty:
        high_lines.append("No new-vs-validated `|rho| >= 0.6` pairs were found; no 02g `redundant_with` tags apply.")
    else:
        for _, rr in high_diag_df.sort_values("abs_rho", ascending=False).iterrows():
            high_lines.append(
                f"- `{rr['new_feature']}` vs `{rr['validated_feature']}`: "
                f"rho={float(rr['pearson_rho']):+.4f}, n={int(rr['n_pair_nonnull']):,}"
            )
    corr_summary = (
        "### Cross-notebook Pearson redundancy diagnostic\\n\\n"
        f"Artifact: `{corr_path.relative_to(REPO_ROOT)}`. "
        f"Full wide matrix: `{full_corr_path.relative_to(REPO_ROOT)}`.\\n\\n"
        f"New-vs-validated `|rho| >= 0.6` count: **{len(high_diag_df)}**.\\n\\n"
        + "\\n".join(high_lines)
        + "\\n\\nTop new-vs-validated correlations by absolute rho:\\n\\n"
        "| New feature | Validated feature | Pairwise n | Pearson rho |\\n"
        "|---|---|---:|---:|\\n"
        + "\\n".join(top_diag_rows)
    )

SECTION_BEGIN = "<!-- BEGIN: 02g context_week_home_neutral -->"
SECTION_END = "<!-- END: 02g context_week_home_neutral -->"

section_body = (
    f"""
## 02g -- Context: week phase + home/away/neutral (`{FEATURE_SET_VERSION}`)

**Section last writer:** `research/notebooks/02g_context_week_home_neutral.ipynb`
**Last writer commit:** `{commit_hash}`
**Last writer generation timestamp:** {now_text}

### Structural finding

Notebook 02g tested **5** context features (**3** season-phase dummies plus
**2** home/neutral flags). **2 of 5** passed R6 stability mechanically, but
both passes have per-fold Brier magnitudes at the noise floor:

- `season_phase_bowl`: **+0.00003 / -0.00070 / +0.00001** (mean dBrier
  essentially zero).
- `fav_is_home`: **-0.00153 / +0.00145 / +0.00015** (two folds with
  `|dBrier| < 0.0002`).

Per the `corrections_log.md` magnitude-skepticism threshold (**+0.005**),
neither feature should be treated as load-bearing signal in N03. The honest
finding is that pre-game-state context features (week-of-season,
home/away/neutral site) do **not** carry meaningful comeback-equity signal at
the trigger level. Both R6-pass features still enter N03 per the project's
L1/ablation-based pruning policy; expect both to be zeroed out or contribute
minimally.

Mechanistic correlation note: `season_phase_bowl` and `is_neutral_site`
correlate at **rho = +0.803**. This is structural rather than surprising:
bowl/postseason games are predominantly neutral-site games.

### Reduced scope

Weather / venue candidates (`is_dome`, `wind_mph`, `temp_f`) were deferred
before scaffold because the full Open-Meteo cache and authoritative venue
coordinate source are not present locally. See `research/future_features.md`
and `research/tech_debt.md` item 9.

### Candidate features (all Category A)

- `{'`, `'.join(CANDIDATE_FEATURES)}`

Source map:

- `season_phase_mid`, `season_phase_late`, `season_phase_bowl`: derived
  directly from `trigger_events.csv` (`season_type`, `week`).
- `fav_is_home`: derived directly from `trigger_events.csv` (`home_is_fav`).
- `is_neutral_site`: cached `/games.neutralSite` joined by `game_id`.

### D3/D4 week-of-season encoding

Regular weeks `1-4` are the omitted early-season reference. Regular weeks
`5-9` set `season_phase_mid = 1`; regular weeks `10-16` set
`season_phase_late = 1`; postseason rows set `season_phase_bowl = 1`.

### D5 home/away/neutral encoding

`fav_is_home` and `is_neutral_site` are separate binary indicators; away
favorite is the reference state.

### Prevalence and null counts

All five features have zero null rows on **{len(feature_matrix_df):,}**
in-scope triggers.

| Feature | Rows == 1 | % |
|---|---:|---:|
"""
    + "\\n".join(prevalence_rows)
    + """

### D10 diff-vs-leaky

The extractor accepts a filter-mode label for notebook consistency but does
not inspect plays or drives. Chrono and leaky builds are byte-identical:

| Feature | Mismatches | Verdict |
|---|---:|---|
"""
    + "\\n".join(d10_rows)
    + """

### Candidate-vs-candidate Pearson matrix summary

| Feature A | Feature B | Pearson rho |
|---|---|---:|
"""
    + "\\n".join(cand_corr_rows)
    + f"""

### Stability table per walk-forward folds (`feature_validation.csv`)

| Feature | Train window -> test season | dBrier test | dECE test | R6 stab |
|---|---|---:|---:|---|
"""
    + "\\n".join(verdict_rows)
    + f"""

### D12 cumulative PASS rollup after 02g

| Feature | Feature set version | Brier-positive folds | ECE-positive folds |
|---|---|---:|---:|
"""
    + "\\n".join(cumul_rows)
    + f"""

**Rows in rollup:** **{len(cumulative_validated):,}**.

{corr_summary}
"""
)

new_section = SECTION_BEGIN + "\\n" + section_body.rstrip() + "\\n" + SECTION_END

if FEATURE_VALIDATION_SCHEMA.exists():
    existing_text = FEATURE_VALIDATION_SCHEMA.read_text(encoding="utf-8")
    if SECTION_BEGIN in existing_text and SECTION_END in existing_text:
        start = existing_text.index(SECTION_BEGIN)
        end = existing_text.index(SECTION_END) + len(SECTION_END)
        updated = existing_text[:start] + new_section + existing_text[end:]
        print("[ok] spliced 02g section in place")
    else:
        updated = existing_text.rstrip() + "\\n\\n" + new_section + "\\n"
        print("[ok] appended 02g section")
else:
    updated = "# feature_validation.csv -- schema sidecar\\n\\n" + new_section + "\\n"
    print("[warn] schema sidecar absent; seeded minimal file")

FEATURE_VALIDATION_SCHEMA.write_text(updated, encoding="utf-8")
print(f"[ok] wrote feature_validation.schema.md ({len(updated):,} chars)")
''')


add("markdown", "m02g000l", """
## Phase 02g-i -- Summary and budget
""")


add("code", "c02g000m", '''
print("=" * 70)
print("Notebook 02g -- context week/home/neutral -- summary")
print("=" * 70)

print(f"\\nIn-scope trigger rows: {len(feature_matrix_df):,}")
print(f"Unique trigger games with cached /games neutralSite: {len(games_meta_by_id):,}")

print("\\nFeature prevalence:")
for feat in CANDIDATE_FEATURES:
    n = int(feature_matrix_df[feat].sum())
    print(f"  {feat:<24} {n:>6,} ({n / len(feature_matrix_df) * 100:5.2f}%)")

print("\\nD10 byte-identical mismatches:")
for feat, n_mis in catA_mismatches.items():
    print(f"  {feat:<24} {n_mis}")

print("\\nPer-feature x per-test-season results:")
print(f"  {'feature':<24} {'window->test':<18} {'d_brier':>10} {'d_ece':>10} {'stab':>6}")
for _, r in eval_df.sort_values(["feature", "test_season"]).iterrows():
    win = f"{r['train_window']}->{int(r['test_season'])}"
    print(f"  {r['feature']:<24} {win:<18} "
          f"{float(r['brier_improvement']):>+10.5f} "
          f"{float(r['calibration_improvement']):>+10.5f} "
          f"{'PASS' if bool(r['passed_stability']) else 'FAIL':>6}")

print("\\nR6 stability summary:")
for feat in CANDIDATE_FEATURES:
    sub = eval_df[eval_df["feature"] == feat]
    n_pos = int((sub["brier_improvement"] > 0).sum())
    n_ece = int((sub["calibration_improvement"] > 0).sum())
    print(f"  {feat:<24} {'PASS' if stability_decision[feat] else 'FAIL':<5} "
          f"({n_pos}/3 dBrier+, {n_ece}/3 dECE+)")

fv = pd.read_csv(FEATURE_VALIDATION_CSV, keep_default_na=False)
fv["brier_improvement"] = fv["brier_improvement"].astype(float)
cumul = []
for (fsv, feat), grp in fv.groupby(["feature_set_version", "feature"]):
    if int((grp["brier_improvement"] > 0).sum()) >= 2:
        cumul.append((fsv, feat))
print(f"\\nCumulative validated PASS groups after 02g: {len(cumul)}")

n_call_log_after = sum(1 for _ in CALL_LOG.open("r", encoding="utf-8")) - 1
print("\\nCall budget -- Notebook 02g execution:")
print(f"  call-log rows before: {n_call_log_before:,}")
print(f"  call-log rows after:  {n_call_log_after:,}")
print(f"  new call-log rows:    {n_call_log_after - n_call_log_before:,}")
assert n_call_log_after == n_call_log_before, (
    "02g reduced scope should not invoke CFBD/Open-Meteo helpers or append call-log rows"
)

print("\\n[ok] notebook 02g complete -- halt for diagnostics/report.")
''')


def _to_lines(s: str) -> list[str]:
    lines = s.split("\n")
    out = [ln + "\n" for ln in lines[:-1]]
    if lines[-1] != "":
        out.append(lines[-1])
    return out


def _cell_dict(cell_type: str, cell_id: str, src: str) -> dict:
    d: dict = {
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
print(f"[ok] wrote {OUT}  ({OUT.stat().st_size:,} bytes, {len(CELLS)} cells)")
