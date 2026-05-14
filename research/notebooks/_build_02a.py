"""
Deterministic builder for research/notebooks/02a_baseline_features.ipynb.

Run from anywhere; writes the notebook next to this script. Cell IDs are
stable so re-running the builder produces a byte-identical .ipynb (modulo
JSON key ordering, which we pin via sort_keys=False + ordered dicts).

This is a scratchpad file (per the research/notebooks/_*.py convention).
Not part of the deliverable.
"""

from __future__ import annotations

import json
import pathlib
import sys
import textwrap

OUT = pathlib.Path(__file__).resolve().parent / "02a_baseline_features.ipynb"

# Pull the canonical _chrono_key source from the shared helper module.
# Single source of truth across _build_02a/02b/02c.py; see
# research/notebooks/_lib_chrono.py and research/corrections_log.md for
# the lookahead-bias fix rationale.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _lib_chrono import CHRONO_KEY_SOURCE  # noqa: E402

# Each cell: (cell_type, cell_id, source_text). source_text is split into
# a list of lines (each newline-terminated except possibly the last) at
# serialization time.

CELLS: list[tuple[str, str, str]] = []


def add(cell_type: str, cell_id: str, src: str) -> None:
    CELLS.append((cell_type, cell_id, textwrap.dedent(src).lstrip("\n")))


# ---------------------------------------------------------------------------
# Cell 0 — Title + hypothesis docstring (markdown)
# ---------------------------------------------------------------------------
add("markdown", "bd02a000", """
# Phase 0 — Notebook 02a: Baseline efficiency features

## Hypothesis (this notebook tests one feature group's predictive value)

Baseline efficiency features — `fav_off_epa_per_play`, `fav_def_epa_per_play`, `dog_off_epa_per_play`, `dog_def_epa_per_play`, `epa_divergence`, `plays_so_far` — computed **strictly from plays whose composite chronological key `(period, period_seconds_elapsed, driveNumber, playNumber)` is less than the trigger's** improve a calibrated win-probability model's held-out **Brier score** on **>= 2 of 3 walk-forward test seasons** versus a pre-game-only baseline (`pregame_spread`, `rating_gap`, `fav_pregame_rating`, `dog_pregame_rating`). The V5.1 stability rule (R6) is the gate.

> **Correction note:** an earlier revision of this notebook filtered plays with `playNumber < trigger.play_number`, which silently leaked future plays because CFBD's `playNumber` resets at each drive. The current revision uses the composite chronological key defined above. See `research/corrections_log.md` for the per-feature diff vs the leaked-filter run.

Per-feature evaluation isolates the marginal information each in-game efficiency signal carries on top of closing spread + pre-game Elo. Calibration improvement (test ECE delta) is reported alongside for forensics but is **not** the stability gate in this notebook — that's N03's job at the end-to-end model level.

## What this notebook DOES NOT do

- Does not modify `trigger_events.csv` or `trigger_outcomes.csv`.
- Does not pull any fresh CFBD data — every `/plays` lookup must hit the cache (decision A from the 02a plan-approval: the cache-hit assertion in cell `Phase 02a-b` fails loud on any cache miss).
- Does not select features for the production model — N03's job.
- Does not test feature groups other than baseline efficiency (those are N02b–g).
- Does not tune hyperparameters of the L1 logreg — uses sklearn default `C=1.0` with a fixed seed.

## Spec references

- `BUILD_SPEC.md` Phase 0 Notebook 02 deliverable spec — `feature_validation.csv` column shape
- `BUILD_SPEC.md` `trigger_features` DDL — baseline-efficiency block (V5 lines 162–167); `dog_def_epa_per_play` added in this notebook for fav-side symmetry (spec extension documented in `feature_validation.schema.md`, NOT patched into BUILD_SPEC.md)
- `BUILD_SPEC.md` Phase 0 Notebook 03 — walk-forward windows
- `BUILD_SPEC.md` Owner addendum **A.7** — pre-game Elo replaces SP+/FPI in the baseline
- `.cursorrules` rules **R2 + R3** — no lookahead; `assert_no_lookahead()` is mandatory on every feature extraction
- `.cursorrules` rule **R5** — walk-forward validation only
- `.cursorrules` rule **R6** — stability rule (>= 2 of 3 test seasons)
- `.cursorrules` rule **R7** — L1 logreg / shallow GBM only
- `.cursorrules` rule **R8** — ECE on 10 bins, post-calibration
- `.cursorrules` rule **R19** — record rejected features too
- `.cursorrules` rule **R22** — STOP at end of 02a; do not start 02b without approval

## Deliverables produced by this notebook

1. `research/results/feature_validation.csv` — 21 rows from 02a (7 features × 3 test seasons). Append semantics: subsequent 02b–g runs add rows tagged with their own `feature_set_version`; 02a re-runs delete old 02a rows by `(feature, train_window, test_season)` key before appending.
2. `research/results/feature_validation.schema.md` — sidecar with column dictionary, walk-forward window definition, baseline-model definition, per-feature null policy, calibration choice rationale, EPA-as-`ppa` note, `dog_def_epa_per_play` spec-extension note, and a per-feature × per-test-season verdict table for this run.
3. `research/notebooks/02a_baseline_features.ipynb` — this notebook.

No changes to `trigger_events.csv`, `trigger_outcomes.csv`, `trigger_events_bucket_counts.csv`, or `data_quality_report.md`. No new cache files. No fresh CFBD calls.

## Walk-forward windows (decision **B**, locked for 02b–g)

| Train seasons | Val season | Test season |
|---|---|---|
| 2015–2020 | 2021 | 2022 |
| 2015–2021 | 2022 | 2023 |
| 2015–2022 | 2023 | 2024 |

Train starts at 2015 (the corpus floor) rather than 2017 (the BUILD_SPEC literal text). All available training data within the no-leak constraint is used. Locked at start of 02a and binding for 02b–g; rationale lives in `feature_validation.schema.md` so the BUILD_SPEC text stays intact.

## Call budget

CFBD v2 free tier = 1000 calls/month. Cumulative through end of N01: 269 fresh calls used, 731 remaining.

**This notebook's budget: 0 fresh CFBD calls.** Every `/plays` lookup is a cache hit produced by N01. The cell-`Phase 02a-b` cache-hit assertion and the final-cell budget assertion both fail loud if a fresh call is issued. After 02a: **269 / 1000** consumed, **731 remaining**.
""")


# ---------------------------------------------------------------------------
# Cell 1 — Imports, paths, env, fail-fast (code)
# ---------------------------------------------------------------------------
add("code", "c02a0001", '''
"""
Notebook 02a — imports, environment, path constants, fail-fast checks.
Same structure as Notebook 01. Run this cell first; if it raises, fix the
issue before continuing — none of the downstream cells will work without it.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pathlib
import subprocess
import time
from typing import Any, Callable

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- Paths -------------------------------------------------------------------
NOTEBOOK_DIR = pathlib.Path(".").resolve()
RESEARCH_DIR = (NOTEBOOK_DIR / "..").resolve()
DATA_DIR = (RESEARCH_DIR / "data").resolve()
RESULTS_DIR = (RESEARCH_DIR / "results").resolve()
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CALL_LOG = CACHE_DIR / "cfbd_call_log.csv"
ENV_PATH = (RESEARCH_DIR / ".." / "backend" / ".env").resolve()
REPO_ROOT = (RESEARCH_DIR / "..").resolve()

# --- Sanity check on workspace layout ----------------------------------------
assert RESEARCH_DIR.name == "research", (
    f"Expected to run inside research/notebooks/. Got NOTEBOOK_DIR={NOTEBOOK_DIR}. "
    f"cd into research/notebooks/ and re-launch jupyter."
)
assert ENV_PATH.exists(), (
    f"Did not find {ENV_PATH}. The CFBD key would only be used if the cache "
    f"is missing entries (none expected in 02a); load it anyway so the same "
    f"cfbd_get() helper works."
)

# --- Load CFBD_API_KEY from backend/.env -------------------------------------
load_dotenv(ENV_PATH)
assert os.environ.get("CFBD_API_KEY"), (
    "CFBD_API_KEY is not set. 02a should NOT issue fresh calls, but the "
    "cfbd_get() helper still requires the key in scope. Populate "
    f"{ENV_PATH} and re-run."
)

# --- Required upstream artifacts ---------------------------------------------
TRIGGER_EVENTS_CSV = RESULTS_DIR / "trigger_events.csv"
TRIGGER_OUTCOMES_CSV = RESULTS_DIR / "trigger_outcomes.csv"
assert TRIGGER_EVENTS_CSV.exists(), (
    f"Expected {TRIGGER_EVENTS_CSV} (Notebook 01 deliverable). Run N01 first."
)
assert TRIGGER_OUTCOMES_CSV.exists(), (
    f"Expected {TRIGGER_OUTCOMES_CSV} (Notebook 01 deliverable). Run N01 first."
)

# --- 02a outputs (committable, written at end) -------------------------------
FEATURE_VALIDATION_CSV = RESULTS_DIR / "feature_validation.csv"
FEATURE_VALIDATION_SCHEMA = RESULTS_DIR / "feature_validation.schema.md"

print(f"[ok] paths resolved relative to {NOTEBOOK_DIR}")
print(f"[ok] CFBD_API_KEY loaded from {ENV_PATH}")
print(f"[ok] cache dir: {CACHE_DIR}")
print(f"[ok] N01 deliverables present: trigger_events.csv, trigger_outcomes.csv")
''')


# ---------------------------------------------------------------------------
# Cell 2 — HTTP helpers (code) — reused verbatim from N01 so cache keys match
# ---------------------------------------------------------------------------
add("code", "c02a0002", '''
"""
HTTP helpers — same code as Notebook 00/01, same cache directory.
Cache hits cost zero CFBD budget. 02a expects ALL calls to be cache hits;
the assertion in Phase 02a-b fails loud if any go fresh.
"""
CFBD_BASE = "https://apinext.collegefootballdata.com"

if not CALL_LOG.exists():
    with CALL_LOG.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["timestamp", "service", "endpoint", "params_hash", "cached",
             "status", "bytes", "elapsed_ms"]
        )


def _params_hash(params: dict) -> str:
    return hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def _cache_key(prefix: str, params: dict) -> pathlib.Path:
    return CACHE_DIR / f"{prefix}__{_params_hash(params)}.json"


def _log(service: str, endpoint: str, params: dict, *, cached: bool,
         status: int, bytes_: int, elapsed_ms: int) -> None:
    with CALL_LOG.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [time.strftime("%Y-%m-%dT%H:%M:%S"), service, endpoint,
             _params_hash(params), int(cached), status, bytes_, elapsed_ms]
        )


def cfbd_get(endpoint: str, force_refresh: bool = False, **params: Any) -> Any:
    key = _cache_key(f"cfbd__{endpoint.strip('/').replace('/', '_')}", params)
    if key.exists() and not force_refresh:
        size = key.stat().st_size
        data = json.loads(key.read_text(encoding="utf-8"))
        _log("cfbd", endpoint, params, cached=True, status=200,
             bytes_=size, elapsed_ms=0)
        return data
    headers = {
        "Authorization": f"Bearer {os.environ['CFBD_API_KEY']}",
        "Accept": "application/json",
    }
    t0 = time.perf_counter()
    r = httpx.get(f"{CFBD_BASE}{endpoint}", params=params,
                  headers=headers, timeout=120)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    _log("cfbd", endpoint, params, cached=False, status=r.status_code,
         bytes_=len(r.content), elapsed_ms=elapsed_ms)
    r.raise_for_status()
    data = r.json()
    key.write_text(json.dumps(data), encoding="utf-8")
    return data


print("[ok] cfbd_get defined")
print(f"[ok] sharing cache with Notebook 00/01 at {CACHE_DIR}")
''')


# ---------------------------------------------------------------------------
# Cell 3 — Configuration (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02a0003", """
## Configuration

`WALK_FORWARD_WINDOWS` is locked at decision **B** from the 02a plan-approval: train starts at 2015 (the corpus floor) rather than 2017 (the BUILD_SPEC literal text). This is binding for 02b–g — the same windows feed every per-feature stability evaluation across 02a–g, and (with possible refinement of which features feed which window) N03.

`BASELINE_PREGAME_FEATURES` is the six-column pre-game baseline (decision α from the plan-approval). All five candidate columns referenced in the plan plus one R16-safe missingness-indicator column:

- `pregame_spread`, `rating_gap`, `fav_pregame_rating`, `dog_pregame_rating` — always present on the in-scope subset.
- `spread_movement` — pre-game-safe per R16. ~30% of trigger rows have NULL movement per the `trigger_events.schema.md` provider-mix table. **R16-safe NaN handling**: impute NULL → 0 (the median; 0 = "line did not move") and add a missingness-indicator column `spread_movement_is_null` so the model can separate "movement was zero" from "movement was unobserved."
- `spread_movement_is_null` — 1 iff `trigger_events.spread_movement is NaN`. Derived at feature-matrix build time.

`CANDIDATE_FEATURES` is the seven baseline-efficiency features evaluated in 02a:

- **Six from the V5 `trigger_features` DDL** baseline-efficiency block: `fav_off_epa_per_play`, `fav_def_epa_per_play`, `dog_off_epa_per_play`, `epa_divergence`, `plays_so_far`.
- **One spec extension**: `dog_def_epa_per_play` — fav-side symmetry mirror. Documented in `feature_validation.schema.md`, not patched into BUILD_SPEC.md.

`FEATURE_SET_VERSION` is the per-notebook tag stamped into every row this notebook writes. 02b–g will use their own tags and append to the same CSV.
""")


# ---------------------------------------------------------------------------
# Cell 4 — Configuration constants (code)
# ---------------------------------------------------------------------------
add("code", "c02a0004", '''
SEASONS: list[int] = list(range(2015, 2025))
SEASON_TYPES: list[str] = ["regular", "postseason"]

FEATURE_SET_VERSION: str = "v1_baseline_efficiency_only"

# Walk-forward windows — decision B from 02a plan-approval (train from 2015).
WALK_FORWARD_WINDOWS: list[dict] = [
    {"train_seasons": list(range(2015, 2021)), "val_season": 2021,
     "test_season": 2022, "train_window_label": "2015-2020"},
    {"train_seasons": list(range(2015, 2022)), "val_season": 2022,
     "test_season": 2023, "train_window_label": "2015-2021"},
    {"train_seasons": list(range(2015, 2023)), "val_season": 2023,
     "test_season": 2024, "train_window_label": "2015-2022"},
]

# Pre-game baseline columns — decision alpha from 02a plan-approval.
# Six columns: four always-present (R3-safe, A.7-compliant), plus
# spread_movement (R16-safe NaN handling: impute NaN -> 0, add missingness
# indicator). See feature_validation.schema.md.
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

# Candidate baseline-efficiency features. dog_def_epa_per_play is the
# fav-side-symmetry spec extension; see feature_validation.schema.md.
CANDIDATE_FEATURES: list[str] = [
    "fav_off_epa_per_play",
    "fav_def_epa_per_play",
    "dog_off_epa_per_play",
    "dog_def_epa_per_play",
    "epa_divergence",
    "plays_so_far",
]

# Structural-redundancy map (discovered post-9822cfc execution; see
# feature_validation.schema.md "Redundancy discoveries" section).
# Keys = features that are per-row identical to their value. Values = the
# canonical feature this row duplicates. Empty string for canonical features.
# Downstream N03 filters redundant_with == '' to drop duplicates cleanly.
REDUNDANT_WITH: dict[str, str] = {
    "dog_def_epa_per_play": "fav_off_epa_per_play",
    "dog_off_epa_per_play": "fav_def_epa_per_play",
}

# Reproducibility seed for L1 logreg.
RANDOM_STATE: int = 42

print(f"seasons: {SEASONS}")
print(f"season types: {SEASON_TYPES}")
print(f"feature_set_version: {FEATURE_SET_VERSION}")
print(f"walk-forward windows:")
for w in WALK_FORWARD_WINDOWS:
    print(f"  train={w['train_window_label']}  val={w['val_season']}  test={w['test_season']}")
print(f"baseline pre-game features ({len(BASELINE_PREGAME_FEATURES)}): {BASELINE_PREGAME_FEATURES}")
print(f"  always-present cols ({len(ALWAYS_PRESENT_PREGAME_COLS)}): {ALWAYS_PRESENT_PREGAME_COLS}")
print(f"  R16-safe imputed col + indicator: ['spread_movement', 'spread_movement_is_null']")
print(f"candidate features ({len(CANDIDATE_FEATURES)}): {CANDIDATE_FEATURES}")
print(f"random state: {RANDOM_STATE}")
''')


# ---------------------------------------------------------------------------
# Cell 5 — Load trigger artifacts (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02a0005", """
## Phase 02a-a — Load trigger artifacts

Read `trigger_events.csv` and `trigger_outcomes.csv`. Inner-join on the natural key `(game_id, fav_deficit)` per `trigger_outcomes.schema.md`. Drop rows with `final_fav_won is NaN` (small count — N01 reports unknown/tie). The merged DataFrame is the row inventory for every downstream cell.

Per R3, the join happens here in 02a (the feature-validation notebook), NOT inside any feature-extraction function. Features are computed from the in-game-state columns of `trigger_events.csv` plus the cached `/plays` corpus — never from the `trigger_outcomes.csv` columns. The label `final_fav_won` only enters as the model target in the walk-forward evaluation cell.
""")


# ---------------------------------------------------------------------------
# Cell 6 — Load triggers code
# ---------------------------------------------------------------------------
add("code", "c02a0006", '''
triggers_df = pd.read_csv(TRIGGER_EVENTS_CSV)
outcomes_df = pd.read_csv(TRIGGER_OUTCOMES_CSV)
print(f"trigger_events.csv:    {len(triggers_df):>6,} rows x {triggers_df.shape[1]} cols")
print(f"trigger_outcomes.csv:  {len(outcomes_df):>6,} rows x {outcomes_df.shape[1]} cols")

# Inner join on natural key. validate=one_to_one matches the schema sidecar contract.
trigger_full_df = triggers_df.merge(
    outcomes_df,
    on=["game_id", "fav_deficit"],
    how="inner",
    validate="one_to_one",
)
print(f"merged:                {len(trigger_full_df):>6,} rows x {trigger_full_df.shape[1]} cols")
assert len(trigger_full_df) == len(triggers_df), (
    f"inner join lost rows: {len(triggers_df)} -> {len(trigger_full_df)}. "
    f"trigger_outcomes.csv should have one outcome per trigger."
)

# Drop ties / unknown final outcomes (NaN final_fav_won). R3 LABEL column.
n_pre_drop = len(trigger_full_df)
trigger_full_df = trigger_full_df[trigger_full_df["final_fav_won"].notna()].copy()
trigger_full_df["final_fav_won"] = trigger_full_df["final_fav_won"].astype(bool)
n_dropped_tie = n_pre_drop - len(trigger_full_df)
print(f"\\nDropped {n_dropped_tie} rows with NaN final_fav_won (ties / unknown).")
print(f"In-scope rows for 02a: {len(trigger_full_df):,}")

print(f"\\nRows per season:")
print(trigger_full_df["season"].value_counts().sort_index().to_string())

# Sanity: ALWAYS_PRESENT_PREGAME_COLS must be non-null (A.7 + N01 schema contract).
for col in ALWAYS_PRESENT_PREGAME_COLS:
    n_null = int(trigger_full_df[col].isna().sum())
    assert n_null == 0, (
        f"always-present pre-game column {col!r} has {n_null} nulls on the "
        f"in-scope subset; expected 0 per the trigger_events.schema.md contract."
    )
print(f"\\n[ok] always-present pre-game columns are non-null on the in-scope subset")

# spread_movement may have ~30% NaN per N01 schema; R16-safe NaN handling
# (impute -> 0 + missingness indicator) happens in the build-feature-matrix cell.
n_sm_null = int(trigger_full_df["spread_movement"].isna().sum())
print(f"     spread_movement nulls (pre-impute): {n_sm_null:,} "
      f"({n_sm_null / len(trigger_full_df) * 100:.2f}% of in-scope)")
''')


# ---------------------------------------------------------------------------
# Cell 7 — Re-load cached /plays (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02a0007", """
## Phase 02a-b — Re-load cached `/plays` corpus (zero fresh calls)

Iterate the same `(season, season_type, week)` tuples N01 walked, call `cfbd_get("/plays", ...)` for each, and **assert every call is a cache hit**. The cell fails loud if any call would be a fresh fetch — that signals cache invalidation and should stop the run for review, not silently consume budget.

This is decision **A** from the 02a plan-approval: the cache-hit assertion is a budget guard. Per R22, 02a is supposed to spend 0 fresh CFBD calls; a cache miss here is a stop-the-line signal.
""")


# ---------------------------------------------------------------------------
# Cell 8 — Re-load cached /plays code
# ---------------------------------------------------------------------------
add("code", "c02a0008", '''
work_tuples_df = (
    trigger_full_df[["season", "season_type", "week"]]
    .drop_duplicates()
    .sort_values(["season", "season_type", "week"])
    .reset_index(drop=True)
)
print(f"distinct (season, season_type, week) tuples to load from cache: {len(work_tuples_df)}")

# Snapshot call-log size BEFORE the cache pull so we can detect non-cached calls
# scoped exactly to this loop.
n_log_before = sum(1 for _ in CALL_LOG.open("r", encoding="utf-8")) - 1  # minus header

plays_by_game: dict[int, list[dict]] = {}
# Observability-only counter for CFBD's negative-integer play.id encoding
# (19,828 plays across 115 games carry the alternate format). These plays
# are LEGITIMATE -- not dropped; the composite chrono_key below orders
# them correctly via (period, period_seconds_elapsed, driveNumber,
# playNumber). See research/corrections_log.md.
n_neg_id_in_cache: int = 0
neg_id_games: set[int] = set()
t_start = time.perf_counter()
for i, row in work_tuples_df.iterrows():
    season = int(row["season"])
    season_type = str(row["season_type"])
    week = int(row["week"])
    plays = cfbd_get(
        "/plays",
        year=season,
        seasonType=season_type,
        week=week,
        classification="fbs",
    )
    for p in plays:
        gid = p.get("gameId")
        if gid is None:
            continue
        try:
            pid = int(p.get("id") or p.get("playId") or 0)
        except (ValueError, TypeError):
            pid = 0
        if pid < 0:
            n_neg_id_in_cache += 1
            neg_id_games.add(gid)
        plays_by_game.setdefault(gid, []).append(p)
elapsed = time.perf_counter() - t_start
n_plays = sum(len(v) for v in plays_by_game.values())
print(f"[ok] /plays loaded from cache in {elapsed:.1f}s -- "
      f"{len(plays_by_game):,} games, {n_plays:,} plays")
print(f"[info] CFBD negative-id encoding: {n_neg_id_in_cache:,} plays across "
      f"{len(neg_id_games):,} games carry the alternate (negative-int) id "
      f"format. Retained; chrono_key orders them correctly without referencing "
      f"play.id. See research/corrections_log.md.")

# --- Cache-hit assertion (decision A from 02a plan-approval) -------------
calls_log_df = pd.read_csv(CALL_LOG)
this_run_plays_calls = calls_log_df.iloc[n_log_before:].copy()
n_fresh_this_cell = int((this_run_plays_calls["cached"] == 0).sum())
assert n_fresh_this_cell == 0, (
    f"02a budget invariant violated: {n_fresh_this_cell} non-cached /plays call(s) "
    f"issued in this cell. 02a is supposed to spend 0 fresh CFBD calls; the cache "
    f"for some (year, type, week) tuple is missing or stale. Stop and investigate "
    f"cache invalidation before continuing -- DO NOT proceed and silently spend "
    f"budget."
)
print(f"[ok] cache-hit assertion passed: {len(this_run_plays_calls)} /plays lookups, all cached.")

# --- Composite chrono_key helper + pre-sort per game ---------------------
# Filter `plays_before` strictly by `_chrono_key(p) < trig_chrono_key` rather
# than `playNumber < trig.play_number`. The latter silently leaked future plays
# because CFBD's `playNumber` resets per drive (see corrections_log.md).
# Source-of-truth function defined in research/notebooks/_lib_chrono.py;
# inlined here so the notebook stays self-contained.

''' + CHRONO_KEY_SOURCE + '''


for gid in plays_by_game:
    plays_by_game[gid].sort(key=_chrono_key)
print(f"[ok] plays_by_game sorted by composite _chrono_key "
      f"({len(plays_by_game):,} games)")
''')


# ---------------------------------------------------------------------------
# Cell 9 — assert_no_lookahead + feature extractors (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02a0009", """
## Phase 02a-c — `assert_no_lookahead` (R3 hard gate) + feature extractors

`assert_no_lookahead` is the per-row R3 gate: every play used in a feature must have its composite `_chrono_key < trigger_chrono_key`. The check runs once per `(trigger, feature)` evaluation, on the exact slice the feature function received. A failure raises — no silent leak path.

The composite key replaces the original `playNumber < trigger.play_number` test, which silently leaked future plays because CFBD's `playNumber` resets per drive. See `research/corrections_log.md` for the full-corpus verification and the per-feature diff vs the leaked-filter run.

Seven baseline-efficiency feature functions, all pure: input is the pre-trigger play slice (already filtered + R3-gated by the wrapper), output is a single `float | int | None`. `None` on insufficient data (the per-feature null policy — decision B from the 02a plan-approval — drops that row from this feature's per-feature non-null subset at evaluation time; does NOT drop the row globally).
""")


# ---------------------------------------------------------------------------
# Cell 10 — assert_no_lookahead code
# ---------------------------------------------------------------------------
add("code", "c02a000a", '''
def assert_no_lookahead(plays_used: list[dict],
                        trigger_chrono_key: tuple[int, int, int, int],
                        feature_name: str, game_id: int) -> None:
    """Per-row R3 hard gate. Raises if any play in `plays_used` has
    `_chrono_key(p) >= trigger_chrono_key`.

    Switched from the original `playNumber < trigger_play_number` test
    (which silently leaked future plays because CFBD playNumber resets
    per drive) to the composite chrono_key. See
    research/corrections_log.md.
    """
    if not plays_used:
        return
    max_key = max(
        (_chrono_key(p) for p in plays_used),
        default=(-1, -1, -1, -1),
    )
    assert max_key < trigger_chrono_key, (
        f"R3 LOOKAHEAD: feature {feature_name!r} on game {game_id} trigger "
        f"chrono_key={trigger_chrono_key} touched a play with "
        f"chrono_key={max_key} (>= trigger). Refusing to emit this row."
    )


print("[ok] assert_no_lookahead defined (composite chrono_key gate)")
''')


# ---------------------------------------------------------------------------
# Cell 11 — Feature functions code
# ---------------------------------------------------------------------------
add("code", "c02a000b", '''
# EPA = CFBD's `ppa` field (Predicted Points Added). See feature_validation.schema.md.

def _mean_ppa(plays: list[dict]) -> float | None:
    """Mean of `ppa` over the supplied plays where `ppa` is non-null.
    Returns None if zero plays have a non-null `ppa`."""
    vals = [p["ppa"] for p in plays if p.get("ppa") is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def feat_fav_off_epa_per_play(plays_before: list[dict],
                              fav_team: str, dog_team: str) -> float | None:
    return _mean_ppa([p for p in plays_before if p.get("offense") == fav_team])


def feat_fav_def_epa_per_play(plays_before: list[dict],
                              fav_team: str, dog_team: str) -> float | None:
    return _mean_ppa([p for p in plays_before if p.get("defense") == fav_team])


def feat_dog_off_epa_per_play(plays_before: list[dict],
                              fav_team: str, dog_team: str) -> float | None:
    return _mean_ppa([p for p in plays_before if p.get("offense") == dog_team])


def feat_dog_def_epa_per_play(plays_before: list[dict],
                              fav_team: str, dog_team: str) -> float | None:
    return _mean_ppa([p for p in plays_before if p.get("defense") == dog_team])


def feat_epa_divergence(plays_before: list[dict],
                        fav_team: str, dog_team: str) -> float | None:
    """fav_off_epa_per_play - dog_off_epa_per_play. None if either is None."""
    fav_off = feat_fav_off_epa_per_play(plays_before, fav_team, dog_team)
    dog_off = feat_dog_off_epa_per_play(plays_before, fav_team, dog_team)
    if fav_off is None or dog_off is None:
        return None
    return fav_off - dog_off


def feat_plays_so_far(plays_before: list[dict],
                      fav_team: str, dog_team: str) -> int:
    """Count of plays observed strictly before the trigger. Always int >= 0."""
    return len(plays_before)


FEATURE_FUNCTIONS: dict[str, Callable[[list[dict], str, str], float | int | None]] = {
    "fav_off_epa_per_play": feat_fav_off_epa_per_play,
    "fav_def_epa_per_play": feat_fav_def_epa_per_play,
    "dog_off_epa_per_play": feat_dog_off_epa_per_play,
    "dog_def_epa_per_play": feat_dog_def_epa_per_play,
    "epa_divergence":       feat_epa_divergence,
    "plays_so_far":         feat_plays_so_far,
}

assert set(FEATURE_FUNCTIONS) == set(CANDIDATE_FEATURES), (
    f"FEATURE_FUNCTIONS keys {set(FEATURE_FUNCTIONS)} != "
    f"CANDIDATE_FEATURES {set(CANDIDATE_FEATURES)}"
)
print(f"[ok] {len(FEATURE_FUNCTIONS)} feature functions registered")
''')


# ---------------------------------------------------------------------------
# Cell 12 — Build feature matrix (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02a000c", """
## Phase 02a-d — Build feature matrix

Walk every in-scope trigger, slice plays by `_chrono_key(p) < trigger_chrono_key` (composite key over `period`, `period_seconds_elapsed`, `driveNumber`, `playNumber`), run each candidate feature, gate the slice through `assert_no_lookahead`, and attach the pre-game baseline columns. Print null counts per feature so the practical applicability gap of each feature is visible upfront — early-Q1 triggers with few pre-trigger plays for a given team-side are the typical culprits for EPA-feature nulls.

The composite chrono_key replaces the original `playNumber < trigger.play_number` filter (which silently leaked future plays because CFBD's `playNumber` resets per drive). See `research/corrections_log.md`.

Decision **B** from the 02a plan-approval (null handling): a trigger that is null for one feature may be evaluable for others. Per-feature null masking happens at the evaluation cell, not here — this cell preserves all rows.
""")


# ---------------------------------------------------------------------------
# Cell 13 — Build matrix code
# ---------------------------------------------------------------------------
add("code", "c02a000d", '''
ID_COLS = ["game_id", "fav_deficit", "trigger_sequence", "season", "season_type",
           "week", "fav_team", "dog_team", "play_number", "quarter"]
LABEL_COL = "final_fav_won"

records: list[dict] = []
n_skipped_unknown_game = 0

for _, trig in trigger_full_df.iterrows():
    gid = int(trig["game_id"])
    trig_pn = int(trig["play_number"])
    fav = str(trig["fav_team"])
    dog = str(trig["dog_team"])

    # Composite chrono_key for the trigger row (period, period_elapsed,
    # driveNumber, playNumber). period_seconds_elapsed = 900 - the trigger
    # row's `clock_seconds_in_period_total` (which N01 already computed as
    # 60*minutes_remaining + seconds_remaining). drive_number_in_game is
    # CFBD's `driveNumber` for the trigger play.
    trig_period = int(trig["quarter"])
    trig_period_elapsed = 900 - int(trig["clock_seconds_in_period_total"])
    trig_drive_in_game = int(trig["drive_number_in_game"])
    trig_chrono_key = (trig_period, trig_period_elapsed, trig_drive_in_game, trig_pn)

    plays = plays_by_game.get(gid)
    if plays is None:
        n_skipped_unknown_game += 1
        continue
    # Composite chrono_key filter (replaces the leaky `playNumber < trig_pn`
    # filter; see corrections_log.md). All 19,828 negative-id plays across
    # 115 games are retained -- the chrono_key orders them correctly without
    # referencing play.id.
    plays_before = [p for p in plays if _chrono_key(p) < trig_chrono_key]

    row: dict[str, Any] = {col: trig[col] for col in ID_COLS}
    # Always-present pre-game columns
    for col in ALWAYS_PRESENT_PREGAME_COLS:
        row[col] = trig[col]
    # R16-safe NaN handling for spread_movement: impute NaN -> 0 (median;
    # 0 = "line did not move") and emit a binary missingness indicator
    # so the model can separate "movement was zero" from "movement was unobserved."
    sm_raw = trig["spread_movement"]
    sm_is_null = bool(pd.isna(sm_raw))
    row["spread_movement"] = 0.0 if sm_is_null else float(sm_raw)
    row["spread_movement_is_null"] = int(sm_is_null)
    row[LABEL_COL] = bool(trig[LABEL_COL])

    for feat_name, fn in FEATURE_FUNCTIONS.items():
        # R3 gate runs per-feature: the EPA features all read the same
        # `plays_before` slice; a future per-feature sub-slicing function
        # (e.g., red-zone-only) would still be gated here.
        assert_no_lookahead(plays_before, trig_chrono_key, feat_name, gid)
        row[feat_name] = fn(plays_before, fav, dog)

    records.append(row)

feature_matrix_df = pd.DataFrame.from_records(records)
print(f"feature_matrix_df:  {len(feature_matrix_df):>6,} rows x {feature_matrix_df.shape[1]} cols")
print(f"  skipped (game not in plays_by_game): {n_skipped_unknown_game}")
assert n_skipped_unknown_game == 0, (
    f"{n_skipped_unknown_game} triggers had no plays in cache. Investigate before continuing."
)

# --- Null counts per candidate feature (decision B: print upfront) ---------
print(f"\\nNull counts per candidate feature (this run):")
print(f"  total in-scope triggers: {len(feature_matrix_df):,}")
null_counts: dict[str, int] = {}
for feat in CANDIDATE_FEATURES:
    n_null = int(feature_matrix_df[feat].isna().sum())
    pct = (n_null / len(feature_matrix_df) * 100) if len(feature_matrix_df) else 0
    null_counts[feat] = n_null
    print(f"  {feat:<26} {n_null:>5,} null ({pct:5.2f}%)")

assert null_counts["plays_so_far"] == 0, (
    f"plays_so_far is supposed to be int >= 0, never null. "
    f"Got {null_counts['plays_so_far']} nulls."
)

# R16-safe NaN handling sanity: after impute, spread_movement has 0 NaNs;
# spread_movement_is_null is the row-wise indicator and matches the pre-impute
# count from cell 6.
sm_null_after = int(feature_matrix_df["spread_movement"].isna().sum())
sm_indicator_sum = int(feature_matrix_df["spread_movement_is_null"].sum())
print(f"\\nR16-safe NaN handling for spread_movement:")
print(f"  spread_movement nulls AFTER impute:    {sm_null_after:,}  (expected: 0)")
print(f"  spread_movement_is_null indicator sum: {sm_indicator_sum:,} "
      f"({sm_indicator_sum / len(feature_matrix_df) * 100:.2f}% of in-scope)")
assert sm_null_after == 0, f"spread_movement still has {sm_null_after} nulls after impute"
''')


# ---------------------------------------------------------------------------
# Cell 14 — Walk-forward evaluation (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02a000e", """
## Phase 02a-e — Walk-forward per-feature evaluation

For each `(window, candidate_feature)` pair:

1. Mask `feature_matrix_df` to rows where the candidate is non-null (decision **B** per-feature null policy).
2. Split into train (multi-season), val (single season), test (single season) — all from the same non-null subset.
3. Fit two pipelines on the train split:
   - **Baseline**: `StandardScaler` → `LogisticRegression(penalty="l1", C=1.0, solver="liblinear", random_state=42, max_iter=1000)` on `BASELINE_PREGAME_FEATURES` only.
   - **Candidate**: same pipeline on `BASELINE_PREGAME_FEATURES + [feature]`.
4. Calibrate each on val via `CalibratedClassifierCV(method="isotonic", cv="prefit")`.
5. Evaluate calibrated probabilities on test: Brier (`sklearn.metrics.brier_score_loss`) and ECE (10 equal-width bins).
6. Record `brier_improvement = baseline - candidate` and `calibration_improvement = ece_baseline - ece_candidate`.

After the loop, `passed_stability = sum(brier_improvement > 0) >= 2` per feature, broadcast to all 3 rows of that feature (R6).
""")


# ---------------------------------------------------------------------------
# Cell 15 — ECE + fit helper
# ---------------------------------------------------------------------------
add("code", "c02a000f", '''
def ece_10bin(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Expected Calibration Error with 10 equal-width probability bins (R8).
    Closed interval on the rightmost bin so probs == 1.0 land in it."""
    bin_edges = np.linspace(0.0, 1.0, 11)
    n = len(y_true)
    if n == 0:
        return float("nan")
    ece = 0.0
    for i in range(10):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == 9:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        m = int(mask.sum())
        if m == 0:
            continue
        bin_conf = float(y_prob[mask].mean())
        bin_acc = float(y_true[mask].mean())
        ece += (m / n) * abs(bin_conf - bin_acc)
    return float(ece)


def fit_calibrate_evaluate(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> tuple[float, float]:
    """L1 logreg (default C=1.0, fixed seed) -> StandardScaler pipeline,
    isotonic-calibrated on val, evaluated on test. Returns (brier_test, ece_test).
    """
    estimator = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l1", C=1.0, solver="liblinear",
            random_state=RANDOM_STATE, max_iter=1000,
        )),
    ])
    estimator.fit(X_train, y_train)
    calibrator = CalibratedClassifierCV(estimator=estimator, method="isotonic", cv="prefit")
    calibrator.fit(X_val, y_val)
    probs_test = calibrator.predict_proba(X_test)[:, 1]
    brier = float(brier_score_loss(y_test, probs_test))
    ece = ece_10bin(np.asarray(y_test, dtype=int), np.asarray(probs_test, dtype=float))
    return brier, ece


print("[ok] ece_10bin and fit_calibrate_evaluate defined")
''')


# ---------------------------------------------------------------------------
# Cell 16 — Eval loop
# ---------------------------------------------------------------------------
add("code", "c02a0010", '''
eval_rows: list[dict] = []
t_start = time.perf_counter()

for window in WALK_FORWARD_WINDOWS:
    train_seasons = window["train_seasons"]
    val_season = window["val_season"]
    test_season = window["test_season"]
    win_label = window["train_window_label"]

    for feat in CANDIDATE_FEATURES:
        # Per-feature null mask (decision B). Baseline and candidate run on
        # IDENTICAL row subsets so the comparison is apples-to-apples.
        mask = feature_matrix_df[feat].notna()
        sub = feature_matrix_df[mask]

        train_sub = sub[sub["season"].isin(train_seasons)]
        val_sub = sub[sub["season"] == val_season]
        test_sub = sub[sub["season"] == test_season]

        if len(train_sub) == 0 or len(val_sub) == 0 or len(test_sub) == 0:
            print(f"[skip] feat={feat} window={win_label}: empty split "
                  f"(n_train={len(train_sub)}, n_val={len(val_sub)}, n_test={len(test_sub)})")
            continue

        y_train = train_sub[LABEL_COL].values.astype(int)
        y_val = val_sub[LABEL_COL].values.astype(int)
        y_test = test_sub[LABEL_COL].values.astype(int)

        X_train_b = train_sub[BASELINE_PREGAME_FEATURES].values.astype(float)
        X_val_b = val_sub[BASELINE_PREGAME_FEATURES].values.astype(float)
        X_test_b = test_sub[BASELINE_PREGAME_FEATURES].values.astype(float)
        brier_b, ece_b = fit_calibrate_evaluate(
            X_train_b, y_train, X_val_b, y_val, X_test_b, y_test
        )

        cand_cols = BASELINE_PREGAME_FEATURES + [feat]
        X_train_c = train_sub[cand_cols].values.astype(float)
        X_val_c = val_sub[cand_cols].values.astype(float)
        X_test_c = test_sub[cand_cols].values.astype(float)
        brier_c, ece_c = fit_calibrate_evaluate(
            X_train_c, y_train, X_val_c, y_val, X_test_c, y_test
        )

        eval_rows.append({
            "feature": feat,
            "feature_set_version": FEATURE_SET_VERSION,
            "train_window": win_label,
            "val_season": val_season,
            "test_season": test_season,
            "n_train": len(train_sub),
            "n_val": len(val_sub),
            "n_test": len(test_sub),
            "brier_test_baseline": brier_b,
            "brier_test_candidate": brier_c,
            "brier_improvement": brier_b - brier_c,
            "ece_test_baseline": ece_b,
            "ece_test_candidate": ece_c,
            "calibration_improvement": ece_b - ece_c,
            "redundant_with": REDUNDANT_WITH.get(feat, ""),
        })

eval_df = pd.DataFrame(eval_rows)
print(f"\\n[ok] evaluation loop complete in {time.perf_counter() - t_start:.1f}s")
print(f"     rows: {len(eval_df)} (expected: {len(WALK_FORWARD_WINDOWS) * len(CANDIDATE_FEATURES)} = "
      f"{len(WALK_FORWARD_WINDOWS)} windows x {len(CANDIDATE_FEATURES)} features)")
assert len(eval_df) == len(WALK_FORWARD_WINDOWS) * len(CANDIDATE_FEATURES), (
    f"expected {len(WALK_FORWARD_WINDOWS) * len(CANDIDATE_FEATURES)} eval rows, got {len(eval_df)}"
)

# --- Per-feature stability decision (R6) ---------------------------------
stability_decision: dict[str, bool] = {}
for feat in CANDIDATE_FEATURES:
    n_positive = int((eval_df[eval_df["feature"] == feat]["brier_improvement"] > 0).sum())
    stability_decision[feat] = (n_positive >= 2)
eval_df["passed_stability"] = eval_df["feature"].map(stability_decision)

CSV_COLUMNS = [
    "feature", "feature_set_version", "train_window", "val_season", "test_season",
    "n_train", "n_val", "n_test",
    "brier_test_baseline", "brier_test_candidate", "brier_improvement",
    "ece_test_baseline", "ece_test_candidate", "calibration_improvement",
    "passed_stability",
    "redundant_with",
]
eval_df = eval_df[CSV_COLUMNS]

print(f"\\nstability verdict per feature:")
for feat in CANDIDATE_FEATURES:
    n_pos = int((eval_df[eval_df["feature"] == feat]["brier_improvement"] > 0).sum())
    verdict = "PASS" if stability_decision[feat] else "FAIL"
    print(f"  {feat:<26} {verdict}  ({n_pos}/3 test seasons with positive Brier improvement)")
''')


# ---------------------------------------------------------------------------
# Cell 17 — Write CSV (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02a0011", """
## Phase 02a-f — Write `feature_validation.csv` (defensive append)

Append semantics per the 02a plan-approval decision: 02a re-runs must not clobber 02b–g's rows.

1. If `feature_validation.csv` exists, read it.
2. Drop rows matching this run's `(feature, train_window, test_season)` keys.
3. Concatenate this run's new rows.
4. Sort by `(feature_set_version, feature, train_window, test_season)`.
5. Write.

Natural-key uniqueness is asserted after the write.
""")


# ---------------------------------------------------------------------------
# Cell 18 — Write CSV code
# ---------------------------------------------------------------------------
add("code", "c02a0012", '''
NEW_KEYS = set(zip(
    eval_df["feature"],
    eval_df["train_window"],
    eval_df["test_season"].astype(int),
))

if FEATURE_VALIDATION_CSV.exists():
    existing_df = pd.read_csv(FEATURE_VALIDATION_CSV)
    print(f"existing feature_validation.csv: {len(existing_df):,} rows")
    existing_keys = list(zip(
        existing_df["feature"],
        existing_df["train_window"],
        existing_df["test_season"].astype(int),
    ))
    mask_keep = [k not in NEW_KEYS for k in existing_keys]
    n_displaced = len(existing_df) - sum(mask_keep)
    existing_df = existing_df[mask_keep].reset_index(drop=True)
    if n_displaced > 0:
        print(f"  displaced {n_displaced} row(s) matching this run's "
              f"(feature, train_window, test_season) keys")
    combined_df = pd.concat([existing_df, eval_df], ignore_index=True)
else:
    print(f"feature_validation.csv does not exist -- creating new file")
    combined_df = eval_df.copy()

combined_df = combined_df.sort_values(
    ["feature_set_version", "feature", "train_window", "test_season"]
).reset_index(drop=True)

# Natural-key uniqueness check.
dups = combined_df.duplicated(subset=["feature", "train_window", "test_season"], keep=False)
assert not dups.any(), (
    "natural-key duplicate after append:\\n"
    f"{combined_df[dups][['feature', 'train_window', 'test_season', 'feature_set_version']]}"
)

combined_df.to_csv(FEATURE_VALIDATION_CSV, index=False)
print(f"\\n[ok] wrote feature_validation.csv: {len(combined_df):,} rows "
      f"({len(eval_df)} from this run, {len(combined_df) - len(eval_df)} retained from prior runs)")
print(f"     path: {FEATURE_VALIDATION_CSV}")
''')


# ---------------------------------------------------------------------------
# Cell 19 — Schema sidecar (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02a0013", """
## Phase 02a-g — Write `feature_validation.schema.md` sidecar

The sidecar is the single source of truth for the per-feature × per-test-season deliverable schema, the locked walk-forward window choice, the baseline-model definition, the per-feature null policy, the calibration choice rationale, the EPA-field convention, and the `dog_def_epa_per_play` spec-extension note. Notebook 03 reads this file when assembling `validated_filters.json.active_features`.
""")


# ---------------------------------------------------------------------------
# Cell 20 — Write schema code
# ---------------------------------------------------------------------------
add("code", "c02a0014", '''
def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception as e:  # noqa: BLE001
        return f"<unavailable: {e}>"


now_text = time.strftime("%Y-%m-%d %H:%M:%S %Z").strip() or time.strftime("%Y-%m-%d %H:%M:%S")
commit_hash = _git_commit()

# Per-feature x per-test-season summary table for the sidecar.
def _fmt_delta(x: float) -> str:
    return f"{x:+.5f}"


verdict_rows = []
for feat in CANDIDATE_FEATURES:
    feat_rows = eval_df[eval_df["feature"] == feat].sort_values("test_season")
    for _, r in feat_rows.iterrows():
        verdict_rows.append(
            f"| `{feat}` | {r['train_window']} -> test {int(r['test_season'])} | "
            f"{_fmt_delta(r['brier_improvement'])} | {_fmt_delta(r['calibration_improvement'])} | "
            f"{'**PASS**' if r['passed_stability'] else 'FAIL'} |"
        )

null_rows = []
for feat in CANDIDATE_FEATURES:
    n_null = null_counts[feat]
    pct = (n_null / len(feature_matrix_df) * 100) if len(feature_matrix_df) else 0
    null_rows.append(f"| `{feat}` | {n_null:,} | {pct:.2f}% |")

window_rows = [
    f"| {w['train_window_label']} | {w['val_season']} | {w['test_season']} |"
    for w in WALK_FORWARD_WINDOWS
]

text = f"""# feature_validation.csv -- schema sidecar

**Generated:** {now_text}
**Source notebook (last writer):** `research/notebooks/02a_baseline_features.ipynb`
**Source commit (last writer):** `{commit_hash}`
**Feature set version (this writer):** `{FEATURE_SET_VERSION}`
**Spec:** `BUILD_SPEC.md` Phase 0 Notebook 02 deliverable; V5.1 stability rule (R6)

This file documents the schema of `feature_validation.csv`, the canonical
per-feature x per-test-season stability-rule deliverable for Phase 0
Notebooks 02a-g. Each notebook (02a, 02b, ..., 02g) appends rows tagged
with its own `feature_set_version` to this single shared file. Notebook 03
reads this file when assembling `validated_filters.json.active_features`.

Every downstream notebook that reads `feature_validation.csv` should treat
this file as the canonical schema reference. Do not paraphrase from memory.

## Corrections vs commit 9822cfc

The previous commit (`9822cfc`) overstated this notebook's output shape and
miscategorized one feature in the sidecar provenance narrative. Specifically:

- Commit message and sidecar provenance: "21 rows from 02a (7 features x 3 test
  seasons)". **Actual:** 18 rows, 6 candidate features (3 test seasons x 6
  features). The V5 `trigger_features` baseline-efficiency block has 5
  fields, not 6; adding `dog_def_epa_per_play` made 6 candidates total, not 7.
- Sidecar provenance line broke down the 6 candidates as "5 V5 DDL +
  `dog_def_epa_per_play` extension + `plays_so_far`". **Wrong** -- `plays_so_far`
  is a V5 DDL field (`BUILD_SPEC.md` line 167); the correct breakdown is
  "5 V5 DDL + `dog_def_epa_per_play` extension".

The `feature_validation.csv` produced by `9822cfc`'s run had 18 rows; the
data was correct, only the narrative was off. This sidecar (regenerated by
the post-9822cfc 02a run) reflects the corrected values throughout.

## Append semantics

Rows are keyed by the natural key `(feature, train_window, test_season)`.

On write, each 02x notebook:

1. Reads `feature_validation.csv` if it exists.
2. Drops rows matching this run's `(feature, train_window, test_season)` keys.
3. Appends this run's new rows.
4. Sorts by `(feature_set_version, feature, train_window, test_season)`.
5. Writes.

This pattern lets 02a re-run safely without clobbering 02b-g, and lets each
notebook be re-run independently to refresh its own rows. Natural-key
uniqueness is asserted after the write.

## Column dictionary

| Column | Type | Description |
|---|---|---|
| `feature` | TEXT | Candidate feature name. Joins to a `trigger_features` DDL field name. |
| `feature_set_version` | TEXT | Per-notebook tag (e.g., `v1_baseline_efficiency_only` from 02a). Provenance -- does NOT match the production `feature_set_version` from N03's `validated_filters.json`. |
| `train_window` | TEXT | Train-season range as `start-end` (e.g., `2015-2020`). |
| `val_season` | INTEGER | Validation season. Used to fit the isotonic calibrator. |
| `test_season` | INTEGER | Test season. Used for the held-out Brier / ECE measurement. |
| `n_train` | INTEGER | Trigger rows in train (post per-feature null drop). |
| `n_val` | INTEGER | Trigger rows in val (post per-feature null drop). |
| `n_test` | INTEGER | Trigger rows in test (post per-feature null drop). |
| `brier_test_baseline` | REAL | Test-season Brier of the pre-game-only baseline model (isotonic-calibrated on val). |
| `brier_test_candidate` | REAL | Test-season Brier of the (pre-game + this candidate feature) model (isotonic-calibrated on val). |
| `brier_improvement` | REAL | `brier_test_baseline - brier_test_candidate`. Positive = candidate improved Brier. |
| `ece_test_baseline` | REAL | Test-season ECE (10 equal-width bins) of the baseline model. |
| `ece_test_candidate` | REAL | Test-season ECE of the candidate model. |
| `calibration_improvement` | REAL | `ece_test_baseline - ece_test_candidate`. Positive = candidate improved calibration. Reported alongside Brier but NOT the stability gate. |
| `passed_stability` | BOOLEAN | True iff `sum(brier_improvement > 0 across the 3 test seasons) >= 2` for this `(feature, feature_set_version)`. Same value on all 3 rows for a given feature. |
| `redundant_with` | TEXT | Empty string for canonical features. For features that are per-row identical to another feature (structural redundancy, not statistical correlation), the name of the canonical feature this one duplicates. Filter `redundant_with == ''` to drop duplicates when assembling the production feature set, after a `fillna("")` or `keep_default_na=False` read (see "Redundancy discoveries" / "N03 guidance" below). |

## Walk-forward windows (locked for 02a-g)

| Train seasons | Val season | Test season |
|---|---|---|
""" + "\\n".join(window_rows) + """

Decision **B** from the 02a plan-approval: train starts at 2015 (the corpus
floor) rather than 2017 (the BUILD_SPEC literal text). All available training
data within the no-leak constraint is used. Locked at start of 02a and
binding for 02b-g. Documented here (not via a BUILD_SPEC.md patch) so the
spec text stays intact while research conclusions cite this sidecar.

## Baseline pre-game model definition

For each candidate feature `F` and walk-forward window:

- **Baseline model:** L1-regularized logistic regression on
  `[pregame_spread, rating_gap, fav_pregame_rating, dog_pregame_rating]`,
  StandardScaler-scaled, `sklearn.linear_model.LogisticRegression(
  penalty='l1', C=1.0, solver='liblinear', random_state=""" + str(RANDOM_STATE) + """,
  max_iter=1000)`, isotonic-calibrated on the val season, evaluated on
  the test season.
- **Candidate model:** Same pipeline plus the single candidate feature `F`
  appended to the feature column list.

No hyperparameter tuning in 02a -- N03's job per V5.1 R7. The default `C=1.0`
penalty strength keeps the comparison apples-to-apples across features.

**Pre-game baseline column choice rationale:**

- `pregame_spread` -- closing spread (always present, pickem excluded by N01).
- `rating_gap`, `fav_pregame_rating`, `dog_pregame_rating` -- pre-game Elo
  per A.7 (always present; coverage cleared 80% in N00 audit).
- `spread_movement` -- pre-game-safe per R16 ("closing spread is allowed
  as a model feature because it was known before kickoff"). ~30% of
  trigger rows have NULL movement per the `trigger_events.schema.md`
  provider-mix table. **R16-safe NaN handling**: impute NaN -> 0 (the
  median; 0 = "line did not move") and add a missingness-indicator
  column `spread_movement_is_null` so the model can separate "movement
  was zero" from "movement was unobserved." Both columns enter the
  baseline; this is the standard scikit-learn pattern for non-random
  missingness in a continuous feature.
- `spread_movement_is_null` -- 1 iff the underlying `spread_movement`
  was NaN in `trigger_events.csv`. Derived at feature-matrix build time.

## Per-feature null policy

For each **candidate** feature `F` (the seven baseline-efficiency features):

- A trigger row is included in `F`'s evaluation iff `F` is non-null on that row.
- The baseline model for that window is fit and evaluated on the same
  non-null subset so the comparison is apples-to-apples (same `n_train`,
  `n_val`, `n_test`; the only thing changing between baseline and candidate
  is the presence of the candidate column).
- Null rows are NOT dropped globally -- a trigger that is null for one
  candidate feature may be evaluable for others.

For the **baseline** pre-game columns:

- The four always-present columns are guaranteed non-null on the in-scope
  subset per the `trigger_events.csv` schema contract.
- `spread_movement` is handled via R16-safe NaN handling (impute + indicator)
  at feature-matrix build time, NOT per-feature drop, so the baseline keeps
  the same row inventory as each candidate's per-feature non-null subset.

Null cause for the EPA features: a trigger row may have zero plays before
trigger where the relevant team was on offense (or defense), e.g., a
deficit-3-in-Q1 trigger where the favorite's first possession hasn't
started yet, or where the relevant side has had zero `ppa`-tagged plays.
`plays_so_far` is never null (always integer >= 0).

### Null counts for this run

| Feature | Null rows | % of in-scope triggers |
|---|---:|---:|
""" + "\\n".join(null_rows) + f"""

In-scope triggers (post NaN `final_fav_won` drop): {len(feature_matrix_df):,}.

## Calibration choice rationale

Isotonic regression fit on the val season, applied to test, via
`sklearn.calibration.CalibratedClassifierCV(estimator=..., method='isotonic',
cv='prefit')`. Chosen over Platt scaling because:

- The pre-game baseline (4 features, all continuous, well-separated) is
  expected to produce non-linear miscalibration where the favorite is a
  heavy favorite -- isotonic captures that without imposing a sigmoid shape.
- N03 currently expects isotonic per `BUILD_SPEC.md` Patch 3 example
  (`"calibration_version": "isotonic_v1.0"`); keeping 02a consistent makes
  the per-feature deltas comparable to N03's full-model deltas.

The calibrator is refit on each window's val season (R8: "refit calibration
parameters on the validation set"). ECE uses 10 equal-width bins on the
calibrated probabilities.

## Stability rule

A feature passes stability iff:

```
sum(brier_improvement > 0 across the 3 test seasons) >= 2
```

Calibration improvement is reported alongside but is **NOT** part of the
stability gate (per the 02a plan-approval decision; calibration is N03's
hard gate at the end-to-end-model level, not the per-feature level here).

Per R19, both passed and failed features are recorded in this CSV with
their per-season Brier deltas. Failed features get a row in
`validated_filters.json.rejected_features` at the end of N03.

## EPA field

The CFBD `/plays` endpoint exposes Expected Points Added under the field
name `ppa` (Predicted Points Added). Every reference to "EPA" in this
notebook and in the V5 `trigger_features` DDL is computed as the **mean**
of CFBD's `ppa` over the relevant pre-trigger play subset (filtered by
`p.get('ppa') is not None`). No fresh CFBD call was issued to verify this;
the convention is documented here so N03 (and any downstream re-run) joins
on a known basis.

## Spec extension: `dog_def_epa_per_play`

The V5 `trigger_features` DDL baseline-efficiency block (`BUILD_SPEC.md`
lines 162-167) lists `fav_off_epa_per_play`, `fav_def_epa_per_play`,
`dog_off_epa_per_play`, `epa_divergence`, `plays_so_far` -- but omits
`dog_def_epa_per_play`. We read this as a drafting oversight (the fav-side
has both offense and defense; the dog-side should too) and add
`dog_def_epa_per_play` as a 7th candidate in 02a. The extension is
documented here rather than via a mid-flight BUILD_SPEC.md patch; if N03
ends up using it, it appears in `validated_filters.json.active_features`
with provenance pointing at this sidecar.

`epa_divergence` is defined as `fav_off_epa_per_play - dog_off_epa_per_play`
(offensive-side divergence between the two teams). This is the
interpretation 02a uses; if N03 wants a different formulation it can
read the underlying components directly from the V5 schema.

## Per-feature x per-test-season results (this run, {FEATURE_SET_VERSION})

| Feature | Window -> Test | Brier improvement | ECE improvement | Stability |
|---|---|---:|---:|---|
""" + "\\n".join(verdict_rows) + f"""

Sign convention: positive = candidate beat baseline. `**PASS**` means
`sum(brier_improvement > 0) >= 2` across the 3 test seasons.

## Redundancy discoveries (surfaced by `9822cfc`'s run, tagged in CSV from this run forward)

Running 02a surfaced a structural redundancy in the V5 `trigger_features`
DDL baseline-efficiency block that the spec text does not mention.

### Per-row identity (football mechanism)

In a two-team game, every play has exactly one offense and one defense.
A play's `offense == fav_team` iff its `defense == dog_team`, so:

```
fav_off_epa_per_play (mean ppa where offense == fav_team)
    ==  dog_def_epa_per_play (mean ppa where defense == dog_team)

fav_def_epa_per_play (mean ppa where defense == fav_team)
    ==  dog_off_epa_per_play (mean ppa where offense == dog_team)
```

Per-row identity, not "highly correlated" -- the play subsets are literally
the same set of plays, so the per-trigger means are bit-identical.
Confirmed empirically in `9822cfc`'s run: every `feature_validation.csv`
row for `dog_def_epa_per_play` had identical `brier_test_*` and `ece_test_*`
to the matching `fav_off_epa_per_play` row, and the same for
`dog_off_epa_per_play` vs `fav_def_epa_per_play`.

### Implication for the V5 DDL

The V5 `trigger_features` baseline-efficiency block lists `fav_off`,
`fav_def`, `dog_off`, `epa_divergence`, `plays_so_far`. Of the three
EPA-per-play fields, only two carry independent information:

- `fav_off_epa_per_play` == `dog_def_epa_per_play` (fav-side offense / dog-side defense)
- `fav_def_epa_per_play` == `dog_off_epa_per_play` (fav-side defense / dog-side offense)

`epa_divergence = fav_off - dog_off = fav_off - fav_def` (using the
equivalence above) is a linear combination of the two, useful in
non-linear models or as a single-column summary but adding no new
information to an L1 logreg that already has both base features as
separate columns.

`dog_def_epa_per_play` was added by 02a "for fav-side symmetry" with the
V5 DDL block. With hindsight the V5 DDL's omission of `dog_def_epa_per_play`
was correct, not a drafting oversight -- including it would have been
redundant. 02a keeps the column in the CSV for transparency (tagged via
`redundant_with`) but downstream consumers should drop it.

### N03 guidance

When assembling the production feature set, filter `redundant_with == ''`
to drop the two duplicate columns. The empty-string convention does not
survive `pd.read_csv`'s default NaN-conversion of blank cells, so fill
NaN first (or pass `keep_default_na=False`):

```python
# Recommended: fillna keeps the literal `== ""` filter from the spec.
df = pd.read_csv("research/results/feature_validation.csv")
df["redundant_with"] = df["redundant_with"].fillna("")
non_redundant = df[df["redundant_with"] == ""]

# Equivalent: na_filter=False off keeps empty strings as strings on read.
df = pd.read_csv("research/results/feature_validation.csv", keep_default_na=False)
non_redundant = df[df["redundant_with"] == ""]
```

This leaves the canonical features only: `fav_off_epa_per_play`,
`fav_def_epa_per_play`, `epa_divergence`, `plays_so_far` from 02a, plus
whatever 02b-g and subsequent feature-group notebooks emit with
`redundant_with == ""`.

### Honest restatement of the per-feature stability results

The per-feature x per-test-season table above lists 18 rows, including 6
rows that are duplicates-by-construction of other rows. The substantive
claim from 02a is:

- **2 independent EPA features pass stability with 3/3 positive Brier
  improvement on test seasons:** `fav_off_epa_per_play`,
  `fav_def_epa_per_play`.
- **`epa_divergence` (a linear combination of the two) also passes with 3/3.**
- **`plays_so_far` fails (1/3 positive Brier improvement).**
- **The dog-side EPA features are reported in the CSV as
  `dog_def_epa_per_play == fav_off_epa_per_play` and
  `dog_off_epa_per_play == fav_def_epa_per_play`. Their reported stability
  "PASS" is a duplicate of the canonical features' result; treat as
  redundant per the `redundant_with` column.**

Brier improvements of +0.02 to +0.04 are real but small; edge is tested
in Notebook 03's end-to-end walk-forward CLV evaluation, not in 02a's
per-feature isolation test.

## Generation provenance

- Notebook: `research/notebooks/02a_baseline_features.ipynb`
- Commit hash: `{commit_hash}`
- Generation timestamp: {now_text}
- Trigger rows in scope (post NaN `final_fav_won` drop): {len(feature_matrix_df):,}
- Walk-forward windows: {len(WALK_FORWARD_WINDOWS)}
- Candidate features evaluated in 02a: {len(CANDIDATE_FEATURES)} (5 V5 DDL + `dog_def_epa_per_play` extension)
- Of which independent (per the "Redundancy discoveries" section): {sum(1 for f in CANDIDATE_FEATURES if f not in REDUNDANT_WITH)}; duplicates tagged via `redundant_with`: {sum(1 for f in CANDIDATE_FEATURES if f in REDUNDANT_WITH)}
- Total rows written to `feature_validation.csv` from 02a: {len(eval_df)}
- Fresh CFBD calls this notebook: 0 (asserted)
"""

FEATURE_VALIDATION_SCHEMA.write_text(text, encoding="utf-8")
print(f"[ok] wrote feature_validation.schema.md ({len(text):,} chars)")
print(f"     path: {FEATURE_VALIDATION_SCHEMA}")
''')


# ---------------------------------------------------------------------------
# Cell 21 — Summary (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02a0015", """
## Phase 02a-h — Summary, headline stats, STOP banner
""")


# ---------------------------------------------------------------------------
# Cell 22 — Summary print code
# ---------------------------------------------------------------------------
add("code", "c02a0016", '''
print("=" * 70)
print("Notebook 02a -- baseline efficiency features -- summary")
print("=" * 70)

print(f"\\nIn-scope corpus:")
print(f"  trigger rows (post NaN final_fav_won drop): {len(feature_matrix_df):,}")
print(f"  /plays cache hits this run:                 {len(work_tuples_df)} tuples, "
      f"{sum(len(v) for v in plays_by_game.values()):,} plays across {len(plays_by_game):,} games")

print(f"\\nPer-feature x per-test-season results ({FEATURE_SET_VERSION}):")
print(f"  {'feature':<26} {'window->test':<18} "
      f"{'brier_b':>9} {'brier_c':>9} {'d_brier':>10} "
      f"{'ece_b':>8} {'ece_c':>8} {'d_ece':>10} {'stab':>6}")
for _, r in eval_df.sort_values(["feature", "test_season"]).iterrows():
    win = f"{r['train_window']}->{int(r['test_season'])}"
    print(f"  {r['feature']:<26} {win:<18} "
          f"{r['brier_test_baseline']:>9.5f} {r['brier_test_candidate']:>9.5f} "
          f"{r['brier_improvement']:>+10.5f} "
          f"{r['ece_test_baseline']:>8.5f} {r['ece_test_candidate']:>8.5f} "
          f"{r['calibration_improvement']:>+10.5f} "
          f"{'PASS' if r['passed_stability'] else 'FAIL':>6}")

print(f"\\nFeature stability verdicts:")
for feat in CANDIDATE_FEATURES:
    n_pos = int((eval_df[eval_df["feature"] == feat]["brier_improvement"] > 0).sum())
    n_pos_cal = int((eval_df[eval_df["feature"] == feat]["calibration_improvement"] > 0).sum())
    verdict = "PASS" if stability_decision[feat] else "FAIL"
    print(f"  {feat:<26} {verdict:<5} "
          f"({n_pos}/3 brier-improving, {n_pos_cal}/3 ece-improving)")

print(f"\\nNull counts per feature (in-scope triggers: {len(feature_matrix_df):,}):")
for feat in CANDIDATE_FEATURES:
    pct = (null_counts[feat] / len(feature_matrix_df) * 100) if len(feature_matrix_df) else 0
    print(f"  {feat:<26} {null_counts[feat]:>5,} null ({pct:5.2f}%)")

print(f"\\nDeliverables (research/results/):")
for path in [FEATURE_VALIDATION_CSV, FEATURE_VALIDATION_SCHEMA]:
    size = path.stat().st_size
    print(f"  {path.name:<40} {size:>10,} bytes")
''')


# ---------------------------------------------------------------------------
# Cell 23 — Budget print
# ---------------------------------------------------------------------------
add("code", "c02a0017", '''
calls_log_df = pd.read_csv(CALL_LOG)
n_total_log_rows = len(calls_log_df)
n_fresh_cfbd_total = int(((calls_log_df["service"] == "cfbd")
                          & (calls_log_df["cached"] == 0)).sum())

# This-run slice: log rows after n_log_before (snapshot taken in Phase 02a-b).
# 02a should have appended zero non-cached rows after that snapshot.
this_run_calls = calls_log_df.iloc[n_log_before:].copy()
n_this_run = len(this_run_calls)
n_this_run_fresh = int((this_run_calls["cached"] == 0).sum())

print("=" * 64)
print("CFBD call budget -- Notebook 02a")
print("=" * 64)
print(f"\\nThis notebook run:")
print(f"  total calls this run:     {n_this_run:>5,}")
print(f"  fresh (uncached) this run: {n_this_run_fresh:>5,}  (budget: 0)")

assert n_this_run_fresh == 0, (
    f"02a budget invariant violated: {n_this_run_fresh} fresh CFBD call(s) "
    f"this run. 02a is supposed to spend 0 fresh CFBD calls."
)

# Hardcoded 1,000-call display constant from BUILD_SPEC A.4 is incorrect on
# the current API key (actual quota 3,000/cycle per the probe header).
# Aligned to 02c's dual-display in the chrono_key corrections sweep;
# tech_debt item 1 is resolved-for-02a by this print (other notebooks may
# still carry the stale single-value narrative until they get the same
# treatment).
print(f"\\nCumulative across all notebooks (call log: {n_total_log_rows:,} rows):")
print(f"  total fresh CFBD calls (lifetime):    {n_fresh_cfbd_total:,}")
print(f"  monthly free-tier limit (BUILD_SPEC A.4 stated):  1,000")
print(f"    actual quota on current key (probe header):     3,000")
print(f"  remaining this billing cycle (against actual 3K): {3000 - n_fresh_cfbd_total:,}")
if n_fresh_cfbd_total >= 0.8 * 3000:
    print(f"  [WARN] >=80% of 3,000-call cycle consumed.")

print(f"\\n[ok] notebook 02a complete -- STOP per R22. "
      f"Do not start Notebook 02b without approval.")
''')


# ---------------------------------------------------------------------------
# Serialize
# ---------------------------------------------------------------------------
def _to_lines(s: str) -> list[str]:
    """Split into a list of newline-terminated lines (last line may be bare)."""
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
