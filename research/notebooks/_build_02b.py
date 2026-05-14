"""
Deterministic builder for research/notebooks/02b_opening_drive_shock.ipynb.

Run from anywhere; writes the notebook next to this script. Cell IDs are
stable so re-running the builder produces a byte-identical .ipynb (modulo
JSON key ordering, which we pin via sort_keys=False + ordered dicts).

Mirrors the architecture of _build_02a.py:
  - 8 phases (02b-a through 02b-h), ~24 cells.
  - Same cache-hit-assertion contract.
  - Same defensive-append pattern for feature_validation.csv.
  - Sidecar uses sentinel-delimited splicing so 02a's content
    (Corrections, Redundancy discoveries, etc.) is preserved verbatim.

This is a scratchpad file (per the research/notebooks/_*.py convention).
Not part of the deliverable.
"""

from __future__ import annotations

import json
import pathlib
import sys
import textwrap

OUT = pathlib.Path(__file__).resolve().parent / "02b_opening_drive_shock.ipynb"

# Pull the canonical _chrono_key source from the shared helper module.
# Single source of truth across _build_02a/02b/02c.py; see
# research/notebooks/_lib_chrono.py and research/corrections_log.md for
# the lookahead-bias fix rationale.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _lib_chrono import CHRONO_KEY_SOURCE  # noqa: E402

CELLS: list[tuple[str, str, str]] = []


def add(cell_type: str, cell_id: str, src: str) -> None:
    CELLS.append((cell_type, cell_id, textwrap.dedent(src).lstrip("\n")))


# ---------------------------------------------------------------------------
# Cell 0 — Title + hypothesis docstring (markdown)
# ---------------------------------------------------------------------------
add("markdown", "bd02b000", """
# Phase 0 — Notebook 02b: Opening-drive shock features

## Hypothesis (this notebook tests one feature group's predictive value)

Opening-drive-shock features -- `dog_received_opening_kickoff`,
`dog_scored_on_opening_drive`, `opening_drive_was_td`,
`opening_drive_was_explosive_td`, `opening_drive_yards`,
`opening_drive_plays`, `opening_drive_seconds`, `fav_def_epa_first_drive`,
`fav_def_epa_after_first_drive`, `defense_stabilized_flag` -- computed
**strictly from plays / drives observable at the trigger play** improve
a calibrated win-probability model's held-out **Brier score** on
**>= 2 of 3 walk-forward test seasons** versus the same pre-game-only
baseline used in 02a (`pregame_spread`, `rating_gap`, `fav_pregame_rating`,
`dog_pregame_rating`, `spread_movement`, `spread_movement_is_null`).
The V5.1 stability rule (R6) is the gate.

> **Correction note:** an earlier revision of this notebook filtered plays with `playNumber < trigger.play_number`. CFBD `playNumber` resets per drive, so that filter (a) silently leaked future plays from drives > trig_drive with low playNumber and (b) silently truncated drive 1 (and other completed drives) on triggers where `trig_pn` was smaller than late-drive plays' playNumber. The current revision uses the composite chrono_key `(period, period_seconds_elapsed, driveNumber, playNumber)`. Drive-summary features that read `drives_for_game` directly are unaffected; the four extractors that iterate `plays_before` (`opening_drive_was_explosive_td`, `fav_def_epa_first_drive`, `fav_def_epa_after_first_drive`, and the derived `defense_stabilized_flag`) are. See `research/corrections_log.md` for the per-feature diff and the truncation-impact diagnostics in this notebook's Phase 02b-d2 cell.

This is the **per-feature** stability claim. The narrative V5.1 hypothesis
("dog opening-drive scoring reverts more than sustained dog dominance") is
Notebook 03's end-to-end job; 02b only isolates each candidate feature's
marginal Brier contribution on top of the pre-game baseline.

## What this notebook DOES NOT do

- Does not modify `trigger_events.csv` or `trigger_outcomes.csv`.
- Does not pull any fresh CFBD data -- every `/plays` AND `/drives` lookup
  must hit the cache produced by N01. The cache-hit assertion in cell
  `Phase 02b-b` fails loud on any cache miss (decision A from 02a's
  plan-approval, carried forward).
- Does not select features for the production model -- N03's job.
- Does not test feature groups other than opening-drive shock (those are
  02a / 02c-g).
- Does not tune hyperparameters of the L1 logreg -- uses sklearn default
  `C=1.0` with a fixed seed, identical to 02a.

## Spec references

- `BUILD_SPEC.md` Phase 0 Notebook 02 deliverable spec -- `feature_validation.csv` column shape
- `BUILD_SPEC.md` `trigger_features` DDL -- opening-drive-shock block (V5 lines 169-179)
- `BUILD_SPEC.md` Phase 0 Notebook 03 -- walk-forward windows
- `.cursorrules` rules **R2 + R3** -- no lookahead; `assert_no_lookahead()` is mandatory on every feature extraction
- `.cursorrules` rule **R5** -- walk-forward validation only
- `.cursorrules` rule **R6** -- stability rule (>= 2 of 3 test seasons)
- `.cursorrules` rule **R7** -- L1 logreg / shallow GBM only
- `.cursorrules` rule **R8** -- ECE on 10 bins, post-calibration
- `.cursorrules` rule **R19** -- record rejected features too
- `.cursorrules` rule **R22** -- STOP at end of 02b; do not start 02c without approval

## Plan-time pre-execution redundancy audit

Per the 02b plan-approval, every candidate feature pair was audited at
plan time for structural identity (same play subset + same arithmetic
=> bit-identical values across the trigger corpus). Verdict: **zero
structural duplicates among 02b's 10 candidates.** Two *conditional*
identities were flagged for the sidecar (distinct from the
`redundant_with` bit-identical tags from 02a):

1. `fav_def_epa_after_first_drive` == `fav_def_epa_per_play` on triggers
   where `dog_received_opening_kickoff == 0` (fav was on defense only
   after drive 1; no fav-defense plays exist in drive 1 on that subset).
2. First-drive EPA features have offense-vs-defense per-row identity
   *within* drive 1 (a play's `offense == dog` iff its `defense == fav`),
   the same identity that drove 02a's `dog_def_epa_per_play` redundancy
   discovery, but restricted to the drive-1 play subset.

Neither warrants a `redundant_with` tag (the identities hold only on
subsets of triggers, not bit-identically across the full corpus). Both
are documented in the sidecar's "Redundancy discoveries" section as
**conditional identities**.

## Plan-time candidate-vs-trigger-fields audit

Some 02b candidates are partially deducible from `trigger_events.csv`
fields alone, on the subset of triggers where the trigger play is
itself part of drive 1. Of the 11,416 triggers, **1,860 (16.3%) fire on
drive 1** (`drive_number_in_game == 1`); the other 9,556 (83.7%) fire
on drive 2 or later, where drive-1 outcomes are NOT readable from
`trigger_events` fields alone (the trigger row carries a single
score snapshot at trigger time, not a per-drive decomposition).

Under D4's null policy, all drive-summary features (`opening_drive_was_td`,
`_was_explosive_td`, `_yards`, `_plays`, `_seconds`,
`dog_scored_on_opening_drive`) are NULL for the 1,860 drive-1 triggers
(lookahead-unsafe to read the full drive's summary when the drive is in
progress). So for the evaluable 9,556 drive-2+ triggers, **none of the
02b candidates are materially deducible from `trigger_events` fields
alone** -- the drives JSON is required to reconstruct drive-1 outcomes.

`dog_received_opening_kickoff` is the lone partial exception:
deducible from `(drive_number_in_game == 1) AND (possession_team == dog_team)`
on the drive-1 subset (1,749 of 1,860 = 94% match), but those are
exactly the rows the null policy drops. On the drive-2+ subset (where
the feature IS evaluated), the trigger row's possession field reflects
later possession state, not opening-drive offense.

Documented for interpretation: if a 02b candidate passes stability and
turns out to be partially deducible, the marginal information is in the
drive-1 vs drive-2+ partition, not in the candidate alone.

## Deliverables produced by this notebook

1. `research/results/feature_validation.csv` -- adds 30 rows from 02b
   (10 features x 3 test seasons), tagged `feature_set_version =
   v1_opening_drive_shock`. 02a's 18 rows (`v1_baseline_efficiency_only`)
   are preserved; defensive-append by `(feature, train_window, test_season)` key.
2. `research/results/feature_validation.schema.md` -- splices a
   sentinel-delimited "02b - Opening-drive shock" section into the existing
   sidecar. 02a's sections (Corrections, Redundancy discoveries, etc.)
   are preserved verbatim.
3. `research/notebooks/02b_opening_drive_shock.ipynb` -- this notebook.

No changes to `trigger_events.csv`, `trigger_outcomes.csv`,
`trigger_events_bucket_counts.csv`, `data_quality_report.md`,
`budget_reconciliation.md`, `tech_debt.md`. No new cache files. No fresh
CFBD calls.

## Walk-forward windows (decision **B**, locked in 02a; carried forward verbatim)

| Train seasons | Val season | Test season |
|---|---|---|
| 2015-2020 | 2021 | 2022 |
| 2015-2021 | 2022 | 2023 |
| 2015-2022 | 2023 | 2024 |

## Baseline (decision **alpha**, locked in 02a; carried forward verbatim)

`BASELINE_PREGAME_FEATURES = [pregame_spread, rating_gap, fav_pregame_rating,
dog_pregame_rating, spread_movement, spread_movement_is_null]`. Same
R16-safe NaN handling for `spread_movement` (impute NaN -> 0; emit
binary missingness indicator).

## Call budget

CFBD v2 free tier per the documented BUILD_SPEC A.4 = 1,000 calls/month;
the actual quota on the current API key is 3,000/cycle per
`research/results/budget_reconciliation.md`. The hardcoded `1,000`
display constant is tracked as item 1 in `research/tech_debt.md`.

**This notebook's budget: 0 fresh CFBD calls.** Every `/plays` AND
`/drives` lookup is a cache hit produced by N01. The cell `Phase 02b-b`
cache-hit assertion and the final-cell budget assertion both fail loud
if a fresh call is issued. Lifetime audited count is 253/3000 going into
this notebook per `research/data/cache/cfbd_call_log.csv`.
""")


# ---------------------------------------------------------------------------
# Cell 1 — Imports, paths, env, fail-fast (code)
# ---------------------------------------------------------------------------
add("code", "c02b0001", '''
"""
Notebook 02b -- imports, environment, path constants, fail-fast checks.
Same structure as Notebook 02a. Run this cell first; if it raises, fix
the issue before continuing -- none of the downstream cells will work
without it.
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
    f"is missing entries (none expected in 02b); load it anyway so the same "
    f"cfbd_get() helper works."
)

# --- Load CFBD_API_KEY from backend/.env -------------------------------------
load_dotenv(ENV_PATH)
assert os.environ.get("CFBD_API_KEY"), (
    "CFBD_API_KEY is not set. 02b should NOT issue fresh calls, but the "
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

# --- 02b outputs (committable, written at end) -------------------------------
# Same files as 02a -- 02b appends to feature_validation.csv and splices a
# new section into feature_validation.schema.md. 02a's content is preserved.
FEATURE_VALIDATION_CSV = RESULTS_DIR / "feature_validation.csv"
FEATURE_VALIDATION_SCHEMA = RESULTS_DIR / "feature_validation.schema.md"

print(f"[ok] paths resolved relative to {NOTEBOOK_DIR}")
print(f"[ok] CFBD_API_KEY loaded from {ENV_PATH}")
print(f"[ok] cache dir: {CACHE_DIR}")
print(f"[ok] N01 deliverables present: trigger_events.csv, trigger_outcomes.csv")
''')


# ---------------------------------------------------------------------------
# Cell 2 — HTTP helpers (code) — reused verbatim from N01/02a so cache keys match
# ---------------------------------------------------------------------------
add("code", "c02b0002", '''
"""
HTTP helpers -- same code as Notebook 00/01/02a, same cache directory.
Cache hits cost zero CFBD budget. 02b expects ALL calls to be cache hits;
the assertion in Phase 02b-b fails loud if any go fresh.
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
print(f"[ok] sharing cache with Notebook 00/01/02a at {CACHE_DIR}")
''')


# ---------------------------------------------------------------------------
# Cell 3 — Configuration (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02b0003", """
## Configuration

`WALK_FORWARD_WINDOWS` and `BASELINE_PREGAME_FEATURES` are **carried verbatim from 02a** -- locked at 02a plan-approval and binding for 02b-g. Not re-decided in this notebook.

`CANDIDATE_FEATURES` is the 10 features from the V5 `trigger_features` DDL opening-drive-shock block (lines 169-179):

- **Drive-1 metadata flags / scalars** (require drive 1 to be complete; null per D4 when `drive_number_in_game <= 1`):
  - `dog_received_opening_kickoff`, `dog_scored_on_opening_drive`, `opening_drive_was_td`, `opening_drive_was_explosive_td`, `opening_drive_yards`, `opening_drive_plays`, `opening_drive_seconds`
- **Drive-1 EPA features** (computed from the lookahead-safe `plays_before` slice, R3-safe by construction; null when no eligible plays):
  - `fav_def_epa_first_drive`, `fav_def_epa_after_first_drive`
- **Derived flag** (deterministic function of the two EPA features):
  - `defense_stabilized_flag` -- D2 option (i): `int(fav_def_epa_after_first_drive < fav_def_epa_first_drive)`; null when either input is null.

`EXPLOSIVE_PASS_YARDS = 20` and `EXPLOSIVE_RUSH_YARDS = 12` -- decision **D1**, type-specific thresholds. PFF/SP+ convention; a 20-yard rush is a different football event from a 20-yard pass, so they get distinct thresholds. Tracked as a module-level constant pair that 02c-g will reuse verbatim. If N03 results suggest tuning, log to `tech_debt.md` rather than mid-flight patch.

`EXPLOSIVE_PASS_PLAY_TYPES = {"Pass Reception", "Passing Touchdown"}` and `EXPLOSIVE_RUSH_PLAY_TYPES = {"Rush", "Rushing Touchdown"}` -- the recognized pass / rush variants for explosive-play classification. Empirically the top-frequency offensive playType values; `Sack` and `Pass Incompletion` are pass attempts but never have `yardsGained >= 20` (sacks are negative; incompletions are 0), so they're excluded for clarity. `Pass Interception Return` is excluded -- the play was a pass but `yardsGained` reflects defensive return yards, not an offensive explosive play. All non-pass / non-rush playTypes (penalty, timeout, kickoff, punt, field goal, etc.) are excluded from explosive consideration per D1.

`REDUNDANT_WITH` is empty for 02b -- the plan-time redundancy audit found zero structural duplicates. Two **conditional identities** are documented in the sidecar's "Redundancy discoveries" section but do not warrant `redundant_with` tags (the identities hold only on subsets of triggers, not bit-identically across the full corpus).

`FEATURE_SET_VERSION = "v1_opening_drive_shock"` -- the per-notebook tag stamped into every row this notebook writes.
""")


# ---------------------------------------------------------------------------
# Cell 4 — Configuration constants (code)
# ---------------------------------------------------------------------------
add("code", "c02b0004", '''
SEASONS: list[int] = list(range(2015, 2025))
SEASON_TYPES: list[str] = ["regular", "postseason"]

FEATURE_SET_VERSION: str = "v1_opening_drive_shock"

# Walk-forward windows -- decision B from 02a plan-approval (train from 2015),
# locked and binding for 02b-g. Carried verbatim from 02a.
WALK_FORWARD_WINDOWS: list[dict] = [
    {"train_seasons": list(range(2015, 2021)), "val_season": 2021,
     "test_season": 2022, "train_window_label": "2015-2020"},
    {"train_seasons": list(range(2015, 2022)), "val_season": 2022,
     "test_season": 2023, "train_window_label": "2015-2021"},
    {"train_seasons": list(range(2015, 2023)), "val_season": 2023,
     "test_season": 2024, "train_window_label": "2015-2022"},
]

# Pre-game baseline columns -- decision alpha from 02a plan-approval,
# carried verbatim. Six columns: four always-present, plus spread_movement
# with R16-safe NaN handling + missingness indicator.
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

# Explosive-play thresholds -- decision D1 from 02b plan-approval.
# Type-specific (PFF/SP+ convention): a 20-yard rush is rare and
# breakdown-driven; a 20-yard pass is schemable. Two module-level
# constants so 02c-g can import and reuse one definition.
EXPLOSIVE_PASS_YARDS: int = 20
EXPLOSIVE_RUSH_YARDS: int = 12
EXPLOSIVE_PASS_PLAY_TYPES: frozenset[str] = frozenset({
    "Pass Reception",
    "Passing Touchdown",
})
EXPLOSIVE_RUSH_PLAY_TYPES: frozenset[str] = frozenset({
    "Rush",
    "Rushing Touchdown",
})

# Candidate opening-drive-shock features (V5 trigger_features DDL lines 169-179).
CANDIDATE_FEATURES: list[str] = [
    "dog_received_opening_kickoff",
    "dog_scored_on_opening_drive",
    "opening_drive_was_td",
    "opening_drive_was_explosive_td",
    "opening_drive_yards",
    "opening_drive_plays",
    "opening_drive_seconds",
    "fav_def_epa_first_drive",
    "fav_def_epa_after_first_drive",
    "defense_stabilized_flag",
]

# Structural-redundancy map for 02b. Empty: the plan-time redundancy audit
# found zero structural duplicates among 02b candidates. Two conditional
# identities (subset-only equalities) are documented in the sidecar's
# "Redundancy discoveries" section but do NOT warrant redundant_with tags
# (the identities hold only on subsets of triggers, not bit-identically
# across the full corpus).
REDUNDANT_WITH: dict[str, str] = {}

# Drive-summary feature names -- D4 NULLs these when drive_number_in_game <= 1
# (drive 1 not complete at trigger time; reading drive-summary fields would
# be lookahead-unsafe per R3).
DRIVE_SUMMARY_FEATURES: frozenset[str] = frozenset({
    "dog_scored_on_opening_drive",
    "opening_drive_was_td",
    "opening_drive_was_explosive_td",
    "opening_drive_yards",
    "opening_drive_plays",
    "opening_drive_seconds",
})

# Reproducibility seed for L1 logreg -- same as 02a.
RANDOM_STATE: int = 42

print(f"seasons: {SEASONS}")
print(f"season types: {SEASON_TYPES}")
print(f"feature_set_version: {FEATURE_SET_VERSION}")
print(f"walk-forward windows (locked from 02a, binding for 02b-g):")
for w in WALK_FORWARD_WINDOWS:
    print(f"  train={w['train_window_label']}  val={w['val_season']}  test={w['test_season']}")
print(f"baseline pre-game features ({len(BASELINE_PREGAME_FEATURES)}): {BASELINE_PREGAME_FEATURES}")
print(f"  R16-safe imputed col + indicator: ['spread_movement', 'spread_movement_is_null']")
print(f"candidate features ({len(CANDIDATE_FEATURES)}): {CANDIDATE_FEATURES}")
print(f"  drive-summary (D4 NULLs when drive_number_in_game <= 1): {sorted(DRIVE_SUMMARY_FEATURES)}")
print(f"  EPA / derived (computed from plays_before): "
      f"{sorted(set(CANDIDATE_FEATURES) - DRIVE_SUMMARY_FEATURES - {'dog_received_opening_kickoff'})}")
print(f"explosive thresholds (D1):")
print(f"  pass yards >= {EXPLOSIVE_PASS_YARDS}; pass types: {sorted(EXPLOSIVE_PASS_PLAY_TYPES)}")
print(f"  rush yards >= {EXPLOSIVE_RUSH_YARDS}; rush types: {sorted(EXPLOSIVE_RUSH_PLAY_TYPES)}")
print(f"redundant_with map ({len(REDUNDANT_WITH)} entries): {REDUNDANT_WITH}")
print(f"random state: {RANDOM_STATE}")
''')


# ---------------------------------------------------------------------------
# Cell 5 — Load trigger artifacts (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02b0005", """
## Phase 02b-a — Load trigger artifacts

Read `trigger_events.csv` and `trigger_outcomes.csv`. Inner-join on the natural key `(game_id, fav_deficit)` per `trigger_outcomes.schema.md`. Drop rows with `final_fav_won is NaN` (small count -- N01 reports unknown/tie). Identical setup to 02a; results in the same `trigger_full_df` row inventory.

Per R3, the join happens here in 02b (the feature-validation notebook), NOT inside any feature-extraction function. Features are computed from in-game state columns of `trigger_events.csv` plus the cached `/plays` and `/drives` corpora -- never from `trigger_outcomes.csv` columns. The label `final_fav_won` only enters as the model target in the walk-forward evaluation cell.
""")


# ---------------------------------------------------------------------------
# Cell 6 — Load triggers code
# ---------------------------------------------------------------------------
add("code", "c02b0006", '''
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
print(f"merged:                {len(trigger_full_df):>6,} rows x {trigger_full_df.shape[1]} cols")
assert len(trigger_full_df) == len(triggers_df), (
    f"inner join lost rows: {len(triggers_df)} -> {len(trigger_full_df)}. "
    f"trigger_outcomes.csv should have one outcome per trigger."
)

n_pre_drop = len(trigger_full_df)
trigger_full_df = trigger_full_df[trigger_full_df["final_fav_won"].notna()].copy()
trigger_full_df["final_fav_won"] = trigger_full_df["final_fav_won"].astype(bool)
n_dropped_tie = n_pre_drop - len(trigger_full_df)
print(f"\\nDropped {n_dropped_tie} rows with NaN final_fav_won (ties / unknown).")
print(f"In-scope rows for 02b: {len(trigger_full_df):,}")

# --- D4 plan-time scale check: how many triggers are on drive 1? ---------
n_drive1 = int((trigger_full_df["drive_number_in_game"] == 1).sum())
n_drive2plus = int((trigger_full_df["drive_number_in_game"] >= 2).sum())
print(f"\\nD4 null-policy scale check:")
print(f"  drive_number_in_game == 1 (drive-summary features NULL): "
      f"{n_drive1:,} ({n_drive1 / len(trigger_full_df) * 100:.1f}%)")
print(f"  drive_number_in_game >= 2 (drive-summary features computable): "
      f"{n_drive2plus:,} ({n_drive2plus / len(trigger_full_df) * 100:.1f}%)")

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
# Cell 7 — Re-load cached /plays + /drives (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02b0007", """
## Phase 02b-b — Re-load cached `/plays` AND `/drives` corpora (zero fresh calls)

Iterate the same `(season, season_type, week)` tuples N01 walked and call:
- `cfbd_get("/plays", ...)` for each (cache: 162 files)
- `cfbd_get("/drives", ...)` for each (cache: ~20 files)

**Assert every call is a cache hit.** The cell fails loud if any call would be a fresh fetch -- that signals cache invalidation and stops the run for review rather than silently consuming budget.

Decision **A** from 02a's plan-approval, carried forward and extended to cover `/drives` in addition to `/plays`. Per R22, 02b is supposed to spend 0 fresh CFBD calls; a cache miss here is a stop-the-line signal.
""")


# ---------------------------------------------------------------------------
# Cell 8 — Re-load /plays + /drives code
# ---------------------------------------------------------------------------
add("code", "c02b0008", '''
work_tuples_df = (
    trigger_full_df[["season", "season_type", "week"]]
    .drop_duplicates()
    .sort_values(["season", "season_type", "week"])
    .reset_index(drop=True)
)
print(f"distinct (season, season_type, week) tuples to load from cache: {len(work_tuples_df)}")

# Snapshot call-log size BEFORE the cache pull so we can detect non-cached
# calls scoped exactly to this loop.
n_log_before = sum(1 for _ in CALL_LOG.open("r", encoding="utf-8")) - 1  # minus header

# --- /plays re-load ----------------------------------------------------
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
elapsed_plays = time.perf_counter() - t_start
n_plays = sum(len(v) for v in plays_by_game.values())
print(f"[ok] /plays loaded from cache in {elapsed_plays:.1f}s -- "
      f"{len(plays_by_game):,} games, {n_plays:,} plays")
print(f"[info] CFBD negative-id encoding: {n_neg_id_in_cache:,} plays across "
      f"{len(neg_id_games):,} games carry the alternate (negative-int) id "
      f"format. Retained; chrono_key orders them correctly without referencing "
      f"play.id. See research/corrections_log.md.")

# --- /drives re-load ---------------------------------------------------
# N01 fetched /drives at the season level (year + seasonType) rather than
# per-week. We use the same N01-shaped requests so cache keys match.
drives_by_game: dict[int, list[dict]] = {}
t_start = time.perf_counter()
season_type_tuples = (
    trigger_full_df[["season", "season_type"]]
    .drop_duplicates()
    .sort_values(["season", "season_type"])
    .reset_index(drop=True)
)
for _, row in season_type_tuples.iterrows():
    season = int(row["season"])
    season_type = str(row["season_type"])
    drives = cfbd_get(
        "/drives",
        year=season,
        seasonType=season_type,
        classification="fbs",
    )
    for d in drives:
        gid = d.get("gameId")
        if gid is None:
            continue
        drives_by_game.setdefault(gid, []).append(d)
elapsed_drives = time.perf_counter() - t_start
n_drives = sum(len(v) for v in drives_by_game.values())
print(f"[ok] /drives loaded from cache in {elapsed_drives:.1f}s -- "
      f"{len(drives_by_game):,} games, {n_drives:,} drives")

# --- Cache-hit assertion (decision A, extended to /drives) -------------
calls_log_df = pd.read_csv(CALL_LOG)
this_run_calls = calls_log_df.iloc[n_log_before:].copy()
n_fresh_this_cell = int((this_run_calls["cached"] == 0).sum())
assert n_fresh_this_cell == 0, (
    f"02b budget invariant violated: {n_fresh_this_cell} non-cached CFBD call(s) "
    f"issued in this cell. 02b is supposed to spend 0 fresh CFBD calls; the "
    f"cache for some (year, type, week) tuple or (year, type) tuple is missing "
    f"or stale. Stop and investigate cache invalidation before continuing -- "
    f"DO NOT proceed and silently spend budget."
)
n_plays_lookups = int((this_run_calls["endpoint"] == "/plays").sum())
n_drives_lookups = int((this_run_calls["endpoint"] == "/drives").sum())
print(f"[ok] cache-hit assertion passed: {n_plays_lookups} /plays lookups, "
      f"{n_drives_lookups} /drives lookups, all cached.")

# --- Composite chrono_key helper + pre-sort per game -------------------
# Filter `plays_before` strictly by `_chrono_key(p) < trig_chrono_key` rather
# than `playNumber < trig.play_number`. The latter silently leaked future
# plays from drives > trig_drive (cross-drive leak) AND silently truncated
# completed earlier drives (drive-1 truncation, etc.) for triggers with
# small trig_pn -- because CFBD's `playNumber` resets per drive. See
# research/corrections_log.md. Source-of-truth in research/notebooks/
# _lib_chrono.py; inlined here so the notebook stays self-contained.

''' + CHRONO_KEY_SOURCE + '''


for gid in plays_by_game:
    plays_by_game[gid].sort(key=_chrono_key)
# --- Pre-sort drives per game by driveNumber ---------------------------
for gid in drives_by_game:
    drives_by_game[gid].sort(
        key=lambda d: (d.get("driveNumber") if d.get("driveNumber") is not None else 10**9)
    )
print(f"[ok] plays_by_game sorted by composite _chrono_key ({len(plays_by_game):,} games)")
print(f"[ok] drives_by_game sorted by driveNumber ({len(drives_by_game):,} games)")
''')


# ---------------------------------------------------------------------------
# Cell 9 — assert_no_lookahead + feature extractors (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02b0009", """
## Phase 02b-c — `assert_no_lookahead` (R3 hard gate) + feature extractors

`assert_no_lookahead` is the per-row R3 gate: every play used in a feature must have its composite `_chrono_key < trigger_chrono_key`. Same definition as 02a (post-corrections).

The composite key replaces the original `playNumber < trigger.play_number` test, which silently leaked future plays AND truncated completed earlier drives because CFBD's `playNumber` resets per drive. See `research/corrections_log.md` for the full-corpus verification.

Ten opening-drive-shock feature functions, all pure. Two categories:

- **Drive-summary features** (7 of 10): take `drive1` (the drive-1 summary record from the cached `/drives` corpus, if available), plus the `trig_drive_in_game` value from the trigger row, plus `fav_team` / `dog_team` strings. Return `None` when drive 1 is not yet complete at trigger time (`trig_drive_in_game <= 1`) -- the drive-summary fields would otherwise be lookahead-unsafe per R3 (D4 null policy).
  - `dog_received_opening_kickoff` is the lone exception: knowable from drive 1's offense team as soon as drive 1 has started (i.e., as soon as at least one play with `driveNumber == 1` exists in `plays_before`). Returns `None` only when `plays_before` has no drive-1 plays AND `drives_for_game` has no drive-1 entry.
- **EPA / derived features** (3 of 10): `fav_def_epa_first_drive`, `fav_def_epa_after_first_drive`, `defense_stabilized_flag`. Computed strictly from the lookahead-safe `plays_before` slice (R3-safe by construction). `None` when no eligible plays exist in the relevant subset.
""")


# ---------------------------------------------------------------------------
# Cell 10 — assert_no_lookahead code
# ---------------------------------------------------------------------------
add("code", "c02b000a", '''
def assert_no_lookahead(plays_used: list[dict],
                        trigger_chrono_key: tuple[int, int, int, int],
                        feature_name: str, game_id: int) -> None:
    """Per-row R3 hard gate. Raises if any play in `plays_used` has
    `_chrono_key(p) >= trigger_chrono_key`.

    Switched from the original `playNumber < trigger_play_number` test
    (which silently leaked future plays AND truncated earlier drives
    because CFBD playNumber resets per drive) to the composite chrono
    key. See research/corrections_log.md.
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
add("code", "c02b000b", '''
# EPA = CFBD's `ppa` field (Predicted Points Added). Same convention as 02a.

def _mean_ppa(plays: list[dict]) -> float | None:
    """Mean of `ppa` over the supplied plays where `ppa` is non-null.
    Returns None if zero plays have a non-null `ppa`."""
    vals = [p["ppa"] for p in plays if p.get("ppa") is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _is_explosive(play: dict) -> bool:
    """D1 explosive-play classifier. Type-specific thresholds:
    pass yardsGained >= EXPLOSIVE_PASS_YARDS, rush yardsGained >= EXPLOSIVE_RUSH_YARDS.
    Plays not in either play-type set (penalties, timeouts, kickoffs, punts,
    field goals, sacks, incompletions, interception returns, etc.) return False.
    """
    pt = play.get("playType", "")
    yg = play.get("yardsGained")
    if yg is None:
        return False
    if pt in EXPLOSIVE_PASS_PLAY_TYPES:
        return int(yg) >= EXPLOSIVE_PASS_YARDS
    if pt in EXPLOSIVE_RUSH_PLAY_TYPES:
        return int(yg) >= EXPLOSIVE_RUSH_YARDS
    return False


def _find_drive1(drives_for_game: list[dict]) -> dict | None:
    """Return the drive-1 record from `drives_for_game`, or None if absent."""
    for d in drives_for_game:
        if d.get("driveNumber") == 1:
            return d
    return None


def feat_dog_received_opening_kickoff(
    plays_before: list[dict], drives_for_game: list[dict],
    trig_drive_in_game: int, fav_team: str, dog_team: str,
) -> int | None:
    """1 iff drive-1 offense == dog_team. Knowable as soon as drive 1 has
    started (i.e., any play with driveNumber == 1 exists in plays_before,
    OR drive 1 exists in drives_for_game). Not gated by D4 because the
    drive-1 offense is fixed at the moment drive 1 starts (not lookahead-
    dependent on drive completion)."""
    # Prefer plays_before (strictly lookahead-safe). If plays_before has
    # any drive-1 play, that play's `offense` field tells us drive 1's
    # offense team. This is identical to drives_for_game's drive-1 offense
    # for any drive that has started.
    for p in plays_before:
        if p.get("driveNumber") == 1:
            return int(p.get("offense") == dog_team)
    # Fallback: if plays_before is empty for drive 1 but the trigger play
    # itself is associated with drive_number_in_game == 1 (which means
    # drive 1 has at least started), we can read drive 1's offense from
    # drives_for_game -- still not lookahead because drive-1's offense
    # is fixed at drive-1 start, not at drive-1 end.
    if trig_drive_in_game >= 1:
        d1 = _find_drive1(drives_for_game)
        if d1 is not None:
            return int(d1.get("offense") == dog_team)
    return None


def feat_dog_scored_on_opening_drive(
    plays_before: list[dict], drives_for_game: list[dict],
    trig_drive_in_game: int, fav_team: str, dog_team: str,
) -> int | None:
    """1 iff drive 1 had `scoring == True` AND drive-1 offense == dog_team.
    NULL when drive 1 not yet complete (D4)."""
    if trig_drive_in_game <= 1:
        return None
    d1 = _find_drive1(drives_for_game)
    if d1 is None:
        return None
    return int(bool(d1.get("scoring")) and d1.get("offense") == dog_team)


def feat_opening_drive_was_td(
    plays_before: list[dict], drives_for_game: list[dict],
    trig_drive_in_game: int, fav_team: str, dog_team: str,
) -> int | None:
    """1 iff drive 1's `driveResult == 'TD'`. Generic per D3(i) -- any
    team's TD on the opening drive (the dog-only variant is captured by
    `dog_scored_on_opening_drive`). NULL when drive 1 not yet complete (D4).
    """
    if trig_drive_in_game <= 1:
        return None
    d1 = _find_drive1(drives_for_game)
    if d1 is None:
        return None
    return int(d1.get("driveResult") == "TD")


def feat_opening_drive_was_explosive_td(
    plays_before: list[dict], drives_for_game: list[dict],
    trig_drive_in_game: int, fav_team: str, dog_team: str,
) -> int | None:
    """1 iff drive 1's `driveResult == 'TD'` AND >= 1 explosive play
    occurred in drive 1. Explosive per D1 type-specific thresholds.
    NULL when drive 1 not yet complete (D4)."""
    if trig_drive_in_game <= 1:
        return None
    d1 = _find_drive1(drives_for_game)
    if d1 is None:
        return None
    if d1.get("driveResult") != "TD":
        return 0
    # Drive-1 plays from plays_before (strictly lookahead-safe under the
    # composite chrono_key filter). Drive 1 is complete at trigger time
    # per D4, so plays_before should contain the entire drive 1 by
    # definition (every drive-1 play's chrono_key is < trig_chrono_key
    # when trig_drive_in_game >= 2). NOTE: the original revision used a
    # `playNumber < trig_pn` filter, which TRUNCATED drive 1 because
    # CFBD playNumber resets per drive -- the truncation could drop late-
    # drive-1 plays (including explosive ones) and flip this feature
    # from 1 to 0 for affected triggers. See corrections_log.md and the
    # truncation diagnostics in Phase 02b-d2.
    d1_plays = [p for p in plays_before if p.get("driveNumber") == 1]
    return int(any(_is_explosive(p) for p in d1_plays))


def feat_opening_drive_yards(
    plays_before: list[dict], drives_for_game: list[dict],
    trig_drive_in_game: int, fav_team: str, dog_team: str,
) -> int | None:
    """Drive 1's `yards` field. NULL when drive 1 not yet complete (D4)."""
    if trig_drive_in_game <= 1:
        return None
    d1 = _find_drive1(drives_for_game)
    if d1 is None or d1.get("yards") is None:
        return None
    return int(d1["yards"])


def feat_opening_drive_plays(
    plays_before: list[dict], drives_for_game: list[dict],
    trig_drive_in_game: int, fav_team: str, dog_team: str,
) -> int | None:
    """Drive 1's `plays` field (count of plays). NULL when drive 1 not yet
    complete (D4)."""
    if trig_drive_in_game <= 1:
        return None
    d1 = _find_drive1(drives_for_game)
    if d1 is None or d1.get("plays") is None:
        return None
    return int(d1["plays"])


def feat_opening_drive_seconds(
    plays_before: list[dict], drives_for_game: list[dict],
    trig_drive_in_game: int, fav_team: str, dog_team: str,
) -> int | None:
    """Drive 1's elapsed time in seconds (`minutes*60 + seconds`). NULL
    when drive 1 not yet complete (D4)."""
    if trig_drive_in_game <= 1:
        return None
    d1 = _find_drive1(drives_for_game)
    if d1 is None:
        return None
    e = d1.get("elapsed") or {}
    m, s = e.get("minutes"), e.get("seconds")
    if m is None or s is None:
        return None
    return int(m) * 60 + int(s)


def feat_fav_def_epa_first_drive(
    plays_before: list[dict], drives_for_game: list[dict],
    trig_drive_in_game: int, fav_team: str, dog_team: str,
) -> float | None:
    """Mean ppa over plays in drive 1 where defense == fav_team. Computed
    strictly from plays_before (R3-safe by construction). NULL when the
    eligible play subset is empty (e.g., fav was on offense in drive 1,
    or drive 1 hasn't started)."""
    sub = [
        p for p in plays_before
        if p.get("driveNumber") == 1 and p.get("defense") == fav_team
    ]
    return _mean_ppa(sub)


def feat_fav_def_epa_after_first_drive(
    plays_before: list[dict], drives_for_game: list[dict],
    trig_drive_in_game: int, fav_team: str, dog_team: str,
) -> float | None:
    """Mean ppa over plays where defense == fav_team AND driveNumber > 1.
    Computed strictly from plays_before. NULL when the eligible play subset
    is empty (e.g., trigger on drive 1, or no fav-defense plays after drive 1)."""
    sub = [
        p for p in plays_before
        if p.get("driveNumber") is not None
        and p.get("driveNumber") > 1
        and p.get("defense") == fav_team
    ]
    return _mean_ppa(sub)


def feat_defense_stabilized_flag(
    plays_before: list[dict], drives_for_game: list[dict],
    trig_drive_in_game: int, fav_team: str, dog_team: str,
) -> int | None:
    """D2(i): 1 iff fav_def_epa_after_first_drive < fav_def_epa_first_drive
    (the favorite's defense was less leaky after drive 1 than during drive 1).
    NULL when either input is NULL."""
    a = feat_fav_def_epa_first_drive(
        plays_before, drives_for_game, trig_drive_in_game, fav_team, dog_team
    )
    b = feat_fav_def_epa_after_first_drive(
        plays_before, drives_for_game, trig_drive_in_game, fav_team, dog_team
    )
    if a is None or b is None:
        return None
    return int(b < a)


FEATURE_FUNCTIONS: dict[
    str,
    Callable[[list[dict], list[dict], int, str, str], float | int | None],
] = {
    "dog_received_opening_kickoff":   feat_dog_received_opening_kickoff,
    "dog_scored_on_opening_drive":    feat_dog_scored_on_opening_drive,
    "opening_drive_was_td":           feat_opening_drive_was_td,
    "opening_drive_was_explosive_td": feat_opening_drive_was_explosive_td,
    "opening_drive_yards":            feat_opening_drive_yards,
    "opening_drive_plays":            feat_opening_drive_plays,
    "opening_drive_seconds":          feat_opening_drive_seconds,
    "fav_def_epa_first_drive":        feat_fav_def_epa_first_drive,
    "fav_def_epa_after_first_drive":  feat_fav_def_epa_after_first_drive,
    "defense_stabilized_flag":        feat_defense_stabilized_flag,
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
add("markdown", "m02b000c", """
## Phase 02b-d — Build feature matrix

Walk every in-scope trigger, slice plays by the composite chrono_key filter `_chrono_key(p) < trigger_chrono_key` (composite over `period`, `period_seconds_elapsed`, `driveNumber`, `playNumber`), look up drive-1 metadata from the cached `/drives` corpus, run each candidate feature, gate every play subset through `assert_no_lookahead`, and attach the pre-game baseline columns.

The composite chrono_key replaces the original `playNumber < trigger.play_number` filter (which silently leaked future plays AND truncated drive 1 because CFBD's `playNumber` resets per drive). See `research/corrections_log.md` and the truncation-impact diagnostics in Phase 02b-d2 below.

Print null counts per feature so the practical applicability gap of each feature is visible upfront. The drive-summary features will null out on the 1,860 drive-1 triggers (~16% of in-scope) per D4; the EPA / derived features will additionally null out on triggers where the relevant play subset is empty.

Decision **B** from the 02a plan-approval (per-feature null policy, carried forward): a trigger that is null for one feature may be evaluable for others. Per-feature null masking happens at the evaluation cell, not here -- this cell preserves all rows.
""")


# ---------------------------------------------------------------------------
# Cell 13 — Build matrix code
# ---------------------------------------------------------------------------
add("code", "c02b000d", '''
ID_COLS = ["game_id", "fav_deficit", "trigger_sequence", "season", "season_type",
           "week", "fav_team", "dog_team", "play_number", "quarter",
           "drive_number_in_game"]
LABEL_COL = "final_fav_won"

records: list[dict] = []
n_skipped_unknown_game = 0
n_skipped_no_drives = 0

for _, trig in trigger_full_df.iterrows():
    gid = int(trig["game_id"])
    trig_pn = int(trig["play_number"])
    fav = str(trig["fav_team"])
    dog = str(trig["dog_team"])
    trig_drive_in_game = int(trig["drive_number_in_game"])

    # Composite chrono_key for the trigger row (period, period_elapsed,
    # driveNumber, playNumber). N01's `clock_seconds_in_period_total` is
    # the trigger's seconds-remaining-in-period, so period_elapsed = 900
    # minus that value.
    trig_period = int(trig["quarter"])
    trig_period_elapsed = 900 - int(trig["clock_seconds_in_period_total"])
    trig_chrono_key = (trig_period, trig_period_elapsed, trig_drive_in_game, trig_pn)

    plays = plays_by_game.get(gid)
    if plays is None:
        n_skipped_unknown_game += 1
        continue
    # Composite chrono_key filter (replaces the leaky `playNumber < trig_pn`
    # filter). All 19,828 negative-id plays across 115 games are retained --
    # the chrono_key orders them correctly without referencing play.id.
    plays_before = [p for p in plays if _chrono_key(p) < trig_chrono_key]
    drives_for_game = drives_by_game.get(gid, [])

    row: dict[str, Any] = {col: trig[col] for col in ID_COLS}
    # Always-present pre-game columns
    for col in ALWAYS_PRESENT_PREGAME_COLS:
        row[col] = trig[col]
    # R16-safe NaN handling for spread_movement (same logic as 02a).
    sm_raw = trig["spread_movement"]
    sm_is_null = bool(pd.isna(sm_raw))
    row["spread_movement"] = 0.0 if sm_is_null else float(sm_raw)
    row["spread_movement_is_null"] = int(sm_is_null)
    row[LABEL_COL] = bool(trig[LABEL_COL])

    for feat_name, fn in FEATURE_FUNCTIONS.items():
        # R3 gate runs per-feature on the play subset the extractor sees.
        # All extractors receive the same `plays_before` slice; drive-1 EPA
        # features filter further by driveNumber inside the function.
        assert_no_lookahead(plays_before, trig_chrono_key, feat_name, gid)
        row[feat_name] = fn(plays_before, drives_for_game, trig_drive_in_game, fav, dog)

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
    is_drive_summary = "(D4 drive-summary)" if feat in DRIVE_SUMMARY_FEATURES else ""
    print(f"  {feat:<32} {n_null:>5,} null ({pct:5.2f}%) {is_drive_summary}")

# Sanity: drive-summary features should all null out exactly on drive-1 triggers
# (plus possibly a handful with missing drive-1 records).
n_drive1 = int((feature_matrix_df["drive_number_in_game"] == 1).sum())
print(f"\\n  drive-1 trigger count (expected null floor for drive-summary feats): {n_drive1:,}")
for feat in DRIVE_SUMMARY_FEATURES:
    assert null_counts[feat] >= n_drive1, (
        f"drive-summary feature {feat!r} has {null_counts[feat]:,} nulls, "
        f"less than the {n_drive1:,} drive-1 triggers that D4 NULLs out. "
        f"Bug in extractor or D4 policy violation."
    )

# R16-safe NaN handling sanity for spread_movement (same as 02a).
sm_null_after = int(feature_matrix_df["spread_movement"].isna().sum())
sm_indicator_sum = int(feature_matrix_df["spread_movement_is_null"].sum())
print(f"\\nR16-safe NaN handling for spread_movement:")
print(f"  spread_movement nulls AFTER impute:    {sm_null_after:,}  (expected: 0)")
print(f"  spread_movement_is_null indicator sum: {sm_indicator_sum:,} "
      f"({sm_indicator_sum / len(feature_matrix_df) * 100:.2f}% of in-scope)")
assert sm_null_after == 0, f"spread_movement still has {sm_null_after} nulls after impute"
''')


# ---------------------------------------------------------------------------
# Cell 13b — Truncation-impact diagnostics (markdown) — added in the
# chrono_key corrections sweep. Quantifies what the prior (leaky) filter
# would have seen vs what the chrono filter sees, per trigger.
# ---------------------------------------------------------------------------
add("markdown", "m02b000d2", """
## Phase 02b-d2 — Truncation-impact diagnostics (corrections-sweep diagnostic)

The original 02b filter `playNumber < trig_pn` had two distinct R3 failure modes:

1. **Cross-drive forward leak.** Plays from drives > trig_drive with low `playNumber` (because CFBD `playNumber` resets per drive) were incorrectly included as "before".
2. **Drive-1 truncation.** Plays from drives < trig_drive with high `playNumber` (drive 1 typically has 6-12 plays; trig_pn can be 1-5 on later drives) were incorrectly excluded.

This cell computes both leaky and chrono `plays_before` slices per trigger (informational only -- the matrix above was built using the chrono filter exclusively) and surfaces three diagnostics tied to specific at-risk features:

- **(1)** Distribution of drive-1 plays in `plays_before` under each filter, with per-trigger delta.
- **(2)** Per-trigger value flips for `opening_drive_was_explosive_td`: how many triggers had `leaky=0` and `chrono=1` (drive-1 explosive play that the leak truncated out) and vice versa.
- **(3)** Per-trigger EPA-mean shifts for `fav_def_epa_first_drive`: distribution of `(chrono_epa - leaky_epa)` across triggers where both filters produced non-null values.

This grounds the prior-vs-new verdict comparison in mechanism rather than outcome alone.
""")


# ---------------------------------------------------------------------------
# Cell 13c — Truncation-impact diagnostics (code)
# ---------------------------------------------------------------------------
add("code", "c02b000d3", '''
# Re-iterate triggers, computing both leaky and chrono plays_before slices
# per trigger so we can quantify the truncation/leak mechanism without
# rebuilding the matrix. The main matrix above was built using the chrono
# filter exclusively; this cell is INFORMATIONAL.

import statistics

drive1_count_leaky: list[int] = []
drive1_count_chrono: list[int] = []
explosive_td_evaluated = 0  # triggers where both filters returned 0/1 (not NULL)
explosive_td_leaky0_chrono1 = 0  # drive-1 explosive play truncated by leak
explosive_td_leaky1_chrono0 = 0  # drive-1 explosive play falsely included by leak

epa_first_deltas: list[float] = []
epa_first_sign_flips = 0
epa_first_large_shifts = 0  # |chrono - leaky| > 0.01

n_drive1_truncation_affected = 0  # triggers where drive-1 plays_before differs

for _, trig in trigger_full_df.iterrows():
    gid = int(trig["game_id"])
    trig_pn = int(trig["play_number"])
    fav = str(trig["fav_team"])
    dog = str(trig["dog_team"])
    trig_drive_in_game = int(trig["drive_number_in_game"])
    trig_period = int(trig["quarter"])
    trig_period_elapsed = 900 - int(trig["clock_seconds_in_period_total"])
    trig_chrono_key = (trig_period, trig_period_elapsed, trig_drive_in_game, trig_pn)

    plays = plays_by_game.get(gid)
    if plays is None:
        continue
    drives_for_game = drives_by_game.get(gid, [])

    leaky_slice = [
        p for p in plays
        if p.get("playNumber") is not None and int(p["playNumber"]) < trig_pn
    ]
    chrono_slice = [p for p in plays if _chrono_key(p) < trig_chrono_key]

    d1_leaky = [p for p in leaky_slice if p.get("driveNumber") == 1]
    d1_chrono = [p for p in chrono_slice if p.get("driveNumber") == 1]

    drive1_count_leaky.append(len(d1_leaky))
    drive1_count_chrono.append(len(d1_chrono))
    if len(d1_leaky) != len(d1_chrono):
        n_drive1_truncation_affected += 1

    # opening_drive_was_explosive_td: evaluable only for triggers on drive > 1
    # where drive 1 ended in a TD.
    if trig_drive_in_game > 1:
        d1 = _find_drive1(drives_for_game)
        if d1 is not None and d1.get("driveResult") == "TD":
            leaky_val = int(any(_is_explosive(p) for p in d1_leaky))
            chrono_val = int(any(_is_explosive(p) for p in d1_chrono))
            explosive_td_evaluated += 1
            if leaky_val == 0 and chrono_val == 1:
                explosive_td_leaky0_chrono1 += 1
            elif leaky_val == 1 and chrono_val == 0:
                explosive_td_leaky1_chrono0 += 1

    # fav_def_epa_first_drive: drive-1 plays where defense == fav, mean(ppa).
    d1_fav_def_leaky = [p for p in d1_leaky if p.get("defense") == fav]
    d1_fav_def_chrono = [p for p in d1_chrono if p.get("defense") == fav]
    leaky_epa = _mean_ppa(d1_fav_def_leaky)
    chrono_epa = _mean_ppa(d1_fav_def_chrono)
    if leaky_epa is not None and chrono_epa is not None:
        delta = chrono_epa - leaky_epa
        epa_first_deltas.append(delta)
        if abs(delta) > 0.01:
            epa_first_large_shifts += 1
        if (leaky_epa > 0) != (chrono_epa > 0) and leaky_epa != 0 and chrono_epa != 0:
            epa_first_sign_flips += 1


N = len(drive1_count_leaky)
delta_counts = [c - l for c, l in zip(drive1_count_chrono, drive1_count_leaky)]
positive = sum(1 for d in delta_counts if d > 0)
negative = sum(1 for d in delta_counts if d < 0)
zero = sum(1 for d in delta_counts if d == 0)

print("=" * 70)
print("TRUNCATION-IMPACT DIAGNOSTICS (leaky vs chrono, per trigger)")
print("=" * 70)

print("\\n[1] Drive-1 plays in plays_before (per trigger):")
print(f"  total triggers analyzed: {N:,}")
print(f"  leaky:  mean={statistics.mean(drive1_count_leaky):>6.2f}  "
      f"median={statistics.median(drive1_count_leaky):>4.1f}  "
      f"max={max(drive1_count_leaky):>3}")
print(f"  chrono: mean={statistics.mean(drive1_count_chrono):>6.2f}  "
      f"median={statistics.median(drive1_count_chrono):>4.1f}  "
      f"max={max(drive1_count_chrono):>3}")
print(f"  delta (chrono - leaky) distribution:")
print(f"    delta > 0 (chrono recovered truncated drive-1 plays): "
      f"{positive:>6,}  ({positive / N * 100:>5.2f}%)")
print(f"    delta < 0 (chrono dropped cross-drive-leaked plays):  "
      f"{negative:>6,}  ({negative / N * 100:>5.2f}%)")
print(f"    delta == 0 (unchanged):                                "
      f"{zero:>6,}  ({zero / N * 100:>5.2f}%)")
print(f"    max positive delta: {max(delta_counts):>3}  "
      f"min negative delta: {min(delta_counts):>3}")
print(f"  TRUNCATION/LEAK-AFFECTED triggers: {n_drive1_truncation_affected:,} "
      f"({n_drive1_truncation_affected / N * 100:.2f}%)")

print("\\n[2] opening_drive_was_explosive_td flip counts:")
print(f"  triggers evaluable under both filters "
      f"(trig drive >= 2, drive-1 was TD):  {explosive_td_evaluated:,}")
print(f"  leaky=0, chrono=1 (explosive truncated out by leak): "
      f"{explosive_td_leaky0_chrono1:,}")
print(f"  leaky=1, chrono=0 (false positive removed by chrono): "
      f"{explosive_td_leaky1_chrono0:,}")
print(f"  net flip: {explosive_td_leaky0_chrono1 - explosive_td_leaky1_chrono0:+,} "
      f"toward chrono=1")
if explosive_td_evaluated > 0:
    flip_pct = (explosive_td_leaky0_chrono1 + explosive_td_leaky1_chrono0) / explosive_td_evaluated * 100
    print(f"  flip rate (any direction): {flip_pct:.2f}% of evaluable triggers")

print("\\n[3] fav_def_epa_first_drive per-trigger EPA-mean shifts:")
if epa_first_deltas:
    abs_d = [abs(d) for d in epa_first_deltas]
    print(f"  triggers with both leaky and chrono non-null: {len(epa_first_deltas):,}")
    print(f"  delta (chrono - leaky):")
    print(f"    mean   = {statistics.mean(epa_first_deltas):+.5f}")
    print(f"    median = {statistics.median(epa_first_deltas):+.5f}")
    print(f"    |delta| max = {max(abs_d):.5f}")
    print(f"  |delta| > 0.01 EPA/play: {epa_first_large_shifts:,} triggers "
          f"({epa_first_large_shifts / len(epa_first_deltas) * 100:.2f}%)")
    print(f"  sign flips (leaky > 0 vs chrono < 0 or vice versa): "
          f"{epa_first_sign_flips:,} triggers "
          f"({epa_first_sign_flips / len(epa_first_deltas) * 100:.2f}%)")
else:
    print("  no triggers had both filters non-null (unexpected)")

print()
print("=" * 70)
''')


# ---------------------------------------------------------------------------
# Cell 14 — Walk-forward evaluation (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02b000e", """
## Phase 02b-e — Walk-forward per-feature evaluation

For each `(window, candidate_feature)` pair:

1. Mask `feature_matrix_df` to rows where the candidate is non-null (decision **B** per-feature null policy, carried forward from 02a).
2. Split into train (multi-season), val (single season), test (single season) -- all from the same non-null subset.
3. Fit two pipelines on the train split:
   - **Baseline**: `StandardScaler` -> `LogisticRegression(penalty="l1", C=1.0, solver="liblinear", random_state=42, max_iter=1000)` on `BASELINE_PREGAME_FEATURES` only.
   - **Candidate**: same pipeline on `BASELINE_PREGAME_FEATURES + [feature]`.
4. Calibrate each on val via `CalibratedClassifierCV(method="isotonic", cv="prefit")`.
5. Evaluate calibrated probabilities on test: Brier (`sklearn.metrics.brier_score_loss`) and ECE (10 equal-width bins).
6. Record `brier_improvement = baseline - candidate` and `calibration_improvement = ece_baseline - ece_candidate`.

After the loop, `passed_stability = sum(brier_improvement > 0) >= 2` per feature, broadcast to all 3 rows of that feature (R6).

10 candidate features x 3 windows = 30 eval rows expected.
""")


# ---------------------------------------------------------------------------
# Cell 15 — ECE + fit helper (same as 02a)
# ---------------------------------------------------------------------------
add("code", "c02b000f", '''
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
    Identical to 02a's helper; will be deduped into a shared module before N03
    per research/tech_debt.md item 2.
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
add("code", "c02b0010", '''
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
    print(f"  {feat:<32} {verdict}  ({n_pos}/3 test seasons with positive Brier improvement)")
''')


# ---------------------------------------------------------------------------
# Cell 17 — Write CSV (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02b0011", """
## Phase 02b-f — Write `feature_validation.csv` (defensive append)

Same defensive-append pattern as 02a:

1. If `feature_validation.csv` exists, read it (with `keep_default_na=False` so the `redundant_with` empty-string convention survives the round-trip).
2. Drop rows matching this run's `(feature, train_window, test_season)` keys.
3. Concatenate this run's 30 new rows (`feature_set_version = v1_opening_drive_shock`).
4. Sort by `(feature_set_version, feature, train_window, test_season)`.
5. Write.

02a's 18 rows (`v1_baseline_efficiency_only`) are preserved because their keys don't overlap with 02b's. Natural-key uniqueness is asserted after the write.
""")


# ---------------------------------------------------------------------------
# Cell 18 — Write CSV code
# ---------------------------------------------------------------------------
add("code", "c02b0012", '''
NEW_KEYS = set(zip(
    eval_df["feature"],
    eval_df["train_window"],
    eval_df["test_season"].astype(int),
))

if FEATURE_VALIDATION_CSV.exists():
    # keep_default_na=False so `redundant_with == ''` survives the round-trip.
    existing_df = pd.read_csv(FEATURE_VALIDATION_CSV, keep_default_na=False)
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

# Preserve empty-string redundant_with on write -- pandas default na_rep="" already
# emits empty cells, which round-trip cleanly with keep_default_na=False on read.
combined_df.to_csv(FEATURE_VALIDATION_CSV, index=False)
print(f"\\n[ok] wrote feature_validation.csv: {len(combined_df):,} rows "
      f"({len(eval_df)} from this run, {len(combined_df) - len(eval_df)} retained from prior runs)")
print(f"     path: {FEATURE_VALIDATION_CSV}")
''')


# ---------------------------------------------------------------------------
# Cell 19 — Schema sidecar (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02b0013", """
## Phase 02b-g — Splice into `feature_validation.schema.md` sidecar

The sidecar accumulates one section per feature-group notebook (02a, 02b, ..., 02g). 02b uses **sentinel-delimited splicing**: read the existing sidecar, find the `<!-- BEGIN: 02b -->` / `<!-- END: 02b -->` markers, replace the contents between them (or append at end if the markers don't exist yet), write back.

Why splicing: 02a regenerates the entire sidecar from scratch each run, so a naive "append to file" pattern would lose its content the next time 02a runs. Splicing keeps each notebook's section under its own ownership while preserving the others.

Known limitation: 02a's writer does NOT yet use splicing -- it still overwrites the entire file. Re-running 02a after 02b will clobber 02b's section. Tracked as item 3 in `research/tech_debt.md`; will be fixed in the cleanup sweep before N03 (along with the 1,000-call display constant and the sklearn deprecation).
""")


# ---------------------------------------------------------------------------
# Cell 20 — Write schema code
# ---------------------------------------------------------------------------
add("code", "c02b0014", '''
def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception as e:  # noqa: BLE001
        return f"<unavailable: {e}>"


now_text = time.strftime("%Y-%m-%d %H:%M:%S %Z").strip() or time.strftime("%Y-%m-%d %H:%M:%S")
commit_hash = _git_commit()


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
    tag = " (D4 drive-summary)" if feat in DRIVE_SUMMARY_FEATURES else ""
    null_rows.append(f"| `{feat}` | {n_null:,} | {pct:.2f}%{tag} |")

n_drive1 = int((feature_matrix_df["drive_number_in_game"] == 1).sum())

# Build the 02b-owned section, delimited by sentinels.
SECTION_BEGIN = "<!-- BEGIN: 02b opening_drive_shock -->"
SECTION_END = "<!-- END: 02b opening_drive_shock -->"

section_body = f"""
## 02b -- Opening-drive shock features

**Section last writer:** `research/notebooks/02b_opening_drive_shock.ipynb`
**Last writer commit:** `{commit_hash}`
**Last writer generation timestamp:** {now_text}
**Feature set version:** `{FEATURE_SET_VERSION}`
**Source DDL:** `BUILD_SPEC.md` `trigger_features` opening-drive-shock block (V5 lines 169-179)

### Candidate features (10)

Drive-1 metadata flags / scalars (D4 NULLs when `drive_number_in_game <= 1`,
i.e., the trigger play is in drive 1 and the drive is not yet complete --
reading drive-summary fields would be lookahead-unsafe per R3):

- `dog_received_opening_kickoff` (lone exception to D4: drive-1 offense is
  fixed at drive-1 start, not at drive-1 end, so safe to read whenever
  drive 1 has at least started)
- `dog_scored_on_opening_drive`
- `opening_drive_was_td` (D3(i) generic: any team's TD, not dog-only)
- `opening_drive_was_explosive_td`
- `opening_drive_yards`
- `opening_drive_plays`
- `opening_drive_seconds`

Drive-1 EPA features (computed strictly from the lookahead-safe `plays_before`
slice; R3-safe by construction; null when the eligible play subset is empty):

- `fav_def_epa_first_drive`
- `fav_def_epa_after_first_drive`

Derived flag (deterministic function of the two EPA features):

- `defense_stabilized_flag` -- D2(i): `int(fav_def_epa_after_first_drive < fav_def_epa_first_drive)`; null when either input is null.

### D1: explosive-play thresholds (type-specific)

```python
EXPLOSIVE_PASS_YARDS = {EXPLOSIVE_PASS_YARDS}
EXPLOSIVE_RUSH_YARDS = {EXPLOSIVE_RUSH_YARDS}
EXPLOSIVE_PASS_PLAY_TYPES = {sorted(EXPLOSIVE_PASS_PLAY_TYPES)}
EXPLOSIVE_RUSH_PLAY_TYPES = {sorted(EXPLOSIVE_RUSH_PLAY_TYPES)}
```

PFF / SP+ convention: a 20-yard rush is a different football event from a
20-yard pass (rare and breakdown-driven vs schemable), so they get distinct
thresholds. `Sack` and `Pass Incompletion` are pass attempts but never have
`yardsGained >= 20` (sacks are negative; incompletions are 0), so they're
excluded for clarity. `Pass Interception Return` is excluded -- the play
was a pass but `yardsGained` reflects defensive return yards, not an
offensive explosive play. All non-pass / non-rush playTypes (penalty,
timeout, kickoff, punt, field goal, etc.) are excluded from explosive
consideration per D1.

If N03 results suggest the thresholds need tuning, log to
`research/tech_debt.md` rather than mid-flight patch (per the 02b plan-
approval decision on D1).

### D4: null policy for drive-1 triggers

The 1,860 triggers (16.3% of in-scope) where `drive_number_in_game == 1`
have drive-summary features set to NULL by the extractor. Rationale: at
trigger time, drive 1 is in progress (the trigger play is part of drive
1); reading drive 1's `driveResult` / `yards` / `plays` / `elapsed` /
`scoring` fields from the CFBD `/drives` record would include plays
AFTER the trigger play, which is a R3 lookahead violation. The
per-feature null policy (decision B from 02a, carried forward) means
these rows are dropped from each drive-summary feature's per-feature
evaluation, not globally.

EPA-based features (`fav_def_epa_first_drive`,
`fav_def_epa_after_first_drive`, `defense_stabilized_flag`) remain
computable from the lookahead-safe `plays_before` slice and are
not affected by D4. They null out only when the eligible play subset
is empty.

`dog_received_opening_kickoff` is exempt from D4: drive 1's offense team
is fixed at drive-1 start (not drive-1 end), so it's lookahead-safe to
read once drive 1 has at least started.

### Per-feature null counts (this run)

In-scope triggers (post NaN `final_fav_won` drop): {len(feature_matrix_df):,}.
Drive-1 trigger count (D4 null floor for drive-summary features): {n_drive1:,}.

| Feature | Null rows | % of in-scope |
|---|---:|---:|
""" + "\\n".join(null_rows) + f"""

**Per-feature evaluation N for D4-affected drive-summary features (6 of 10):** {len(feature_matrix_df) - n_drive1:,} evaluable, {n_drive1:,} NULL'd by D4 (`drive_number_in_game <= 1`). Saves N03 the back-derivation. The remaining 4 candidates -- `dog_received_opening_kickoff` (D4 exception) and the three EPA-derived features (`fav_def_epa_first_drive`, `fav_def_epa_after_first_drive`, `defense_stabilized_flag`) -- use per-feature N from the null-counts table above.

### Per-feature x per-test-season results (this run, {FEATURE_SET_VERSION})

| Feature | Window -> Test | Brier improvement | ECE improvement | Stability |
|---|---|---:|---:|---|
""" + "\\n".join(verdict_rows) + f"""

Sign convention: positive = candidate beat baseline. `**PASS**` means
`sum(brier_improvement > 0) >= 2` across the 3 test seasons.

### Redundancy discoveries (02b plan-time audit + this run)

Plan-time audit verdict: **zero structural duplicates among 02b's 10
candidates.** `REDUNDANT_WITH = {{}}` for this feature set version. All
30 rows have `redundant_with == ""`.

Two **conditional identities** were flagged at plan time. Distinct from
the bit-identical `redundant_with` tags from 02a (where the identity
held across the FULL trigger corpus), these are identities that hold
only on a SUBSET of triggers:

1. `fav_def_epa_after_first_drive == fav_def_epa_per_play` on triggers
   where `dog_received_opening_kickoff == 0` (fav was on defense only
   after drive 1; no fav-defense plays exist in drive 1 on that subset).
   On the complement (fav was on defense in drive 1),
   `fav_def_epa_after_first_drive` is a strict play-subset average of
   `fav_def_epa_per_play`, not identity.

2. Within drive 1, the offense-vs-defense per-row identity from 02a's
   redundancy discovery still holds (`offense == fav` iff `defense == dog`
   on drive 1's plays), so `fav_def_epa_first_drive ==` (a not-emitted
   feature) `dog_off_epa_first_drive` restricted to drive 1. We do not
   emit dog-side drive-1 EPA features, so there's no actual duplicate
   in 02b's CSV; the identity is documented as a forward note for
   02c-g and N03 in case the dog-side drive-1 mirrors come up.

Neither warrants a `redundant_with` tag because the identities hold only
on subsets of triggers, not bit-identically across the full corpus.
N03's production-feature assembly should still filter `redundant_with ==
""` (this drops only 02a's two duplicates; 02b's 10 rows all stay).

### Post-correction findings and retractions

**This section is re-emitted under the chrono_key correction sweep.**
See `research/corrections_log.md` section 1 for the full lookahead-leak
write-up and section 3's 02b subsection for the per-feature verdict
deltas. The two findings below revise interpretive claims that were
made in the superseded 02b commit `e1710a2`.

**1. Retraction of `fav_def_epa_first_drive` calibration-standout claim.**
The prior commit body (`e1710a2`) flagged `fav_def_epa_first_drive` as
"the only feature across 02a + 02b to pass 3/3 on BOTH Brier AND ECE"
and described it as "structurally important for N03 because Kelly stake
sizing reads directly off calibrated probabilities, so a feature that
improves ranking AND calibration is rarer and more valuable than one
that improves ranking alone." That claim is **empirically falsified**
under the corrected `_chrono_key` filter:

- Brier improvement: 3/3 leaky -> 0/3 corrected. The corrected Brier
  improvements across the three folds collapsed in sign on all three
  folds (negative on each).
- ECE improvement: 3/3 leaky -> 2/3 corrected (one fold flipped).
- Overall: PASS -> **FAIL**.

Mechanism: the within-drive truncation arm of the lookahead leak (see
`research/corrections_log.md` section 1) systematically removed
late-drive-1 plays from `plays_before` for triggers in later drives
with low `playNumber`. Late-drive-1 plays disproportionately include
sacks, third-down conversions, and stops -- removing them biased the
leaky `fav_def_epa_first_drive` mean upward in a way that happened to
correlate with comeback outcomes. The corrected feature, which
includes all real drive-1 plays before the trigger, shows no signal.

**N03 implication:** `fav_def_epa_first_drive` is removed from the
validated set. Calibration support for Kelly stake sizing must come
from elsewhere in the validated set (no single feature in the
corrected 02a + 02b sets passes 3/3 on both Brier and ECE).

**2. `defense_stabilized_flag` mechanistic note.** The derived feature
`defense_stabilized_flag = int(fav_def_epa_after_first_drive < fav_def_epa_first_drive)`
survives the correction at 3/3 PASS, *even though one of its two
inputs (`fav_def_epa_first_drive`) failed 0/3*. The empirical pattern:

- `fav_def_epa_first_drive` (input A): FAIL 0/3 -- corrected EPA means
  no longer carry comeback signal.
- `fav_def_epa_after_first_drive` (input B): PASS (corrected; verdict
  preserved with magnitudes adjusted -- see the per-feature x
  per-test-season table above).
- `defense_stabilized_flag = int(B < A)`: PASS 3/3 -- the *direction
  of the inequality* between the two corrected EPA means carries
  signal even though the level of input A alone does not.

This is a non-obvious finding for N03 model interpretation: a binary
"defense improved after drive 1" indicator can be informative even
when neither raw EPA value is, because the binary captures the
relative-trajectory information without exposing the model to the
noise in input A's level. N03 should treat `defense_stabilized_flag`
as a defense-trajectory feature, not as an EPA aggregate, when
interpreting feature importances.

### Candidate-vs-trigger-fields deducibility (plan-time audit)

On the 9,556 drive-2+ triggers (where drive-summary features are
evaluated), **none of the 02b candidates are materially deducible from
`trigger_events.csv` fields alone** -- the drives JSON is required to
reconstruct drive-1 outcomes. On the 1,860 drive-1 triggers,
`dog_received_opening_kickoff` is partially deducible from
`(possession_team == dog_team)` (1,749 of 1,860 = 94% match), but those
rows are the same ones where the other drive-summary features are
NULL per D4.

Interpretation: if a 02b candidate passes stability and N03 finds it
useful, the marginal information is in the drive-1 vs drive-2+ split,
not in the candidate alone.

### Section provenance

- Last writer: this 02b run (timestamp + commit above).
- Splicing strategy: sentinel-delimited; re-running 02b refreshes only
  this section. Re-running 02a in its current form WILL clobber this
  section -- known issue tracked in `research/tech_debt.md`.
"""

new_section = SECTION_BEGIN + "\\n" + section_body.rstrip() + "\\n" + SECTION_END

if FEATURE_VALIDATION_SCHEMA.exists():
    existing_text = FEATURE_VALIDATION_SCHEMA.read_text(encoding="utf-8")
    if SECTION_BEGIN in existing_text and SECTION_END in existing_text:
        start = existing_text.index(SECTION_BEGIN)
        end = existing_text.index(SECTION_END) + len(SECTION_END)
        updated = existing_text[:start] + new_section + existing_text[end:]
        print(f"[ok] spliced 02b section in place (existing markers found)")
    else:
        updated = existing_text.rstrip() + "\\n\\n" + new_section + "\\n"
        print(f"[ok] appended 02b section at end of sidecar (markers added)")
else:
    # If 02b runs without 02a's sidecar existing, write a stub header so the
    # file is self-contained. In normal Phase 0 sequencing this branch is
    # never hit (02a always runs first).
    header = (
        "# feature_validation.csv -- schema sidecar\\n\\n"
        "(02a section missing -- run 02a to regenerate the baseline-efficiency content.)\\n\\n"
    )
    updated = header + new_section + "\\n"
    print(f"[warn] sidecar did not exist; wrote stub header + 02b section.")

FEATURE_VALIDATION_SCHEMA.write_text(updated, encoding="utf-8")
print(f"[ok] wrote feature_validation.schema.md ({len(updated):,} chars)")
print(f"     path: {FEATURE_VALIDATION_SCHEMA}")
''')


# ---------------------------------------------------------------------------
# Cell 21 — Summary (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02b0015", """
## Phase 02b-h — Summary, headline stats, STOP banner
""")


# ---------------------------------------------------------------------------
# Cell 22 — Summary print code
# ---------------------------------------------------------------------------
add("code", "c02b0016", '''
print("=" * 70)
print("Notebook 02b -- opening-drive shock features -- summary")
print("=" * 70)

print(f"\\nIn-scope corpus:")
print(f"  trigger rows (post NaN final_fav_won drop): {len(feature_matrix_df):,}")
print(f"  /plays cache hits this run:                 "
      f"{n_plays_lookups} lookups, "
      f"{sum(len(v) for v in plays_by_game.values()):,} plays across "
      f"{len(plays_by_game):,} games")
print(f"  /drives cache hits this run:                "
      f"{n_drives_lookups} lookups, "
      f"{sum(len(v) for v in drives_by_game.values()):,} drives across "
      f"{len(drives_by_game):,} games")

n_drive1_final = int((feature_matrix_df["drive_number_in_game"] == 1).sum())
print(f"\\nD4 null-policy scale (additional ask from 02b plan-approval):")
print(f"  drive_number_in_game == 1 (drive-summary feats NULL): "
      f"{n_drive1_final:,} ({n_drive1_final / len(feature_matrix_df) * 100:.2f}%)")

print(f"\\nPer-feature x per-test-season results ({FEATURE_SET_VERSION}):")
print(f"  {'feature':<32} {'window->test':<18} "
      f"{'brier_b':>9} {'brier_c':>9} {'d_brier':>10} "
      f"{'ece_b':>8} {'ece_c':>8} {'d_ece':>10} {'stab':>6}")
for _, r in eval_df.sort_values(["feature", "test_season"]).iterrows():
    win = f"{r['train_window']}->{int(r['test_season'])}"
    print(f"  {r['feature']:<32} {win:<18} "
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
    print(f"  {feat:<32} {verdict:<5} "
          f"({n_pos}/3 brier-improving, {n_pos_cal}/3 ece-improving)")

print(f"\\nNull counts per feature (in-scope triggers: {len(feature_matrix_df):,}):")
for feat in CANDIDATE_FEATURES:
    pct = (null_counts[feat] / len(feature_matrix_df) * 100) if len(feature_matrix_df) else 0
    tag = " (D4 drive-summary)" if feat in DRIVE_SUMMARY_FEATURES else ""
    print(f"  {feat:<32} {null_counts[feat]:>5,} null ({pct:5.2f}%){tag}")

print(f"\\nDeliverables (research/results/):")
for path in [FEATURE_VALIDATION_CSV, FEATURE_VALIDATION_SCHEMA]:
    size = path.stat().st_size
    print(f"  {path.name:<40} {size:>10,} bytes")
''')


# ---------------------------------------------------------------------------
# Cell 23 — Budget print + STOP banner
# ---------------------------------------------------------------------------
add("code", "c02b0017", '''
calls_log_df = pd.read_csv(CALL_LOG)
n_total_log_rows = len(calls_log_df)
n_fresh_cfbd_total = int(((calls_log_df["service"] == "cfbd")
                          & (calls_log_df["cached"] == 0)).sum())

# This-run slice: log rows after n_log_before (snapshot taken in Phase 02b-b).
this_run_calls = calls_log_df.iloc[n_log_before:].copy()
n_this_run = len(this_run_calls)
n_this_run_fresh = int((this_run_calls["cached"] == 0).sum())

print("=" * 64)
print("CFBD call budget -- Notebook 02b")
print("=" * 64)
print(f"\\nThis notebook run:")
print(f"  total calls this run:     {n_this_run:>5,}  ({n_plays_lookups} /plays + {n_drives_lookups} /drives)")
print(f"  fresh (uncached) this run: {n_this_run_fresh:>5,}  (budget: 0)")

assert n_this_run_fresh == 0, (
    f"02b budget invariant violated: {n_this_run_fresh} fresh CFBD call(s) "
    f"this run. 02b is supposed to spend 0 fresh CFBD calls."
)

# NOTE: the hardcoded 1,000-call monthly limit below is incorrect on the
# current API key (actual quota: 3,000/cycle per research/results/
# budget_reconciliation.md). Tracked as item 1 in research/tech_debt.md;
# display-only, no budget-enforcement dependency.
print(f"\\nCumulative across all notebooks (call log: {n_total_log_rows:,} rows):")
print(f"  total fresh CFBD calls (lifetime):    {n_fresh_cfbd_total:,}")
print(f"  monthly free-tier limit (BUILD_SPEC A.4 stated):  1,000")
print(f"    actual quota on current key (probe header):     3,000")
print(f"  remaining this billing cycle (against actual 3K): {3000 - n_fresh_cfbd_total:,}")
if n_fresh_cfbd_total >= 0.8 * 3000:
    print(f"  [WARN] >=80% of 3,000-call cycle consumed.")

print(f"\\n[ok] notebook 02b complete -- STOP per R22. "
      f"Do not start Notebook 02c without approval.")
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
