"""
Deterministic builder for research/notebooks/02c_explosive_vs_sustained.ipynb.

Mirrors _build_02b.py:
  - Same cache-hit-assertion contract.
  - Same defensive-append pattern for feature_validation.csv.
  - Sentinel-delimited splicing of the schema sidecar so 02a + 02b content is
    preserved verbatim.

02c-specific additions vs 02b:
  - Adds a play-type registry + runtime defensive enumeration (D4 ask):
    halts if any scoring playType in the cache is not in our enumeration.
  - Implements drive-level attribution for the V5 DDL block-3 points buckets
    (D2/D3/D4) with the D12 PAT/2pt-to-preceding-TD rule.
  - Adds momentum features (continuous + R16-safe indicator + binary
    prior-drive). Continuous uses train-window-specific median imputation
    per D8 augmentation; the per-window imputation value is stored in
    a NEW `imputation_value` column on feature_validation.csv.
  - Adds the P2 Roebber-2022 framing note to the momentum section of the
    sidecar splice.

This is a scratchpad file (per the research/notebooks/_*.py convention).
Not part of the deliverable.
"""

from __future__ import annotations

import json
import pathlib
import sys
import textwrap

OUT = pathlib.Path(__file__).resolve().parent / "02c_explosive_vs_sustained.ipynb"

# Pull the canonical _chrono_key source from the shared helper module
# (single-source-of-truth across 02a/02b/02c build scripts). See
# research/notebooks/_lib_chrono.py for the function definition and
# the corrections rationale.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _lib_chrono import CHRONO_KEY_SOURCE  # noqa: E402

CELLS: list[tuple[str, str, str]] = []


def add(cell_type: str, cell_id: str, src: str) -> None:
    CELLS.append((cell_type, cell_id, textwrap.dedent(src).lstrip("\n")))


# ---------------------------------------------------------------------------
# Cell 0 — Title + hypothesis docstring (markdown)
# ---------------------------------------------------------------------------
add("markdown", "bd02c000", """
# Phase 0 — Notebook 02c: Explosive vs sustained drives + post-explosive momentum

## Hypothesis (two parallel per-feature stability claims under R6)

**H1 (V5 DDL block 3):** how the underdog has scored its points so far
(explosive plays / sustained drives / returns) carries comeback-equity
signal beyond the pre-game baseline. Football mechanism: 14 points from
two 70-yard TD passes is structurally different from 14 points from two
12-play 75-yard drives -- the former implies a favorite defense leaking
chunk plays (which tend to recur); the latter implies a favorite defense
that lost a possession battle (more state-dependent).

**H2 (post-explosive momentum):** an underdog explosive play has a
decaying effect on the favorite defense's subsequent performance. The
recency of the underdog's most recent explosive play (and whether the
immediately preceding drive contained one) carries additional signal
beyond the binary opening-drive feature already validated in 02b.

Both H1 and H2 use the pre-game-only baseline locked in 02a
(`pregame_spread`, `rating_gap`, `fav_pregame_rating`, `dog_pregame_rating`,
`spread_movement`, `spread_movement_is_null`). Per-feature null policy
applies (decision **B** from 02a). For the continuous momentum feature,
R16-safe NaN handling applies (impute to train-window median + paired
`*_is_null` indicator) -- see D8 below for the per-window imputation
provenance.

The narrative V5.1 hypothesis ("dog explosive scoring reverts more than
sustained dog dominance") is Notebook 03's end-to-end job; 02c only
isolates each candidate feature's marginal Brier contribution.

## What this notebook DOES NOT do

- Does not modify `trigger_events.csv` or `trigger_outcomes.csv`.
- Does not pull any fresh CFBD data -- every `/plays` AND `/drives` lookup
  must hit the cache produced by N01. Cache-hit assertion fails loud on
  any miss.
- Does not select features for the production model -- N03's job.
- Does not test feature groups other than explosive-vs-sustained +
  momentum (those are 02a / 02b / 02d-g).
- Does not tune hyperparameters of the L1 logreg -- uses sklearn default
  `C=1.0` with a fixed seed, identical to 02a / 02b.

## Spec references

- `BUILD_SPEC.md` Phase 0 Notebook 02 deliverable spec -- `feature_validation.csv` shape
- `BUILD_SPEC.md` `trigger_features` DDL -- explosive-vs-sustained block (V5 lines 181-187)
- `research/future_features.md` -- "Momentum / decaying-shock features (target: 02c)"
- `.cursorrules` **R2 + R3** -- no lookahead; `assert_no_lookahead()` mandatory on every feature extraction
- `.cursorrules` **R5** -- walk-forward validation only
- `.cursorrules` **R6** -- stability rule (>=2 of 3 test seasons)
- `.cursorrules` **R7** -- L1 logreg / shallow GBM only
- `.cursorrules` **R8** -- ECE on 10 bins, post-calibration
- `.cursorrules` **R16** -- pre-game-safe NaN handling for non-random missingness
- `.cursorrules` **R19** -- record rejected features too
- `.cursorrules` **R22** -- STOP at end of 02c; do not start 02d without approval

## Decision-points log (from the 02c plan-approval)

- **D1** -- explosive-play thresholds reused verbatim from 02b: `EXPLOSIVE_PASS_YARDS = 20`, `EXPLOSIVE_RUSH_YARDS = 12`. Type-specific (PFF/SP+ convention).
- **D2** -- `dog_points_from_explosives` uses drive-level attribution: sum of points from dog-offense drives that contain at least one dog explosive play.
- **D3** -- `dog_points_from_sustained`: points from dog-offense drives containing zero dog explosive plays (complement of D2 inside the offensive bucket; return-TD drives are not in either bucket per D4).
- **D4** -- `dog_points_from_returns` covers return-style touchdowns (KO / punt / interception / fumble / blocked punt / blocked FG return TDs) and defensive 2pt returns and safeties scored by the dog. **D4 augmentation:** enumerate the cache's scoring playTypes at runtime (Phase 02c-c) and halt loud if any aren't in our registry.
- **D5** -- `dog_explosive_play_count` counts only dog-offense explosive plays (using D1 thresholds).
- **D6** -- `dog_avg_drive_yards` / `dog_avg_drive_plays` are means over completed dog-offense drives only; NULL if zero completed dog-offense drives by trigger time.
- **D7** -- `dog_points_from_*` are 0 (not NULL) once at least one dog-offense drive has completed (meaningful: drives happened but no points from this bucket). NULL only when zero dog-offense drives have completed.
- **D8** -- Primary momentum form: continuous `seconds_since_last_dog_explosive_play` + R16-safe paired indicator `seconds_since_last_dog_explosive_play_is_null`. Imputation: **median computed per train window** (no leakage); per-window value stored in the new `imputation_value` column on `feature_validation.csv`.
- **D9** -- Secondary momentum form: binary `prior_drive_had_dog_explosive_play`. NULL for drive-1 triggers (no prior drive). Per-feature null drop.
- **D10** -- Pair-candidate treatment: when the candidate is `seconds_since_last_dog_explosive_play`, the candidate model adds BOTH the continuous column and the `_is_null` indicator; the CSV row is keyed on the continuous-feature name. Single stability verdict per pair.
- **D11** -- Categorical-window variants (`had_explosive_in_last_120s/300s/600s`) considered and rejected pre-execution. The post-execution trigger logic (P3 from plan-approval review) determines whether categorical goes to `future_features.md` or `tech_debt.md` based on D8 vs D9 results.
- **D12** -- PAT/2pt attribution: PAT/2pt points attribute to the SAME bucket as the preceding TD. Explosive TD + successful PAT = 7 points to `dog_points_from_explosives`, not a 6/1 split. Documented in the schema-sidecar accounting section.

## Plan-time pre-execution redundancy audit

Per the 02c plan-approval, every candidate feature pair was audited at
plan time for structural identity. Verdict: **zero structural duplicates
among 02c's 8 candidates.** Several **conditional identities** were
flagged for the sidecar (distinct from bit-identical `redundant_with`
tags from 02a):

1. `dog_explosive_play_count` vs 02b's `opening_drive_was_explosive_td`:
   conditional overlap on drive-1-with-dog-explosive-TD triggers; not
   structural identity.
2. `dog_avg_drive_yards` vs 02b's `opening_drive_yards`: equality when
   `drive_number_in_game == 2` AND dog had drive 1 (else not).
3. `dog_avg_drive_plays` vs 02b's `opening_drive_plays`: same conditional
   identity structure.

`dog_points_from_explosives + dog_points_from_sustained + dog_points_from_returns`
sums to (a quantity close to) `dog_score_at_trigger` per the accounting
identity, modulo the trigger play itself: see the sidecar's "Accounting"
section for the exact relation and D12's role.

## Deliverables produced by this notebook

1. `research/results/feature_validation.csv` -- adds 24 rows from 02c
   (8 features x 3 test seasons), tagged `feature_set_version =
   v1_explosive_vs_sustained`. **Adds a new `imputation_value` column**
   (REAL, NULL except for the continuous momentum feature's rows). 02a's
   18 rows + 02b's 30 rows are preserved by the defensive-append.
2. `research/results/feature_validation.schema.md` -- splices a
   sentinel-delimited "02c -- Explosive vs sustained + post-explosive
   momentum" section into the existing sidecar. 02a's + 02b's sections
   are preserved verbatim.
3. `research/notebooks/02c_explosive_vs_sustained.ipynb` -- this notebook.

No changes to `trigger_events.csv`, `trigger_outcomes.csv`,
`trigger_events_bucket_counts.csv`, `data_quality_report.md`,
`budget_reconciliation.md`, `tech_debt.md`, `future_features.md`. No
new cache files. No fresh CFBD calls.

## Walk-forward windows (decision **B**, locked in 02a; carried verbatim)

| Train seasons | Val season | Test season |
|---|---|---|
| 2015-2020 | 2021 | 2022 |
| 2015-2021 | 2022 | 2023 |
| 2015-2022 | 2023 | 2024 |

## Baseline (decision **alpha**, locked in 02a; carried verbatim)

`BASELINE_PREGAME_FEATURES = [pregame_spread, rating_gap, fav_pregame_rating,
dog_pregame_rating, spread_movement, spread_movement_is_null]`. Same
R16-safe handling for `spread_movement`.

## Call budget

**This notebook's budget: 0 fresh CFBD calls.** Every `/plays` AND
`/drives` lookup is a cache hit produced by N01. Lifetime audited count
is 253/3000 going into this notebook per
`research/data/cache/cfbd_call_log.csv`.
""")


# ---------------------------------------------------------------------------
# Cell 1 — Imports, paths, env, fail-fast (code)
# ---------------------------------------------------------------------------
add("code", "c02c0001", '''
"""
Notebook 02c -- imports, environment, path constants, fail-fast checks.
Same structure as Notebook 02a / 02b. Run this cell first.
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

assert RESEARCH_DIR.name == "research", (
    f"Expected to run inside research/notebooks/. Got NOTEBOOK_DIR={NOTEBOOK_DIR}. "
    f"cd into research/notebooks/ and re-launch jupyter."
)
assert ENV_PATH.exists(), (
    f"Did not find {ENV_PATH}. The CFBD key would only be used if the cache "
    f"is missing entries (none expected in 02c); load it anyway so the same "
    f"cfbd_get() helper works."
)

load_dotenv(ENV_PATH)
assert os.environ.get("CFBD_API_KEY"), (
    "CFBD_API_KEY is not set. 02c should NOT issue fresh calls, but the "
    "cfbd_get() helper still requires the key in scope. Populate "
    f"{ENV_PATH} and re-run."
)

TRIGGER_EVENTS_CSV = RESULTS_DIR / "trigger_events.csv"
TRIGGER_OUTCOMES_CSV = RESULTS_DIR / "trigger_outcomes.csv"
assert TRIGGER_EVENTS_CSV.exists(), (
    f"Expected {TRIGGER_EVENTS_CSV} (Notebook 01 deliverable). Run N01 first."
)
assert TRIGGER_OUTCOMES_CSV.exists(), (
    f"Expected {TRIGGER_OUTCOMES_CSV} (Notebook 01 deliverable). Run N01 first."
)

FEATURE_VALIDATION_CSV = RESULTS_DIR / "feature_validation.csv"
FEATURE_VALIDATION_SCHEMA = RESULTS_DIR / "feature_validation.schema.md"

print(f"[ok] paths resolved relative to {NOTEBOOK_DIR}")
print(f"[ok] CFBD_API_KEY loaded from {ENV_PATH}")
print(f"[ok] cache dir: {CACHE_DIR}")
print(f"[ok] N01 deliverables present: trigger_events.csv, trigger_outcomes.csv")
''')


# ---------------------------------------------------------------------------
# Cell 2 — HTTP helpers (code, same code path as 00/01/02a/02b)
# ---------------------------------------------------------------------------
add("code", "c02c0002", '''
"""
HTTP helpers -- same code as Notebook 00/01/02a/02b, same cache directory.
02c expects ALL calls to be cache hits; the assertion in Phase 02c-b fails
loud if any go fresh.
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
print(f"[ok] sharing cache with Notebook 00/01/02a/02b at {CACHE_DIR}")
''')


# ---------------------------------------------------------------------------
# Cell 3 — Configuration (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02c0003", """
## Configuration

`WALK_FORWARD_WINDOWS` and `BASELINE_PREGAME_FEATURES` are **carried verbatim from 02a** -- locked at 02a plan-approval and binding for 02b-g. Not re-decided here.

`EXPLOSIVE_PASS_YARDS = 20`, `EXPLOSIVE_RUSH_YARDS = 12`, `EXPLOSIVE_PASS_PLAY_TYPES`, `EXPLOSIVE_RUSH_PLAY_TYPES` -- **carried verbatim from 02b** (decision **D1**, type-specific thresholds; PFF/SP+ convention). 02c-g import these constants directly so they're defined once in 02b and reused unchanged.

`SCORING_PLAYTYPE_REGISTRY` -- the play-type categorization used by the D2/D3/D4/D12 attribution logic. Seven attribution categories plus an `exclude` sentinel:

- `offensive_td` -- offensive TDs (Passing / Rushing / own-fumble-recovery-in-EZ); attributed to explosives bucket if drive contained dog explosive play, else sustained.
- `fg` -- field goals; same drive-level attribution rule.
- `return_td` -- defensive / special-teams TDs (interception / fumble / kickoff / punt / blocked-punt / blocked-FG returns, plus three CFBD alt-encodings); attributed to returns bucket.
- `pat_1pt` / `pat_2pt` -- PAT or 2pt conversion; per D12, attributed to the SAME bucket as the preceding TD.
- `safety_def` -- defensive safety (2pt awarded to the team NOT on offense); attributed to returns bucket.
- `pat_def_ret` -- defensive PAT return (rare; 2pt return on a PAT attempt); attributed to returns bucket.
- `exclude` -- alt-encoding playTypes whose point value cannot be unambiguously determined from playType alone (same-bucket-different-value or cross-bucket ambiguity), or anomalous CFBD `scoring=True` false positives. The points-bucket helper short-circuits before attribution. 1,562 plays / ~2% of recognized scoring; revisit conditions in `research/tech_debt.md` item 4. Per-playType evidence in `research/results/_investigate_02c_unknown_scoring.csv`.

The registry is the canonical enumeration -- Phase 02c-c runs a defensive enumeration against the cache and halts loud if a scoring play type appears that isn't in the registry. Same pattern as N01's scoring registry per D4 augmentation.

`CANDIDATE_FEATURES` (8 total):

- **V5 DDL block 3 (6 features):** `dog_points_from_explosives`, `dog_points_from_sustained`, `dog_points_from_returns`, `dog_explosive_play_count`, `dog_avg_drive_yards`, `dog_avg_drive_plays`.
- **Momentum (2 features):** `seconds_since_last_dog_explosive_play` (continuous, R16-safe imputed with paired `_is_null` indicator -- decision D8/D10), `prior_drive_had_dog_explosive_play` (binary, decision D9).

`REDUNDANT_WITH = {}` -- the plan-time redundancy audit found zero structural duplicates among 02c candidates. Three **conditional identities** were documented (against 02b's drive-1 features); see the sidecar's "Redundancy discoveries" section.

`FEATURE_SET_VERSION = "v1_explosive_vs_sustained"` -- the per-notebook tag stamped into every row this notebook writes.
""")


# ---------------------------------------------------------------------------
# Cell 4 — Configuration constants (code)
# ---------------------------------------------------------------------------
add("code", "c02c0004", '''
SEASONS: list[int] = list(range(2015, 2025))
SEASON_TYPES: list[str] = ["regular", "postseason"]

FEATURE_SET_VERSION: str = "v1_explosive_vs_sustained"

# Walk-forward windows -- decision B from 02a plan-approval; carried verbatim.
WALK_FORWARD_WINDOWS: list[dict] = [
    {"train_seasons": list(range(2015, 2021)), "val_season": 2021,
     "test_season": 2022, "train_window_label": "2015-2020"},
    {"train_seasons": list(range(2015, 2022)), "val_season": 2022,
     "test_season": 2023, "train_window_label": "2015-2021"},
    {"train_seasons": list(range(2015, 2023)), "val_season": 2023,
     "test_season": 2024, "train_window_label": "2015-2022"},
]

# Pre-game baseline columns -- decision alpha; carried verbatim from 02a.
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

# Explosive-play thresholds -- decision D1 from 02b plan-approval; carried verbatim.
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

# --- Scoring play-type registry (D2/D3/D4/D12) -----------------------------
# Maps known scoring playType -> category. The runtime enumeration in
# Phase 02c-c halts if a scoring playType in the cache isn't in this map.
#
# Categories:
#   offensive_td -- offensive TDs (Passing / Rushing); +6 to dog bucket
#                   depending on whether the drive had a dog explosive
#                   (explosives) or didn't (sustained). PAT/2pt then
#                   attribute to the same bucket per D12.
#   fg           -- field goals; +3 to dog bucket using same explosive-
#                   drive rule as offensive_td. (No PAT follows.)
#   return_td    -- defensive / special-teams TDs; +6 to returns bucket.
#                   `play.offense` is the team that lost the ball;
#                   scoring team = the OTHER team.
#   pat_1pt      -- PAT (extra point); +1 to bucket of preceding TD (D12).
#   pat_2pt      -- successful 2pt conversion; +2 to bucket of preceding
#                   TD (D12). CFBD encodes 2pt success in various play-
#                   type strings; enumeration is defensive.
#   safety_def   -- safety (2pt awarded to the team NOT on offense).
#                   For dog: +2 to returns bucket (defensive score).
#   pat_def_ret  -- defensive PAT return (very rare; 2pt return on a PAT
#                   attempt). +2 to returns bucket.
#   exclude      -- alt-encoding playTypes whose point value cannot be
#                   unambiguously determined from playType alone (same-
#                   bucket-different-value or cross-bucket ambiguity), or
#                   anomalous CFBD scoring=True false positives. The
#                   points-bucket helper short-circuits cat == 'exclude'
#                   before attribution. 1,562 plays total (~2% of
#                   recognized scoring); revisit conditions documented
#                   in research/tech_debt.md item 4.
#
# Seed the registry with the documented + commonly observed CFBD playType
# strings PLUS the 18 alt-encoding playTypes the 02c plan-approval
# investigation discovered. Per-playType evidence:
#   research/results/_investigate_02c_unknown_scoring.csv
#   research/results/_investigate_02c_unknown_scoring.summary.json
# The runtime enumeration in Phase 02c-c catches anything missing.
SCORING_PLAYTYPE_REGISTRY: dict[str, str] = {
    # Offensive TDs (6 pts to offense)
    "Passing Touchdown":              "offensive_td",
    "Rushing Touchdown":              "offensive_td",
    "Fumble Recovery (Own)":          "offensive_td",  # alt-encoding: offense recovers own fumble in EZ; n=10 sample 10/10 own-team-TD; drive-attribution check showed 8/9 dog-offense cases land on already-explosive drives
    # Field goals (3 pts to offense)
    "Field Goal Good":                "fg",
    # Defensive / special-teams TDs (6 pts to defense / receiving team)
    "Interception Return Touchdown":  "return_td",
    "Fumble Return Touchdown":        "return_td",
    "Kickoff Return Touchdown":       "return_td",
    "Punt Return Touchdown":          "return_td",
    "Blocked Punt Touchdown":         "return_td",
    "Blocked Field Goal Touchdown":   "return_td",
    "Missed Field Goal Return Touchdown": "return_td",
    "Fumble Recovery Touchdown":      "return_td",
    "Fumble Recovery (Opponent)":     "return_td",  # alt-encoding; n=50 sample 50/50 fumble-return-TD (4 false-positive exceptions had non-(KICK) trailers but still TD)
    "Pass Interception Return":       "return_td",  # alt-encoding; full pop 3/3 INT-return-TD
    "Kickoff Return (Offense)":       "return_td",  # alt-encoding (kickoff fumbled by returner, kicking team recovers for TD); full pop 6/6
    "Defensive 2pt Conversion":       "pat_def_ret",
    # PAT (1 pt)
    "Extra Point Good":               "pat_1pt",
    "PAT Good":                       "pat_1pt",
    # 2pt conversion (2 pts)
    "Two Point Pass":                 "pat_2pt",
    "Two Point Rush":                 "pat_2pt",
    "Two-Point Pass":                 "pat_2pt",
    "Two-Point Rush":                 "pat_2pt",
    "2pt Conversion Good":            "pat_2pt",
    # Safety (2 pts to defense)
    "Safety":                         "safety_def",
    # Excluded alt-encoding playTypes (registry sentinel = no attribution)
    # Same-bucket-different-value or cross-bucket ambiguity:
    "Uncategorized":                  "exclude",  # n=50: 49 pat_1pt + 1 pat_2pt (different point value, same PAT bucket)
    "Punt":                           "exclude",  # n=50: 49 return_td + 1 safety_def (different point value, same returns bucket)
    "Kickoff":                        "exclude",  # n=50: ~47 return_td + 1 safety_def + 2 ambiguous (no clear scoring trailer)
    "Blocked Punt":                   "exclude",  # n=10: 8 return_td + 2 safety_def
    "Sack":                           "exclude",  # n=10: 9 sack-strip-return-TD + 1 anomaly (CFBD bundling artifact)
    "Pass Reception":                 "exclude",  # n=10: 9 offensive_td + 1 safety_def (cross-bucket)
    "Interception":                   "exclude",  # n=10: 9 return_td + 1 pat_def_ret (different point value, same returns bucket)
    "Blocked Field Goal":             "exclude",  # n=10: 9 return_td + 1 anomaly (recovered blocked PAT)
    "Rush":                           "exclude",  # n=10: 7 offensive_td + 3 safety_def-against-offense (cross-bucket)
    "Pass Incompletion":              "exclude",  # n=3 full pop: anomalies (TD, no-text, safety)
    "Penalty":                        "exclude",  # n=3 full pop: declined penalties; CFBD scoring=True false positive
    "End Period":                     "exclude",  # n=1 full pop: anomalous encoding
    "Timeout":                        "exclude",  # n=1 full pop: anomalous encoding
    "placeholder":                    "exclude",  # n=1 full pop: anomalous encoding
}
RETURN_TD_PLAY_TYPES: frozenset[str] = frozenset({
    pt for pt, cat in SCORING_PLAYTYPE_REGISTRY.items() if cat == "return_td"
})

# Candidate features (V5 trigger_features DDL lines 181-187 + momentum).
CANDIDATE_FEATURES: list[str] = [
    # V5 DDL block 3 (6 features)
    "dog_points_from_explosives",
    "dog_points_from_sustained",
    "dog_points_from_returns",
    "dog_explosive_play_count",
    "dog_avg_drive_yards",
    "dog_avg_drive_plays",
    # Momentum (2 features)
    "seconds_since_last_dog_explosive_play",
    "prior_drive_had_dog_explosive_play",
]

# Continuous momentum feature uses R16-safe imputation (D8/D10).
# When this feature is the candidate, the candidate model adds BOTH
# the continuous column and the `_is_null` indicator; the median is
# computed per train window (no leakage) and stored in the eval row.
MOMENTUM_CONTINUOUS_FEATURE: str = "seconds_since_last_dog_explosive_play"
MOMENTUM_CONTINUOUS_INDICATOR: str = "seconds_since_last_dog_explosive_play_is_null"

# Structural-redundancy map for 02c. Empty: the plan-time redundancy audit
# found zero structural duplicates among 02c candidates.
REDUNDANT_WITH: dict[str, str] = {}

# Reproducibility seed -- same as 02a / 02b.
RANDOM_STATE: int = 42

print(f"seasons: {SEASONS}")
print(f"season types: {SEASON_TYPES}")
print(f"feature_set_version: {FEATURE_SET_VERSION}")
print(f"walk-forward windows (locked from 02a, binding for 02b-g):")
for w in WALK_FORWARD_WINDOWS:
    print(f"  train={w['train_window_label']}  val={w['val_season']}  test={w['test_season']}")
print(f"baseline pre-game features ({len(BASELINE_PREGAME_FEATURES)}): {BASELINE_PREGAME_FEATURES}")
print(f"candidate features ({len(CANDIDATE_FEATURES)}):")
for f in CANDIDATE_FEATURES:
    tag = ""
    if f == MOMENTUM_CONTINUOUS_FEATURE:
        tag = " (R16-safe pair with " + MOMENTUM_CONTINUOUS_INDICATOR + "; D8/D10)"
    print(f"  - {f}{tag}")
print(f"explosive thresholds (D1, carried from 02b):")
print(f"  pass yards >= {EXPLOSIVE_PASS_YARDS}; pass types: {sorted(EXPLOSIVE_PASS_PLAY_TYPES)}")
print(f"  rush yards >= {EXPLOSIVE_RUSH_YARDS}; rush types: {sorted(EXPLOSIVE_RUSH_PLAY_TYPES)}")
print(f"scoring playType registry: {len(SCORING_PLAYTYPE_REGISTRY)} entries across "
      f"{len(set(SCORING_PLAYTYPE_REGISTRY.values()))} categories "
      f"({sorted(set(SCORING_PLAYTYPE_REGISTRY.values()))})")
print(f"return_td playTypes: {sorted(RETURN_TD_PLAY_TYPES)}")
print(f"redundant_with map ({len(REDUNDANT_WITH)} entries): {REDUNDANT_WITH}")
print(f"random state: {RANDOM_STATE}")
''')


# ---------------------------------------------------------------------------
# Cell 5 — Load triggers (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02c0005", """
## Phase 02c-a — Load trigger artifacts

Identical join setup to 02a / 02b: read `trigger_events.csv` and `trigger_outcomes.csv`, inner-join on `(game_id, fav_deficit)`, drop rows with `final_fav_won is NaN`. The label only enters as the model target in the walk-forward eval cell; it does NOT enter any feature extractor.

Print the drive-1 trigger count for symmetry with 02b's reporting (also relevant to D9 -- `prior_drive_had_dog_explosive_play` is NULL for drive-1 triggers since there's no prior drive).
""")


# ---------------------------------------------------------------------------
# Cell 6 — Load triggers code
# ---------------------------------------------------------------------------
add("code", "c02c0006", '''
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
print(f"In-scope rows for 02c: {len(trigger_full_df):,}")

# Drive-1 scale (D9 null floor for prior_drive_had_dog_explosive_play).
n_drive1 = int((trigger_full_df["drive_number_in_game"] == 1).sum())
n_drive2plus = int((trigger_full_df["drive_number_in_game"] >= 2).sum())
print(f"\\nDrive-1 scale (relevant to D9 null policy):")
print(f"  drive_number_in_game == 1 (prior_drive_had_dog_explosive_play NULL): "
      f"{n_drive1:,} ({n_drive1 / len(trigger_full_df) * 100:.1f}%)")
print(f"  drive_number_in_game >= 2: "
      f"{n_drive2plus:,} ({n_drive2plus / len(trigger_full_df) * 100:.1f}%)")

# Sanity: ALWAYS_PRESENT_PREGAME_COLS must be non-null (A.7 + N01 contract).
for col in ALWAYS_PRESENT_PREGAME_COLS:
    n_null = int(trigger_full_df[col].isna().sum())
    assert n_null == 0, (
        f"always-present pre-game column {col!r} has {n_null} nulls on the "
        f"in-scope subset; expected 0 per the trigger_events.schema.md contract."
    )
print(f"\\n[ok] always-present pre-game columns are non-null on the in-scope subset")

n_sm_null = int(trigger_full_df["spread_movement"].isna().sum())
print(f"     spread_movement nulls (pre-impute): {n_sm_null:,} "
      f"({n_sm_null / len(trigger_full_df) * 100:.2f}% of in-scope)")
''')


# ---------------------------------------------------------------------------
# Cell 7 — Cache re-load (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02c0007", """
## Phase 02c-b — Re-load cached `/plays` AND `/drives` (zero fresh calls)

Identical setup to 02b: iterate the (season, season_type, week) tuples for `/plays` and (season, season_type) tuples for `/drives`. **Assert every call is a cache hit.** The cell fails loud on any cache miss -- 02c's budget is 0 fresh CFBD calls.
""")


# ---------------------------------------------------------------------------
# Cell 8 — Cache re-load code (verbatim from 02b)
# ---------------------------------------------------------------------------
add("code", "c02c0008", '''
work_tuples_df = (
    trigger_full_df[["season", "season_type", "week"]]
    .drop_duplicates()
    .sort_values(["season", "season_type", "week"])
    .reset_index(drop=True)
)
print(f"distinct (season, season_type, week) tuples to load from cache: {len(work_tuples_df)}")

n_log_before = sum(1 for _ in CALL_LOG.open("r", encoding="utf-8")) - 1  # minus header

plays_by_game: dict[int, list[dict]] = {}
# Note: no negative-play.id pre-filter. The 19,828 plays across 115 games
# with negative-id encoding (e.g., id='-6654') are LEGITIMATE plays in real
# games -- CFBD uses two id formats (standard 18-digit positive integers
# and compact negative integers) but both have complete play data
# (playText, drive metadata, yardage, scoring flags). The composite
# chrono_key below orders these correctly via (period, period_seconds_
# elapsed, driveNumber, playNumber) without referencing play.id.
# See research/corrections_log.md "CFBD play.id encoding observation".
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
        # Track negative-id encoding stats for the execution report only;
        # plays are NOT dropped.
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

calls_log_df = pd.read_csv(CALL_LOG)
this_run_calls = calls_log_df.iloc[n_log_before:].copy()
n_fresh_this_cell = int((this_run_calls["cached"] == 0).sum())
assert n_fresh_this_cell == 0, (
    f"02c budget invariant violated: {n_fresh_this_cell} non-cached CFBD call(s) "
    f"issued in this cell. 02c is supposed to spend 0 fresh CFBD calls; the "
    f"cache for some (year, type, week) or (year, type) tuple is missing or "
    f"stale. Stop and investigate cache invalidation before continuing."
)
n_plays_lookups = int((this_run_calls["endpoint"] == "/plays").sum())
n_drives_lookups = int((this_run_calls["endpoint"] == "/drives").sum())
print(f"[ok] cache-hit assertion passed: {n_plays_lookups} /plays lookups, "
      f"{n_drives_lookups} /drives lookups, all cached.")

''' + CHRONO_KEY_SOURCE + '''


for gid in plays_by_game:
    plays_by_game[gid].sort(key=_chrono_key)
for gid in drives_by_game:
    drives_by_game[gid].sort(
        key=lambda d: (d.get("driveNumber") if d.get("driveNumber") is not None else 10**9)
    )
print(f"[ok] plays_by_game sorted by composite _chrono_key "
      f"({len(plays_by_game):,} games)")
print(f"[ok] drives_by_game sorted by driveNumber ({len(drives_by_game):,} games)")
''')


# ---------------------------------------------------------------------------
# Cell 9 — Scoring playtype enumeration (markdown) — NEW for 02c (D4 augmentation)
# ---------------------------------------------------------------------------
add("markdown", "m02c0009", """
## Phase 02c-c — Defensive scoring-playType enumeration (D4 augmentation)

Before computing any features that touch the scoring registry, enumerate every distinct `playType` in the cache where `scoring == True`. For each:

1. Look it up in `SCORING_PLAYTYPE_REGISTRY` and tally counts per category.
2. **If any observed scoring playType is NOT in the registry, halt loud** with the offending playType, its count, and a sample season/game. The same defensive enumeration pattern as N01's scoring registry per the D4 plan-approval augmentation.

This catches CFBD adding a new playType in a season we haven't yet processed (or an old season being re-classified after the fact), before the points-bucket logic in Phase 02c-e silently misattributes anything. The check runs against the full cache (all 10 seasons), not just the in-scope trigger games, so it's a corpus-wide registry validation.
""")


# ---------------------------------------------------------------------------
# Cell 10 — Scoring playtype enumeration (code)
# ---------------------------------------------------------------------------
add("code", "c02c000a", '''
# Enumerate every distinct (playType, scoring) combination across the cache.
# This is the registry-validation gate; the points-bucket logic in Phase 02c-e
# relies on every scoring playType being categorizable.
scoring_play_counts: dict[str, int] = {}
sample_play_by_type: dict[str, tuple[int, int]] = {}  # playType -> (gameId, playId)
for gid, plays in plays_by_game.items():
    for p in plays:
        if not p.get("scoring"):
            continue
        pt = p.get("playType", "")
        scoring_play_counts[pt] = scoring_play_counts.get(pt, 0) + 1
        if pt not in sample_play_by_type:
            sample_play_by_type[pt] = (gid, p.get("id") or p.get("playId") or -1)

# Categorize each scoring playType. Halt on any unknowns.
unknown_scoring_playtypes: list[tuple[str, int]] = []
counts_by_category: dict[str, int] = {}
counts_by_playtype: dict[str, tuple[str, int]] = {}  # playType -> (category, count)
for pt, ct in sorted(scoring_play_counts.items(), key=lambda x: (-x[1], x[0])):
    cat = SCORING_PLAYTYPE_REGISTRY.get(pt)
    if cat is None:
        unknown_scoring_playtypes.append((pt, ct))
        counts_by_playtype[pt] = ("UNKNOWN", ct)
    else:
        counts_by_category[cat] = counts_by_category.get(cat, 0) + ct
        counts_by_playtype[pt] = (cat, ct)

print("Scoring playType counts across cache (all seasons), by category:")
for cat in sorted(set(SCORING_PLAYTYPE_REGISTRY.values())):
    n = counts_by_category.get(cat, 0)
    members = sorted(pt for pt, c in SCORING_PLAYTYPE_REGISTRY.items() if c == cat)
    print(f"  [{cat}] total {n:>6,} plays across {len(members)} known playTypes")
    for pt in members:
        n_pt = scoring_play_counts.get(pt, 0)
        if n_pt > 0:
            print(f"    {pt:<48} {n_pt:>6,}")
        else:
            print(f"    {pt:<48} {'(0)':>6}")

# Return-TD playType counts (the D4 specific ask from plan-approval).
print(f"\\nReturn-TD playType counts (decision D4):")
for pt in sorted(RETURN_TD_PLAY_TYPES):
    n_pt = scoring_play_counts.get(pt, 0)
    print(f"  {pt:<48} {n_pt:>6,}")
print(f"  (total return-TD plays across cache: {counts_by_category.get('return_td', 0):,})")

# Halt-loud on unknown scoring playTypes (defensive enumeration per D4 augmentation).
if unknown_scoring_playtypes:
    print(f"\\n[FATAL] {len(unknown_scoring_playtypes)} scoring playType(s) not in SCORING_PLAYTYPE_REGISTRY:")
    for pt, ct in unknown_scoring_playtypes:
        gid, pid = sample_play_by_type[pt]
        print(f"  {pt!r:<50} count={ct:,}  sample: gameId={gid} playId={pid}")
    raise AssertionError(
        f"Unknown scoring playType(s) detected: {[pt for pt, _ in unknown_scoring_playtypes]!r}. "
        f"Add each to SCORING_PLAYTYPE_REGISTRY with the right category and re-run. "
        f"Same defensive-enumeration pattern as N01's scoring registry per D4 augmentation."
    )
print(f"\\n[ok] all {len(scoring_play_counts)} observed scoring playTypes are in the registry "
      f"({sum(scoring_play_counts.values()):,} scoring plays total)")
''')


# ---------------------------------------------------------------------------
# Cell 11 — assert_no_lookahead (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02c000b", """
## Phase 02c-d — `assert_no_lookahead` (R3 hard gate) + feature extractors

`assert_no_lookahead` is the per-row R3 gate -- same definition as 02a / 02b.

Eight feature functions plus helpers. Two structural categories:

- **Drive-attribution features (4 of 8):** `dog_points_from_*` (three buckets) and `dog_explosive_play_count`. Computed in a single pass over `plays_before` with drive-level explosive-detection, applying the D2/D3/D4/D12 attribution rules. The four scalar outputs are returned as a tuple by a single helper; per-feature null policy applies independently to each scalar.
- **Drive-mean features (2 of 8):** `dog_avg_drive_yards`, `dog_avg_drive_plays`. Mean over completed dog-offense drives (drives whose final play has `playNumber < trigger_play_number`).
- **Momentum features (2 of 8):** `seconds_since_last_dog_explosive_play` (continuous, NULL when no prior dog explosive; later imputed per-window in the eval loop per D8) and `prior_drive_had_dog_explosive_play` (binary, NULL for drive-1 triggers per D9).

Every extractor calls `assert_no_lookahead` on the play subset it sees (R3 hard gate). Drives are only consulted via their member plays, never via `drive.endPlayId` or similar lookahead-coupled fields. The points-bucket logic uses `plays_before` exclusively; the avg-drive-yards logic uses `drives_for_game` filtered to `driveNumber < trigger_drive_in_game` (drives strictly before trigger drive are complete by definition).
""")


# ---------------------------------------------------------------------------
# Cell 12 — assert_no_lookahead code (verbatim from 02b)
# ---------------------------------------------------------------------------
add("code", "c02c000c", '''
def assert_no_lookahead(plays_used: list[dict],
                        trigger_chrono_key: tuple[int, int, int, int],
                        feature_name: str, game_id: int) -> None:
    """Per-row R3 hard gate. Raises if any play in `plays_used` has
    `_chrono_key(p) >= trigger_chrono_key`.

    Switched from the original `playNumber < trigger_play_number` test
    (which silently leaked future plays because CFBD playNumber resets
    per drive) to the composite chrono_key. See
    research/corrections_log.md for the lookahead-bias fix history.
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
# Cell 13 — Feature extractors (code)
# ---------------------------------------------------------------------------
add("code", "c02c000d", '''
# --- Helpers ----------------------------------------------------------------

def _is_explosive(play: dict) -> bool:
    """D1 explosive-play classifier (verbatim from 02b)."""
    pt = play.get("playType", "")
    yg = play.get("yardsGained")
    if yg is None:
        return False
    if pt in EXPLOSIVE_PASS_PLAY_TYPES:
        return int(yg) >= EXPLOSIVE_PASS_YARDS
    if pt in EXPLOSIVE_RUSH_PLAY_TYPES:
        return int(yg) >= EXPLOSIVE_RUSH_YARDS
    return False


def _play_game_seconds(play: dict) -> int | None:
    """Convert play's (period, clock.minutes, clock.seconds) to total
    regulation seconds elapsed since kickoff. Clock counts DOWN per period;
    period 1 starts at 15:00, period 4 ends at 0:00 of regulation. OT plays
    are excluded from features per R15 -- if a play is in OT (period > 4),
    return None (extractors filter these out)."""
    period = play.get("period")
    clock = play.get("clock") or {}
    m = clock.get("minutes")
    s = clock.get("seconds")
    if period is None or m is None or s is None:
        return None
    if int(period) > 4:
        return None  # R15: OT plays excluded
    return (int(period) - 1) * 900 + (900 - int(m) * 60 - int(s))


# --- Drive-attribution pass: one helper returns all four scalars -----------

def compute_points_buckets(
    plays_before: list[dict], fav: str, dog: str,
) -> tuple[int | None, int | None, int | None, int]:
    """Single-pass drive-level attribution (D2/D3/D4/D12).

    Returns (points_explosives, points_sustained, points_returns,
             dog_explosive_play_count).

    All three points buckets are 0 (not None) once at least one dog-offense
    drive has completed (D7); None when zero dog-offense drives have
    completed.

    Algorithm:
      1. First pass: identify per-drive whether the dog had an explosive
         play in that drive (drive_had_dog_explosive map).
      2. Second pass in play order: for each scoring play, attribute points
         to the right bucket per the SCORING_PLAYTYPE_REGISTRY category and
         the drive's explosive flag. PAT/2pt (categories pat_1pt / pat_2pt)
         attribute to the bucket of the immediately preceding dog TD
         (last_dog_td_bucket), per D12.
    """
    # Pre-compute per-drive explosive flag (dog offense only).
    drive_had_dog_explosive: dict[int, bool] = {}
    for p in plays_before:
        dn = p.get("driveNumber")
        if dn is None:
            continue
        if p.get("offense") == dog and _is_explosive(p):
            drive_had_dog_explosive[int(dn)] = True

    # Dog explosive play count (only dog-offense plays).
    dog_explosive_play_count = sum(
        1 for p in plays_before
        if p.get("offense") == dog and _is_explosive(p)
    )

    # Walk plays in order, attribute scoring plays to buckets.
    points_explosives = 0
    points_sustained = 0
    points_returns = 0
    last_dog_td_bucket: str | None = None  # for PAT attribution per D12

    for p in plays_before:
        if not p.get("scoring"):
            continue
        pt = p.get("playType", "")
        cat = SCORING_PLAYTYPE_REGISTRY.get(pt)
        if cat is None:
            # Phase 02c-c already halted on unknowns; defensive no-op here.
            continue
        if cat == "exclude":
            # Alt-encoding scoring playTypes with ambiguous point values
            # (same-bucket-different-value or cross-bucket) or anomalous
            # CFBD scoring=True false positives. See registry comment +
            # research/tech_debt.md item 4. Per-playType evidence in
            # research/results/_investigate_02c_unknown_scoring.csv.
            continue
        play_offense = p.get("offense")
        play_drive_no = p.get("driveNumber")
        # Determine which bucket (if any) this play contributes to for the dog.
        if cat == "offensive_td":
            if play_offense != dog:
                continue
            had_explosive = bool(drive_had_dog_explosive.get(int(play_drive_no), False)) if play_drive_no is not None else False
            if had_explosive:
                points_explosives += 6
                last_dog_td_bucket = "explosives"
            else:
                points_sustained += 6
                last_dog_td_bucket = "sustained"
        elif cat == "fg":
            if play_offense != dog:
                continue
            had_explosive = bool(drive_had_dog_explosive.get(int(play_drive_no), False)) if play_drive_no is not None else False
            if had_explosive:
                points_explosives += 3
            else:
                points_sustained += 3
            # FG drives don't have a following PAT; don't update
            # last_dog_td_bucket. (If a FG drive is followed by a PAT-marked
            # play in the data -- shouldn't happen -- it would attribute
            # to the previous dog TD's bucket, which is the right behavior.)
        elif cat == "return_td":
            # Scoring team = team that did NOT have the ball at the start
            # of this play (for INT/Fumble/Blocked-* return TDs) or the
            # team that did not kick (for KO/Punt return TDs). In all
            # cases the convention is: scoring team is the OPPOSITE of
            # `play.offense`.
            scoring_team = fav if play_offense == dog else dog
            if scoring_team == dog:
                points_returns += 6
                last_dog_td_bucket = "returns"
        elif cat == "pat_1pt":
            # PAT good -- 1 point to the bucket of the preceding TD (D12).
            # Convention: PAT plays in CFBD typically have offense = the
            # team that scored the TD. If offense == dog AND there's a
            # preceding dog TD bucket, add to that bucket.
            if play_offense == dog and last_dog_td_bucket is not None:
                if last_dog_td_bucket == "explosives":
                    points_explosives += 1
                elif last_dog_td_bucket == "sustained":
                    points_sustained += 1
                else:  # returns
                    points_returns += 1
            # No update to last_dog_td_bucket.
        elif cat == "pat_2pt":
            # 2pt good -- 2 points to bucket of preceding TD (D12).
            if play_offense == dog and last_dog_td_bucket is not None:
                if last_dog_td_bucket == "explosives":
                    points_explosives += 2
                elif last_dog_td_bucket == "sustained":
                    points_sustained += 2
                else:
                    points_returns += 2
        elif cat == "safety_def":
            # Safety = 2 points awarded to the defense. Scoring team is
            # the OPPOSITE of play.offense.
            scoring_team = fav if play_offense == dog else dog
            if scoring_team == dog:
                points_returns += 2
        elif cat == "pat_def_ret":
            # Defensive PAT return = 2 points to the defense (rare).
            scoring_team = fav if play_offense == dog else dog
            if scoring_team == dog:
                points_returns += 2

    # Decide whether to return None vs 0 per D7. "At least one dog-offense
    # drive has completed" iff any dog-offense play is in plays_before AND
    # that play's drive is complete (its driveNumber < trigger_drive_in_game).
    # We don't have trigger_drive_in_game here -- the caller knows it. So
    # we use a proxy: the dog has had at least one play in `plays_before`
    # in a drive that has a NON-dog-offense play occurring later in
    # plays_before (i.e., possession switched, proving drive completion).
    # Simpler proxy: any dog-offense scoring play already attributed
    # implies at least one completed dog drive; or, if dog has any plays in
    # plays_before from a drive that the most recent play is NOT in (a
    # later drive exists), the dog drive is complete.
    #
    # In practice the caller checks: are there any dog-offense plays whose
    # driveNumber is < trigger_drive_in_game? That's the right "has at
    # least one completed dog drive" test. We expose the integer counts
    # here and let the caller (feature_matrix builder) decide None vs 0
    # using the trigger row's drive_number_in_game.
    return points_explosives, points_sustained, points_returns, dog_explosive_play_count


# --- Drive-mean features (D6) ----------------------------------------------

def _completed_dog_drives(drives_for_game: list[dict], trig_drive_in_game: int, dog: str) -> list[dict]:
    """Dog-offense drives with driveNumber < trigger_drive_in_game.
    A drive's `driveNumber < trig_drive_in_game` is the lookahead-safe
    "completed" test -- the trigger play is in drive `trig_drive_in_game`,
    so any drive with a strictly smaller number has ended before trigger.
    """
    return [
        d for d in drives_for_game
        if d.get("driveNumber") is not None
        and int(d["driveNumber"]) < trig_drive_in_game
        and d.get("offense") == dog
    ]


def feat_dog_avg_drive_yards(
    plays_before: list[dict], drives_for_game: list[dict],
    trig_drive_in_game: int, fav: str, dog: str,
) -> float | None:
    """Mean `yards` over completed dog-offense drives. NULL if zero."""
    drives = _completed_dog_drives(drives_for_game, trig_drive_in_game, dog)
    vals = [int(d["yards"]) for d in drives if d.get("yards") is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def feat_dog_avg_drive_plays(
    plays_before: list[dict], drives_for_game: list[dict],
    trig_drive_in_game: int, fav: str, dog: str,
) -> float | None:
    """Mean `plays` over completed dog-offense drives. NULL if zero."""
    drives = _completed_dog_drives(drives_for_game, trig_drive_in_game, dog)
    vals = [int(d["plays"]) for d in drives if d.get("plays") is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


# --- Momentum features (D8 / D9) -------------------------------------------

def feat_seconds_since_last_dog_explosive_play(
    plays_before: list[dict], drives_for_game: list[dict],
    trig_drive_in_game: int, fav: str, dog: str,
    trigger_secs: int,
) -> float | None:
    """Game-clock seconds elapsed since the dog's most recent dog-offense
    explosive play, computed at trigger time. NULL when no prior dog
    explosive play. (R16-safe imputation -- per-window median -- applied
    in the eval loop per D8.)
    """
    last_secs = -1
    for p in plays_before:
        if p.get("offense") != dog:
            continue
        if not _is_explosive(p):
            continue
        secs = _play_game_seconds(p)
        if secs is None:
            continue
        if secs > last_secs:
            last_secs = secs
    if last_secs < 0:
        return None
    return float(trigger_secs - last_secs)


def feat_prior_drive_had_dog_explosive_play(
    plays_before: list[dict], drives_for_game: list[dict],
    trig_drive_in_game: int, fav: str, dog: str,
) -> int | None:
    """1 iff the immediately preceding drive (driveNumber == trig - 1)
    contained at least one dog-offense explosive play. NULL when the
    trigger play is in drive 1 (no prior drive). Per D9.
    """
    if trig_drive_in_game <= 1:
        return None
    prior_dn = trig_drive_in_game - 1
    for p in plays_before:
        if p.get("driveNumber") != prior_dn:
            continue
        if p.get("offense") == dog and _is_explosive(p):
            return 1
    return 0


# Single-pass extractor for the four drive-attribution scalars (D2/D3/D4/D12).
# Defined here as a sentinel; feature_matrix construction uses the helper
# `compute_points_buckets` directly and the per-trigger None vs 0 decision
# is made there using trig_drive_in_game.
print("[ok] feature extractors defined "
      "(compute_points_buckets + 4 single-feature extractors)")
''')


# ---------------------------------------------------------------------------
# Cell 14 — Build feature matrix (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02c000e", """
## Phase 02c-e — Build feature matrix

Walk every in-scope trigger, slice plays via the **composite `_chrono_key` filter** `(period, period_seconds_elapsed, driveNumber, playNumber) < trigger_chrono_key`, run all extractors, gate every play subset through `assert_no_lookahead`, attach the pre-game baseline columns, and compute the `seconds_since_last_dog_explosive_play_is_null` indicator (R16-safe pair for D8/D10).

The composite chrono_key replaces the original `playNumber < trigger.play_number` filter, which silently leaked future plays because CFBD's `playNumber` resets per drive (verified in `_verify_playnumber_semantics.py`). See `research/corrections_log.md` for the lookahead-bias fix history and the full-corpus verification (8,537 games, 1.54M plays) bounding residual disagreement at 0.394% of triggers.

Decision **D7**: `dog_points_from_*` are 0 (not NULL) once at least one dog-offense drive has completed (drive proxy: there exists a dog-offense play in `plays_before` whose `driveNumber < trigger_drive_in_game`). NULL when zero completed dog-offense drives.

Print:
1. Null counts per candidate feature.
2. Drive-1 trigger count (D9 NULL floor for `prior_drive_had_dog_explosive_play`).
3. Implied accounting delta: `dog_score_at_trigger - (dog_points_from_explosives + dog_points_from_sustained + dog_points_from_returns)`. Per D12 this should equal the dog's points scored ON the trigger play (typically 0, 6, or 7 depending on whether the trigger play was itself a dog scoring play).
""")


# ---------------------------------------------------------------------------
# Cell 15 — Build matrix code
# ---------------------------------------------------------------------------
add("code", "c02c000f", '''
ID_COLS = ["game_id", "fav_deficit", "trigger_sequence", "season", "season_type",
           "week", "fav_team", "dog_team", "play_number", "quarter",
           "drive_number_in_game", "dog_score_at_trigger",
           "seconds_remaining_in_regulation"]
LABEL_COL = "final_fav_won"

records: list[dict] = []
n_skipped_unknown_game = 0
accounting_deltas: list[int] = []  # dog_score_at_trigger - sum(buckets)

for _, trig in trigger_full_df.iterrows():
    gid = int(trig["game_id"])
    trig_pn = int(trig["play_number"])
    fav = str(trig["fav_team"])
    dog = str(trig["dog_team"])
    trig_drive_in_game = int(trig["drive_number_in_game"])
    trigger_secs = 3600 - int(trig["seconds_remaining_in_regulation"])

    # Composite chrono_key for the trigger row (period, period_elapsed,
    # drive_number_in_game, play_number). clock_seconds_in_period_total
    # in trigger_events.csv = 60*minutes_remaining + seconds_remaining,
    # so period_elapsed = 900 - that value.
    trig_period = int(trig["quarter"])
    trig_period_elapsed = 900 - int(trig["clock_seconds_in_period_total"])
    trig_chrono_key = (trig_period, trig_period_elapsed, trig_drive_in_game, trig_pn)

    plays = plays_by_game.get(gid)
    if plays is None:
        n_skipped_unknown_game += 1
        continue
    # Composite chrono_key filter (replaces the leaky `playNumber < trig_pn`
    # filter; negative-id plays are already excluded at load time).
    plays_before = [p for p in plays if _chrono_key(p) < trig_chrono_key]
    drives_for_game = drives_by_game.get(gid, [])

    # R3 gate on the play subset (single check covers all extractors below).
    assert_no_lookahead(plays_before, trig_chrono_key, "<02c-extractors>", gid)

    row: dict[str, Any] = {col: trig[col] for col in ID_COLS}
    for col in ALWAYS_PRESENT_PREGAME_COLS:
        row[col] = trig[col]
    sm_raw = trig["spread_movement"]
    sm_is_null = bool(pd.isna(sm_raw))
    row["spread_movement"] = 0.0 if sm_is_null else float(sm_raw)
    row["spread_movement_is_null"] = int(sm_is_null)
    row[LABEL_COL] = bool(trig[LABEL_COL])

    # Drive-attribution pass (D2/D3/D4/D7/D12).
    pts_exp, pts_sus, pts_ret, expl_count = compute_points_buckets(
        plays_before, fav, dog
    )
    # Decide D7 None vs 0: is there at least one COMPLETED dog-offense drive?
    has_completed_dog_drive = any(
        p.get("offense") == dog
        and p.get("driveNumber") is not None
        and int(p["driveNumber"]) < trig_drive_in_game
        for p in plays_before
    )
    if has_completed_dog_drive:
        row["dog_points_from_explosives"] = int(pts_exp)
        row["dog_points_from_sustained"] = int(pts_sus)
        row["dog_points_from_returns"] = int(pts_ret)
        # Accounting check: trigger row's dog_score_at_trigger - buckets.
        # >=0 expected: the trigger play itself may have added points the
        # buckets don't capture (since buckets are pre-trigger).
        delta = int(trig["dog_score_at_trigger"]) - int(pts_exp + pts_sus + pts_ret)
        accounting_deltas.append(delta)
    else:
        row["dog_points_from_explosives"] = None
        row["dog_points_from_sustained"] = None
        row["dog_points_from_returns"] = None

    # dog_explosive_play_count is always defined (0 if no dog plays seen).
    row["dog_explosive_play_count"] = int(expl_count)

    # Drive-mean features (D6).
    row["dog_avg_drive_yards"] = feat_dog_avg_drive_yards(
        plays_before, drives_for_game, trig_drive_in_game, fav, dog
    )
    row["dog_avg_drive_plays"] = feat_dog_avg_drive_plays(
        plays_before, drives_for_game, trig_drive_in_game, fav, dog
    )

    # Momentum features.
    row["seconds_since_last_dog_explosive_play"] = (
        feat_seconds_since_last_dog_explosive_play(
            plays_before, drives_for_game, trig_drive_in_game, fav, dog,
            trigger_secs,
        )
    )
    row["prior_drive_had_dog_explosive_play"] = (
        feat_prior_drive_had_dog_explosive_play(
            plays_before, drives_for_game, trig_drive_in_game, fav, dog,
        )
    )

    records.append(row)

feature_matrix_df = pd.DataFrame.from_records(records)
print(f"feature_matrix_df:  {len(feature_matrix_df):>6,} rows x {feature_matrix_df.shape[1]} cols")
print(f"  skipped (game not in plays_by_game): {n_skipped_unknown_game}")
assert n_skipped_unknown_game == 0, (
    f"{n_skipped_unknown_game} triggers had no plays in cache. Investigate."
)

# R16-safe pair indicator for the continuous momentum feature (D8/D10).
# Built here (not in the eval loop) because the indicator value is per-row;
# the imputation value is per-train-window (computed in the eval loop).
feature_matrix_df[MOMENTUM_CONTINUOUS_INDICATOR] = (
    feature_matrix_df[MOMENTUM_CONTINUOUS_FEATURE].isna().astype(int)
)

# Null counts per candidate feature.
print(f"\\nNull counts per candidate feature (this run):")
print(f"  total in-scope triggers: {len(feature_matrix_df):,}")
null_counts: dict[str, int] = {}
for feat in CANDIDATE_FEATURES:
    n_null = int(feature_matrix_df[feat].isna().sum())
    pct = (n_null / len(feature_matrix_df) * 100) if len(feature_matrix_df) else 0
    null_counts[feat] = n_null
    extra = ""
    if feat == MOMENTUM_CONTINUOUS_FEATURE:
        extra = " (D8 R16-safe impute in eval loop)"
    elif feat == "prior_drive_had_dog_explosive_play":
        extra = " (D9 NULL on drive-1 triggers)"
    elif feat in {"dog_points_from_explosives", "dog_points_from_sustained", "dog_points_from_returns"}:
        extra = " (D7 NULL when no completed dog drive)"
    print(f"  {feat:<42} {n_null:>5,} null ({pct:5.2f}%){extra}")

n_drive1 = int((feature_matrix_df["drive_number_in_game"] == 1).sum())
print(f"\\n  drive-1 trigger count (expected null floor for prior_drive_had_dog_explosive_play): {n_drive1:,}")
assert null_counts["prior_drive_had_dog_explosive_play"] >= n_drive1, (
    f"prior_drive_had_dog_explosive_play has {null_counts['prior_drive_had_dog_explosive_play']:,} nulls, "
    f"less than {n_drive1:,} drive-1 triggers. Bug in extractor or D9 violation."
)

# R16-safe sanity (carried from 02a/02b).
sm_null_after = int(feature_matrix_df["spread_movement"].isna().sum())
sm_indicator_sum = int(feature_matrix_df["spread_movement_is_null"].sum())
print(f"\\nR16-safe NaN handling for spread_movement (baseline):")
print(f"  spread_movement nulls AFTER impute:    {sm_null_after:,}  (expected: 0)")
print(f"  spread_movement_is_null indicator sum: {sm_indicator_sum:,} "
      f"({sm_indicator_sum / len(feature_matrix_df) * 100:.2f}% of in-scope)")
assert sm_null_after == 0, f"spread_movement still has {sm_null_after} nulls after impute"

# Accounting delta diagnostics (D12 implies bucket-sum + trigger-play points == dog_score_at_trigger).
import collections as _c
delta_counter = _c.Counter(accounting_deltas)
print(f"\\nAccounting delta = dog_score_at_trigger - sum(3 buckets) "
      f"({len(accounting_deltas):,} triggers with completed dog drives):")
print(f"  per D12: PAT/2pt attribute to preceding TD's bucket, so positive deltas")
print(f"  reflect the trigger play itself being a dog scoring play.")
print(f"  Top deltas (value: count, % of accounted):")
for d, c in sorted(delta_counter.items(), key=lambda kv: (-kv[1], kv[0]))[:10]:
    pct = c / len(accounting_deltas) * 100 if accounting_deltas else 0
    print(f"    delta={d:>4}  count={c:>6,}  ({pct:5.2f}%)")
n_negative = sum(c for d, c in delta_counter.items() if d < 0)
if n_negative:
    print(f"  [WARN] {n_negative:,} triggers have NEGATIVE delta -- "
          f"buckets exceeded dog_score_at_trigger. Investigate registry / "
          f"attribution edge case before relying on these features in N03.")
else:
    print(f"  [ok] no negative deltas -- bucket sum never exceeds dog_score_at_trigger")
''')


# ---------------------------------------------------------------------------
# Cell 16 — Walk-forward eval (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02c0010", """
## Phase 02c-f — Walk-forward per-feature evaluation

For each `(window, candidate_feature)` pair, two modes:

**Mode A — per-feature null drop (decision B from 02a):** for 7 of 8 candidates, mask `feature_matrix_df` to rows where the candidate is non-null, then split by season. Used for `dog_points_from_*`, `dog_explosive_play_count`, `dog_avg_drive_yards`, `dog_avg_drive_plays`, `prior_drive_had_dog_explosive_play`.

**Mode B — R16-safe per-train-window median imputation (decision D8):** for the continuous momentum feature `seconds_since_last_dog_explosive_play`, DON'T drop rows. Compute the per-train-window median (no leakage -- train-only), impute NULLs to that median across all three splits (train / val / test), and add the paired `_is_null` indicator alongside the continuous column. The per-window imputation value is stored in the new `imputation_value` column on `feature_validation.csv` for downstream interpretability.

Eval pipeline (both modes): `StandardScaler` -> `LogisticRegression(penalty="l1", C=1.0, solver="liblinear", random_state=42, max_iter=1000)`, then `CalibratedClassifierCV(method="isotonic", cv="prefit")` on the val set; eval Brier + ECE on the test set. Identical helper to 02a / 02b.

8 candidates x 3 windows = 24 eval rows expected.
""")


# ---------------------------------------------------------------------------
# Cell 17 — ECE + fit helper code
# ---------------------------------------------------------------------------
add("code", "c02c0011", '''
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
    Identical to 02a / 02b's helper; will be deduped into a shared module
    before N03 per research/tech_debt.md item 2."""
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
# Cell 18 — Eval loop code
# ---------------------------------------------------------------------------
add("code", "c02c0012", '''
eval_rows: list[dict] = []
t_start = time.perf_counter()

for window in WALK_FORWARD_WINDOWS:
    train_seasons = window["train_seasons"]
    val_season = window["val_season"]
    test_season = window["test_season"]
    win_label = window["train_window_label"]

    for feat in CANDIDATE_FEATURES:
        if feat == MOMENTUM_CONTINUOUS_FEATURE:
            # Mode B: R16-safe per-train-window median imputation. No row drop.
            sub = feature_matrix_df  # don't drop; impute below
            train_sub = sub[sub["season"].isin(train_seasons)].copy()
            val_sub = sub[sub["season"] == val_season].copy()
            test_sub = sub[sub["season"] == test_season].copy()
            # Train-window median (no leakage, computed only from train).
            train_non_null = train_sub[feat].dropna()
            if len(train_non_null) == 0:
                # Defensive: empty train non-null (extreme edge case). Skip.
                print(f"[skip] feat={feat} window={win_label}: train non-null is empty")
                continue
            median_imp = float(train_non_null.median())
            # Apply imputation across all three splits.
            train_sub[feat] = train_sub[feat].fillna(median_imp)
            val_sub[feat] = val_sub[feat].fillna(median_imp)
            test_sub[feat] = test_sub[feat].fillna(median_imp)
            # Candidate columns: baseline + continuous + paired indicator.
            cand_cols = BASELINE_PREGAME_FEATURES + [feat, MOMENTUM_CONTINUOUS_INDICATOR]
            imputation_value: float | None = median_imp
        else:
            # Mode A: per-feature null drop (decision B from 02a).
            mask = feature_matrix_df[feat].notna()
            sub = feature_matrix_df[mask]
            train_sub = sub[sub["season"].isin(train_seasons)]
            val_sub = sub[sub["season"] == val_season]
            test_sub = sub[sub["season"] == test_season]
            cand_cols = BASELINE_PREGAME_FEATURES + [feat]
            imputation_value = None

        if len(train_sub) == 0 or len(val_sub) == 0 or len(test_sub) == 0:
            print(f"[skip] feat={feat} window={win_label}: empty split "
                  f"(n_train={len(train_sub)}, n_val={len(val_sub)}, n_test={len(test_sub)})")
            continue

        y_train = train_sub[LABEL_COL].values.astype(int)
        y_val = val_sub[LABEL_COL].values.astype(int)
        y_test = test_sub[LABEL_COL].values.astype(int)

        # Baseline model.
        X_train_b = train_sub[BASELINE_PREGAME_FEATURES].values.astype(float)
        X_val_b = val_sub[BASELINE_PREGAME_FEATURES].values.astype(float)
        X_test_b = test_sub[BASELINE_PREGAME_FEATURES].values.astype(float)
        brier_b, ece_b = fit_calibrate_evaluate(
            X_train_b, y_train, X_val_b, y_val, X_test_b, y_test
        )

        # Candidate model.
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
            "imputation_value": imputation_value,
        })

eval_df = pd.DataFrame(eval_rows)
print(f"\\n[ok] evaluation loop complete in {time.perf_counter() - t_start:.1f}s")
print(f"     rows: {len(eval_df)} (expected: {len(WALK_FORWARD_WINDOWS) * len(CANDIDATE_FEATURES)} = "
      f"{len(WALK_FORWARD_WINDOWS)} windows x {len(CANDIDATE_FEATURES)} features)")
assert len(eval_df) == len(WALK_FORWARD_WINDOWS) * len(CANDIDATE_FEATURES), (
    f"expected {len(WALK_FORWARD_WINDOWS) * len(CANDIDATE_FEATURES)} eval rows, got {len(eval_df)}"
)

# Per-feature stability decision (R6).
stability_decision: dict[str, bool] = {}
for feat in CANDIDATE_FEATURES:
    n_positive = int((eval_df[eval_df["feature"] == feat]["brier_improvement"] > 0).sum())
    stability_decision[feat] = (n_positive >= 2)
eval_df["passed_stability"] = eval_df["feature"].map(stability_decision)

# Column order for CSV write -- imputation_value is NEW vs 02a/02b.
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

print(f"\\nstability verdict per feature:")
for feat in CANDIDATE_FEATURES:
    n_pos = int((eval_df[eval_df["feature"] == feat]["brier_improvement"] > 0).sum())
    verdict = "PASS" if stability_decision[feat] else "FAIL"
    extra = ""
    if feat == MOMENTUM_CONTINUOUS_FEATURE:
        per_win = eval_df[eval_df["feature"] == feat][["train_window", "imputation_value"]].values.tolist()
        extra = f"  per-window medians: {per_win}"
    print(f"  {feat:<42} {verdict}  ({n_pos}/3 test seasons with positive Brier){extra}")
''')


# ---------------------------------------------------------------------------
# Cell 19 — CSV write (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02c0013", """
## Phase 02c-g — Write `feature_validation.csv` (defensive append + new column)

Same defensive-append pattern as 02a / 02b:

1. Read existing CSV with `keep_default_na=False` so the `redundant_with` empty-string convention round-trips. (Note: `imputation_value` is REAL; rows from 02a / 02b will not have this column.)
2. Drop rows matching this run's `(feature, train_window, test_season)` keys.
3. Concatenate this run's 24 new rows. `pd.concat` unions columns; rows from 02a / 02b get NaN for the new `imputation_value`.
4. Sort by `(feature_set_version, feature, train_window, test_season)`.
5. Write.

02a / 02b rows are preserved (their keys don't overlap with 02c's). Natural-key uniqueness is asserted after the write. The new column is documented in the schema sidecar (Phase 02c-h).
""")


# ---------------------------------------------------------------------------
# Cell 20 — CSV write code
# ---------------------------------------------------------------------------
add("code", "c02c0014", '''
NEW_KEYS = set(zip(
    eval_df["feature"],
    eval_df["train_window"],
    eval_df["test_season"].astype(int),
))

if FEATURE_VALIDATION_CSV.exists():
    # keep_default_na=False preserves redundant_with == ''.
    # Existing rows from 02a / 02b won't have imputation_value; that's OK --
    # pd.concat unions columns and fills missing with NaN.
    existing_df = pd.read_csv(FEATURE_VALIDATION_CSV, keep_default_na=False)
    print(f"existing feature_validation.csv: {len(existing_df):,} rows, {existing_df.shape[1]} cols")
    existing_keys = list(zip(
        existing_df["feature"],
        existing_df["train_window"],
        existing_df["test_season"].astype(int),
    ))
    mask_keep = [k not in NEW_KEYS for k in existing_keys]
    n_displaced = len(existing_df) - sum(mask_keep)
    existing_df = existing_df[mask_keep].reset_index(drop=True)
    if n_displaced > 0:
        print(f"  displaced {n_displaced} row(s) matching this run's keys")
    combined_df = pd.concat([existing_df, eval_df], ignore_index=True)
else:
    print(f"feature_validation.csv does not exist -- creating new file")
    combined_df = eval_df.copy()

# Ensure imputation_value column exists for all rows (NaN for non-02c rows).
if "imputation_value" not in combined_df.columns:
    combined_df["imputation_value"] = pd.NA

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
print(f"     columns: {list(combined_df.columns)}")
print(f"     path: {FEATURE_VALIDATION_CSV}")
''')


# ---------------------------------------------------------------------------
# Cell 21 — Schema sidecar (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02c0015", """
## Phase 02c-h — Splice `feature_validation.schema.md` sidecar

02c's sidecar section is sentinel-delimited (same pattern as 02b). Includes:

1. Candidate list with D2/D3/D4/D6/D7/D8/D9/D10/D12 references.
2. **Accounting section (D12):** PAT/2pt attribution rule documented with worked examples.
3. **Imputation provenance (D8):** per-train-window median values used for `seconds_since_last_dog_explosive_play`, plus a one-line note about the new `imputation_value` CSV column.
4. **Momentum framing note (P2), revised post-correction:** binary form (literature-supported) failed 0/3 stability and continuous form (unstudied) passed 2/3 Brier-improving folds with weak magnitudes under the corrected `_chrono_key` filter; Roebber 2022's NFL-WP-streaks setting may not transfer to CFB comeback-trigger contexts. See `research/corrections_log.md` section 1.
5. **Redundancy discoveries:** the three conditional identities against 02b's drive-1 features.
6. Per-feature null counts + stability table.

02a / 02b sections preserved by sentinel splicing. Same known limitation as before: 02a's writer doesn't yet use splicing -- tracked as tech_debt item 3.
""")


# ---------------------------------------------------------------------------
# Cell 22 — Schema sidecar code
# ---------------------------------------------------------------------------
add("code", "c02c0016", '''
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


# Per-feature x per-test-season table rows.
verdict_rows = []
for feat in CANDIDATE_FEATURES:
    feat_rows = eval_df[eval_df["feature"] == feat].sort_values("test_season")
    for _, r in feat_rows.iterrows():
        imp_val = r["imputation_value"]
        imp_cell = f"{float(imp_val):.1f}" if pd.notna(imp_val) else ""
        verdict_rows.append(
            f"| `{feat}` | {r['train_window']} -> test {int(r['test_season'])} | "
            f"{_fmt_delta(r['brier_improvement'])} | {_fmt_delta(r['calibration_improvement'])} | "
            f"{imp_cell} | "
            f"{'**PASS**' if r['passed_stability'] else 'FAIL'} |"
        )

# Per-feature null counts.
null_rows = []
for feat in CANDIDATE_FEATURES:
    n_null = null_counts[feat]
    pct = (n_null / len(feature_matrix_df) * 100) if len(feature_matrix_df) else 0
    if feat == MOMENTUM_CONTINUOUS_FEATURE:
        tag = " (D8 R16-safe impute in eval loop)"
    elif feat == "prior_drive_had_dog_explosive_play":
        tag = " (D9 drive-1 NULL)"
    elif feat in {"dog_points_from_explosives", "dog_points_from_sustained", "dog_points_from_returns"}:
        tag = " (D7 no-completed-dog-drive NULL)"
    else:
        tag = ""
    null_rows.append(f"| `{feat}` | {n_null:,} | {pct:.2f}%{tag} |")

n_drive1 = int((feature_matrix_df["drive_number_in_game"] == 1).sum())

# Per-window imputation values for the continuous momentum feature (D8 provenance).
imp_provenance_rows = []
for _, r in eval_df[eval_df["feature"] == MOMENTUM_CONTINUOUS_FEATURE].sort_values("train_window").iterrows():
    imp_provenance_rows.append(
        f"| {r['train_window']} | {int(r['n_train']):,} | {float(r['imputation_value']):.1f} |"
    )

# Top accounting-delta rows.
import collections as _c
delta_counter = _c.Counter(accounting_deltas)
delta_rows = []
for d, c in sorted(delta_counter.items(), key=lambda kv: (-kv[1], kv[0]))[:8]:
    pct = c / len(accounting_deltas) * 100 if accounting_deltas else 0
    delta_rows.append(f"| {d} | {c:,} | {pct:.2f}% |")

# Build the 02c-owned section.
SECTION_BEGIN = "<!-- BEGIN: 02c explosive_vs_sustained -->"
SECTION_END = "<!-- END: 02c explosive_vs_sustained -->"

section_body = f"""
## 02c -- Explosive vs sustained drives + post-explosive momentum

**Section last writer:** `research/notebooks/02c_explosive_vs_sustained.ipynb`
**Last writer commit:** `{commit_hash}`
**Last writer generation timestamp:** {now_text}
**Feature set version:** `{FEATURE_SET_VERSION}`
**Source DDL:** `BUILD_SPEC.md` `trigger_features` explosive-vs-sustained block (V5 lines 181-187) + `research/future_features.md` momentum hypothesis

### Candidate features (8)

V5 DDL block 3 (drive-level points attribution + drive shape; 6 features):

- `dog_points_from_explosives` (D2 drive-level attribution; D7 NULL when no completed dog drive)
- `dog_points_from_sustained` (D3 complement; D7 NULL)
- `dog_points_from_returns` (D4 return-style TDs + safeties + defensive 2pt; D7 NULL)
- `dog_explosive_play_count` (D5; always defined, 0 when no dog plays)
- `dog_avg_drive_yards` (D6; NULL when no completed dog drives)
- `dog_avg_drive_plays` (D6; NULL when no completed dog drives)

Momentum (post-explosive decay; 2 features):

- `seconds_since_last_dog_explosive_play` (D8 primary; R16-safe paired
  with `seconds_since_last_dog_explosive_play_is_null`; per-train-window
  median imputation; per-window value stored in the new `imputation_value`
  column on `feature_validation.csv`)
- `prior_drive_had_dog_explosive_play` (D9 secondary; binary; NULL on
  drive-1 triggers per the established drive-1 null-policy pattern)

### D12: PAT / 2pt attribution rule (accounting)

PAT (1pt) and successful 2pt conversion (2pt) points attribute to the
**same bucket as the preceding TD**, not split off. Examples:

- Dog scores an explosive TD (drive contains a 35-yard rushing TD that
  meets `EXPLOSIVE_RUSH_YARDS >= 12`), followed by a successful PAT.
  Bucket: `dog_points_from_explosives += 7` (6 for the TD + 1 for PAT).
- Dog scores a sustained TD (drive had no explosive play), followed by
  a successful 2pt. Bucket: `dog_points_from_sustained += 8`.
- Dog scores on a kickoff return TD (return_td category), followed by
  PAT. Bucket: `dog_points_from_returns += 7`.

This makes the three buckets bit-exact-equivalent to a partition of the
dog's pre-trigger points: `dog_points_from_explosives +
dog_points_from_sustained + dog_points_from_returns` equals the dog's
score at the start of the trigger play, which is `dog_score_at_trigger`
minus any dog points scored on the trigger play itself.

Accounting-delta distribution on the {len(accounting_deltas):,} triggers
with at least one completed dog drive (delta = `dog_score_at_trigger -
sum(buckets)`; expected to be 0, 1, 2, 3, 6, 7, or 8 corresponding to
the trigger play's own dog points):

| Delta | Triggers | % |
|---:|---:|---:|
""" + "\\n".join(delta_rows) + f"""

Negative deltas would indicate the registry mis-attributed points (a
return TD credited to the dog when it was actually fav-team scoring, or
similar). The Phase 02c-c defensive enumeration plus the registry
review at 02c plan-approval should keep this at zero -- if a non-zero
negative count appears here, treat as a stop-the-line bug.

#### Excluded alt-encoding scoring playTypes (registry category = `exclude`)

The 02c-c registry-validation gate originally halted on 18 CFBD
`playType` strings flagged `scoring == True` that weren't in the
bootstrapped registry. The 02c plan-approval investigation
(`research/notebooks/_investigate_02c_unknown_scoring.py`,
`research/results/_investigate_02c_unknown_scoring.csv` +
`.summary.json`) classified each by playText sample and routed:

- 4 alt-encodings into existing categories: `Fumble Recovery (Opponent)`,
  `Kickoff Return (Offense)`, `Pass Interception Return` -> `return_td`
  (265 plays); `Fumble Recovery (Own)` -> `offensive_td` (34 plays).
- 14 alt-encodings into a new sentinel `exclude` category (1,562 plays
  total). The points-bucket helper short-circuits `cat == 'exclude'`
  before attribution.

Excluded playTypes have either ambiguous point values within the same
destination bucket (e.g., `Punt` is 49/50 punt-return-TD worth 6+1 PAT
and 1/50 punt-for-SAFETY worth 2 -- both routing to the returns bucket
but with different point counts) or cross-bucket destinations (e.g.,
`Rush` flagged `scoring=True` is sometimes an offensive rushing TD and
sometimes a safety **against** the rushing team -- routing to opposite
buckets). PlayText branching to disambiguate was rejected at 02c
plan-approval per R17: excluding 1,562 / ~78,000 = ~2% of recognized
scoring plays leaves features with ample data, while playText parsing
is a maintenance liability with silent-miscategorization risk that the
registry-validation gate cannot catch.

Categorization evidence + n=50 verification on the four highest-volume
alt-encodings (`Uncategorized`, `Punt`, `Fumble Recovery (Opponent)`,
`Kickoff`) is preserved in the investigation CSV/JSON. If
`dog_points_from_returns` fails stability or N03 needs higher-volume
returns signal, see `research/tech_debt.md` item 4 for the revisit
condition (text-branching registry extension).

### D8 provenance: per-train-window imputation values for the continuous momentum feature

Median of non-null `seconds_since_last_dog_explosive_play` computed
**within the training subset only** (no leakage). The same median is
applied to train, val, and test rows in that window. Stored in the new
`imputation_value` column on `feature_validation.csv` (REAL; NULL for
all non-02c rows and for 02c rows that don't use imputation).

| Train window | n_train (after impute) | Median seconds (imputed value) |
|---|---:|---:|
""" + "\\n".join(imp_provenance_rows) + f"""

The paired indicator `seconds_since_last_dog_explosive_play_is_null` is
built per-row at feature-matrix build time (Phase 02c-e), not per-window
-- a NULL continuous value remains NULL in the indicator regardless of
which window the row falls into. The indicator value is therefore the
same across windows for any given trigger.

### P2 framing note (momentum interpretation in N03), revised post-correction

The plan-time framing assumed both momentum forms might pass stability,
in which case the binary (drives-level) form would be the stronger
result because it aligned with Roebber 2022 (PLOS ONE 17(6): e0269604),
which established non-random momentum streaks at the **possession-level**
unit of analysis in NFL football. The corrected results invert that
expectation:

- **Binary form (literature-supported) failed 0/3 stability** under the
  corrected `_chrono_key` filter. `prior_drive_had_dog_explosive_play`
  produced near-zero Brier deltas (-0.00009 / -0.00014 / -0.00010) -- no
  signal at the possession-level unit of analysis in this CFB
  comeback-trigger context.
- **Continuous form (unstudied) passed 2/3 Brier-improving folds under
  R6 stability** with weak magnitudes (+0.00468 / +0.00780 / -0.00269)
  -- passes the rule's floor but doesn't strengthen across all test
  seasons.

Two interpretive consequences:

1. Roebber 2022's NFL-WP-streaks setting may not transfer to CFB
   comeback-trigger contexts. Possible explanation: trigger conditioning
   already selects for games where the underdog has been productive,
   compressing the variance the binary form needs to detect.
   Continuous-seconds preserves recency information the binary form
   discards, which is why it survives weakly while the binary form does
   not.

2. The continuous form's 2/3 survival with one negative-Brier fold is
   itself a marginal result. N03 should treat this signal with
   skepticism and consider testing a binned middle-ground form
   (categorical-window momentum) as a follow-up; see
   `research/future_features.md` "Categorical-window momentum features"
   for the live alternative-shape hypothesis.

Prior text in this section assumed Roebber 2022 would transfer cleanly
and described the continuous form passing as a "novel finding" -- both
framings were leak-era reasoning. See `research/corrections_log.md`
section 1 for the lookahead leak that distorted the prior published
interpretation of these features.

### Per-feature null counts (this run)

In-scope triggers (post NaN `final_fav_won` drop): {len(feature_matrix_df):,}.
Drive-1 trigger count (D9 null floor for prior-drive feature): {n_drive1:,}.

| Feature | Null rows | % of in-scope |
|---|---:|---:|
""" + "\\n".join(null_rows) + f"""

### Per-feature x per-test-season results (this run, {FEATURE_SET_VERSION})

`imp_value` is the per-train-window imputation median for the continuous
momentum feature (blank for the other 7 candidates).

| Feature | Window -> Test | Brier improvement | ECE improvement | Imp value | Stability |
|---|---|---:|---:|---:|---|
""" + "\\n".join(verdict_rows) + f"""

Sign convention: positive = candidate beat baseline. `**PASS**` means
`sum(brier_improvement > 0) >= 2` across the 3 test seasons.

### Redundancy discoveries (02c plan-time audit)

Plan-time verdict: **zero structural duplicates among 02c's 8
candidates.** `REDUNDANT_WITH = {{}}` for this feature set version.
All 24 rows have `redundant_with == ""`.

Three **conditional identities** were flagged at plan time against
02b's validated set:

1. `dog_explosive_play_count` vs 02b's `opening_drive_was_explosive_td`:
   02b is binary, restricted to drive 1. 02c is count, all completed
   pre-trigger drives. Overlap only on drive-1 triggers when the dog
   had drive 1 with an explosive TD. Conditional, not structural.
2. `dog_avg_drive_yards` vs 02b's `opening_drive_yards`: equality on
   the subset where `drive_number_in_game == 2` AND the dog had drive
   1 (so the only completed dog drive is drive 1, whose yards == the
   2c average over 1 drive).
3. `dog_avg_drive_plays` vs 02b's `opening_drive_plays`: same
   conditional-identity structure as (2).

None warrant a `redundant_with` tag (the identities hold only on
subsets of triggers, not bit-identically across the full corpus).
N03 filtering `redundant_with == ""` still drops only 02a's two
duplicates; all 02c rows pass through.

### New CSV column: `imputation_value`

`imputation_value` is a REAL column added to `feature_validation.csv`
in this commit. Semantics:

- NULL for rows whose candidate model used no imputation
  (all 02a + 02b rows, and all 02c rows except those for the continuous
  momentum feature `seconds_since_last_dog_explosive_play`).
- The per-train-window median (in seconds) used to impute the continuous
  momentum feature's NULLs for the corresponding train window.

Downstream consumers (N03, etc.) should treat NULL in this column as
"feature used drop-not-impute" and a value as "feature used impute with
this train-window-specific median." On `pd.read_csv` with
`keep_default_na=False`, empty cells appear as `""` (not NaN); cast to
float and treat `""` as NaN, or use `keep_default_na=True` for this
column specifically.

### Section provenance

- Last writer: this 02c run (timestamp + commit above).
- Splicing strategy: sentinel-delimited; re-running 02c refreshes only
  this section. Re-running 02a in its current form WILL clobber 02b's
  and 02c's sections -- tracked as `research/tech_debt.md` item 3.
"""

new_section = SECTION_BEGIN + "\\n" + section_body.rstrip() + "\\n" + SECTION_END

if FEATURE_VALIDATION_SCHEMA.exists():
    existing_text = FEATURE_VALIDATION_SCHEMA.read_text(encoding="utf-8")
    if SECTION_BEGIN in existing_text and SECTION_END in existing_text:
        start = existing_text.index(SECTION_BEGIN)
        end = existing_text.index(SECTION_END) + len(SECTION_END)
        updated = existing_text[:start] + new_section + existing_text[end:]
        print(f"[ok] spliced 02c section in place (existing markers found)")
    else:
        updated = existing_text.rstrip() + "\\n\\n" + new_section + "\\n"
        print(f"[ok] appended 02c section at end of sidecar (markers added)")
else:
    header = (
        "# feature_validation.csv -- schema sidecar\\n\\n"
        "(02a + 02b sections missing -- run 02a / 02b to regenerate.)\\n\\n"
    )
    updated = header + new_section + "\\n"
    print(f"[warn] sidecar did not exist; wrote stub header + 02c section.")

FEATURE_VALIDATION_SCHEMA.write_text(updated, encoding="utf-8")
print(f"[ok] wrote feature_validation.schema.md ({len(updated):,} chars)")
print(f"     path: {FEATURE_VALIDATION_SCHEMA}")
''')


# ---------------------------------------------------------------------------
# Cell 23 — Summary (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02c0017", """
## Phase 02c-i — Summary, headline stats, STOP banner
""")


# ---------------------------------------------------------------------------
# Cell 24 — Summary print code
# ---------------------------------------------------------------------------
add("code", "c02c0018", '''
print("=" * 70)
print("Notebook 02c -- explosive vs sustained + post-explosive momentum -- summary")
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
print(f"\\nDrive-1 trigger count (D9 null floor for prior-drive momentum feature):")
print(f"  drive_number_in_game == 1: {n_drive1_final:,} "
      f"({n_drive1_final / len(feature_matrix_df) * 100:.2f}%)")

print(f"\\nD12 accounting (top accounting deltas):")
import collections as _c
_dc = _c.Counter(accounting_deltas)
for d, c in sorted(_dc.items(), key=lambda kv: (-kv[1], kv[0]))[:5]:
    pct = c / len(accounting_deltas) * 100 if accounting_deltas else 0
    print(f"  delta={d:>4}  {c:>6,}  ({pct:5.2f}%)")

print(f"\\nPer-feature x per-test-season results ({FEATURE_SET_VERSION}):")
print(f"  {'feature':<42} {'window->test':<18} "
      f"{'d_brier':>10} {'d_ece':>10} {'imp':>8} {'stab':>6}")
for _, r in eval_df.sort_values(["feature", "test_season"]).iterrows():
    win = f"{r['train_window']}->{int(r['test_season'])}"
    imp = float(r["imputation_value"]) if pd.notna(r["imputation_value"]) else None
    imp_str = f"{imp:.0f}" if imp is not None else "-"
    print(f"  {r['feature']:<42} {win:<18} "
          f"{r['brier_improvement']:>+10.5f} {r['calibration_improvement']:>+10.5f} "
          f"{imp_str:>8} "
          f"{'PASS' if r['passed_stability'] else 'FAIL':>6}")

print(f"\\nFeature stability verdicts:")
for feat in CANDIDATE_FEATURES:
    n_pos = int((eval_df[eval_df["feature"] == feat]["brier_improvement"] > 0).sum())
    n_pos_cal = int((eval_df[eval_df["feature"] == feat]["calibration_improvement"] > 0).sum())
    verdict = "PASS" if stability_decision[feat] else "FAIL"
    print(f"  {feat:<42} {verdict:<5} "
          f"({n_pos}/3 brier-improving, {n_pos_cal}/3 ece-improving)")

print(f"\\nNull counts per feature (in-scope triggers: {len(feature_matrix_df):,}):")
for feat in CANDIDATE_FEATURES:
    pct = (null_counts[feat] / len(feature_matrix_df) * 100) if len(feature_matrix_df) else 0
    extra = ""
    if feat == MOMENTUM_CONTINUOUS_FEATURE:
        extra = " (D8 R16-safe impute in eval loop)"
    elif feat == "prior_drive_had_dog_explosive_play":
        extra = " (D9 drive-1 NULL)"
    elif feat in {"dog_points_from_explosives", "dog_points_from_sustained", "dog_points_from_returns"}:
        extra = " (D7 no-completed-dog-drive NULL)"
    print(f"  {feat:<42} {null_counts[feat]:>5,} null ({pct:5.2f}%){extra}")

print(f"\\nP3 trigger logic (categorical-window variants disposition):")
d8_pass = stability_decision[MOMENTUM_CONTINUOUS_FEATURE]
d9_pass = stability_decision["prior_drive_had_dog_explosive_play"]
if d8_pass and d9_pass:
    print(f"  D8 PASS + D9 PASS -> log categorical-window variants to tech_debt.md")
    print(f"  ('could refine binning if N03 wants finer-grained windows')")
elif (not d8_pass) and d9_pass:
    print(f"  D8 FAIL + D9 PASS -> categorical-windows becomes live hypothesis")
    print(f"  -> log to future_features.md (decay shape was wrong; binned windows worth testing)")
elif d8_pass and (not d9_pass):
    print(f"  D8 PASS + D9 FAIL -> continuous form passes alone, binary form fails")
    print(f"  -> log categorical-windows to future_features.md as live hypothesis")
    print(f"  -> (inverse of D8 FAIL + D9 PASS; same logical move -- the shape we")
    print(f"  -> chose was wrong on the binary side, finer-grained alternative")
    print(f"  -> on the continuous side becomes worth testing)")
else:
    print(f"  D8 FAIL + D9 FAIL -> momentum hypothesis REJECTED as feature group")
    print(f"  -> route both to validated_filters.json rejected_features at end of Phase 0")
    print(f"  -> log categorical-windows alongside as also-rejected")
print(f"  (Disposition is for execution-report time; no file edits in this notebook.)")

print(f"\\nDeliverables (research/results/):")
for path in [FEATURE_VALIDATION_CSV, FEATURE_VALIDATION_SCHEMA]:
    size = path.stat().st_size
    print(f"  {path.name:<40} {size:>10,} bytes")
''')


# ---------------------------------------------------------------------------
# Cell 25 — Budget print + STOP banner
# ---------------------------------------------------------------------------
add("code", "c02c0019", '''
calls_log_df = pd.read_csv(CALL_LOG)
n_total_log_rows = len(calls_log_df)
n_fresh_cfbd_total = int(((calls_log_df["service"] == "cfbd")
                          & (calls_log_df["cached"] == 0)).sum())

this_run_calls = calls_log_df.iloc[n_log_before:].copy()
n_this_run = len(this_run_calls)
n_this_run_fresh = int((this_run_calls["cached"] == 0).sum())

print("=" * 64)
print("CFBD call budget -- Notebook 02c")
print("=" * 64)
print(f"\\nThis notebook run:")
print(f"  total calls this run:     {n_this_run:>5,}  ({n_plays_lookups} /plays + {n_drives_lookups} /drives)")
print(f"  fresh (uncached) this run: {n_this_run_fresh:>5,}  (budget: 0)")

assert n_this_run_fresh == 0, (
    f"02c budget invariant violated: {n_this_run_fresh} fresh CFBD call(s) "
    f"this run. 02c is supposed to spend 0 fresh CFBD calls."
)

# Hardcoded 1,000-call display constant is incorrect on the current API key
# (actual quota 3,000/cycle). Tracked as item 1 in research/tech_debt.md.
print(f"\\nCumulative across all notebooks (call log: {n_total_log_rows:,} rows):")
print(f"  total fresh CFBD calls (lifetime):    {n_fresh_cfbd_total:,}")
print(f"  monthly free-tier limit (BUILD_SPEC A.4 stated):  1,000")
print(f"    actual quota on current key (probe header):     3,000")
print(f"  remaining this billing cycle (against actual 3K): {3000 - n_fresh_cfbd_total:,}")
if n_fresh_cfbd_total >= 0.8 * 3000:
    print(f"  [WARN] >=80% of 3,000-call cycle consumed.")

print(f"\\n[ok] notebook 02c complete -- STOP per R22. "
      f"Do not start Notebook 02d without approval.")
''')


# ---------------------------------------------------------------------------
# Serialize
# ---------------------------------------------------------------------------
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
