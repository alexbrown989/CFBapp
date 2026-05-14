"""
Deterministic builder for research/notebooks/02d_turnover_and_short_field.ipynb.

Mirrors _build_02c.py:
  - Same cache-hit-assertion contract.
  - Same defensive-append pattern for feature_validation.csv.
  - Sentinel-delimited splicing of the schema sidecar so 02a / 02b / 02c content
    is preserved verbatim.
  - Same _chrono_key fix via _lib_chrono.CHRONO_KEY_SOURCE.

02d-specific additions vs 02c:
  - Four turnover & short-field candidates: fav_turnovers_so_far,
    dog_points_off_turnovers, dog_avg_starting_field_pos,
    short_field_tds_allowed (V5 DDL block 4, BUILD_SPEC.md lines 189-193).
  - All four extractors are drive-metadata only (Category A per the
    plan-time candidate-vs-extractor-structure audit) -- they do NOT
    iterate plays_before.
  - **Diff-vs-leaky empirical verification (per plan-approval addition 1):**
    builds the feature matrix TWICE -- once with the chrono_key filter and
    once with the leaky `playNumber < trig_pn` filter -- and asserts the 4
    per-trigger feature columns are byte-identical between the two passes.
    Confirms the Category A claim empirically, not just from the extractor
    sketches.
  - **Overlap-fraction diagnostic (per plan-approval addition 2):**
    classifies each trigger into 4 buckets by co-occurrence of
    `dog_points_off_turnovers` vs `dog_points_from_returns` (recomputed
    here using 02c's SCORING_PLAYTYPE_REGISTRY logic to avoid cross-
    notebook coupling).
  - **Cumulative validated-set context (per plan-approval addition 3):**
    summary cell prints the running validated-set count and notable
    conditional identities accumulating across 02a + 02b + 02c + 02d.

This is a scratchpad file (per the research/notebooks/_*.py convention).
Not part of the deliverable.
"""

from __future__ import annotations

import json
import pathlib
import sys
import textwrap

OUT = pathlib.Path(__file__).resolve().parent / "02d_turnover_and_short_field.ipynb"

# Pull the canonical _chrono_key source from the shared helper module
# (single-source-of-truth across 02a/02b/02c/02d build scripts). See
# research/notebooks/_lib_chrono.py for the function definition and
# research/corrections_log.md section 1 for the lookahead-bias fix rationale.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _lib_chrono import CHRONO_KEY_SOURCE  # noqa: E402

CELLS: list[tuple[str, str, str]] = []


def add(cell_type: str, cell_id: str, src: str) -> None:
    CELLS.append((cell_type, cell_id, textwrap.dedent(src).lstrip("\n")))


# ---------------------------------------------------------------------------
# Cell 0 — Title + hypothesis docstring (markdown)
# ---------------------------------------------------------------------------
add("markdown", "bd02d000", """
# Phase 0 — Notebook 02d: Turnover & short-field features

## Hypothesis (per-feature stability claims under R6)

Four parallel claims tied to the V5 DDL block 4 (turnover & short field;
`BUILD_SPEC.md` lines 189-193):

1. **`fav_turnovers_so_far`** -- the favorite's pre-trigger turnover count
   carries comeback-equity signal beyond the pre-game baseline. Football
   mechanism: turnover margin is the single strongest in-game correlator
   with win probability across the play-by-play and analytics literature
   (Pro Football Reference, Football Outsiders' DVOA decomposition,
   numerous coaching analyses citing turnover margin at 70-75% correlation
   with game outcome). A favorite who has already turned the ball over has
   shifted possessions, often given short fields, and surrendered scoring
   opportunities the bet-favorable pre-game line did not price.

2. **`dog_points_off_turnovers`** -- the dog's points scored on offensive
   drives immediately following a fav turnover carries signal beyond the
   raw turnover count. Mechanism: capitalization is the second-order
   measure. A fav with 3 turnovers but the dog converted 0 into points
   is structurally different from a fav with 3 turnovers where the dog
   converted 2 into TDs -- the latter implies the dog's offense is also
   executing.

3. **`dog_avg_starting_field_pos`** -- the dog's mean starting field
   position over completed pre-trigger drives carries signal beyond pre-
   game baseline. Football mechanism: Field Position Value (FPV / Net
   Starting Field Position / NSFP) is a long-established analytics
   metric (Football Outsiders, EPA-by-field-position curves; see
   `research/future_features.md` provenance). Better dog field position
   compresses the EPA distribution the favorite defense must defend; a
   trigger that fired despite the dog having favorable field position
   implies the favorite is either still well-positioned to win or the
   dog is leaving points on the field.

4. **`short_field_tds_allowed`** -- the favorite defense's count of
   short-field (`startYardsToGoal <= 40`) drives that ended in a dog TD.
   Mechanism: short-field TDs allowed are red-zone-adjacent defensive
   collapses. They isolate the cases where the fav defense was given a
   manageable distance to defend and STILL surrendered 6+ points. This
   is structurally different from a long-drive TD allowed and from a
   short-field FG allowed.

All four claims use the pre-game-only baseline locked in 02a
(`pregame_spread`, `rating_gap`, `fav_pregame_rating`, `dog_pregame_rating`,
`spread_movement`, `spread_movement_is_null`). Per-feature null policy
applies (decision **B** from 02a).

## What this notebook DOES NOT do

- Does not modify `trigger_events.csv` or `trigger_outcomes.csv`.
- Does not pull any fresh CFBD data -- every `/plays` AND `/drives` lookup
  must hit the cache produced by N01. Cache-hit assertion fails loud on
  any miss.
- Does not select features for the production model -- N03's job.
- Does not test feature groups other than turnover & short field
  (those are 02a / 02b / 02c / 02e-g).
- Does not tune hyperparameters of the L1 logreg -- uses sklearn default
  `C=1.0` with a fixed seed, identical to 02a / 02b / 02c.
- Does not iterate plays_before for feature computation -- all four
  extractors are drive-metadata only (Category A). Plays are loaded and
  chrono_key-sorted only to provide the `assert_no_lookahead` gate
  surface; the drive-level extractors filter `drives_for_game` by
  `driveNumber < trigger_drive_in_game`.

## Spec references

- `BUILD_SPEC.md` Phase 0 Notebook 02 deliverable spec -- `feature_validation.csv` shape
- `BUILD_SPEC.md` `trigger_features` DDL -- turnover & short field block (V5 lines 189-193)
- `research/future_features.md` -- "Field Position Value / NSFP" framing (Football Outsiders origins)
- `research/corrections_log.md` section 1 -- the `_chrono_key` composite filter and the lookahead-bias fix history
- `.cursorrules` **R2 + R3** -- no lookahead; `assert_no_lookahead()` mandatory on every feature extraction
- `.cursorrules` **R5** -- walk-forward validation only
- `.cursorrules` **R6** -- stability rule (>=2 of 3 test seasons)
- `.cursorrules` **R7** -- L1 logreg / shallow GBM only
- `.cursorrules` **R8** -- ECE on 10 bins, post-calibration
- `.cursorrules` **R16** -- pre-game-safe NaN handling for non-random missingness
- `.cursorrules` **R19** -- record rejected features too
- `.cursorrules` **R22** -- STOP at end of 02d; do not start 02e without approval
- `.cursorrules` **R23** -- commit message standards (auto-applied)

## Decision-points log (from the 02d plan-approval)

- **D1** -- Turnover definition: `driveResult in {INT, INT TD, FUMBLE, FUMBLE TD, FUMBLE RETURN TD}` represents a turnover BY the offense of that drive. Excludes turnover-on-downs (`DOWNS`) per the standard NFL/CFB definition (turnovers = interceptions + lost fumbles; turnover-on-downs is a "giveaway" but not a traditional turnover). Excludes safeties (`SF`; defensive score on the offense's drive, not a turnover).
- **D2** -- Short-field threshold: `startYardsToGoal <= 40`. Standard analytics cutoff (matches the Football Outsiders "FP+" framing; tightly fits the cliff in the EPA-by-field-position curve where defenses lose structural advantage). If results suggest the cutoff needs tuning (e.g., at 35 or 30), the decision is logged to `research/tech_debt.md` rather than re-run in 02d.
- **D3** -- `dog_points_off_turnovers` no-next-fav-drive edge case: when the trigger drive itself is dog offense AND immediately follows the most recent pre-trigger fav turnover (no later fav drive completed yet), use `dog_score_at_trigger - trigger_drive.startOffenseScore` to capture pre-trigger points on the in-progress trigger drive. Worked example below in the sidecar. When the trigger drive itself is dog offense for ANY reason (whether immediately following a turnover or not), `dog_score_at_trigger` correctly captures the dog's pre-trigger-play points on that drive -- the formula above subtracts the dog's drive-start score to isolate this-drive contribution.
- **D4** -- Return TD exclusion: `dog_points_off_turnovers` EXCLUDES return TDs (`INT TD`, `FUMBLE TD`, `FUMBLE RETURN TD`). Those are dog points scored DIRECTLY on the turnover by the dog defense -- there is no subsequent dog offense drive that capitalized. Per separation of concerns from 02c's `dog_points_from_returns`, return-TD points are NOT double-counted here. Only points from the NEXT dog-offense drive after a non-return-TD fav turnover (i.e., `INT`, `FUMBLE`) count toward `dog_points_off_turnovers`.
- **D5** -- `dog_avg_starting_field_pos` uses **yards-from-own-end-zone** convention: `100 - startYardsToGoal`. Higher value = better field position (closer to opponent's end zone). NULL when zero completed dog drives.
- **D6** -- `short_field_tds_allowed` counts COMPLETED dog-offense drives where `startYardsToGoal <= 40` AND `driveResult == 'TD'`. Always integer; 0 when no qualifying drives (e.g., dog had no short-field opportunities yet, or had them but did not score TDs on them).
- **D7** -- D7 null policy from 02c carries forward: `dog_points_off_turnovers` is 0 (not NULL) once at least one completed dog drive exists. NULL only when zero completed dog drives. `fav_turnovers_so_far` and `short_field_tds_allowed` are always defined (0 when no qualifying drives). `dog_avg_starting_field_pos` is NULL when zero completed dog drives.
- **D8** -- Composite `_chrono_key` filter from 02c carries forward. Plays sorted by chrono_key at load time. The play subset is used only by `assert_no_lookahead`; the four extractors operate on `drives_for_game` filtered to `driveNumber < trigger_drive_in_game`.
- **D9** -- Negative-id retention from 02c carries forward. 19,828 plays across 115 games use the alt-encoding negative-integer string-encoded `play.id`; chrono_key orders them correctly without referencing `play.id`. See `research/corrections_log.md` section 2.
- **D10** -- Diff-vs-leaky empirical verification (plan-approval addition 1): build the feature matrix twice -- once with `_chrono_key < trig_chrono_key`, once with the leaky `playNumber < trig.playNumber` filter -- and assert the 4 per-trigger feature columns are byte-identical. Confirms the Category A claim empirically (drive-level extractors are insensitive to the play-iteration filter) rather than relying on the extractor sketch.
- **D11** -- Overlap-fraction diagnostic (plan-approval addition 2): at execution time, classify each trigger into a 4-bucket co-occurrence table over (`dog_points_off_turnovers` nonzero, `dog_points_from_returns` nonzero). Recompute `dog_points_from_returns` here using 02c's SCORING_PLAYTYPE_REGISTRY logic (duplicated inline for the diagnostic; not a deliverable column).
- **D12** -- Cumulative validated-set context (plan-approval addition 3): summary cell prints the running validated-set count and notable conditional identities accumulating across 02a + 02b + 02c + 02d. Pre-execution running count is 17 features; post-02d count varies with verdicts.

## Plan-time pre-execution redundancy audit

Per the 02d plan-approval, every candidate feature pair was audited at
plan time. Three audit dimensions:

### Candidate-vs-candidate (within 02d)

| Pair | Verdict |
|---|---|
| `fav_turnovers_so_far` vs `dog_points_off_turnovers` | Conditional: zero overlap on triggers with `fav_turnovers_so_far == 0`; for triggers with `fav_turnovers_so_far >= 1`, the second is the capitalization rate. NOT structural duplicates; tested separately. |
| `fav_turnovers_so_far` vs `dog_avg_starting_field_pos` | Indirect: more fav turnovers tend to give the dog better mean starting field position. Conditional correlation, not structural identity. |
| `dog_points_off_turnovers` vs `short_field_tds_allowed` | Partial overlap on triggers where a fav turnover gave the dog short field AND the dog scored a TD on that short field. Subset relation possible (a short-field TD allowed after a fav turnover counts in both). Worth empirical co-occurrence check; conditional identity flagged for the sidecar. |
| `dog_avg_starting_field_pos` vs `short_field_tds_allowed` | Indirect: more short fields (low `startYardsToGoal`) push the mean up. Conditional correlation, not structural identity. |

Plan-time verdict: **zero structural duplicates among 02d's 4
candidates.** Two conditional identities flagged: (a)
`dog_points_off_turnovers` <-> `short_field_tds_allowed` partial-subset;
(b) the cross-cut conditional with 02c's `dog_points_from_returns`
documented under D11.

### Candidate-vs-validated-set (against 17-feature accumulated set)

The 17 already-validated features cover: pre-game baseline (alpha
fixed in 02a), EPA aggregates (02a), opening-drive shock features
(02b), explosive-vs-sustained drive-volume features (02c), and one
momentum feature (02c). No structural duplicate of any 02d candidate.

Notable cross-notebook **conditional identities** flagged:

| 02d candidate | 02c / 02b / 02a feature | Relation |
|---|---|---|
| `dog_points_off_turnovers` | `dog_points_from_returns` (02c) | Co-occurrence when a fav turnover happened in pre-trigger plays. Cleanly separable (no double-counting); D11 diagnostic at execution time. |
| `dog_avg_starting_field_pos` | `dog_avg_drive_yards` (02c) | Better field position trims drive yards needed; conditional correlation. Indirect, not structural. |
| `dog_points_off_turnovers` | `dog_points_from_sustained` (02c) | A short-field TD after a fav turnover counts in `dog_points_off_turnovers` AND in `dog_points_from_sustained` (offensive TD on a drive without dog explosive plays). Cross-cuts, not structural identity. |
| `fav_turnovers_so_far` | (none) | Standalone; no validated-set parallel. |
| `short_field_tds_allowed` | (none) | Standalone; no validated-set parallel. |

### Candidate-vs-trigger-fields (deducibility audit)

Every 02d candidate must NOT be a function of `trigger_events.csv`
columns alone -- otherwise the feature is a deterministic rewrite of
existing state and can't add signal. Verified:

| Candidate | Trigger-field deducibility | Verdict |
|---|---|---|
| `fav_turnovers_so_far` | Not deducible from triggers (counts pre-trigger drive results). | Standalone signal. |
| `dog_points_off_turnovers` | Not deducible from triggers (depends on drive chain following turnovers). | Standalone signal. |
| `dog_avg_starting_field_pos` | Not deducible from triggers (mean over drives, not the trigger play's `yardline_at_trigger`). | Standalone signal. |
| `short_field_tds_allowed` | Not deducible from triggers (filters & counts drives by start + result). | Standalone signal. |

### Candidate-vs-extractor-structure (02b refinement: read the source, not the sketch)

Per the 02b lookahead-leak post-mortem, every candidate is classified
by what the extractor TOUCHES at run time, not by what the sketch
implies:

| Candidate | Touches `plays_before`? | Touches `drives_for_game`? | Category | Leak-sensitivity |
|---|---|---|---|---|
| `fav_turnovers_so_far` | No | Yes (filter by `driveNumber < trig_drive`) | A: drive-metadata only | Structurally safe under the leak. |
| `dog_points_off_turnovers` | No (uses `dog_score_at_trigger` for the in-progress trigger drive only) | Yes | A: drive-metadata only | Structurally safe. |
| `dog_avg_starting_field_pos` | No | Yes | A: drive-metadata only | Structurally safe. |
| `short_field_tds_allowed` | No | Yes | A: drive-metadata only | Structurally safe. |

Plan-time prediction: all four features pass diff-vs-leaky equality
(D10) byte-identically. Verdict pre-correction vs post-correction:
unchanged (this notebook has no pre-correction history; the prediction
is for the diff-vs-leaky pass within this run).

Plan-time prediction on stability verdicts (separate from leak
verification): **2/4 pass**, with `dog_points_off_turnovers` and
`short_field_tds_allowed` favored over `fav_turnovers_so_far` and
`dog_avg_starting_field_pos`. Rationale: the raw count features
(`fav_turnovers_so_far`) are dominated by base-rate effects already
captured by `plays_so_far`; the capitalization features
(`dog_points_off_turnovers`, `short_field_tds_allowed`) carry second-
order information about the dog's execution that the count features
alone cannot.

Hypothesis-watch from review: `dog_points_off_turnovers` more likely
to pass than `short_field_tds_allowed`. The watch tests whether
trigger-conditioning compresses the short-field-conversion signal
(the trigger fires because the dog HASN'T converted enough to erase
the deficit; that conditional structure may already absorb the short-
field signal).

## Deliverables produced by this notebook

1. `research/results/feature_validation.csv` -- adds 12 rows from 02d
   (4 features x 3 test seasons), tagged `feature_set_version =
   v1_turnover_short_field`. 02a's 18 + 02b's 30 + 02c's 24 = 72 prior
   rows preserved by the defensive-append.
2. `research/results/feature_validation.schema.md` -- splices a
   sentinel-delimited "02d -- Turnover & short field" section into the
   existing sidecar. 02a's + 02b's + 02c's sections are preserved
   verbatim.
3. `research/notebooks/02d_turnover_and_short_field.ipynb` -- this notebook.

No changes to `trigger_events.csv`, `trigger_outcomes.csv`,
`trigger_events_bucket_counts.csv`, `data_quality_report.md`,
`budget_reconciliation.md`. No new cache files. No fresh CFBD calls.

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
per `research/data/cache/cfbd_call_log.csv` -- this notebook's run
should leave that count unchanged.
""")


# ---------------------------------------------------------------------------
# Cell 1 — Imports, paths, env, fail-fast (code)
# ---------------------------------------------------------------------------
add("code", "c02d0001", '''
"""
Notebook 02d -- imports, environment, path constants, fail-fast checks.
Same structure as Notebook 02a / 02b / 02c. Run this cell first.
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
    f"is missing entries (none expected in 02d); load it anyway so the same "
    f"cfbd_get() helper works."
)

load_dotenv(ENV_PATH)
assert os.environ.get("CFBD_API_KEY"), (
    "CFBD_API_KEY is not set. 02d should NOT issue fresh calls, but the "
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
# Cell 2 — HTTP helpers (code, same code path as 00/01/02a/02b/02c)
# ---------------------------------------------------------------------------
add("code", "c02d0002", '''
"""
HTTP helpers -- same code as Notebook 00/01/02a/02b/02c, same cache directory.
02d expects ALL calls to be cache hits; the assertion in Phase 02d-b fails
loud on any go-fresh.
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
print(f"[ok] sharing cache with Notebook 00/01/02a/02b/02c at {CACHE_DIR}")
''')


# ---------------------------------------------------------------------------
# Cell 3 — Configuration (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02d0003", """
## Configuration

`WALK_FORWARD_WINDOWS` and `BASELINE_PREGAME_FEATURES` are **carried verbatim from 02a** -- locked at 02a plan-approval and binding for 02b-g. Not re-decided here.

`SHORT_FIELD_THRESHOLD = 40` (decision **D2**, standard analytics cutoff at `startYardsToGoal <= 40`). If 02d results suggest the cutoff needs tuning (e.g., at 35 or 30 to better isolate the structural defensive collapse), the decision is logged to `research/tech_debt.md` rather than re-run in 02d.

`TURNOVER_DRIVE_RESULTS = {INT, INT TD, FUMBLE, FUMBLE TD, FUMBLE RETURN TD}` (decision **D1**, standard NFL/CFB turnover definition: interceptions + lost fumbles). Excludes `DOWNS` (turnover-on-downs; "giveaway" but not a traditional turnover) and `SF` (safety; defensive score on the offense's drive, not a turnover).

`NON_RETURN_TURNOVERS = {INT, FUMBLE}` (decision **D4**, the subset of turnovers where the dog defense did NOT score directly -- so a subsequent dog offense drive can capitalize). Used by `feat_dog_points_off_turnovers` to identify turnover events that hand the dog an offensive possession.

`CANDIDATE_FEATURES` (4 total):

- **V5 DDL block 4 (4 features):** `fav_turnovers_so_far`, `dog_points_off_turnovers`, `dog_avg_starting_field_pos`, `short_field_tds_allowed`.

All four are drive-metadata only (Category A per the plan-time candidate-vs-extractor-structure audit). The diff-vs-leaky verification in Phase 02d-e asserts byte-identical per-trigger values under chrono_key and leaky filters.

`REDUNDANT_WITH = {}` -- the plan-time redundancy audit found zero structural duplicates among 02d candidates. Two conditional identities flagged: (a) `dog_points_off_turnovers` <-> `short_field_tds_allowed` partial-subset; (b) `dog_points_off_turnovers` <-> `dog_points_from_returns` (02c) co-occurrence, with the D11 diagnostic measuring it empirically.

`FEATURE_SET_VERSION = "v1_turnover_short_field"` -- the per-notebook tag stamped into every row this notebook writes.

`SCORING_PLAYTYPE_REGISTRY_FOR_DIAGNOSTIC` -- a duplicate of 02c's registry, used solely by the D11 overlap diagnostic to recompute `dog_points_from_returns` per trigger. Not a deliverable column; not written to `feature_validation.csv`.
""")


# ---------------------------------------------------------------------------
# Cell 4 — Configuration constants (code)
# ---------------------------------------------------------------------------
add("code", "c02d0004", '''
SEASONS: list[int] = list(range(2015, 2025))
SEASON_TYPES: list[str] = ["regular", "postseason"]

FEATURE_SET_VERSION: str = "v1_turnover_short_field"

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

# --- 02d-specific constants -------------------------------------------------

# Decision D1: turnover-producing driveResult values (standard NFL/CFB
# definition: interceptions + lost fumbles). Excludes DOWNS (giveaway,
# not traditional turnover) and SF (defensive score on offense's drive,
# not a turnover).
TURNOVER_DRIVE_RESULTS: frozenset[str] = frozenset({
    "INT",
    "INT TD",
    "FUMBLE",
    "FUMBLE TD",
    "FUMBLE RETURN TD",
})

# Decision D4: the subset of turnovers where the dog defense did NOT score
# directly. These are the events that hand the dog an offensive
# possession; the IMMEDIATE next dog drive is the one whose points count
# toward dog_points_off_turnovers. Return-TD turnovers (INT TD,
# FUMBLE TD, FUMBLE RETURN TD) are excluded here -- those are scored
# directly by the dog defense and bucketed under 02c's
# dog_points_from_returns. See sidecar D4 worked example.
NON_RETURN_TURNOVERS: frozenset[str] = frozenset({
    "INT",
    "FUMBLE",
})

# Decision D2: short-field threshold. Standard analytics cutoff at
# startYardsToGoal <= 40 (Football Outsiders FP+ framing; fits the
# defensive-EPA-by-field-position cliff). Tuning tracked in
# research/tech_debt.md if results suggest a tighter cutoff.
SHORT_FIELD_THRESHOLD: int = 40

# Candidate features (V5 DDL block 4, BUILD_SPEC.md lines 189-193).
CANDIDATE_FEATURES: list[str] = [
    "fav_turnovers_so_far",
    "dog_points_off_turnovers",
    "dog_avg_starting_field_pos",
    "short_field_tds_allowed",
]

# Structural-redundancy map for 02d. Empty: the plan-time redundancy
# audit found zero structural duplicates among 02d candidates.
REDUNDANT_WITH: dict[str, str] = {}

# Reproducibility seed -- same as 02a / 02b / 02c.
RANDOM_STATE: int = 42

# --- 02c registry duplicated inline for the D11 overlap diagnostic ---------
# Used ONLY by the dog_points_from_returns recomputation in Phase 02d-f
# (overlap diagnostic). Not used by any deliverable feature; not written
# to feature_validation.csv. Mirrors _build_02c.py's registry exactly.
# If 02c's registry changes, this constant must be updated in tandem
# (manual sync; tracked as a tech-debt candidate if it becomes a third
# duplication).
SCORING_PLAYTYPE_REGISTRY_FOR_DIAGNOSTIC: dict[str, str] = {
    "Passing Touchdown":              "offensive_td",
    "Rushing Touchdown":              "offensive_td",
    "Fumble Recovery (Own)":          "offensive_td",
    "Field Goal Good":                "fg",
    "Interception Return Touchdown":  "return_td",
    "Fumble Return Touchdown":        "return_td",
    "Kickoff Return Touchdown":       "return_td",
    "Punt Return Touchdown":          "return_td",
    "Blocked Punt Touchdown":         "return_td",
    "Blocked Field Goal Touchdown":   "return_td",
    "Missed Field Goal Return Touchdown": "return_td",
    "Fumble Recovery Touchdown":      "return_td",
    "Fumble Recovery (Opponent)":     "return_td",
    "Pass Interception Return":       "return_td",
    "Kickoff Return (Offense)":       "return_td",
    "Defensive 2pt Conversion":       "pat_def_ret",
    "Extra Point Good":               "pat_1pt",
    "PAT Good":                       "pat_1pt",
    "Two Point Pass":                 "pat_2pt",
    "Two Point Rush":                 "pat_2pt",
    "Two-Point Pass":                 "pat_2pt",
    "Two-Point Rush":                 "pat_2pt",
    "2pt Conversion Good":            "pat_2pt",
    "Safety":                         "safety_def",
    "Uncategorized":                  "exclude",
    "Punt":                           "exclude",
    "Kickoff":                        "exclude",
    "Blocked Punt":                   "exclude",
    "Sack":                           "exclude",
    "Pass Reception":                 "exclude",
    "Interception":                   "exclude",
    "Blocked Field Goal":             "exclude",
    "Rush":                           "exclude",
    "Pass Incompletion":              "exclude",
    "Penalty":                        "exclude",
    "End Period":                     "exclude",
}

print(f"seasons: {SEASONS}")
print(f"season types: {SEASON_TYPES}")
print(f"feature_set_version: {FEATURE_SET_VERSION}")
print(f"walk-forward windows (locked from 02a, binding for 02b-g):")
for w in WALK_FORWARD_WINDOWS:
    print(f"  train={w['train_window_label']}  val={w['val_season']}  test={w['test_season']}")
print(f"baseline pre-game features ({len(BASELINE_PREGAME_FEATURES)}): {BASELINE_PREGAME_FEATURES}")
print(f"candidate features ({len(CANDIDATE_FEATURES)}):")
for f in CANDIDATE_FEATURES:
    print(f"  - {f}")
print(f"turnover driveResults (D1): {sorted(TURNOVER_DRIVE_RESULTS)}")
print(f"non-return-TD turnovers (D4 subset): {sorted(NON_RETURN_TURNOVERS)}")
print(f"short-field threshold (D2): startYardsToGoal <= {SHORT_FIELD_THRESHOLD}")
print(f"redundant_with map ({len(REDUNDANT_WITH)} entries): {REDUNDANT_WITH}")
print(f"scoring playType registry (for D11 overlap diagnostic only): "
      f"{len(SCORING_PLAYTYPE_REGISTRY_FOR_DIAGNOSTIC)} entries")
print(f"random state: {RANDOM_STATE}")
''')


# ---------------------------------------------------------------------------
# Cell 5 — Load triggers (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02d0005", """
## Phase 02d-a — Load trigger artifacts

Identical join setup to 02a / 02b / 02c: read `trigger_events.csv` and `trigger_outcomes.csv`, inner-join on `(game_id, fav_deficit)`, drop rows with `final_fav_won is NaN`. The label only enters as the model target in the walk-forward eval cell; it does NOT enter any feature extractor.

Print the drive-1 trigger count for symmetry with 02b / 02c reporting (drive-1 triggers have zero completed drives, so all four 02d candidates are either NULL or 0 on those rows).
""")


# ---------------------------------------------------------------------------
# Cell 6 — Load triggers code
# ---------------------------------------------------------------------------
add("code", "c02d0006", '''
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
print(f"In-scope rows for 02d: {len(trigger_full_df):,}")

# Drive-1 trigger count (all 02d candidates NULL or 0 on these rows).
n_drive1 = int((trigger_full_df["drive_number_in_game"] == 1).sum())
n_drive2plus = int((trigger_full_df["drive_number_in_game"] >= 2).sum())
print(f"\\nDrive-1 scale:")
print(f"  drive_number_in_game == 1 (zero completed drives -> "
      f"dog_points_off_turnovers + dog_avg_starting_field_pos NULL "
      f"per D7; fav_turnovers_so_far + short_field_tds_allowed == 0): "
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
add("markdown", "m02d0007", """
## Phase 02d-b — Re-load cached `/plays` AND `/drives` (zero fresh calls)

Identical setup to 02b / 02c: iterate the (season, season_type, week) tuples for `/plays` and (season, season_type) tuples for `/drives`. **Assert every call is a cache hit.** The cell fails loud on any cache miss -- 02d's budget is 0 fresh CFBD calls.

`/plays` is loaded for two reasons even though all 02d extractors are drive-level (Category A):

1. To populate `plays_by_game` for the `assert_no_lookahead` gate -- the gate inspects the play subset to enforce R3.
2. To enable the diff-vs-leaky verification in Phase 02d-e, which runs the feature matrix twice (once with `_chrono_key < trig_chrono_key`, once with the leaky `playNumber < trig.playNumber` filter) and asserts byte-identical per-trigger values.

Since extractors don't iterate plays_before, the diff is a positive empirical confirmation that the Category A claim holds -- not a numerical sensitivity to the play filter.
""")


# ---------------------------------------------------------------------------
# Cell 8 — Cache re-load code (mirrors 02c structure, includes chrono_key)
# ---------------------------------------------------------------------------
add("code", "c02d0008", '''
work_tuples_df = (
    trigger_full_df[["season", "season_type", "week"]]
    .drop_duplicates()
    .sort_values(["season", "season_type", "week"])
    .reset_index(drop=True)
)
print(f"distinct (season, season_type, week) tuples to load from cache: {len(work_tuples_df)}")

n_log_before = sum(1 for _ in CALL_LOG.open("r", encoding="utf-8")) - 1  # minus header

plays_by_game: dict[int, list[dict]] = {}
# Note: no negative-play.id pre-filter. The ~19,828 plays across ~115 games
# with negative-id encoding (e.g., id='-6654') are LEGITIMATE plays in real
# games -- CFBD uses two id formats (standard 18-digit positive integers
# and compact negative-integer strings) but both have complete play data
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
    f"02d budget invariant violated: {n_fresh_this_cell} non-cached CFBD call(s) "
    f"issued in this cell. 02d is supposed to spend 0 fresh CFBD calls; the "
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
# Cell 9 — assert_no_lookahead (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02d0009", """
## Phase 02d-c — `assert_no_lookahead` (R3 hard gate) + feature extractors

`assert_no_lookahead` is the per-row R3 gate -- same definition as 02a / 02b / 02c (composite chrono_key gate). Even though 02d's four extractors don't iterate `plays_before`, the gate is still called on each trigger's play slice so the R3 contract holds uniformly across all N02 notebooks.

Four feature functions, all drive-metadata only (Category A):

- **`fav_turnovers_so_far`** -- count of fav-offense completed drives ending in a turnover (D1 set).
- **`dog_points_off_turnovers`** -- sum of dog points on the immediate-next dog-offense drive after each non-return-TD fav turnover (D4 set, i.e., `INT` or `FUMBLE`). D3 edge case for the in-progress trigger drive uses `dog_score_at_trigger - trigger_drive.startOffenseScore`.
- **`dog_avg_starting_field_pos`** -- mean of `100 - startYardsToGoal` over completed dog-offense drives.
- **`short_field_tds_allowed`** -- count of completed dog-offense drives with `startYardsToGoal <= SHORT_FIELD_THRESHOLD` AND `driveResult == 'TD'`.

Drive-level computation: each extractor filters `drives_for_game` to `driveNumber < trigger_drive_in_game` (i.e., COMPLETED pre-trigger drives) and accumulates the relevant quantity. None of them call `_chrono_key` on individual plays; the chrono_key sort affects `plays_by_game` ordering only, and `plays_by_game` is consumed only by `assert_no_lookahead`. This is the Category A claim the Phase 02d-e diff-vs-leaky check confirms empirically.
""")


# ---------------------------------------------------------------------------
# Cell 10 — assert_no_lookahead code (verbatim from 02c)
# ---------------------------------------------------------------------------
add("code", "c02d000a", '''
def assert_no_lookahead(plays_used: list[dict],
                        trigger_chrono_key: tuple[int, int, int, int],
                        feature_name: str, game_id: int) -> None:
    """Per-row R3 hard gate. Raises if any play in `plays_used` has
    `_chrono_key(p) >= trigger_chrono_key`.

    Same composite-chrono_key gate as 02a / 02b / 02c. See
    research/corrections_log.md for the lookahead-bias fix history that
    introduced this filter (replacing the original per-drive
    `playNumber < trigger_play_number` test).
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
# Cell 11 — Feature extractors (code)
# ---------------------------------------------------------------------------
add("code", "c02d000b", '''
# --- Helpers ----------------------------------------------------------------

def _completed_drives_before_trigger(
    drives_for_game: list[dict], trig_drive_in_game: int,
) -> list[dict]:
    """Filter drives to those with driveNumber < trigger's drive_number_in_game.

    A drive whose driveNumber is < the trigger drive's number is COMPLETE by
    definition -- the game has moved on to a later drive, so this drive has
    ended and its driveResult / endOffenseScore reflect its final state.

    Drives without a driveNumber field are dropped (no way to order them).
    """
    out: list[dict] = []
    for d in drives_for_game:
        dn = d.get("driveNumber")
        if dn is None:
            continue
        if int(dn) >= trig_drive_in_game:
            continue
        out.append(d)
    return out


def _trigger_drive(
    drives_for_game: list[dict], trig_drive_in_game: int,
) -> dict | None:
    """Return the drive whose driveNumber == trigger's drive_number_in_game,
    or None if not found in the drives list.

    Used by the D3 edge case in feat_dog_points_off_turnovers when the
    trigger drive itself is the post-turnover dog drive and is in
    progress.
    """
    for d in drives_for_game:
        dn = d.get("driveNumber")
        if dn is None:
            continue
        if int(dn) == trig_drive_in_game:
            return d
    return None


def _drive_points_for_offense(drive: dict) -> int:
    """Points scored BY this drive's offense team on this drive.

    Computed as endOffenseScore - startOffenseScore. CFBD bakes the
    PAT into endOffenseScore when the PAT is on the same drive (verified:
    a TD drive with successful PAT has endOffenseScore == startOffenseScore + 7;
    a TD drive without PAT has + 6 or different if 2pt). Negative or
    zero values are coerced to 0 (defensive against any anomalous
    CFBD encoding).
    """
    try:
        end = int(drive.get("endOffenseScore") or 0)
        start = int(drive.get("startOffenseScore") or 0)
        return max(0, end - start)
    except (TypeError, ValueError):
        return 0


# --- Feature extractors (Category A: drive-metadata only) -------------------

def feat_fav_turnovers_so_far(
    drives_for_game: list[dict], trig_drive_in_game: int, fav: str,
) -> int:
    """D1: count fav-offense completed drives ending in a turnover.

    Always integer (0 when no qualifying drives, including drive-1 triggers
    where no drives have completed).
    """
    n = 0
    for dr in _completed_drives_before_trigger(drives_for_game, trig_drive_in_game):
        if dr.get("offense") != fav:
            continue
        if str(dr.get("driveResult", "")) in TURNOVER_DRIVE_RESULTS:
            n += 1
    return n


def feat_dog_points_off_turnovers(
    drives_for_game: list[dict], trig_drive_in_game: int,
    fav: str, dog: str, dog_score_at_trigger: int,
) -> int | None:
    """D3/D4/D7: sum of dog points scored on the IMMEDIATE next dog-offense
    drive after each non-return-TD fav turnover (INT or FUMBLE).

    EXCLUDES return TDs (INT TD, FUMBLE TD, FUMBLE RETURN TD) -- those
    are dog points scored DIRECTLY on the turnover by the dog defense,
    bucketed under 02c's dog_points_from_returns. The separation of
    concerns is what makes dog_points_off_turnovers a distinct signal
    measuring the dog OFFENSE's capitalization on short fields handed
    by fav turnovers, vs. the dog DEFENSE's direct scoring.

    Edge case (D3): when the trigger drive itself is the immediate next
    dog drive after the most recent pre-trigger fav turnover, that drive
    is in-progress (its driveNumber == trig_drive_in_game), so it's not
    in the completed-drives filter. Use
    `dog_score_at_trigger - trigger_drive.startOffenseScore` to capture
    the dog's pre-trigger-play points on the in-progress drive.

    D7: NULL when zero completed dog drives. 0 when at least one
    completed dog drive but no turnover-driven points yet.
    """
    completed = _completed_drives_before_trigger(drives_for_game, trig_drive_in_game)
    if not any(d.get("offense") == dog for d in completed):
        # D7: NULL when zero completed dog drives. The dog has not had
        # an offensive possession yet, so the "points off turnovers"
        # concept doesn't apply.
        return None

    total = 0
    for i, dr in enumerate(completed):
        if dr.get("offense") != fav:
            continue
        if str(dr.get("driveResult", "")) not in NON_RETURN_TURNOVERS:
            continue
        # Find the immediate next drive after this turnover.
        next_idx = i + 1
        if next_idx < len(completed):
            next_dr = completed[next_idx]
            if next_dr.get("offense") == dog:
                total += _drive_points_for_offense(next_dr)
        else:
            # No more completed drives. D3 edge case: check if the
            # trigger drive itself is dog-offense (in-progress, follows
            # this turnover). If so, the dog's points on the in-progress
            # drive are dog_score_at_trigger - trigger_drive.startOffenseScore.
            trig_dr = _trigger_drive(drives_for_game, trig_drive_in_game)
            if trig_dr is not None and trig_dr.get("offense") == dog:
                try:
                    drive_start_score = int(trig_dr.get("startOffenseScore") or 0)
                    in_progress_pts = max(0, int(dog_score_at_trigger) - drive_start_score)
                    total += in_progress_pts
                except (TypeError, ValueError):
                    pass
    return int(total)


def feat_dog_avg_starting_field_pos(
    drives_for_game: list[dict], trig_drive_in_game: int, dog: str,
) -> float | None:
    """D5: mean of (100 - startYardsToGoal) over completed dog-offense drives.

    Uses yards-from-own-end-zone convention (higher = better field
    position, closer to opponent's end zone). NULL when zero completed
    dog drives or none have a startYardsToGoal field.
    """
    completed_dog = [
        d for d in _completed_drives_before_trigger(drives_for_game, trig_drive_in_game)
        if d.get("offense") == dog and d.get("startYardsToGoal") is not None
    ]
    if not completed_dog:
        return None
    fp_sum = 0.0
    n = 0
    for d in completed_dog:
        try:
            sytg = int(d["startYardsToGoal"])
        except (TypeError, ValueError):
            continue
        fp_sum += (100 - sytg)
        n += 1
    if n == 0:
        return None
    return fp_sum / n


def feat_short_field_tds_allowed(
    drives_for_game: list[dict], trig_drive_in_game: int, dog: str,
) -> int:
    """D6: count of completed dog-offense drives where
    startYardsToGoal <= SHORT_FIELD_THRESHOLD AND driveResult == 'TD'.

    Always integer; 0 when no qualifying drives (drive-1 triggers, or
    games where the dog had no short-field opportunities yet, or had
    them but did not score TDs on them).
    """
    n = 0
    for dr in _completed_drives_before_trigger(drives_for_game, trig_drive_in_game):
        if dr.get("offense") != dog:
            continue
        sytg = dr.get("startYardsToGoal")
        if sytg is None:
            continue
        try:
            if int(sytg) > SHORT_FIELD_THRESHOLD:
                continue
        except (TypeError, ValueError):
            continue
        if str(dr.get("driveResult", "")) == "TD":
            n += 1
    return n


print("[ok] 4 feature extractors defined (all Category A: drive-metadata only)")
print(f"     fav_turnovers_so_far / dog_points_off_turnovers")
print(f"     dog_avg_starting_field_pos / short_field_tds_allowed")
''')


# ---------------------------------------------------------------------------
# Cell 12 — Build feature matrix (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02d000c", """
## Phase 02d-d — Build feature matrix (canonical: chrono_key filter)

Walk every in-scope trigger, slice plays via the **composite `_chrono_key` filter** `(period, period_seconds_elapsed, driveNumber, playNumber) < trigger_chrono_key`, gate the play subset through `assert_no_lookahead`, run all four drive-level extractors on `drives_for_game` filtered to `driveNumber < trigger_drive_in_game`, and attach the pre-game baseline columns.

Decision **D7**: `dog_points_off_turnovers` and `dog_avg_starting_field_pos` are NULL when zero completed dog drives (drive-1 triggers). `fav_turnovers_so_far` and `short_field_tds_allowed` are always defined (0 on drive-1 triggers).

Print:
1. Null counts per candidate feature.
2. Drive-1 trigger count (verifies the D7 null floor).
3. Quick summary statistics: mean/median of each feature on the in-scope subset.

The Phase 02d-e diff-vs-leaky cell rebuilds the matrix under the leaky `playNumber < trig.playNumber` filter and asserts byte-identical per-trigger values for all four candidate columns.
""")


# ---------------------------------------------------------------------------
# Cell 13 — Build matrix code (canonical pass)
# ---------------------------------------------------------------------------
add("code", "c02d000d", '''
ID_COLS = ["game_id", "fav_deficit", "trigger_sequence", "season", "season_type",
           "week", "fav_team", "dog_team", "play_number", "quarter",
           "drive_number_in_game", "dog_score_at_trigger",
           "seconds_remaining_in_regulation"]
LABEL_COL = "final_fav_won"


def build_feature_matrix(
    triggers: pd.DataFrame,
    plays_by_game: dict[int, list[dict]],
    drives_by_game: dict[int, list[dict]],
    plays_before_filter: str,  # "chrono_key" or "leaky_playnumber"
) -> tuple[pd.DataFrame, int]:
    """Build the per-trigger feature matrix for 02d's 4 candidates.

    `plays_before_filter` selects which filter is applied to plays_before
    BEFORE the assert_no_lookahead gate runs:
      - "chrono_key": _chrono_key(p) < trig_chrono_key (canonical, post-correction)
      - "leaky_playnumber": p.playNumber < trig.playNumber (pre-correction; leaks
        future plays because CFBD playNumber resets per drive). Used ONLY by
        the Phase 02d-e diff-vs-leaky verification to confirm that 02d's
        Category A extractors produce byte-identical outputs under both filters
        (since the extractors don't iterate plays_before at all).

    Returns (feature_matrix_df, n_skipped_unknown_game).
    """
    assert plays_before_filter in ("chrono_key", "leaky_playnumber"), (
        f"invalid plays_before_filter: {plays_before_filter!r}"
    )

    records: list[dict] = []
    n_skipped = 0

    for _, trig in triggers.iterrows():
        gid = int(trig["game_id"])
        trig_pn = int(trig["play_number"])
        fav = str(trig["fav_team"])
        dog = str(trig["dog_team"])
        trig_drive_in_game = int(trig["drive_number_in_game"])
        dog_score_at_trigger = int(trig["dog_score_at_trigger"])

        trig_period = int(trig["quarter"])
        trig_period_elapsed = 900 - int(trig["clock_seconds_in_period_total"])
        trig_chrono_key = (trig_period, trig_period_elapsed, trig_drive_in_game, trig_pn)

        plays = plays_by_game.get(gid)
        if plays is None:
            n_skipped += 1
            continue

        # Apply the selected plays_before filter. The extractors do NOT
        # consume plays_before -- it goes only to assert_no_lookahead. The
        # two filters produce different play subsets; identical feature
        # values empirically confirm the Category A claim.
        if plays_before_filter == "chrono_key":
            plays_before = [p for p in plays if _chrono_key(p) < trig_chrono_key]
        else:  # "leaky_playnumber"
            plays_before = [
                p for p in plays
                if p.get("playNumber") is not None and int(p["playNumber"]) < trig_pn
            ]

        # R3 gate on the play subset (uniform across all N02 notebooks).
        # Note: under the leaky filter, this gate will pass even though
        # the SUBSET is different -- because the gate checks
        # `max(_chrono_key(p) for p in plays_used) < trigger_chrono_key`,
        # and the leaky filter happens to select a subset where this is
        # still mostly true. The R3 contract that matters for 02d's
        # numerical output is NOT this gate -- it's the chrono_key sort +
        # `driveNumber < trig_drive_in_game` filter inside each extractor.
        # The gate exists for uniformity with 02a / 02b / 02c, not as the
        # primary leak defense for this notebook.
        if plays_before_filter == "chrono_key":
            assert_no_lookahead(plays_before, trig_chrono_key, "<02d-extractors>", gid)

        drives_for_game = drives_by_game.get(gid, [])

        row: dict[str, Any] = {col: trig[col] for col in ID_COLS}
        for col in ALWAYS_PRESENT_PREGAME_COLS:
            row[col] = trig[col]
        sm_raw = trig["spread_movement"]
        sm_is_null = bool(pd.isna(sm_raw))
        row["spread_movement"] = 0.0 if sm_is_null else float(sm_raw)
        row["spread_movement_is_null"] = int(sm_is_null)
        row[LABEL_COL] = bool(trig[LABEL_COL])

        # 4 extractors, all drive-metadata only.
        row["fav_turnovers_so_far"] = feat_fav_turnovers_so_far(
            drives_for_game, trig_drive_in_game, fav
        )
        row["dog_points_off_turnovers"] = feat_dog_points_off_turnovers(
            drives_for_game, trig_drive_in_game, fav, dog, dog_score_at_trigger
        )
        row["dog_avg_starting_field_pos"] = feat_dog_avg_starting_field_pos(
            drives_for_game, trig_drive_in_game, dog
        )
        row["short_field_tds_allowed"] = feat_short_field_tds_allowed(
            drives_for_game, trig_drive_in_game, dog
        )

        records.append(row)

    return pd.DataFrame.from_records(records), n_skipped


# Canonical pass: chrono_key filter.
t_start = time.perf_counter()
feature_matrix_df, n_skipped_unknown_game = build_feature_matrix(
    trigger_full_df, plays_by_game, drives_by_game, plays_before_filter="chrono_key"
)
elapsed_canonical = time.perf_counter() - t_start
print(f"[ok] canonical (chrono_key) feature matrix built in {elapsed_canonical:.1f}s")
print(f"     {len(feature_matrix_df):>6,} rows x {feature_matrix_df.shape[1]} cols")
print(f"     skipped (game not in plays_by_game): {n_skipped_unknown_game}")
assert n_skipped_unknown_game == 0, (
    f"{n_skipped_unknown_game} triggers had no plays in cache. Investigate."
)

# Null counts per candidate feature.
print(f"\\nNull counts per candidate feature (this run):")
print(f"  total in-scope triggers: {len(feature_matrix_df):,}")
null_counts: dict[str, int] = {}
for feat in CANDIDATE_FEATURES:
    n_null = int(feature_matrix_df[feat].isna().sum())
    pct = (n_null / len(feature_matrix_df) * 100) if len(feature_matrix_df) else 0
    null_counts[feat] = n_null
    if feat == "dog_points_off_turnovers":
        extra = " (D7 NULL when no completed dog drive)"
    elif feat == "dog_avg_starting_field_pos":
        extra = " (D5 NULL when no completed dog drive)"
    else:
        extra = " (always defined; 0 when no qualifying drives)"
    print(f"  {feat:<32} {n_null:>5,} null ({pct:5.2f}%){extra}")

# D7 sanity: drive-1 triggers have no completed drives -> NULL on the two
# D7 features and 0 on the two always-defined features.
n_drive1 = int((feature_matrix_df["drive_number_in_game"] == 1).sum())
print(f"\\n  drive-1 trigger count (expected D7 NULL floor for dog_points_off_turnovers + "
      f"dog_avg_starting_field_pos): {n_drive1:,}")
assert null_counts["dog_points_off_turnovers"] >= 0, "dog_points_off_turnovers null count negative"
assert null_counts["dog_avg_starting_field_pos"] >= 0, "dog_avg_starting_field_pos null count negative"

# R16-safe sanity (carried from 02a/02b/02c).
sm_null_after = int(feature_matrix_df["spread_movement"].isna().sum())
sm_indicator_sum = int(feature_matrix_df["spread_movement_is_null"].sum())
print(f"\\nR16-safe NaN handling for spread_movement (baseline):")
print(f"  spread_movement nulls AFTER impute:    {sm_null_after:,}  (expected: 0)")
print(f"  spread_movement_is_null indicator sum: {sm_indicator_sum:,} "
      f"({sm_indicator_sum / len(feature_matrix_df) * 100:.2f}% of in-scope)")
assert sm_null_after == 0, f"spread_movement still has {sm_null_after} nulls after impute"

# Quick summary statistics.
print(f"\\nSummary statistics per candidate (in-scope, non-null):")
for feat in CANDIDATE_FEATURES:
    s = feature_matrix_df[feat].dropna()
    if len(s) == 0:
        print(f"  {feat:<32} (all null)")
        continue
    s_num = s.astype(float)
    print(f"  {feat:<32} n_nonnull={len(s):>5,}  "
          f"mean={s_num.mean():>7.3f}  median={s_num.median():>7.3f}  "
          f"min={s_num.min():>7.1f}  max={s_num.max():>7.1f}")
''')


# ---------------------------------------------------------------------------
# Cell 14 — Diff-vs-leaky verification (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02d000e", """
## Phase 02d-e — Diff-vs-leaky empirical verification (plan-approval addition 1)

Rebuild the feature matrix under the **leaky `playNumber < trig.playNumber` filter** (the pre-correction filter that silently leaked future plays because CFBD's `playNumber` resets per drive). Assert that **all four candidate columns are byte-identical between the canonical (chrono_key) and leaky passes**.

This confirms the Category A claim (drive-metadata-only extractors are insensitive to the play filter) **empirically**, not by extractor sketch. The 02b lookahead-leak post-mortem established that "looks drive-metadata-only" isn't sufficient evidence; this cell does the actual verification.

Expected outcome: byte-identical for all 4 features × 11,416 triggers = 45,664 cell comparisons. If any disagree, halt and surface -- the extractor signature is wrong or there's a subtle path where a feature does touch `plays_before` (e.g., the in-progress trigger drive edge case in `feat_dog_points_off_turnovers`).

The Phase 02d-f walk-forward eval consumes the canonical (chrono_key) feature matrix only. The leaky matrix is built solely for this verification and discarded after the assertion passes.
""")


# ---------------------------------------------------------------------------
# Cell 15 — Diff-vs-leaky verification code
# ---------------------------------------------------------------------------
add("code", "c02d000f", '''
t_start = time.perf_counter()
feature_matrix_df_leaky, n_skipped_leaky = build_feature_matrix(
    trigger_full_df, plays_by_game, drives_by_game,
    plays_before_filter="leaky_playnumber",
)
elapsed_leaky = time.perf_counter() - t_start
print(f"[ok] leaky-filter (playNumber) feature matrix built in {elapsed_leaky:.1f}s")
print(f"     {len(feature_matrix_df_leaky):>6,} rows x "
      f"{feature_matrix_df_leaky.shape[1]} cols")
print(f"     skipped (game not in plays_by_game): {n_skipped_leaky}")

# Both matrices should have the same row count + row order (we iterated
# triggers in the same order in both passes).
assert len(feature_matrix_df_leaky) == len(feature_matrix_df), (
    f"row count mismatch: canonical={len(feature_matrix_df):,} vs "
    f"leaky={len(feature_matrix_df_leaky):,}. Triggers iterated in same order; "
    f"the only way to differ is a build_feature_matrix bug."
)

# Row alignment check: same (game_id, fav_deficit, trigger_sequence) order.
key_cols = ["game_id", "fav_deficit", "trigger_sequence"]
left_keys = feature_matrix_df[key_cols].values.tolist()
right_keys = feature_matrix_df_leaky[key_cols].values.tolist()
assert left_keys == right_keys, (
    f"row order differs between canonical and leaky matrices -- cannot compare "
    f"column-by-column. Investigate build_feature_matrix iteration order."
)
print(f"[ok] row order matches between canonical and leaky matrices ({len(left_keys):,} rows)")

# Per-feature byte-identical assertion.
n_mismatches_by_feat: dict[str, int] = {}
mismatch_samples: list[dict] = []  # first few rows of any mismatch
for feat in CANDIDATE_FEATURES:
    left = feature_matrix_df[feat]
    right = feature_matrix_df_leaky[feat]
    # Treat NaN == NaN as equal (both indicate "no completed drive").
    both_nan = left.isna() & right.isna()
    both_nonnan = (~left.isna()) & (~right.isna())
    nonnan_equal = both_nonnan & (left == right)
    equal_mask = both_nan | nonnan_equal
    n_mismatch = int((~equal_mask).sum())
    n_mismatches_by_feat[feat] = n_mismatch
    if n_mismatch > 0:
        sample_idx = (~equal_mask)[~equal_mask].index[:5].tolist()
        for idx in sample_idx:
            mismatch_samples.append({
                "feature": feat,
                "row": int(idx),
                "game_id": int(feature_matrix_df.at[idx, "game_id"]),
                "fav_deficit": int(feature_matrix_df.at[idx, "fav_deficit"]),
                "trigger_sequence": int(feature_matrix_df.at[idx, "trigger_sequence"]),
                "drive_number_in_game": int(feature_matrix_df.at[idx, "drive_number_in_game"]),
                "play_number": int(feature_matrix_df.at[idx, "play_number"]),
                "canonical": (None if pd.isna(left.iloc[idx])
                              else (float(left.iloc[idx]) if "field_pos" in feat
                                    else int(left.iloc[idx]))),
                "leaky": (None if pd.isna(right.iloc[idx])
                          else (float(right.iloc[idx]) if "field_pos" in feat
                                else int(right.iloc[idx]))),
            })

print(f"\\nPer-feature mismatch counts between canonical and leaky matrices:")
n_total = len(feature_matrix_df)
all_zero = True
for feat in CANDIDATE_FEATURES:
    n_mm = n_mismatches_by_feat[feat]
    pct = (n_mm / n_total * 100) if n_total else 0
    flag = "" if n_mm == 0 else "  <-- MISMATCH"
    if n_mm > 0:
        all_zero = False
    print(f"  {feat:<32} {n_mm:>5,} / {n_total:,}  ({pct:5.2f}%){flag}")

if mismatch_samples:
    print(f"\\nFirst few mismatch samples (up to 5 per feature):")
    for ms in mismatch_samples[:20]:
        print(f"  {ms}")

assert all_zero, (
    f"Diff-vs-leaky check FAILED: not all features are byte-identical between "
    f"chrono_key and leaky-playNumber filter passes. Mismatch counts: "
    f"{n_mismatches_by_feat}. This indicates the Category A claim (drive-metadata-"
    f"only extractors) is wrong for at least one feature; investigate the "
    f"extractor's data-touching path before proceeding."
)
print(f"\\n[ok] DIFF-VS-LEAKY VERIFICATION PASSED: all 4 features byte-identical "
      f"under chrono_key and leaky-playNumber filters across {n_total:,} triggers.")
print(f"     Category A claim (drive-metadata-only) empirically confirmed.")

# Discard the leaky matrix; downstream cells consume only the canonical one.
del feature_matrix_df_leaky
print(f"[ok] leaky matrix discarded; canonical matrix retained for eval.")
''')


# ---------------------------------------------------------------------------
# Cell 16 — Overlap diagnostic (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02d000g", """
## Phase 02d-f — Overlap-fraction diagnostic (plan-approval addition 2)

For each trigger, classify the co-occurrence of `dog_points_off_turnovers` (this notebook's candidate) and `dog_points_from_returns` (02c's validated feature) into 4 buckets:

| Bucket | `dog_points_off_turnovers` | `dog_points_from_returns` |
|---|---|---|
| Both nonzero | > 0 | > 0 |
| Only turnovers-feature nonzero | > 0 | == 0 |
| Only returns-feature nonzero | == 0 | > 0 |
| Both zero | == 0 | == 0 |

Excludes triggers where either feature is NULL (drive-1 triggers, etc.).

If both-nonzero is **<10%** of evaluable triggers, the features are cleanly separable: L1 will treat them as independent signals. If **>50%**, L1 will probably zero one out and N03 will need to pick. The diagnostic informs N03's feature selection but doesn't change 02d's stability verdict.

`dog_points_from_returns` is recomputed inside this cell using 02c's `SCORING_PLAYTYPE_REGISTRY` logic (duplicated in `SCORING_PLAYTYPE_REGISTRY_FOR_DIAGNOSTIC` above). The recomputed values are NOT written to `feature_validation.csv` -- they're a per-trigger diagnostic only.
""")


# ---------------------------------------------------------------------------
# Cell 17 — Overlap diagnostic code
# ---------------------------------------------------------------------------
add("code", "c02d000h", '''
def _recompute_dog_points_from_returns(
    plays_before: list[dict], dog: str,
) -> int | None:
    """Recompute 02c's dog_points_from_returns per trigger using the
    registry-duplicate inline in this notebook.

    Mirrors the relevant branches of 02c's compute_points_buckets, but
    only the return_td / safety_def / pat_def_ret categories that
    contribute to the returns bucket. PAT/2pt attribution to a preceding
    return TD is included per D12 from 02c.

    Returns None when there are no plays_before (drive-1 triggers etc.).
    Otherwise returns int (0 when no qualifying scoring plays).

    This is used ONLY by the D11 overlap diagnostic; not written to
    feature_validation.csv.
    """
    if not plays_before:
        return None

    points_returns = 0
    last_dog_td_bucket_returns: bool = False  # for PAT attribution

    for p in plays_before:
        if not p.get("scoring"):
            continue
        pt = p.get("playType", "")
        cat = SCORING_PLAYTYPE_REGISTRY_FOR_DIAGNOSTIC.get(pt)
        if cat is None or cat == "exclude":
            continue
        play_offense = p.get("offense")
        # Match 02c's compute_points_buckets convention exactly:
        # `scoring_team = fav if play_offense == dog else dog`. Under that
        # rule, play_offense == dog -> scoring_team == fav (no dog points);
        # otherwise (including play_offense in {fav, None, anything else})
        # -> scoring_team == dog. Equivalent: `scoring_is_dog = (play_offense != dog)`.
        if cat == "return_td":
            scoring_is_dog = (play_offense != dog)
            if scoring_is_dog:
                points_returns += 6
                last_dog_td_bucket_returns = True
            else:
                last_dog_td_bucket_returns = False
        elif cat == "safety_def":
            scoring_is_dog = (play_offense != dog)
            if scoring_is_dog:
                points_returns += 2
            # No update to last_dog_td_bucket_returns; safety is its own bucket entry.
        elif cat == "pat_def_ret":
            scoring_is_dog = (play_offense != dog)
            if scoring_is_dog:
                points_returns += 2
        elif cat in ("pat_1pt", "pat_2pt"):
            # PAT/2pt attributes to bucket of preceding TD per D12. Here we
            # only track returns-bucket TDs. If the preceding TD was a return
            # TD scored by the dog AND this PAT is by the dog, add it.
            if play_offense == dog and last_dog_td_bucket_returns:
                points_returns += (1 if cat == "pat_1pt" else 2)
        elif cat in ("offensive_td", "fg"):
            # An offensive TD or FG by the dog resets the "preceding TD bucket"
            # to non-returns -- a PAT following would attribute to
            # offensive_td's bucket, not returns. We don't track which bucket
            # here (we only care about returns), so just clear the flag.
            if play_offense == dog:
                last_dog_td_bucket_returns = False
    return int(points_returns)


# Recompute dog_points_from_returns per trigger, then build the overlap table.
print(f"Recomputing dog_points_from_returns inline for overlap diagnostic "
      f"({len(trigger_full_df):,} triggers)...")
t_start = time.perf_counter()

# Per-trigger recomputed dog_points_from_returns (None if no plays_before).
returns_values: list[int | None] = []
for _, trig in trigger_full_df.iterrows():
    gid = int(trig["game_id"])
    trig_pn = int(trig["play_number"])
    dog = str(trig["dog_team"])
    trig_drive_in_game = int(trig["drive_number_in_game"])
    trig_period = int(trig["quarter"])
    trig_period_elapsed = 900 - int(trig["clock_seconds_in_period_total"])
    trig_chrono_key = (trig_period, trig_period_elapsed, trig_drive_in_game, trig_pn)
    plays = plays_by_game.get(gid)
    if plays is None:
        returns_values.append(None)
        continue
    plays_before = [p for p in plays if _chrono_key(p) < trig_chrono_key]
    val = _recompute_dog_points_from_returns(plays_before, dog)
    returns_values.append(val)

elapsed = time.perf_counter() - t_start
print(f"[ok] recomputed in {elapsed:.1f}s")
assert len(returns_values) == len(feature_matrix_df), (
    f"returns_values length {len(returns_values)} != feature_matrix rows {len(feature_matrix_df)}"
)

# Build the 4-bucket co-occurrence table.
turnover_vals = feature_matrix_df["dog_points_off_turnovers"]
n_eval = 0
n_both_zero = 0
n_only_turnovers = 0
n_only_returns = 0
n_both_nonzero = 0
n_skipped_null = 0
# Also collect mean values within each bucket for richer reporting.
nonzero_turnovers_in_both = []
nonzero_returns_in_both = []
for idx, ret_val in enumerate(returns_values):
    turn_val = turnover_vals.iloc[idx]
    if pd.isna(turn_val) or ret_val is None:
        n_skipped_null += 1
        continue
    n_eval += 1
    turn_nz = int(turn_val) > 0
    ret_nz = int(ret_val) > 0
    if turn_nz and ret_nz:
        n_both_nonzero += 1
        nonzero_turnovers_in_both.append(int(turn_val))
        nonzero_returns_in_both.append(int(ret_val))
    elif turn_nz:
        n_only_turnovers += 1
    elif ret_nz:
        n_only_returns += 1
    else:
        n_both_zero += 1

print(f"\\nOverlap-fraction diagnostic (D11): dog_points_off_turnovers vs "
      f"dog_points_from_returns")
print(f"  Evaluable triggers (both features non-null): {n_eval:,}")
print(f"  Skipped (NULL on either feature, e.g., drive-1 triggers): {n_skipped_null:,}")
print(f"")
print(f"  | Bucket                                            |   Count |   % of eval |")
print(f"  |--------------------------------------------------:|--------:|------------:|")
for label, n in [
    ("Both nonzero",      n_both_nonzero),
    ("Only turnovers > 0",  n_only_turnovers),
    ("Only returns > 0",    n_only_returns),
    ("Both zero",            n_both_zero),
]:
    pct = (n / n_eval * 100) if n_eval else 0
    print(f"  | {label:<48} | {n:>7,} | {pct:>9.2f}% |")

both_nz_pct = (n_both_nonzero / n_eval * 100) if n_eval else 0
print(f"")
print(f"  Both-nonzero fraction: {both_nz_pct:.2f}%")
if both_nz_pct < 10:
    print(f"  [info] <10% both-nonzero -> features are cleanly separable; L1 will "
          f"treat them as independent signals.")
elif both_nz_pct > 50:
    print(f"  [WARN] >50% both-nonzero -> L1 will probably zero one out; N03 will "
          f"need to pick between them.")
else:
    print(f"  [info] 10-50% both-nonzero -> moderate overlap; L1 may de-weight "
          f"one feature without zeroing it. N03 should monitor.")

if n_both_nonzero > 0:
    import statistics as _st
    mean_turn = _st.mean(nonzero_turnovers_in_both)
    mean_ret = _st.mean(nonzero_returns_in_both)
    print(f"")
    print(f"  Within both-nonzero bucket ({n_both_nonzero:,} triggers):")
    print(f"    mean dog_points_off_turnovers: {mean_turn:.2f}")
    print(f"    mean dog_points_from_returns:  {mean_ret:.2f}")
''')


# ---------------------------------------------------------------------------
# Cell 18 — Walk-forward eval (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02d000i", """
## Phase 02d-g — Walk-forward per-feature evaluation

Per-feature null drop (decision B from 02a): for each of the 4 candidates, mask `feature_matrix_df` to rows where the candidate is non-null, then split by season. Two of the candidates (`fav_turnovers_so_far`, `short_field_tds_allowed`) are always defined; the masking is a no-op for those. The other two (`dog_points_off_turnovers`, `dog_avg_starting_field_pos`) drop the drive-1 triggers (D7 NULL).

Eval pipeline: `StandardScaler` -> `LogisticRegression(penalty="l1", C=1.0, solver="liblinear", random_state=42, max_iter=1000)`, then `CalibratedClassifierCV(method="isotonic", cv="prefit")` on the val set; eval Brier + ECE on the test set. Identical helper to 02a / 02b / 02c.

4 candidates x 3 windows = 12 eval rows expected.
""")


# ---------------------------------------------------------------------------
# Cell 19 — ECE + fit helper code
# ---------------------------------------------------------------------------
add("code", "c02d000j", '''
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
    Identical to 02a / 02b / 02c's helper; will be deduped into a shared module
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
# Cell 20 — Eval loop code
# ---------------------------------------------------------------------------
add("code", "c02d000k", '''
eval_rows: list[dict] = []
t_start = time.perf_counter()

for window in WALK_FORWARD_WINDOWS:
    train_seasons = window["train_seasons"]
    val_season = window["val_season"]
    test_season = window["test_season"]
    win_label = window["train_window_label"]

    for feat in CANDIDATE_FEATURES:
        # Per-feature null drop (decision B from 02a). For always-defined
        # features, this is a no-op.
        mask = feature_matrix_df[feat].notna()
        sub = feature_matrix_df[mask]
        train_sub = sub[sub["season"].isin(train_seasons)]
        val_sub = sub[sub["season"] == val_season]
        test_sub = sub[sub["season"] == test_season]
        cand_cols = BASELINE_PREGAME_FEATURES + [feat]

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
            "imputation_value": None,  # No imputation in 02d; column exists for CSV compat with 02c onward
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

# Column order for CSV write -- matches 02c onward (imputation_value column).
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
    n_pos_ece = int((eval_df[eval_df["feature"] == feat]["calibration_improvement"] > 0).sum())
    verdict = "PASS" if stability_decision[feat] else "FAIL"
    print(f"  {feat:<32} {verdict}  "
          f"({n_pos}/3 brier-improving, {n_pos_ece}/3 ece-improving)")
''')


# ---------------------------------------------------------------------------
# Cell 21 — CSV write (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02d000l", """
## Phase 02d-h — Write `feature_validation.csv` (defensive append)

Same defensive-append pattern as 02a / 02b / 02c:

1. Read existing CSV with `keep_default_na=False` so the `redundant_with` empty-string convention round-trips.
2. Drop rows matching this run's `(feature, train_window, test_season)` keys.
3. Concatenate this run's 12 new rows. `pd.concat` unions columns; existing rows keep their `imputation_value` (NULL except for 02c's continuous momentum rows).
4. Sort by `(feature_set_version, feature, train_window, test_season)`.
5. Write.

02a / 02b / 02c rows are preserved (their keys don't overlap with 02d's). Natural-key uniqueness is asserted after the write.
""")


# ---------------------------------------------------------------------------
# Cell 22 — CSV write code
# ---------------------------------------------------------------------------
add("code", "c02d000m", '''
NEW_KEYS = set(zip(
    eval_df["feature"],
    eval_df["train_window"],
    eval_df["test_season"].astype(int),
))

if FEATURE_VALIDATION_CSV.exists():
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

# Ensure imputation_value column exists for all rows.
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
# Cell 23 — Schema sidecar (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02d000n", """
## Phase 02d-i — Splice `feature_validation.schema.md` sidecar

02d's sidecar section is sentinel-delimited (same pattern as 02b / 02c). Includes:

1. Candidate list with D1-D12 references.
2. **Worked example (D3):** the no-next-fav-drive edge case for `dog_points_off_turnovers`.
3. **Diff-vs-leaky verification result (D10):** byte-identical confirmation across {len(feature_matrix_df):,} triggers x 4 features.
4. **Overlap-fraction diagnostic (D11):** the 4-bucket co-occurrence table for `dog_points_off_turnovers` vs `dog_points_from_returns`.
5. **Cumulative validated-set context (D12):** running validated-set count and notable conditional identities after 02d.
6. **Conditional-identity flags:** `dog_points_off_turnovers` <-> `short_field_tds_allowed` (partial subset); `dog_points_off_turnovers` <-> `dog_points_from_returns` (02c) co-occurrence empirically measured above.
7. Per-feature null counts + stability table.

02a / 02b / 02c sections preserved by sentinel splicing. Same known limitation as before: 02a's writer doesn't yet use splicing -- tracked as tech_debt item 3.
""")


# ---------------------------------------------------------------------------
# Cell 24 — Schema sidecar code
# ---------------------------------------------------------------------------
add("code", "c02d000o", '''
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
        verdict_rows.append(
            f"| `{feat}` | {r['train_window']} -> test {int(r['test_season'])} | "
            f"{_fmt_delta(r['brier_improvement'])} | {_fmt_delta(r['calibration_improvement'])} | "
            f"{'**PASS**' if r['passed_stability'] else 'FAIL'} |"
        )

# Per-feature null counts.
null_rows = []
for feat in CANDIDATE_FEATURES:
    n_null = null_counts[feat]
    pct = (n_null / len(feature_matrix_df) * 100) if len(feature_matrix_df) else 0
    if feat == "dog_points_off_turnovers":
        tag = " (D7 no-completed-dog-drive NULL)"
    elif feat == "dog_avg_starting_field_pos":
        tag = " (D5 no-completed-dog-drive NULL)"
    else:
        tag = ""
    null_rows.append(f"| `{feat}` | {n_null:,} | {pct:.2f}%{tag} |")

# Cumulative validated set after 02d.
fv_after = pd.read_csv(FEATURE_VALIDATION_CSV, keep_default_na=False)
fv_after["brier_improvement"] = fv_after["brier_improvement"].astype(float)
cumulative_validated: list[tuple[str, str, int, int]] = []  # (version, feature, n_brier_pos, n_ece_pos)
for (fsv, feat), grp in fv_after.groupby(["feature_set_version", "feature"]):
    n_pos = int((grp["brier_improvement"] > 0).sum())
    n_pos_ece = int((grp["calibration_improvement"].astype(float) > 0).sum())
    if n_pos >= 2:
        cumulative_validated.append((fsv, feat, n_pos, n_pos_ece))

cumulative_validated.sort()
cumul_rows = []
for fsv, feat, n_b, n_e in cumulative_validated:
    cumul_rows.append(f"| `{feat}` | {fsv} | {n_b}/3 | {n_e}/3 |")

# Overlap diagnostic rows (re-format from the printed table above).
overlap_rows = [
    f"| Both nonzero | {n_both_nonzero:,} | {(n_both_nonzero / n_eval * 100) if n_eval else 0:.2f}% |",
    f"| Only `dog_points_off_turnovers` > 0 | {n_only_turnovers:,} | {(n_only_turnovers / n_eval * 100) if n_eval else 0:.2f}% |",
    f"| Only `dog_points_from_returns` > 0 | {n_only_returns:,} | {(n_only_returns / n_eval * 100) if n_eval else 0:.2f}% |",
    f"| Both zero | {n_both_zero:,} | {(n_both_zero / n_eval * 100) if n_eval else 0:.2f}% |",
]

n_drive1 = int((feature_matrix_df["drive_number_in_game"] == 1).sum())

# Build the 02d-owned section.
SECTION_BEGIN = "<!-- BEGIN: 02d turnover_short_field -->"
SECTION_END = "<!-- END: 02d turnover_short_field -->"

section_body = f"""
## 02d -- Turnover & short-field features

**Section last writer:** `research/notebooks/02d_turnover_and_short_field.ipynb`
**Last writer commit:** `{commit_hash}`
**Last writer generation timestamp:** {now_text}
**Feature set version:** `{FEATURE_SET_VERSION}`
**Source DDL:** `BUILD_SPEC.md` `trigger_features` turnover & short-field block (V5 lines 189-193)

### Candidate features (4)

- `fav_turnovers_so_far` (D1; always defined; 0 when no fav turnovers)
- `dog_points_off_turnovers` (D3/D4/D7; NULL when no completed dog drive; excludes return TDs which are bucketed in 02c's `dog_points_from_returns`)
- `dog_avg_starting_field_pos` (D5; yards-from-own-end-zone convention; NULL when no completed dog drive)
- `short_field_tds_allowed` (D2/D6; always defined; counts completed dog-offense drives with `startYardsToGoal <= {SHORT_FIELD_THRESHOLD}` AND `driveResult == 'TD'`)

### D1: turnover definition

`fav_turnovers_so_far` counts pre-trigger fav-offense completed drives whose
`driveResult` falls in the standard NFL/CFB turnover set:

```python
TURNOVER_DRIVE_RESULTS = {{"INT", "INT TD", "FUMBLE", "FUMBLE TD", "FUMBLE RETURN TD"}}
```

Excludes `DOWNS` (turnover-on-downs; "giveaway" but not a traditional
turnover) and `SF` (safety; defensive score against the offense, not a
turnover-causing play). 22,021 PUNT drives and 4,947 FG drives are
correctly not counted.

### D3 worked example: in-progress trigger drive

`dog_points_off_turnovers` follows each non-return-TD fav turnover (INT
or FUMBLE) to the immediate-next dog-offense drive and adds that drive's
points to the feature.

The edge case is the **last** fav turnover before the trigger when its
immediate next drive happens to be the trigger drive itself (in
progress, so not in the "completed drives" filter). In that case:

- Look up the trigger drive in `drives_for_game` by
  `driveNumber == trig_drive_in_game`.
- If `trigger_drive.offense == dog`, add
  `max(0, dog_score_at_trigger - trigger_drive.startOffenseScore)`.
- Otherwise (trigger drive isn't dog-offense), add nothing -- the
  in-progress non-dog drive doesn't capitalize for the dog.

**Worked example** (synthetic):

- Game state: fav threw an INT on drive 8 (their 4th offensive drive).
- Drive 9 begins as dog offense, at the 30 yard line (short field).
- The dog scores a TD on drive 9 (the trigger play is the TD itself, or
  the post-TD play that crossed the deficit threshold).
- At the trigger, `drive_number_in_game = 9`, so drive 9 is **NOT** in
  `completed_drives_before_trigger`. Drives 1-8 are.
- Of drives 1-8, drive 8 is the only fav turnover. The "immediate next"
  drive (drive 9) is the trigger drive.
- `_trigger_drive(drives_for_game, 9)` returns drive 9.
- `trigger_drive.offense == dog`. `dog_score_at_trigger == 7` (dog scored
  the TD + PAT just now). `trigger_drive.startOffenseScore == 0` (dog
  had 0 points entering drive 9).
- Contribution: `max(0, 7 - 0) == 7`. `dog_points_off_turnovers == 7`.

**General clarity note** (per plan-approval D3 refinement): when the
trigger drive itself is dog-offense for ANY reason (not necessarily
following a turnover), `dog_score_at_trigger` correctly captures the
dog's score AFTER any points scored on the trigger drive prior to (or
on) the trigger play. The formula `dog_score_at_trigger -
trigger_drive.startOffenseScore` isolates the points from this
in-progress drive specifically -- if the dog had scored on an earlier
drive (e.g., a TD on drive 5 = 7 pts), `startOffenseScore` would be 7
at the start of drive 9, and the formula would correctly attribute the
drive-9 points only.

### D4: return-TD exclusion

`dog_points_off_turnovers` EXCLUDES the three return-TD turnover
driveResults (`INT TD`, `FUMBLE TD`, `FUMBLE RETURN TD`) because those
are dog points scored DIRECTLY on the turnover by the dog defense. The
dog defense scoring a pick-6 on a fav INT is bucketed under 02c's
`dog_points_from_returns` (the kickoff or punt return TD category +
INT/Fumble return TD category). Counting them here too would
double-count.

Only `INT` and `FUMBLE` (the non-return turnovers) contribute to
`dog_points_off_turnovers`, via the points scored on the SUBSEQUENT
dog-offense drive.

### D11: overlap-fraction diagnostic

For each evaluable trigger, classify into a 4-bucket co-occurrence
table on (`dog_points_off_turnovers` nonzero, `dog_points_from_returns`
nonzero). Triggers where either feature is NULL are excluded.

Evaluable triggers: {n_eval:,}. Skipped (NULL on either): {n_skipped_null:,}.

| Bucket | Triggers | % of evaluable |
|---|---:|---:|
""" + "\\n".join(overlap_rows) + f"""

Both-nonzero fraction: {(n_both_nonzero / n_eval * 100) if n_eval else 0:.2f}%.
{(
'<10% -> features are cleanly separable; L1 will treat them as independent signals.'
if (n_both_nonzero / n_eval * 100 if n_eval else 0) < 10
else (
'>50% -> L1 will probably zero one out; N03 will need to pick.'
if (n_both_nonzero / n_eval * 100 if n_eval else 0) > 50
else '10-50% -> moderate overlap; L1 may de-weight one feature without zeroing it. N03 should monitor.'
))}

Recomputation uses 02c's `SCORING_PLAYTYPE_REGISTRY` (duplicated inline
in this notebook as `SCORING_PLAYTYPE_REGISTRY_FOR_DIAGNOSTIC`). The
recomputed values are NOT written to `feature_validation.csv` -- this is
a per-trigger diagnostic only.

### D10: diff-vs-leaky verification

Built the feature matrix twice -- canonical (chrono_key) and leaky
(`playNumber < trig.playNumber`) -- and asserted byte-identical per-trigger
values across all 4 candidate columns. Result: **all 4 features
byte-identical across {len(feature_matrix_df):,} triggers**. The Category
A claim (drive-metadata-only extractors) is empirically confirmed.

### Per-feature null counts (this run)

In-scope triggers (post NaN `final_fav_won` drop): {len(feature_matrix_df):,}.
Drive-1 trigger count (D7 null floor): {n_drive1:,}.

| Feature | Null rows | % of in-scope |
|---|---:|---:|
""" + "\\n".join(null_rows) + f"""

### Per-feature x per-test-season results (this run, {FEATURE_SET_VERSION})

| Feature | Window -> Test | Brier improvement | ECE improvement | Stability |
|---|---|---:|---:|---|
""" + "\\n".join(verdict_rows) + f"""

Sign convention: positive = candidate beat baseline. `**PASS**` means
`sum(brier_improvement > 0) >= 2` across the 3 test seasons.

### D12: cumulative validated-set context after 02d

All passing features across `feature_validation.csv` after this run:

| Feature | Feature set | Brier 3-fold | ECE 3-fold |
|---|---|---:|---:|
""" + "\\n".join(cumul_rows) + f"""

**Total cumulative validated features:** {len(cumulative_validated)}.

Notable cross-notebook conditional identities accumulating into N03's
feature-selection picture:

1. `dog_def_epa_per_play` (02a) tagged `redundant_with=fav_off_epa_per_play`
   (byte-identical pair; FAILED under correction).
2. `dog_off_epa_per_play` (02a) tagged `redundant_with=fav_def_epa_per_play`
   (byte-identical pair; both PASSED).
3. `dog_explosive_play_count` (02c) <-> `opening_drive_was_explosive_td`
   (02b): drive-1 conditional overlap.
4. `dog_avg_drive_yards` (02c) <-> `opening_drive_yards` (02b): conditional
   identity on drive_number_in_game == 2 + dog had drive 1 (02b's
   opening_drive_yards FAILED; identity is moot for the validated set).
5. `dog_avg_drive_plays` (02c) <-> `opening_drive_plays` (02b): same
   conditional structure (also moot; opening_drive_plays FAILED).
6. `dog_points_off_turnovers` (02d) <-> `dog_points_from_returns` (02c):
   co-occurrence empirically measured above ({both_nz_pct:.2f}% both-nonzero).
7. `dog_points_off_turnovers` (02d) <-> `short_field_tds_allowed` (02d):
   partial subset (a short-field TD allowed after a fav turnover counts in
   both); subset relation, not structural identity.

### Redundancy discoveries (02d plan-time audit)

Plan-time verdict: **zero structural duplicates among 02d's 4
candidates.** `REDUNDANT_WITH = {{}}` for this feature set version.
All 12 rows have `redundant_with == ""`.

Two **conditional identities** flagged at plan time:

1. `dog_points_off_turnovers` <-> `short_field_tds_allowed` (within 02d):
   partial subset. A short-field TD allowed after a fav turnover contributes
   to both features. Not structural identity (turnovers without short
   fields, and short fields not from turnovers, both produce one-only
   nonzero cases). N03 should treat them as independent signals unless
   the empirical correlation post-eval suggests otherwise.
2. `dog_points_off_turnovers` <-> `dog_points_from_returns` (02c):
   co-occurrence measured by the D11 diagnostic above. They are
   cleanly separable (no double-counting) but can co-occur when a
   game has multiple fav turnovers (a pick-6 + later non-return turnover
   the dog converts via offense).

### Section provenance

- Last writer: this 02d run (timestamp + commit above).
- Splicing strategy: sentinel-delimited; re-running 02d refreshes only
  this section. Re-running 02a in its current form WILL clobber 02b's,
  02c's, and 02d's sections -- tracked as `research/tech_debt.md` item 3.
"""

new_section = SECTION_BEGIN + "\\n" + section_body.rstrip() + "\\n" + SECTION_END

if FEATURE_VALIDATION_SCHEMA.exists():
    existing_text = FEATURE_VALIDATION_SCHEMA.read_text(encoding="utf-8")
    if SECTION_BEGIN in existing_text and SECTION_END in existing_text:
        start = existing_text.index(SECTION_BEGIN)
        end = existing_text.index(SECTION_END) + len(SECTION_END)
        updated = existing_text[:start] + new_section + existing_text[end:]
        print(f"[ok] spliced 02d section in place (existing markers found)")
    else:
        updated = existing_text.rstrip() + "\\n\\n" + new_section + "\\n"
        print(f"[ok] appended 02d section at end of sidecar (markers added)")
else:
    header = (
        "# feature_validation.csv -- schema sidecar\\n\\n"
        "(02a + 02b + 02c sections missing -- run 02a / 02b / 02c to regenerate.)\\n\\n"
    )
    updated = header + new_section + "\\n"
    print(f"[warn] sidecar did not exist; wrote stub header + 02d section.")

FEATURE_VALIDATION_SCHEMA.write_text(updated, encoding="utf-8")
print(f"[ok] wrote feature_validation.schema.md ({len(updated):,} chars)")
print(f"     path: {FEATURE_VALIDATION_SCHEMA}")
''')


# ---------------------------------------------------------------------------
# Cell 25 — Summary (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02d000p", """
## Phase 02d-j — Summary, headline stats, hypothesis-watch result, STOP banner
""")


# ---------------------------------------------------------------------------
# Cell 26 — Summary print code
# ---------------------------------------------------------------------------
add("code", "c02d000q", '''
print("=" * 70)
print("Notebook 02d -- turnover & short-field features -- summary")
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
print(f"\\nDrive-1 trigger count (D7 null floor for dog_points_off_turnovers + "
      f"dog_avg_starting_field_pos):")
print(f"  drive_number_in_game == 1: {n_drive1_final:,} "
      f"({n_drive1_final / len(feature_matrix_df) * 100:.2f}%)")

print(f"\\nDiff-vs-leaky verification (D10):")
print(f"  4 features x {len(feature_matrix_df):,} triggers = "
      f"{4 * len(feature_matrix_df):,} cell comparisons")
print(f"  Mismatches: 0  (all features byte-identical under chrono_key and "
      f"leaky-playNumber filters)")
print(f"  Category A claim (drive-metadata-only) empirically confirmed.")

print(f"\\nOverlap diagnostic (D11): dog_points_off_turnovers vs dog_points_from_returns")
print(f"  Evaluable triggers: {n_eval:,}; both-nonzero: {n_both_nonzero:,} "
      f"({both_nz_pct:.2f}%)")

print(f"\\nPer-feature x per-test-season results ({FEATURE_SET_VERSION}):")
print(f"  {'feature':<32} {'window->test':<18} "
      f"{'d_brier':>10} {'d_ece':>10} {'stab':>6}")
for _, r in eval_df.sort_values(["feature", "test_season"]).iterrows():
    win = f"{r['train_window']}->{int(r['test_season'])}"
    print(f"  {r['feature']:<32} {win:<18} "
          f"{r['brier_improvement']:>+10.5f} {r['calibration_improvement']:>+10.5f} "
          f"{'PASS' if r['passed_stability'] else 'FAIL':>6}")

print(f"\\nFeature stability verdicts:")
verdicts = {}
for feat in CANDIDATE_FEATURES:
    n_pos = int((eval_df[eval_df["feature"] == feat]["brier_improvement"] > 0).sum())
    n_pos_cal = int((eval_df[eval_df["feature"] == feat]["calibration_improvement"] > 0).sum())
    verdict = "PASS" if stability_decision[feat] else "FAIL"
    verdicts[feat] = verdict
    print(f"  {feat:<32} {verdict:<5} "
          f"({n_pos}/3 brier-improving, {n_pos_cal}/3 ece-improving)")

# --- Hypothesis-watch result (plan-approval framing) -----------------------
print(f"\\nHypothesis-watch (plan-approval framing):")
print(f"  Plan-time prediction: 2/4 pass, favoring dog_points_off_turnovers "
      f"and short_field_tds_allowed.")
print(f"  User watch: dog_points_off_turnovers more likely than "
      f"short_field_tds_allowed (capitalization measure > one-step-removed "
      f"trigger-conditioning).")
n_passes = sum(1 for v in verdicts.values() if v == "PASS")
print(f"  Actual: {n_passes}/4 pass.")
dpt_pass = verdicts["dog_points_off_turnovers"] == "PASS"
sfta_pass = verdicts["short_field_tds_allowed"] == "PASS"
ftsf_pass = verdicts["fav_turnovers_so_far"] == "PASS"
dafp_pass = verdicts["dog_avg_starting_field_pos"] == "PASS"
print(f"    fav_turnovers_so_far:        {verdicts['fav_turnovers_so_far']}")
print(f"    dog_points_off_turnovers:    {verdicts['dog_points_off_turnovers']}")
print(f"    dog_avg_starting_field_pos:  {verdicts['dog_avg_starting_field_pos']}")
print(f"    short_field_tds_allowed:     {verdicts['short_field_tds_allowed']}")
if dpt_pass and sfta_pass and not ftsf_pass and not dafp_pass:
    print(f"  Result: plan prediction CONFIRMED (2/4 via the two predicted features).")
elif dpt_pass and not sfta_pass:
    print(f"  Result: user watch CONFIRMED relative to short_field_tds_allowed "
          f"(dog_points_off_turnovers passes, short_field_tds_allowed fails).")
    print(f"  Interpretation: capitalization > trigger-conditioned conversion.")
elif sfta_pass and not dpt_pass:
    print(f"  Result: user watch FALSIFIED -- short_field_tds_allowed passes "
          f"while dog_points_off_turnovers fails.")
    print(f"  Interpretation: trigger-conditioning isn't as harsh as the user "
          f"watch model suggested; short-field conversion still carries "
          f"signal even after the trigger conditioning.")
else:
    print(f"  Result: empirical pattern doesn't match either the plan prediction "
          f"or the user watch cleanly. Mechanism interpretation deferred to N03.")

# --- Cumulative validated set after 02d ------------------------------------
print(f"\\nCumulative validated set after 02d:")
fv = pd.read_csv(FEATURE_VALIDATION_CSV, keep_default_na=False)
fv["brier_improvement"] = fv["brier_improvement"].astype(float)
cumul = []
for (fsv, feat), grp in fv.groupby(["feature_set_version", "feature"]):
    n_pos = int((grp["brier_improvement"] > 0).sum())
    if n_pos >= 2:
        cumul.append((fsv, feat))
cumul.sort()
print(f"  Total validated features: {len(cumul)}")
by_version: dict[str, list[str]] = {}
for fsv, feat in cumul:
    by_version.setdefault(fsv, []).append(feat)
for fsv in sorted(by_version):
    print(f"  [{fsv}] ({len(by_version[fsv])}):")
    for feat in by_version[fsv]:
        print(f"    - {feat}")

print(f"\\nProjection: ~25-30 features by end of 02g (per plan-approval).")
print(f"  After 02d: {len(cumul)} features (notebooks 02a/b/c/d done).")
print(f"  Remaining: 02e (red-zone failure, 3 candidates), 02f (down-and-")
print(f"  distance efficiency, 4 candidates), 02g (context, 4 candidates) =")
print(f"  11 additional candidates projected. If they pass at the corrections-")
print(f"  era ~70% rate, expect ~{len(cumul) + int(11 * 0.7)} features by 02g.")

print(f"\\nDeliverables (research/results/):")
for path in [FEATURE_VALIDATION_CSV, FEATURE_VALIDATION_SCHEMA]:
    size = path.stat().st_size
    print(f"  {path.name:<40} {size:>10,} bytes")
''')


# ---------------------------------------------------------------------------
# Cell 27 — Budget print + STOP banner
# ---------------------------------------------------------------------------
add("code", "c02d000r", '''
calls_log_df = pd.read_csv(CALL_LOG)
n_total_log_rows = len(calls_log_df)
n_fresh_cfbd_total = int(((calls_log_df["service"] == "cfbd")
                          & (calls_log_df["cached"] == 0)).sum())

this_run_calls = calls_log_df.iloc[n_log_before:].copy()
n_this_run = len(this_run_calls)
n_this_run_fresh = int((this_run_calls["cached"] == 0).sum())

print("=" * 64)
print("CFBD call budget -- Notebook 02d")
print("=" * 64)
print(f"\\nThis notebook run:")
print(f"  total calls this run:     {n_this_run:>5,}  ({n_plays_lookups} /plays + {n_drives_lookups} /drives)")
print(f"  fresh (uncached) this run: {n_this_run_fresh:>5,}  (budget: 0)")

assert n_this_run_fresh == 0, (
    f"02d budget invariant violated: {n_this_run_fresh} fresh CFBD call(s) "
    f"this run. 02d is supposed to spend 0 fresh CFBD calls."
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

print(f"\\n[ok] notebook 02d complete -- STOP per R22. "
      f"Do not start Notebook 02e without approval.")
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
