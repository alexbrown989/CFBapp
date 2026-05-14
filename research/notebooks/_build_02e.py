"""
Deterministic builder for research/notebooks/02e_red_zone_failure.ipynb.

Mirrors _build_02d.py:
  - Same cache-hit-assertion contract.
  - Same defensive-append pattern for feature_validation.csv.
  - Sentinel-delimited splicing of the schema sidecar so 02a / 02b / 02c / 02d
    content is preserved verbatim.
  - Same _chrono_key fix via _lib_chrono.CHRONO_KEY_SOURCE.

02e-specific additions vs 02d:
  - Three red-zone-failure candidates: `fav_red_zone_trips`,
    `fav_red_zone_tds`, `fav_yards_per_point` (V5 DDL block 5,
    BUILD_SPEC.md lines 195-198).
  - Mixed-category extractors: `fav_red_zone_trips` and `fav_red_zone_tds`
    are **Category B** (drive metadata filter + iterate plays within
    completed drives for red-zone-entry detection); `fav_yards_per_point`
    is **Category A** (drive metadata only).
  - **D10 disagreement-distribution diagnostic (per plan-approval
    addition 1):** for the two Category B features, the diff-vs-leaky
    pass does NOT assert byte-identical -- it quantifies the magnitude
    distribution of (chrono - leaky) per trigger. Off-by-1 / 2 / 3+
    bucketing tells us how broken the leaky version would have been
    if shipped. For `fav_yards_per_point` (Cat A), byte-identical IS
    asserted.
  - **D11 red-zone conversion diagnostic:** at trigger time, classify
    each trigger by (`fav_red_zone_trips`, `fav_red_zone_tds`) state:
    0 trips, 0+ trips with 0 TDs (stalled), 0+ trips with all-TD (perfect),
    partial conversion. Plus distribution of the implicit
    `red_zone_pct = TDs / trips` ratio.
  - **D7 two-bucket null breakdown (per plan-approval addition 2):**
    for `fav_yards_per_point`, count separately (a) NULL because no
    completed fav drives (early-game / drive-1 triggers), and (b) NULL
    because completed fav drives exist but cumulative fav offensive
    points == 0 (the "fav offense is stalled" diagnostic state).
  - **D8 paired-indicator imputation:** `fav_yards_per_point` uses
    R16-safe per-train-window median imputation + paired
    `fav_yards_per_point_is_null` indicator (same Mode B pattern as
    02c's `seconds_since_last_dog_explosive_play`).
  - **D12 cumulative validated-set context:** summary cell prints the
    running validated-set count after 02e (21 + N) and notable
    conditional identities accumulating across 02a + 02b + 02c +
    02d + 02e.

This is a scratchpad file (per the research/notebooks/_*.py convention).
Not part of the deliverable.
"""

from __future__ import annotations

import json
import pathlib
import sys
import textwrap

OUT = pathlib.Path(__file__).resolve().parent / "02e_red_zone_failure.ipynb"

# Pull the canonical _chrono_key source from the shared helper module
# (single-source-of-truth across 02a/02b/02c/02d/02e build scripts). See
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
add("markdown", "bd02e000", """
# Phase 0 — Notebook 02e: Red-zone failure features

## Hypothesis (per-feature stability claims under R6)

Three parallel claims tied to the V5 DDL block 5 (red-zone failure;
`BUILD_SPEC.md` lines 195-198):

1. **`fav_red_zone_trips`** -- the favorite's count of pre-trigger
   completed drives that REACHED the red zone (any play with
   `yardsToGoal <= 20`) carries comeback-equity signal beyond the
   pre-game baseline. Football mechanism: red-zone trips are a direct
   opportunity-volume measure for the favorite's offense. A favorite
   who has been deep in opponent territory many times and is STILL
   losing by `>= 7` (the trigger condition) has been failing to
   convert opportunities -- a structural signal of offensive
   underperformance the pre-game line did not price.

2. **`fav_red_zone_tds`** -- the favorite's count of pre-trigger
   completed drives that ended in a TD AND reached the red zone.
   Football mechanism: red-zone TD conversion is the cleanest single-
   feature measure of finishing ability. Brian Burke / Football
   Outsiders / PFF treat red-zone TD% as foundational. Conditional on
   the trigger (fav down 7+), `fav_red_zone_tds` directly measures
   whether the fav offense CAN finish drives; the trigger fires
   because it hasn't.

3. **`fav_yards_per_point`** -- inverse efficiency: total favorite
   offensive yards divided by total favorite offensive points across
   all completed pre-trigger drives. Lower = more efficient. Bill
   Connelly's classic measure (under 14 = elite, over 18 = struggling).
   Conditional on the trigger, high values indicate the fav offense
   is moving the ball but failing to convert into points -- the
   defining structural state of the comeback-equity setup. **Imputed
   with R16-safe per-train-window median + paired
   `fav_yards_per_point_is_null` indicator** when the denominator is
   zero (D8 paired-indicator pattern from 02c).

All three claims use the pre-game-only baseline locked in 02a
(`pregame_spread`, `rating_gap`, `fav_pregame_rating`, `dog_pregame_rating`,
`spread_movement`, `spread_movement_is_null`). Per-feature null policy
applies (decision **B** from 02a).

## What this notebook DOES NOT do

- Does not modify `trigger_events.csv` or `trigger_outcomes.csv`.
- Does not pull any fresh CFBD data -- every `/plays` AND `/drives` lookup
  must hit the cache produced by N01. Cache-hit assertion fails loud on
  any miss.
- Does not select features for the production model -- N03's job.
- Does not test feature groups other than red-zone failure (those are
  02a / 02b / 02c / 02d / 02f-g).
- Does not tune hyperparameters of the L1 logreg -- uses sklearn default
  `C=1.0` with a fixed seed, identical to 02a / 02b / 02c / 02d.
- Does not parameterize the red-zone threshold -- locked at
  `yardsToGoal <= 20` (standard NFL/CFB convention; user-approved at
  plan time, not parameterized). If results suggest tuning, log to
  `research/tech_debt.md`.

## Spec references

- `BUILD_SPEC.md` Phase 0 Notebook 02 deliverable spec -- `feature_validation.csv` shape
- `BUILD_SPEC.md` `trigger_features` DDL -- red-zone failure block (V5 lines 195-198)
- `research/corrections_log.md` section 1 -- the `_chrono_key` composite filter and the lookahead-bias fix history
- `research/corrections_log.md` "02d prediction-vs-result calibration" -- soft prior on small-magnitude features
- `.cursorrules` **R2 + R3** -- no lookahead; `assert_no_lookahead()` mandatory on every feature extraction
- `.cursorrules` **R5** -- walk-forward validation only
- `.cursorrules` **R6** -- stability rule (>=2 of 3 test seasons)
- `.cursorrules` **R7** -- L1 logreg / shallow GBM only
- `.cursorrules` **R8** -- ECE on 10 bins, post-calibration
- `.cursorrules` **R16** -- pre-game-safe NaN handling for non-random missingness
- `.cursorrules` **R19** -- record rejected features too
- `.cursorrules` **R22** -- STOP at end of 02e; do not start 02f without approval
- `.cursorrules` **R23** -- commit message standards (auto-applied)

## Decision-points log (from the 02e plan-approval)

- **D1** -- Red-zone threshold: `yardsToGoal <= 20` (standard NFL/CFB convention; user-approved as locked, not parameterized).
- **D2** -- Drive-completion test: `driveNumber < trigger_drive_in_game`. Identical to 02d. The trigger drive itself is in-progress and excluded from all three features' calculations.
- **D3** -- Red-zone trip definition (D5 from the plan): a completed fav-offense drive `D` counts as a trip iff `∃ play p ∈ D : p.driveNumber == D.driveNumber AND p.offense == D.offense AND p.yardsToGoal <= 20`. The `p.offense == D.offense` guard (plan-approved D5) protects against any cross-offense plays misattributed to the drive (defensive returns, kickoffs).
- **D4** -- Red-zone TD definition (D6 from the plan): drive ended in TD by the offense iff `D.driveResult ∈ {'TD', 'END OF GAME TD'}` AND `D.offense == fav`. Excludes `INT TD`, `FUMBLE TD`, `FUMBLE RETURN TD`, `PUNT TD` (those are defensive/special-teams TDs by the OPPOSING team). Includes `END OF GAME TD` (fav offensive TD that happens to end the game).
- **D5** -- `fav_yards_per_point` numerator/denominator:
    - Numerator: `sum(drive.yards)` over completed fav-offense drives with non-null `yards`.
    - Denominator: `sum(max(0, drive.endOffenseScore - drive.startOffenseScore))` over completed fav-offense drives. Per-drive subtraction captures fav offensive points only (dog defensive/return TDs don't affect `endOffenseScore`).
    - Non-negative clamp on per-drive point delta guards against anomalous CFBD encoding.
- **D6** -- (Reserved; absorbed into D4 above for consistency with the plan-approval D-numbering.)
- **D7** -- Null policy:
    - `fav_red_zone_trips`: default int 0 (literally true: 0 trips when no completed fav drives).
    - `fav_red_zone_tds`: default int 0 (literally true: 0 TDs).
    - `fav_yards_per_point`: NULL when **either** (a) no completed fav-offense drives OR (b) cumulative fav offensive points == 0 (denominator-zero, ratio undefined).
- **D7-bucket-breakdown (plan-approval addition 2):** at execution time, count the two NULL conditions separately:
    - **Bucket (a):** no completed fav drives (early-game / drive-1 triggers).
    - **Bucket (b):** completed fav drives exist but cumulative fav offensive points == 0 (the "fav offense is stalled" diagnostic state).
    The diagnostic informs whether the imputation strategy is fitting genuinely-informative non-data (bucket b is the informative case) or just default-fills early-game gaps (bucket a). No design change to D7/D8 unless bucket (b) is large (>200 triggers) AND `fav_yards_per_point` fails at small magnitudes -- in which case worst-case-imputation alternative becomes a tech-debt entry for N03.
- **D8** -- Imputation: `fav_yards_per_point` only. R16-safe per-train-window median + paired `fav_yards_per_point_is_null` indicator. Mode B from 02c (same pattern as `seconds_since_last_dog_explosive_play`). Per-window median stored in the `imputation_value` column on `feature_validation.csv`.
- **D9** -- Composite `_chrono_key` filter from 02c carries forward. Plays sorted by chrono_key at load time. The play subset gates `assert_no_lookahead`; the Category B extractors (`fav_red_zone_trips`, `fav_red_zone_tds`) iterate `plays_before` to detect red-zone entry per completed drive.
- **D10 disagreement-distribution diagnostic (plan-approval addition 1):**
    - Category A (`fav_yards_per_point`): build the feature matrix twice (chrono_key and leaky-playNumber filters); **assert byte-identical** values. Confirms the Category A claim empirically.
    - Category B (`fav_red_zone_trips`, `fav_red_zone_tds`): build twice; do NOT assert byte-identical. **Quantify the bidirectional magnitude distribution** of `diff = chrono - leaky` per trigger. **Positive diff** (chrono > leaky): playNumber-based TRUNCATION within drives (02b mechanism) -- undercounts red-zone entry that happened on high `playNumber` plays in completed drives. **Negative diff** (leaky > chrono): CROSS-DRIVE FORWARD CONTAMINATION -- `playNumber < trig_pn` is not a global chronological threshold, so the leaky `plays_before` can include plays from *later* calendar-time drives that happen to have low `playNumber`, creating spurious red-zone detection on completed drives. No monotonicity guarantee between the two filters; both directions are reported (off-by-1 / 2 / 3+ per sign). Plus the match fraction.
- **D11** -- Red-zone conversion diagnostic: at trigger time, classify each trigger by (`fav_red_zone_trips`, `fav_red_zone_tds`) state. Report:
    - Fraction with zero trips (conversion rate undefined).
    - Fraction with `trips > 0 AND tds == 0` (zero conversion).
    - Fraction with `trips > 0 AND tds == trips` (perfect conversion).
    - Fraction with `0 < tds < trips` (partial conversion).
    - Median `tds / trips` among non-zero-trips triggers.
- **D12** -- Cumulative validated-set context: summary cell prints running validated-set count and notable conditional identities accumulating across 02a + 02b + 02c + 02d + 02e. Pre-execution count is 21 features; post-02e count varies with verdicts.

## Plan-time pre-execution redundancy audit

Per the established 02d / 02b / 02c / 02a pattern, every candidate
feature pair was audited at plan time. Four audit dimensions:

### Candidate-vs-candidate (within 02e)

| Pair | Verdict |
|---|---|
| `fav_red_zone_trips` ⊇ `fav_red_zone_tds` | Every TD requires a trip. Trips >= TDs. **Conditional inclusion, not identity** -- (trips, tds) basis carries same info as (tds, tds/trips) in linear-model space. No `redundant_with` tag. |
| `fav_red_zone_tds` vs `fav_yards_per_point` | TDs reduce yards/point ratio (denominator grows by 6-8). Correlated, not redundant -- yards/point also responds to FGs, total drive yards, turnovers. |
| `fav_red_zone_trips` vs `fav_yards_per_point` | More trips ~ longer drives ~ more yards numerator. Correlated via yards term. |

Plan-time verdict: **zero structural duplicates among 02e's 3 candidates.**
Three pairs are conditionally identifiable but not byte-identical.

### Candidate-vs-validated-set (against 21-feature accumulated set after 02d)

Expected meaningful correlations (0.3 <= |rho| < 0.6 band, per 02d precedent):

| New | Validated | Expected mechanism |
|---|---|---|
| `fav_red_zone_trips` | `plays_so_far` | Longer games -> more drives -> more red-zone opportunities. 02d's `fav_turnovers_so_far` <-> `plays_so_far` rho=+0.501 sets the prior. |
| `fav_red_zone_tds` | `plays_so_far` | Same game-length effect, attenuated by conversion rate. |
| `fav_red_zone_tds` | `dog_points_from_explosives` | Score-correlation across teams (clock-eaten by fav scoring reduces dog opportunities). Sign: negative. |
| `fav_yards_per_point` | `fav_def_epa_per_play` | Mostly orthogonal (one is fav offense, other is fav defense). Expect |rho| < 0.3. |

No expected `|rho| >= 0.6` pre-execution.

### Candidate-vs-trigger-fields (deducibility audit)

Every 02e candidate must NOT be a deterministic function of
`trigger_events.csv` columns alone. Verified:

| Candidate | Trigger-field deducibility | Verdict |
|---|---|---|
| `fav_red_zone_trips` | Not deducible from triggers (requires drive-level play iteration). | Standalone signal. |
| `fav_red_zone_tds` | `tds * 6` is a lower bound on `fav_score_at_trigger`'s offensive component, but trigger row has total fav score (incl. defensive/return points). Not identity. | Standalone signal. |
| `fav_yards_per_point` | Requires `fav_offensive_points` separately from `fav_score_at_trigger`. Not derivable from triggers alone. | Standalone signal. |

### Candidate-vs-extractor-structure (02b refinement: read the source, not the sketch)

Per the 02b lookahead-leak post-mortem, every candidate is classified
by what the extractor TOUCHES at run time:

| Candidate | Touches `plays_before`? | Touches `drives_for_game`? | Category | Leak-sensitivity |
|---|---|---|---|---|
| `fav_red_zone_trips` | **Yes** (iterates plays in completed drives for `yardsToGoal <= 20` detection) | Yes (filter `driveNumber < trig_dn` + `offense == fav`) | **B: drive-metadata + play-iteration** | **NOT structurally safe under leaky filter.** Leaky `playNumber < trig_pn` truncates plays within drives, undercounting red-zone-entry plays at high `playNumber`. |
| `fav_red_zone_tds` | **Yes** (same as trips) | Yes | **B** | Same. The drive's `driveResult` is leak-independent, but the red-zone-entry check is leak-sensitive. |
| `fav_yards_per_point` | No (drive-level `yards`, `startOffenseScore`, `endOffenseScore` only) | Yes | **A: drive-metadata only** | Structurally safe. |

This is the **first Category B feature set in the 02-sequence.** The
D10 diff-vs-leaky pass will report disagreement counts and magnitude
distributions for the two Cat B features (not byte-identical
assertions). For the Cat A `fav_yards_per_point`, byte-identical IS
asserted.

Plan-time prediction (Cursor's prior, 3/3 PASS):

- `fav_red_zone_trips`: PASS (3/3 or 2/3). Volume metric with trigger-conditioning selecting for fav-offense underperformance.
- `fav_red_zone_tds`: PASS (3/3). Strongest mechanism -- direct red-zone finishing measure.
- `fav_yards_per_point`: PASS marginal (2/3). Inverse-efficiency with imputation overhead; could fail in 2022 fold (the noisiest fold across prior runs).

Reviewer hypothesis-watch (2/3 PASS):

- `fav_red_zone_trips` will pass with magnitudes similar to 02d's `fav_turnovers_so_far` (+0.005 to +0.010 range) and correlate `rho`=0.4-0.5 with `plays_so_far`. Real signal, partially redundant.
- `fav_red_zone_tds` will be the strongest 02e finding (>+0.010 Brier on best fold). Cleanest fit to the trigger-conditioning hypothesis.
- `fav_yards_per_point` will be the weakest pass or the one that fails. Imputation-dependent features have been weak across the project.

Per the corrections_log.md soft prior, the cross-notebook correlation
diagnostic is run on **any** divergence from the Cursor-prior of 3/3
(symmetric application -- 0/3 or 1/3 PASS also warrants the
diagnostic).

## Deliverables produced by this notebook

1. `research/results/feature_validation.csv` -- adds 9 rows from 02e
   (3 features x 3 test seasons), tagged `feature_set_version =
   v1_red_zone_failure`. 02a's 18 + 02b's 30 + 02c's 24 + 02d's 12
   = 84 prior rows preserved by the defensive-append.
2. `research/results/feature_validation.schema.md` -- splices a
   sentinel-delimited "02e -- Red-zone failure" section into the
   existing sidecar. 02a / 02b / 02c / 02d sections are preserved
   verbatim.
3. `research/notebooks/02e_red_zone_failure.ipynb` -- this notebook.

No changes to `trigger_events.csv`, `trigger_outcomes.csv`. No new
cache files. No fresh CFBD calls.

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
add("code", "c02e0001", '''
"""
Notebook 02e -- imports, environment, path constants, fail-fast checks.
Same structure as Notebook 02a / 02b / 02c / 02d. Run this cell first.
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
    f"is missing entries (none expected in 02e); load it anyway so the same "
    f"cfbd_get() helper works."
)

load_dotenv(ENV_PATH)
assert os.environ.get("CFBD_API_KEY"), (
    "CFBD_API_KEY is not set. 02e should NOT issue fresh calls, but the "
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
# Cell 2 — HTTP helpers (code, same code path as 00/01/02a/02b/02c/02d)
# ---------------------------------------------------------------------------
add("code", "c02e0002", '''
"""
HTTP helpers -- same code as Notebook 00/01/02a/02b/02c/02d, same cache directory.
02e expects ALL calls to be cache hits; the assertion in Phase 02e-b fails
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
print(f"[ok] sharing cache with Notebook 00/01/02a/02b/02c/02d at {CACHE_DIR}")
''')


# ---------------------------------------------------------------------------
# Cell 3 — Configuration (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02e0003", """
## Configuration

`WALK_FORWARD_WINDOWS` and `BASELINE_PREGAME_FEATURES` are **carried verbatim from 02a** -- locked at 02a plan-approval and binding for 02b-g. Not re-decided here.

`RED_ZONE_THRESHOLD = 20` (decision **D1**, standard NFL/CFB convention `yardsToGoal <= 20`). User-approved at plan time as **locked, not parameterized**. If 02e results suggest the cutoff needs tuning (e.g., to 15 or 25), the decision is logged to `research/tech_debt.md` rather than re-run in 02e.

`FAV_TD_DRIVE_RESULTS = {'TD', 'END OF GAME TD'}` (decision **D4**). A drive ending in a fav TD has one of these `driveResult` values. Excludes `INT TD`, `FUMBLE TD`, `FUMBLE RETURN TD`, `PUNT TD` (those are defensive/special-teams TDs by the OPPOSING team, which would credit the dog, not the fav). Includes `END OF GAME TD` (fav offensive TD that happens to end the game).

`CANDIDATE_FEATURES` (3 total):

- **V5 DDL block 5 (3 features):** `fav_red_zone_trips`, `fav_red_zone_tds`, `fav_yards_per_point`.

**Mixed-category extractor structure:**

- `fav_red_zone_trips` and `fav_red_zone_tds` are **Category B** (drive-metadata filter PLUS play iteration to detect red-zone entry). The diff-vs-leaky verification in Phase 02e-e reports **bidirectional** magnitude distributions (chrono > leaky from truncation within drives; leaky > chrono from cross-drive forward contamination because `playNumber` resets each drive).
- `fav_yards_per_point` is **Category A** (drive-metadata only). Byte-identical IS asserted for this feature.

`REDUNDANT_WITH` -- plan-time redundancy audit found **zero structural duplicates** among 02e candidates. **Three** pairwise conditions are plausible but non-identity correlations (see sidecar **Redundancy discoveries**). Execution applies **additional** **`redundant_with`** tags when **Pearson |rho|** vs the consolidated validated set hits **≥ 0.6** (02d precedent) — **`fav_red_zone_trips`** and **`fav_red_zone_tds`** → **`plays_so_far`**; **`fav_yards_per_point`** stays untagged (**every |rho|** vs pre-02e PASS stays **below 0.6**).

`FEATURE_SET_VERSION = "v1_red_zone_failure"` -- the per-notebook tag stamped into every row this notebook writes.

`YARDS_PER_POINT_FEATURE = "fav_yards_per_point"` and `YARDS_PER_POINT_INDICATOR = "fav_yards_per_point_is_null"` -- the D8/D10 paired-indicator pattern. Identical structure to 02c's `seconds_since_last_dog_explosive_play` + `_is_null` pair.
""")


# ---------------------------------------------------------------------------
# Cell 4 — Configuration constants (code)
# ---------------------------------------------------------------------------
add("code", "c02e0004", '''
SEASONS: list[int] = list(range(2015, 2025))
SEASON_TYPES: list[str] = ["regular", "postseason"]

FEATURE_SET_VERSION: str = "v1_red_zone_failure"

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

# --- 02e-specific constants -------------------------------------------------

# Decision D1: red-zone threshold (yardsToGoal <= 20). Standard NFL/CFB
# convention; user-approved at plan time as locked, not parameterized.
RED_ZONE_THRESHOLD: int = 20

# Decision D4: driveResult values that represent a fav OFFENSIVE TD.
# Excludes defensive/special-teams TDs by the opposing team.
FAV_TD_DRIVE_RESULTS: frozenset[str] = frozenset({"TD", "END OF GAME TD"})

# Candidate features (V5 DDL block 5, BUILD_SPEC.md lines 195-198).
CANDIDATE_FEATURES: list[str] = [
    "fav_red_zone_trips",
    "fav_red_zone_tds",
    "fav_yards_per_point",
]

# Post-execution redundancy tags (*|rho| ≥ 0.6* versus validated set via
# _diag_02e_correlations.py). Empty for fav_yards_per_point.
REDUNDANT_WITH: dict[str, str] = {
    "fav_red_zone_trips": "plays_so_far",
    "fav_red_zone_tds": "plays_so_far",
}

# Per-extractor category classification (plan-time audit).
EXTRACTOR_CATEGORY: dict[str, str] = {
    "fav_red_zone_trips": "B",   # drive-metadata + play-iteration
    "fav_red_zone_tds": "B",     # drive-metadata + play-iteration
    "fav_yards_per_point": "A",  # drive-metadata only
}

# Decision D8/D10 paired-indicator imputation (Mode B from 02c). The
# yards-per-point feature is imputed per-train-window median; the
# paired indicator flags rows where imputation fired.
YARDS_PER_POINT_FEATURE: str = "fav_yards_per_point"
YARDS_PER_POINT_INDICATOR: str = "fav_yards_per_point_is_null"

# Reproducibility seed -- same as 02a / 02b / 02c / 02d.
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
    print(f"  - {f}  (category: {EXTRACTOR_CATEGORY[f]})")
print(f"red-zone threshold (D1, locked): yardsToGoal <= {RED_ZONE_THRESHOLD}")
print(f"fav TD driveResults (D4): {sorted(FAV_TD_DRIVE_RESULTS)}")
print(f"redundant_with map ({len(REDUNDANT_WITH)} entries): {REDUNDANT_WITH}")
print(f"yards-per-point paired-indicator imputation (D8 Mode B):")
print(f"  continuous: {YARDS_PER_POINT_FEATURE}")
print(f"  indicator:  {YARDS_PER_POINT_INDICATOR}")
print(f"random state: {RANDOM_STATE}")
''')


# ---------------------------------------------------------------------------
# Cell 5 — Load triggers (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02e0005", """
## Phase 02e-a — Load trigger artifacts

Identical join setup to 02a / 02b / 02c / 02d: read `trigger_events.csv` and `trigger_outcomes.csv`, inner-join on `(game_id, fav_deficit)`, drop rows with `final_fav_won is NaN`. The label only enters as the model target in the walk-forward eval cell; it does NOT enter any feature extractor.

Print the drive-1 trigger count for symmetry with 02b / 02c / 02d reporting (drive-1 triggers have zero completed drives, so all three 02e candidates are 0 or NULL on those rows).
""")


# ---------------------------------------------------------------------------
# Cell 6 — Load triggers code
# ---------------------------------------------------------------------------
add("code", "c02e0006", '''
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
print(f"In-scope rows for 02e: {len(trigger_full_df):,}")

# Drive-1 trigger count (all 02e candidates 0 or NULL on these rows).
n_drive1 = int((trigger_full_df["drive_number_in_game"] == 1).sum())
n_drive2plus = int((trigger_full_df["drive_number_in_game"] >= 2).sum())
print(f"\\nDrive-1 scale:")
print(f"  drive_number_in_game == 1 (zero completed drives -> "
      f"fav_red_zone_trips + fav_red_zone_tds == 0, "
      f"fav_yards_per_point NULL bucket-(a)): "
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
add("markdown", "m02e0007", """
## Phase 02e-b — Re-load cached `/plays` AND `/drives` (zero fresh calls)

Identical setup to 02b / 02c / 02d: iterate the (season, season_type, week) tuples for `/plays` and (season, season_type) tuples for `/drives`. **Assert every call is a cache hit.** The cell fails loud on any cache miss -- 02e's budget is 0 fresh CFBD calls.

`/plays` is loaded for two reasons:

1. The two Category B extractors (`fav_red_zone_trips`, `fav_red_zone_tds`) iterate plays within completed drives to detect red-zone entry (`yardsToGoal <= 20`).
2. The diff-vs-leaky verification in Phase 02e-e runs the feature matrix twice (once with `_chrono_key < trig_chrono_key`, once with the leaky `playNumber < trig.playNumber` filter) and reports the magnitude distribution of the disagreement.
""")


# ---------------------------------------------------------------------------
# Cell 8 — Cache re-load code (mirrors 02d structure, includes chrono_key)
# ---------------------------------------------------------------------------
add("code", "c02e0008", '''
work_tuples_df = (
    trigger_full_df[["season", "season_type", "week"]]
    .drop_duplicates()
    .sort_values(["season", "season_type", "week"])
    .reset_index(drop=True)
)
print(f"distinct (season, season_type, week) tuples to load from cache: {len(work_tuples_df)}")

n_log_before = sum(1 for _ in CALL_LOG.open("r", encoding="utf-8")) - 1  # minus header

plays_by_game: dict[int, list[dict]] = {}
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
    f"02e budget invariant violated: {n_fresh_this_cell} non-cached CFBD call(s) "
    f"issued in this cell. 02e is supposed to spend 0 fresh CFBD calls; the "
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
# Cell 9 — assert_no_lookahead + extractors (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02e0009", """
## Phase 02e-c — `assert_no_lookahead` (R3 hard gate) + feature extractors

`assert_no_lookahead` is the per-row R3 gate -- same definition as 02a / 02b / 02c / 02d (composite chrono_key gate).

Three feature functions:

- **`fav_red_zone_trips`** (Category B) -- count of fav-offense completed drives where some play satisfied `p.driveNumber == D.driveNumber AND p.offense == D.offense AND p.yardsToGoal <= RED_ZONE_THRESHOLD`. The `p.offense == D.offense` guard (D3, plan-approval D5) protects against cross-offense plays misattributed to the drive.
- **`fav_red_zone_tds`** (Category B) -- subset of the above where `D.driveResult ∈ FAV_TD_DRIVE_RESULTS`. A drive must reach the red zone AND end in fav TD.
- **`fav_yards_per_point`** (Category A) -- `sum(drive.yards) / sum(max(0, endOffenseScore - startOffenseScore))` over completed fav-offense drives. NULL when either bucket-(a) no completed fav drives OR bucket-(b) denominator == 0.

Drive-level computation: each extractor filters `drives_for_game` to `driveNumber < trigger_drive_in_game` (i.e., COMPLETED pre-trigger drives). The two Category B extractors then iterate the plays within each completed drive (filtered to that drive's plays via `_chrono_key`-sorted `plays_before`) for red-zone-entry detection.
""")


# ---------------------------------------------------------------------------
# Cell 10 — assert_no_lookahead code (verbatim from 02d)
# ---------------------------------------------------------------------------
add("code", "c02e000a", '''
def assert_no_lookahead(plays_used: list[dict],
                        trigger_chrono_key: tuple[int, int, int, int],
                        feature_name: str, game_id: int) -> None:
    """Per-row R3 hard gate. Raises if any play in `plays_used` has
    `_chrono_key(p) >= trigger_chrono_key`.

    Same composite-chrono_key gate as 02a / 02b / 02c / 02d. See
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
add("code", "c02e000b", '''
# --- Helpers ----------------------------------------------------------------

def _completed_drives_before_trigger(
    drives_for_game: list[dict], trig_drive_in_game: int,
) -> list[dict]:
    """Filter drives to those with driveNumber < trigger's drive_number_in_game.

    A drive whose driveNumber is < the trigger drive's number is COMPLETE by
    definition. Drives without a driveNumber field are dropped.
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


def _drive_reached_red_zone(
    drive: dict, plays_before: list[dict], threshold: int,
) -> bool:
    """True iff some play `p` in `plays_before` satisfies all of:
      - p.driveNumber == drive.driveNumber
      - p.offense == drive.offense (D5 guard against cross-offense
        attribution; defensive returns, kickoffs)
      - p.yardsToGoal <= threshold

    Iterates plays_before for the drive's plays. Pre-trigger only --
    plays_before is already chrono_key-filtered upstream.
    """
    dn = drive.get("driveNumber")
    if dn is None:
        return False
    dn_int = int(dn)
    drive_offense = drive.get("offense")
    if drive_offense is None:
        return False
    for p in plays_before:
        if p.get("driveNumber") != dn_int:
            continue
        if p.get("offense") != drive_offense:
            continue
        ytg = p.get("yardsToGoal")
        if ytg is None:
            continue
        try:
            if int(ytg) <= threshold:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _drive_points_for_offense(drive: dict) -> int:
    """Points scored BY this drive's offense team on this drive.

    Computed as max(0, endOffenseScore - startOffenseScore). Non-negative
    clamp guards against any anomalous CFBD encoding.
    """
    try:
        end = int(drive.get("endOffenseScore") or 0)
        start = int(drive.get("startOffenseScore") or 0)
        return max(0, end - start)
    except (TypeError, ValueError):
        return 0


# --- Feature extractors -----------------------------------------------------

def feat_fav_red_zone_trips(
    plays_before: list[dict], drives_for_game: list[dict],
    trig_drive_in_game: int, fav: str,
) -> int:
    """D3: count completed fav-offense drives that reached the red zone.

    Category B: drive-metadata filter + play iteration. Iterates plays_before
    for each completed fav-offense drive to detect red-zone entry via the
    `_drive_reached_red_zone` helper (D5 cross-offense guard).

    Always integer; 0 when no completed fav drives or none reached the
    red zone.
    """
    n = 0
    for dr in _completed_drives_before_trigger(drives_for_game, trig_drive_in_game):
        if dr.get("offense") != fav:
            continue
        if _drive_reached_red_zone(dr, plays_before, RED_ZONE_THRESHOLD):
            n += 1
    return n


def feat_fav_red_zone_tds(
    plays_before: list[dict], drives_for_game: list[dict],
    trig_drive_in_game: int, fav: str,
) -> int:
    """D3/D4: count completed fav-offense drives that reached the red zone
    AND ended in a fav TD.

    Category B: same iteration pattern as fav_red_zone_trips, but with the
    additional `driveResult ∈ FAV_TD_DRIVE_RESULTS` filter.

    Excludes INT TD / FUMBLE TD / FUMBLE RETURN TD / PUNT TD (defensive
    or special-teams TDs by the opposing team). Includes END OF GAME TD
    (fav offensive TD that happens to end the game).

    Always integer; 0 when no qualifying drives.
    """
    n = 0
    for dr in _completed_drives_before_trigger(drives_for_game, trig_drive_in_game):
        if dr.get("offense") != fav:
            continue
        if str(dr.get("driveResult", "")) not in FAV_TD_DRIVE_RESULTS:
            continue
        if _drive_reached_red_zone(dr, plays_before, RED_ZONE_THRESHOLD):
            n += 1
    return n


def feat_fav_yards_per_point(
    drives_for_game: list[dict], trig_drive_in_game: int, fav: str,
) -> float | None:
    """D5: sum(drive.yards) / sum(max(0, endOffenseScore - startOffenseScore))
    over completed fav-offense drives.

    Category A: drive-metadata only. No play iteration.

    D7 null policy (two buckets):
      - Bucket (a): NULL when no completed fav-offense drives exist.
      - Bucket (b): NULL when the denominator is 0 (fav has had drives but
        scored 0 offensive points -- a genuine "fav offense stalled" state).

    The eval loop imputes NULL to the per-train-window median (D8) and
    adds a paired `fav_yards_per_point_is_null` indicator alongside the
    continuous value.
    """
    completed_fav = [
        d for d in _completed_drives_before_trigger(drives_for_game, trig_drive_in_game)
        if d.get("offense") == fav
    ]
    if not completed_fav:
        return None  # bucket (a)
    yards_sum = 0
    points_sum = 0
    for d in completed_fav:
        y = d.get("yards")
        if y is not None:
            try:
                yards_sum += int(y)
            except (TypeError, ValueError):
                pass
        points_sum += _drive_points_for_offense(d)
    if points_sum <= 0:
        return None  # bucket (b)
    return float(yards_sum) / float(points_sum)


def _classify_yards_per_point_null_bucket(
    drives_for_game: list[dict], trig_drive_in_game: int, fav: str,
) -> str:
    """Diagnostic helper: returns 'a' (no completed fav drives), 'b'
    (drives exist but 0 fav points), or 'nonnull' (feature has a value).

    Used by the D7 two-bucket null breakdown diagnostic in Phase 02e-d.
    Not called inside feat_fav_yards_per_point (that function returns
    None for both NULL buckets; the bucket distinction is for reporting
    only, not for the deliverable feature value).
    """
    completed_fav = [
        d for d in _completed_drives_before_trigger(drives_for_game, trig_drive_in_game)
        if d.get("offense") == fav
    ]
    if not completed_fav:
        return "a"
    points_sum = 0
    for d in completed_fav:
        points_sum += _drive_points_for_offense(d)
    if points_sum <= 0:
        return "b"
    return "nonnull"


print("[ok] 3 feature extractors defined")
print(f"     fav_red_zone_trips   (Category B: drive-metadata + play-iteration)")
print(f"     fav_red_zone_tds     (Category B: drive-metadata + play-iteration)")
print(f"     fav_yards_per_point  (Category A: drive-metadata only)")
print(f"[ok] D7 two-bucket NULL classifier defined "
      f"(_classify_yards_per_point_null_bucket)")
''')


# ---------------------------------------------------------------------------
# Cell 12 — Build matrix (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02e000c", """
## Phase 02e-d — Build feature matrix (canonical: chrono_key filter)

Walk every in-scope trigger, slice plays via the **composite `_chrono_key` filter** `(period, period_seconds_elapsed, driveNumber, playNumber) < trigger_chrono_key`, gate the play subset through `assert_no_lookahead`, run all three extractors, and attach the pre-game baseline columns.

The two Category B extractors (`fav_red_zone_trips`, `fav_red_zone_tds`) iterate plays within completed fav-offense drives for red-zone-entry detection. The Category A extractor (`fav_yards_per_point`) operates on drive metadata only.

Also computes the paired indicator `fav_yards_per_point_is_null` (D8/D10 R16-safe pair) and the per-row D7 NULL bucket classification (a vs b vs nonnull). The latter is used for the diagnostic table only; the deliverable feature value treats both NULL types uniformly.

Print:
1. Null counts per candidate feature (with D7 two-bucket breakdown for `fav_yards_per_point`).
2. Drive-1 trigger count (verifies the D7 bucket-(a) floor).
3. Quick summary statistics: mean/median of each feature on the in-scope, non-null subset.

The Phase 02e-e diff-vs-leaky cell rebuilds the matrix under the leaky `playNumber < trig.playNumber` filter and reports magnitude disagreement distributions for the two Cat B features. The Cat A feature gets a byte-identical assertion.
""")


# ---------------------------------------------------------------------------
# Cell 13 — Build matrix code (canonical pass)
# ---------------------------------------------------------------------------
add("code", "c02e000d", '''
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
    """Build the per-trigger feature matrix for 02e's 3 candidates.

    `plays_before_filter` selects which filter is applied to plays_before
    BEFORE the assert_no_lookahead gate runs:
      - "chrono_key": _chrono_key(p) < trig_chrono_key (canonical, post-correction)
      - "leaky_playnumber": p.playNumber < trig.playNumber (pre-correction; leaks
        future plays AND truncates plays in completed drives because CFBD
        playNumber resets per drive). Used ONLY by the Phase 02e-e diff-vs-
        leaky verification.

    For Category B features (fav_red_zone_trips, fav_red_zone_tds), the
    filter choice MATERIALLY changes the result -- the leaky filter
    truncates plays within drives, undercounting red-zone-entry plays.
    For the Category A feature (fav_yards_per_point), the filter is
    irrelevant -- it doesn't touch plays_before at all.

    Returns (feature_matrix_df, n_skipped_unknown_game). The
    fav_yards_per_point_is_null indicator and ypp_null_bucket diagnostic
    column are also written by this function.
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

        trig_period = int(trig["quarter"])
        trig_period_elapsed = 900 - int(trig["clock_seconds_in_period_total"])
        trig_chrono_key = (trig_period, trig_period_elapsed, trig_drive_in_game, trig_pn)

        plays = plays_by_game.get(gid)
        if plays is None:
            n_skipped += 1
            continue

        # Apply the selected plays_before filter.
        if plays_before_filter == "chrono_key":
            plays_before = [p for p in plays if _chrono_key(p) < trig_chrono_key]
        else:  # "leaky_playnumber"
            plays_before = [
                p for p in plays
                if p.get("playNumber") is not None and int(p["playNumber"]) < trig_pn
            ]

        # R3 gate on the play subset (uniform across all N02 notebooks).
        # Skipped under the leaky filter -- the gate exists for the
        # chrono_key (canonical) path only. The leaky pass is for the
        # D10 comparison and intentionally violates R3.
        if plays_before_filter == "chrono_key":
            assert_no_lookahead(plays_before, trig_chrono_key, "<02e-extractors>", gid)

        drives_for_game = drives_by_game.get(gid, [])

        row: dict[str, Any] = {col: trig[col] for col in ID_COLS}
        for col in ALWAYS_PRESENT_PREGAME_COLS:
            row[col] = trig[col]
        sm_raw = trig["spread_movement"]
        sm_is_null = bool(pd.isna(sm_raw))
        row["spread_movement"] = 0.0 if sm_is_null else float(sm_raw)
        row["spread_movement_is_null"] = int(sm_is_null)
        row[LABEL_COL] = bool(trig[LABEL_COL])

        # 3 extractors. Category B features pass plays_before; Cat A doesn't.
        row["fav_red_zone_trips"] = feat_fav_red_zone_trips(
            plays_before, drives_for_game, trig_drive_in_game, fav
        )
        row["fav_red_zone_tds"] = feat_fav_red_zone_tds(
            plays_before, drives_for_game, trig_drive_in_game, fav
        )
        ypp_value = feat_fav_yards_per_point(
            drives_for_game, trig_drive_in_game, fav
        )
        row["fav_yards_per_point"] = ypp_value
        row["fav_yards_per_point_is_null"] = int(ypp_value is None)
        # D7 two-bucket classification (diagnostic only).
        row["ypp_null_bucket"] = _classify_yards_per_point_null_bucket(
            drives_for_game, trig_drive_in_game, fav
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
    if feat == "fav_yards_per_point":
        extra = " (D7 NULL: bucket (a) no completed fav drives OR bucket (b) 0 fav offensive points)"
    else:
        extra = " (always defined; 0 when no qualifying drives)"
    print(f"  {feat:<32} {n_null:>5,} null ({pct:5.2f}%){extra}")

# D7 two-bucket NULL breakdown for fav_yards_per_point (plan-approval addition 2).
print(f"\\nD7 two-bucket NULL breakdown for fav_yards_per_point:")
bucket_counts = feature_matrix_df["ypp_null_bucket"].value_counts().to_dict()
n_a = int(bucket_counts.get("a", 0))
n_b = int(bucket_counts.get("b", 0))
n_nn = int(bucket_counts.get("nonnull", 0))
n_total = len(feature_matrix_df)
print(f"  (a) no completed fav drives:       {n_a:>5,} ({n_a / n_total * 100:5.2f}% of in-scope)")
print(f"  (b) drives exist, 0 fav points:    {n_b:>5,} ({n_b / n_total * 100:5.2f}% of in-scope)")
print(f"      total NULL:                     {n_a + n_b:>5,} ({(n_a + n_b) / n_total * 100:5.2f}%)")
print(f"  nonnull (drives + fav points > 0): {n_nn:>5,} ({n_nn / n_total * 100:5.2f}%)")

assert n_a + n_b + n_nn == n_total, (
    f"D7 bucket counts inconsistent: a={n_a} + b={n_b} + nonnull={n_nn} != {n_total}"
)
assert null_counts["fav_yards_per_point"] == n_a + n_b, (
    f"D7 bucket sum {n_a + n_b} != fav_yards_per_point null count "
    f"{null_counts['fav_yards_per_point']}"
)
print(f"[ok] D7 bucket-sum sanity passed (a + b == total NULL).")

# Bucket (b) tech-debt threshold check (per plan-approval addition 2).
if n_b > 200:
    print(f"[info] bucket (b) is large ({n_b:,} > 200 triggers). If "
          f"fav_yards_per_point fails at small magnitudes, the worst-case-"
          f"imputation alternative becomes a tech-debt entry for N03 to consider.")
else:
    print(f"[info] bucket (b) is small ({n_b:,} <= 200 triggers); per-train-"
          f"window median imputation (D8) is the locked strategy.")

# D7 sanity: drive-1 triggers should all be in bucket (a) [no completed drives].
n_drive1 = int((feature_matrix_df["drive_number_in_game"] == 1).sum())
n_drive1_bucket_a = int(
    ((feature_matrix_df["drive_number_in_game"] == 1) &
     (feature_matrix_df["ypp_null_bucket"] == "a")).sum()
)
print(f"\\n  drive-1 trigger count: {n_drive1:,}")
print(f"  drive-1 triggers in bucket (a) (expected: all of them): "
      f"{n_drive1_bucket_a:,}")
assert n_drive1_bucket_a == n_drive1, (
    f"drive-1 triggers not all in bucket (a): {n_drive1_bucket_a} of {n_drive1}. "
    f"Investigate -- drive-1 triggers have no completed drives, so the only "
    f"NULL path should be bucket (a)."
)

# R16-safe sanity (carried from 02a/02b/02c/02d).
sm_null_after = int(feature_matrix_df["spread_movement"].isna().sum())
sm_indicator_sum = int(feature_matrix_df["spread_movement_is_null"].sum())
print(f"\\nR16-safe NaN handling for spread_movement (baseline):")
print(f"  spread_movement nulls AFTER impute:    {sm_null_after:,}  (expected: 0)")
print(f"  spread_movement_is_null indicator sum: {sm_indicator_sum:,} "
      f"({sm_indicator_sum / len(feature_matrix_df) * 100:.2f}% of in-scope)")
assert sm_null_after == 0, f"spread_movement still has {sm_null_after} nulls after impute"

# Yards-per-point indicator sanity.
ypp_ind_sum = int(feature_matrix_df["fav_yards_per_point_is_null"].sum())
ypp_null_n = null_counts["fav_yards_per_point"]
assert ypp_ind_sum == ypp_null_n, (
    f"fav_yards_per_point_is_null sum {ypp_ind_sum} != null count {ypp_null_n}"
)
print(f"\\nR16-safe pair for fav_yards_per_point (D8 Mode B):")
print(f"  fav_yards_per_point_is_null indicator sum: {ypp_ind_sum:,} "
      f"({ypp_ind_sum / len(feature_matrix_df) * 100:.2f}% of in-scope)")

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
          f"min={s_num.min():>7.2f}  max={s_num.max():>7.2f}")
''')


# ---------------------------------------------------------------------------
# Cell 14 — Diff-vs-leaky verification (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02e000e", """
## Phase 02e-e — Diff-vs-leaky verification (mixed Cat A + Cat B with magnitude distribution)

Rebuild the feature matrix under the **leaky `playNumber < trig.playNumber` filter** (the pre-correction filter that silently leaks future plays AND truncates plays within drives because CFBD's `playNumber` resets per drive). Two-mode comparison:

- **Category A** (`fav_yards_per_point`): **assert byte-identical**. The extractor doesn't touch plays_before, so the filter is irrelevant. Disagreement here would indicate a structural bug.
- **Category B** (`fav_red_zone_trips`, `fav_red_zone_tds`): two leak mechanisms coexist in this filter:
    1. **Truncation (typically chrono > leaky):** within completed drives, high-`playNumber` plays drop out when `trig_pn` is small -- red-zone-entry plays can be missed.
    2. **Forward contamination (typically leaky > chrono):** because `playNumber` resets each drive, the filter is not globally chronological; leaky `plays_before` can wrongly include plays from **later calendar-time drives** whose `playNumber` happens to satisfy `< trig_pn`, spuriously crediting red-zone entry on drives that canonically should not yet have those plays visible.

Quantify **`diff = chrono - leaky` bidirectionally:** match; +1 / +2 / +3+ (chrono greater); -1 / -2 / <= -3 (leaky greater).

**No `chrono >= leaky` assertion** — both signs are mechanically possible for Category B features that iterate `plays_before`; the distribution is the diagnostic.
""")


# ---------------------------------------------------------------------------
# Cell 15 — Diff-vs-leaky verification code
# ---------------------------------------------------------------------------
add("code", "c02e000f", '''
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

# Row alignment.
assert len(feature_matrix_df_leaky) == len(feature_matrix_df), (
    f"row count mismatch: canonical={len(feature_matrix_df):,} vs "
    f"leaky={len(feature_matrix_df_leaky):,}."
)
key_cols = ["game_id", "fav_deficit", "trigger_sequence"]
left_keys = feature_matrix_df[key_cols].values.tolist()
right_keys = feature_matrix_df_leaky[key_cols].values.tolist()
assert left_keys == right_keys, (
    f"row order differs between canonical and leaky matrices."
)
print(f"[ok] row order matches between canonical and leaky matrices ({len(left_keys):,} rows)")

n_total = len(feature_matrix_df)

# --- Category A: byte-identical assertion ----------------------------------
print(f"\\n--- Category A diff (byte-identical assertion) ---")
catA_features = [f for f in CANDIDATE_FEATURES if EXTRACTOR_CATEGORY[f] == "A"]
catA_mismatches: dict[str, int] = {}
for feat in catA_features:
    left = feature_matrix_df[feat]
    right = feature_matrix_df_leaky[feat]
    both_nan = left.isna() & right.isna()
    both_nonnan = (~left.isna()) & (~right.isna())
    nonnan_equal = both_nonnan & (left == right)
    equal_mask = both_nan | nonnan_equal
    n_mismatch = int((~equal_mask).sum())
    catA_mismatches[feat] = n_mismatch
    pct = (n_mismatch / n_total * 100) if n_total else 0
    flag = "" if n_mismatch == 0 else "  <-- MISMATCH (Cat A should be byte-identical)"
    print(f"  {feat:<32} {n_mismatch:>5,} / {n_total:,}  ({pct:5.2f}%){flag}")

assert all(v == 0 for v in catA_mismatches.values()), (
    f"Cat A byte-identical assertion FAILED: {catA_mismatches}. The Category A "
    f"claim is violated -- some feature DOES touch plays_before. Investigate "
    f"the extractor's data-touching path before proceeding."
)
print(f"[ok] Cat A byte-identical confirmed: {catA_features} all match.")

# --- Category B: magnitude-distribution diagnostic --------------------------
print(f"\\n--- Category B disagreement-magnitude distribution (D10) ---")
catB_features = [f for f in CANDIDATE_FEATURES if EXTRACTOR_CATEGORY[f] == "B"]
catB_diffs: dict[str, list[int]] = {}
catB_distributions: dict[str, dict[str, int]] = {}

for feat in catB_features:
    left = feature_matrix_df[feat].astype(int)
    right = feature_matrix_df_leaky[feat].astype(int)
    diff = left - right  # chrono - leaky
    catB_diffs[feat] = diff.tolist()

    n_negative = int((diff < 0).sum())
    n_positive = int((diff > 0).sum())
    n_match = int((diff == 0).sum())

    # Positive diff: chrono > leaky (truncation within drives).
    n_p1 = int((diff == 1).sum())
    n_p2 = int((diff == 2).sum())
    n_p3p = int((diff >= 3).sum())
    n_pos_total = n_p1 + n_p2 + n_p3p

    # Negative diff: leaky > chrono (forward contamination).
    n_m1 = int((diff == -1).sum())
    n_m2 = int((diff == -2).sum())
    n_m3p = int((diff <= -3).sum())
    n_neg_total = n_m1 + n_m2 + n_m3p

    n_any_disagree = n_pos_total + n_neg_total

    pct_match = (n_match / n_total * 100) if n_total else 0

    catB_distributions[feat] = {
        "match": n_match,
        "p_off1": n_p1, "p_off2": n_p2, "p_off3plus": n_p3p, "p_total": n_pos_total,
        "n_off1": n_m1, "n_off2": n_m2, "n_off3plus": n_m3p, "n_total": n_neg_total,
        "n_negative_triggers": n_negative, "n_positive_triggers": n_positive,
        "any_disagree": n_any_disagree,
    }

    print(f"\\n  {feat}:")
    print(f"    diff = chrono - leaky   (match={n_match:,}, chrono>leaky={n_positive:,}, leaky>chrono={n_negative:,})")
    print(f"    | direction / bucket              | count   | % of all triggers |")
    print(f"    |-----------------------------------|---------|------------------:|")
    print(f"    | match (diff == 0)                 | {n_match:>5,}   | {pct_match:>15.2f}% |")
    print(f"    | chrono > leaky: +1                | {n_p1:>5,}   | {(n_p1 / n_total * 100) if n_total else 0:>15.2f}% |")
    print(f"    | chrono > leaky: +2                | {n_p2:>5,}   | {(n_p2 / n_total * 100) if n_total else 0:>15.2f}% |")
    print(f"    | chrono > leaky: +3+               | {n_p3p:>5,}   | {(n_p3p / n_total * 100) if n_total else 0:>15.2f}% |")
    print(f"    | SUBTOTAL chrono > leaky           | {n_pos_total:>5,}   | {(n_pos_total / n_total * 100) if n_total else 0:>15.2f}% |")
    print(f"    | leaky > chrono: -1                | {n_m1:>5,}   | {(n_m1 / n_total * 100) if n_total else 0:>15.2f}% |")
    print(f"    | leaky > chrono: -2                | {n_m2:>5,}   | {(n_m2 / n_total * 100) if n_total else 0:>15.2f}% |")
    print(f"    | leaky > chrono: -3 or less        | {n_m3p:>5,}   | {(n_m3p / n_total * 100) if n_total else 0:>15.2f}% |")
    print(f"    | SUBTOTAL leaky > chrono           | {n_neg_total:>5,}   | {(n_neg_total / n_total * 100) if n_total else 0:>15.2f}% |")
    print(f"    | TOTAL any disagree               | {n_any_disagree:>5,}   | {(n_any_disagree / n_total * 100) if n_total else 0:>15.2f}% |")

    if n_any_disagree > 0:
        mismatch_idx_any = diff[diff != 0].index[:5].tolist()
        print(f"    First {len(mismatch_idx_any)} disagreement examples:")
        for idx in mismatch_idx_any:
            dd = int(diff.iloc[idx])
            print(f"      row {int(idx)}: game={int(feature_matrix_df.at[idx, 'game_id'])} "
                  f"drive_n={int(feature_matrix_df.at[idx, 'drive_number_in_game'])} "
                  f"play_n={int(feature_matrix_df.at[idx, 'play_number'])} "
                  f"chrono={int(left.iloc[idx])} leaky={int(right.iloc[idx])} "
                  f"diff={dd}")

print(f"\\nInterpretation (D10 bidirectional):")
print(f"  Positive diff -- chrono REDUCES the count vs leaky within the truncation")
print(f"    mechanism (02b-class playNumber truncation within drives).")
print(f"  Negative diff -- leaky INFLATES the count vs chrono via forward")
print(f"    contamination (playNumber is not chronological across drives).")
print(f"  The canonical `_chrono_key` filter is the only leakage-safe comparator.")

for _bf in catB_features:
    _nn = catB_distributions[_bf]["n_negative_triggers"]
    if _nn > 0:
        print(f"[info] {_bf}: {_nn} triggers where leaky > chrono (forward contamination; expected).")

# Discard the leaky matrix.
del feature_matrix_df_leaky
print(f"\\n[ok] leaky matrix discarded; canonical matrix retained for eval.")
''')


# ---------------------------------------------------------------------------
# Cell 16 — Red-zone conversion diagnostic (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02e000g", """
## Phase 02e-f — Red-zone conversion diagnostic (D11)

For each trigger, classify the (`fav_red_zone_trips`, `fav_red_zone_tds`) state into 4 buckets:

| Bucket | `trips` | `tds` |
|---|---|---|
| Zero trips | 0 | 0 |
| Zero conversion | > 0 | 0 |
| Partial conversion | > 0 | `0 < tds < trips` |
| Perfect conversion | > 0 | `tds == trips` |

Plus the distribution of the implicit `red_zone_pct = tds / trips` ratio among non-zero-trips triggers (median, 25th percentile, 75th percentile).

Interpretation:

- **Zero trips** -- the fav offense hasn't reached the red zone yet. Heavy in early-game triggers.
- **Zero conversion** -- the fav has been deep in opponent territory but failed to score TDs. The "stalled-red-zone" diagnostic state.
- **Partial conversion** -- mixed efficiency.
- **Perfect conversion** -- the fav is converting every red-zone opportunity. Conditional on the trigger (fav down 7+), this implies the fav has had FEW red-zone trips and is losing for other reasons (dog explosives, dog defensive scoring, etc.).

The diagnostic informs how the trigger-conditioning interacts with red-zone offense -- whether the trigger fires because the fav is stalling in the red zone (zero-conversion bucket large) or because the fav hasn't reached the red zone enough (zero-trips bucket large).
""")


# ---------------------------------------------------------------------------
# Cell 17 — Red-zone conversion diagnostic code
# ---------------------------------------------------------------------------
add("code", "c02e000h", '''
trips_vals = feature_matrix_df["fav_red_zone_trips"].astype(int)
tds_vals = feature_matrix_df["fav_red_zone_tds"].astype(int)

n_zero_trips = int((trips_vals == 0).sum())
n_with_trips = int((trips_vals > 0).sum())
n_zero_conv = int(((trips_vals > 0) & (tds_vals == 0)).sum())
n_perfect_conv = int(((trips_vals > 0) & (tds_vals == trips_vals)).sum())
n_partial_conv = int(((trips_vals > 0) & (tds_vals > 0) & (tds_vals < trips_vals)).sum())

n_total = len(feature_matrix_df)

print(f"Red-zone conversion diagnostic (D11): {n_total:,} triggers")
print(f"")
print(f"  | Bucket               |   Count |   % of total |")
print(f"  |----------------------|---------|-------------:|")
for label, n in [
    ("Zero trips",          n_zero_trips),
    ("Zero conversion",     n_zero_conv),
    ("Partial conversion",  n_partial_conv),
    ("Perfect conversion",  n_perfect_conv),
]:
    pct = (n / n_total * 100) if n_total else 0
    print(f"  | {label:<20} | {n:>7,} | {pct:>10.2f}% |")

# Distribution of red_zone_pct among non-zero-trips triggers.
nonzero_mask = trips_vals > 0
if int(nonzero_mask.sum()) > 0:
    pct_series = (tds_vals[nonzero_mask].astype(float) /
                  trips_vals[nonzero_mask].astype(float))
    pct_25 = float(pct_series.quantile(0.25))
    pct_50 = float(pct_series.quantile(0.50))
    pct_75 = float(pct_series.quantile(0.75))
    pct_mean = float(pct_series.mean())
    print(f"\\n  red_zone_pct (tds/trips) distribution among {int(nonzero_mask.sum()):,} "
          f"non-zero-trips triggers:")
    print(f"    p25:  {pct_25:.3f}")
    print(f"    p50 (median): {pct_50:.3f}")
    print(f"    p75:  {pct_75:.3f}")
    print(f"    mean: {pct_mean:.3f}")
else:
    pct_50 = float("nan")
    print(f"\\n  No non-zero-trips triggers (degenerate; cannot compute conversion %).")

# Interpretation summary.
print(f"\\nInterpretation:")
if n_zero_trips / n_total > 0.30:
    print(f"  [info] Large zero-trips bucket ({n_zero_trips / n_total * 100:.1f}%): "
          f"many triggers fire BEFORE the fav reaches the red zone. The trigger")
    print(f"  is sensitive to dog scoring (and other non-red-zone factors), not just")
    print(f"  red-zone failure.")
if n_zero_conv / n_total > 0.15:
    print(f"  [info] Large zero-conversion bucket ({n_zero_conv / n_total * 100:.1f}%): "
          f"a meaningful fraction of triggers fire after the fav has stalled in")
    print(f"  the red zone. This is the structural state fav_red_zone_tds isolates.")
''')


# ---------------------------------------------------------------------------
# Cell 18 — Walk-forward eval (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02e000i", """
## Phase 02e-g — Walk-forward per-feature evaluation

Per-feature handling:

- **`fav_red_zone_trips`, `fav_red_zone_tds`** (Mode A, decision B from 02a): always defined; masking is a no-op. Eval row carries `imputation_value = None`.
- **`fav_yards_per_point`** (Mode B, decision D8 from 02c carried forward): **R16-safe per-train-window median imputation**. Don't drop rows; compute the per-train-window median from training-only non-null values, impute NULLs to that median across train/val/test, and add the paired `fav_yards_per_point_is_null` indicator. The per-window imputation value is stored in the `imputation_value` column on `feature_validation.csv`.

Eval pipeline: `StandardScaler` -> `LogisticRegression(penalty="l1", C=1.0, solver="liblinear", random_state=42, max_iter=1000)`, then `CalibratedClassifierCV(method="isotonic", cv="prefit")` on the val set; eval Brier + ECE on the test set. Identical helper to 02a / 02b / 02c / 02d.

3 candidates x 3 windows = 9 eval rows expected.
""")


# ---------------------------------------------------------------------------
# Cell 19 — ECE + fit helper code
# ---------------------------------------------------------------------------
add("code", "c02e000j", '''
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
    Identical to 02a / 02b / 02c / 02d's helper; will be deduped into a shared
    module before N03 per research/tech_debt.md item 2."""
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
add("code", "c02e000k", '''
eval_rows: list[dict] = []
t_start = time.perf_counter()

for window in WALK_FORWARD_WINDOWS:
    train_seasons = window["train_seasons"]
    val_season = window["val_season"]
    test_season = window["test_season"]
    win_label = window["train_window_label"]

    for feat in CANDIDATE_FEATURES:
        if feat == YARDS_PER_POINT_FEATURE:
            # Mode B: R16-safe per-train-window median imputation. No row drop.
            sub = feature_matrix_df  # don't drop; impute below
            train_sub = sub[sub["season"].isin(train_seasons)].copy()
            val_sub = sub[sub["season"] == val_season].copy()
            test_sub = sub[sub["season"] == test_season].copy()
            # Train-window median (no leakage, computed only from train).
            train_non_null = train_sub[feat].dropna()
            if len(train_non_null) == 0:
                print(f"[skip] feat={feat} window={win_label}: train non-null is empty")
                continue
            median_imp = float(train_non_null.median())
            # Apply imputation across all three splits.
            train_sub[feat] = train_sub[feat].fillna(median_imp)
            val_sub[feat] = val_sub[feat].fillna(median_imp)
            test_sub[feat] = test_sub[feat].fillna(median_imp)
            # Candidate columns: baseline + continuous + paired indicator.
            cand_cols = BASELINE_PREGAME_FEATURES + [feat, YARDS_PER_POINT_INDICATOR]
            imputation_value: float | None = median_imp
        else:
            # Mode A: per-feature null drop (decision B from 02a). Always
            # a no-op for fav_red_zone_trips / fav_red_zone_tds (no nulls).
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
    extra = ""
    if feat == YARDS_PER_POINT_FEATURE:
        per_win = eval_df[eval_df["feature"] == feat][["train_window", "imputation_value"]].values.tolist()
        extra = f"  per-window medians: {per_win}"
    print(f"  {feat:<32} {verdict}  "
          f"({n_pos}/3 brier-improving, {n_pos_ece}/3 ece-improving){extra}")
''')


# ---------------------------------------------------------------------------
# Cell 21 — CSV write (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02e000l", """
## Phase 02e-h — Write `feature_validation.csv` (defensive append)

Same defensive-append pattern as 02a / 02b / 02c / 02d:

1. Read existing CSV with `keep_default_na=False` so the `redundant_with` empty-string convention round-trips.
2. Drop rows matching this run's `(feature, train_window, test_season)` keys.
3. Concatenate this run's 9 new rows. `pd.concat` unions columns.
4. Sort by `(feature_set_version, feature, train_window, test_season)`.
5. Write.

02a / 02b / 02c / 02d rows are preserved (their keys don't overlap with 02e's). Natural-key uniqueness is asserted after the write.
""")


# ---------------------------------------------------------------------------
# Cell 22 — CSV write code
# ---------------------------------------------------------------------------
add("code", "c02e000m", '''
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
add("markdown", "m02e000n", """
## Phase 02e-i — Splice `feature_validation.schema.md` sidecar

02e's sidecar section is sentinel-delimited (same pattern as 02b / 02c / 02d). Includes:

1. Candidate list with D1-D12 references.
2. **D7 two-bucket NULL breakdown** for `fav_yards_per_point` (plan-approval addition 2).
3. **D10 disagreement-magnitude distribution** for the two Cat B features (plan-approval addition 1).
4. **D11 red-zone conversion diagnostic** table.
5. **D12 cumulative validated-set context** after 02e.
6. **Conditional-identity flags:** `fav_red_zone_trips` ⊇ `fav_red_zone_tds` (inclusion); other cross-notebook flags.
7. Per-feature null counts + stability table.

02a / 02b / 02c / 02d sections preserved by sentinel splicing. Same known limitation as before: 02a's writer doesn't yet use splicing -- tracked as tech_debt item 3.
""")


# ---------------------------------------------------------------------------
# Cell 24 — Schema sidecar code
# ---------------------------------------------------------------------------
add("code", "c02e000o", '''
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
    if feat == "fav_yards_per_point":
        tag = " (D7 bucket-(a) no completed fav drives OR bucket-(b) zero fav points)"
    else:
        tag = ""
    null_rows.append(f"| `{feat}` | {n_null:,} | {pct:.2f}%{tag} |")

# D10 magnitude-distribution rows for Cat B features (bidirectional table).
catB_dist_rows = []
for feat in catB_features:
    d = catB_distributions[feat]
    nl = len(feature_matrix_df)
    catB_dist_rows.append(
        f"| `{feat}` | {d['match']:,} ({d['match'] / nl * 100:.2f}%) | "
        f"{d['p_off1']:,} | {d['p_off2']:,} | {d['p_off3plus']:,} | {d['p_total']:,} | "
        f"{d['n_off1']:,} | {d['n_off2']:,} | {d['n_off3plus']:,} | {d['n_total']:,} | "
        f"{d['any_disagree']:,} |"
    )

# Cumulative validated set after 02e.
fv_after = pd.read_csv(FEATURE_VALIDATION_CSV, keep_default_na=False)
fv_after["brier_improvement"] = fv_after["brier_improvement"].astype(float)
cumulative_validated: list[tuple[str, str, int, int]] = []
for (fsv, feat), grp in fv_after.groupby(["feature_set_version", "feature"]):
    n_pos = int((grp["brier_improvement"] > 0).sum())
    n_pos_ece = int((grp["calibration_improvement"].astype(float) > 0).sum())
    if n_pos >= 2:
        cumulative_validated.append((fsv, feat, n_pos, n_pos_ece))

cumulative_validated.sort()
cumul_rows = []
for fsv, feat, n_fold_b_pass, n_fold_e_pass in cumulative_validated:
    cumul_rows.append(f"| `{feat}` | {fsv} | {n_fold_b_pass}/3 | {n_fold_e_pass}/3 |")

# D11 red-zone conversion rows.
n_total_rz = len(feature_matrix_df)
rz_rows = [
    f"| Zero trips | {n_zero_trips:,} | {n_zero_trips / n_total_rz * 100:.2f}% |",
    f"| Zero conversion (trips > 0, tds = 0) | {n_zero_conv:,} | {n_zero_conv / n_total_rz * 100:.2f}% |",
    f"| Partial conversion (0 < tds < trips) | {n_partial_conv:,} | {n_partial_conv / n_total_rz * 100:.2f}% |",
    f"| Perfect conversion (tds = trips) | {n_perfect_conv:,} | {n_perfect_conv / n_total_rz * 100:.2f}% |",
]

# Build the 02e-owned section.
SECTION_BEGIN = "<!-- BEGIN: 02e red_zone_failure -->"
SECTION_END = "<!-- END: 02e red_zone_failure -->"

section_body = f"""
## 02e -- Red-zone failure features

**Section last writer:** `research/notebooks/02e_red_zone_failure.ipynb`
**Last writer commit:** `{commit_hash}`
**Last writer generation timestamp:** {now_text}
**Feature set version:** `{FEATURE_SET_VERSION}`
**Source DDL:** `BUILD_SPEC.md` `trigger_features` red-zone failure block (V5 lines 195-198)

### Candidate features (3)

- `fav_red_zone_trips` (D1/D3; **Category B**; always defined; 0 when no fav red-zone drives)
- `fav_red_zone_tds` (D1/D3/D4; **Category B**; always defined; 0 when no fav red-zone TDs)
- `fav_yards_per_point` (D5/D7/D8; **Category A**; NULL when bucket-(a) no completed fav drives OR bucket-(b) zero fav offensive points; per-train-window median imputation + paired `fav_yards_per_point_is_null` indicator)

### D1: red-zone threshold

Locked at `yardsToGoal <= {RED_ZONE_THRESHOLD}` (standard NFL/CFB convention; user-approved at plan time as locked, not parameterized).

### D3 + D5: red-zone trip detection

A completed fav-offense drive `D` counts as a red-zone trip iff
`∃ play p ∈ plays_before : p.driveNumber == D.driveNumber AND
p.offense == D.offense AND p.yardsToGoal <= {RED_ZONE_THRESHOLD}`.

The `p.offense == D.offense` guard (plan-approval D5) protects against
defensive returns and kickoff plays that might be mis-attributed to a
drive in CFBD's data. This is a Category B feature: it iterates plays
within completed drives, making the leak-correction (`_chrono_key`)
necessary -- the leaky `playNumber < trig.playNumber` filter truncates
plays within drives because CFBD `playNumber` resets per drive.

### D4: fav TD attribution

A drive counts as a fav TD iff `D.driveResult ∈ {sorted(FAV_TD_DRIVE_RESULTS)}`
AND `D.offense == fav`. Excludes defensive/special-teams TDs by the
opposing team (`INT TD`, `FUMBLE TD`, `FUMBLE RETURN TD`, `PUNT TD`).
Includes `END OF GAME TD` (fav offensive TD that happens to end the game).

### D7 two-bucket NULL breakdown for `fav_yards_per_point` (plan-approval addition 2)

In-scope triggers (post NaN `final_fav_won` drop): {len(feature_matrix_df):,}.

| Bucket | Description | Triggers | % of in-scope |
|---|---|---:|---:|
| (a) | No completed fav drives (drive-1 / early-game) | {n_a:,} | {n_a / n_total_rz * 100:.2f}% |
| (b) | Drives exist; 0 fav offensive points | {n_b:,} | {n_b / n_total_rz * 100:.2f}% |
| nonnull | Drives + fav points > 0 | {n_nn:,} | {n_nn / n_total_rz * 100:.2f}% |

**Interpretation:** bucket (a) is the early-game / drive-1 floor (no
data yet). Bucket (b) is the informative "fav offense stalled" state
-- drives exist but the fav has scored 0 offensive points. The
imputation strategy (D8 per-train-window median) treats both bucket
types uniformly, with the paired `fav_yards_per_point_is_null`
indicator preserving the missingness signal. If bucket (b) is large
(>200 triggers) AND `fav_yards_per_point` fails at small magnitudes,
the worst-case-imputation alternative (impute to a high "bad
efficiency" value instead of the train-window median) becomes a
tech-debt entry for N03.

Bucket (b) threshold check (>200): { "yes -- flagged for N03 tech-debt review" if n_b > 200 else "no -- imputation strategy is locked"}.

**Post-execution note:** bucket **(b)** at **{n_b:,}** (**{n_b / n_total_rz * 100:.2f}%**) is orders-of-magnitude above the plan’s **>200** discretionary review gate (**`research/tech_debt.md` item 8**). Stability **PASSED** anyway — paired **`fav_yards_per_point_is_null`** likely carries much of the useful signal versus the median-imputed scalar alone; treat indicator attribution as an **N03** ablation target.

### D8 paired-indicator imputation

`fav_yards_per_point` follows 02c's Mode B pattern:

1. NULL when D7 bucket (a) OR bucket (b) fires.
2. R16-safe per-train-window median imputed for NULL rows in train/val/test.
3. Paired `fav_yards_per_point_is_null` indicator added alongside the continuous value.
4. Per-window imputation value stored in the `imputation_value` column on `feature_validation.csv`.

### D10 disagreement-magnitude distribution (plan-approval addition 1)

The Cat A feature (`fav_yards_per_point`) was byte-identical between
canonical and leaky filters across all {len(feature_matrix_df):,}
triggers, confirming the Category A claim.

The two Cat B features had non-trivial **bidirectional** disagreement
distributions (diff = chrono - leaky). **Positive diff** (chrono > leaky)
is the playNumber **truncation** mechanism within drives. **Negative
diff** (leaky > chrono) is **forward contamination** across drives
because `playNumber` resets per drive and is not a global chronological
threshold. There is **no** `chrono >= leaky` monotonicity guarantee.

| Feature | Match | chr>lck +1 | +2 | +3+ | sub | lck>chr -1 | -2 | <=-3 | sub | any diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\\n".join(catB_dist_rows) + f"""

**Interpretation:** the disagreement quantifies how broken the leaky
version would have been if shipped -- both under-counting (truncation)
and over-counting (forward leak) are visible. Generalizes 02b's
mechanistic story for play-iteration features.

**Residual `lck > chr` magnitude-1 slice (empirical 02e run):** all
**leaky > chrono** rows are **−1** — **6** trigger rows for **`fav_red_zone_trips`**
and **3** for **`fav_red_zone_tds`**, spanning **four** distinct `game_id`s for
**trips** (`401404065`, `401411150`, `401415624`, `401644775`; two IDs each
host **two** in-scope triggers at the same drive/play) and **`401404065`**
+ **`401415624`** for **TDs**. Only **`401415624`** intersects the CFBD
negative-integer **`play.id`** encoding set (`corrections_log.md` §2). The
**749** adjacent **(c)** cases and **~0.394%** residual bound in
`corrections_log.md` §1 concern **ordering** vs alternative lex sorts — not
the same population as these D10 rows, which come from **cross-drive forward
contamination** under the leaky `playNumber` filter. Light overlap with
negative-id games (**1**/4 trip-level IDs) is consistent with **orthogonal**
mechanisms co-occurring rarely, not identity.

### D11 red-zone conversion diagnostic

For each trigger, classified by (`fav_red_zone_trips`, `fav_red_zone_tds`):

| Bucket | Triggers | % of total |
|---|---:|---:|
""" + "\\n".join(rz_rows) + f"""

Median `red_zone_pct = tds/trips` among non-zero-trips triggers: {pct_50:.3f}.

**Interpretation:** zero-trips bucket reflects triggers fired BEFORE
the fav reached the red zone (dog scoring did the work; not red-zone
failure per se). Zero-conversion bucket isolates the "fav stalled in
red zone" state -- the structural condition `fav_red_zone_tds` measures.

**Trigger-conditioning caveat:** With **zero trips** at **{n_zero_trips / n_total_rz * 100:.2f}%** of triggers, the prevalent read is **not** “RZ visits but fail to finish” but “**no RZ volume yet** — loss is explained elsewhere (dog scoring, field position, etc.)”. The three features still encode pre-trigger offensive state, but the **dominant** bucket is **absence of RZ exposure**; expect **mixture** semantics under the live trigger rule.

### Per-feature null counts (this run)

In-scope triggers (post NaN `final_fav_won` drop): {len(feature_matrix_df):,}.
Drive-1 trigger count: {n_drive1:,}.

| Feature | Null rows | % of in-scope |
|---|---:|---:|
""" + "\\n".join(null_rows) + f"""

### Per-feature x per-test-season results (this run, {FEATURE_SET_VERSION})

| Feature | Window -> Test | Brier improvement | ECE improvement | Stability |
|---|---|---:|---:|---|
""" + "\\n".join(verdict_rows) + f"""

Sign convention: positive = candidate beat baseline. `**PASS**` means
`sum(brier_improvement > 0) >= 2` across the 3 test seasons.

**Interpretive hedge (R6 verdict vs fold magnitudes):** The `**PASS**`
entries above are mechanically correct under the approved R6 floor (**≥ 2**/3 folds with strictly positive **Δ Brier**).
Do **not** read “3 × PASS” here as uniformly strong signal.
Per the methodological soft gate in **`corrections_log.md`** (**Δ Brier < +0.005** merits skepticism —
noise-fold territory): **`fav_red_zone_tds`** has **three** positive folds
yet **two** sit **below** +0.005 (**2023**, **2024**); only the **2022**
fold is clearly above bar. **`fav_red_zone_trips`** and
**`fav_yards_per_point`** — each **PASS** at **2/3** folds — exhibit a **weak**
or **negative** **2024** test fold (**trips −0.00229**, **ypp −0.00266**),
consistent with sampling noise surviving R6 rather than uniformly improving
risk. **`N03`** should treat aggregation as **credentialing under R6** only and
cross-check **2024 weakness** (see project-wide fold diagnostic
**`research/notebooks/_diag_02e_fold_pattern.py`**) plus the **correlation**
artifact **`research/results/_02e_correlations.csv`** for L1 penalties.

### D12: cumulative validated-set context after 02e

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
3. `dog_explosive_play_count` (02c) <-> `opening_drive_was_explosive_td` (02b):
   drive-1 conditional overlap.
4. `dog_points_off_turnovers` (02d) <-> `dog_points_from_returns` (02c):
   co-occurrence measured at 2.67% (D11 from 02d); independent in correlation.
5. `fav_red_zone_trips` (02e) ⊇ `fav_red_zone_tds` (02e): inclusion
   relation (every TD requires a trip). NOT structural identity.
6. **`fav_red_zone_trips`** /**`fav_red_zone_tds`** (02e): post-hoc **|ρ| ≥ 0.6**
   vs **`plays_so_far`** — CSV **`redundant_with=plays_so_far`** (**see
   Redundancy tagging** subsection below).

### Redundancy tagging (per 02d-established |ρ| ≥ 0.6 protocol)

Post-execution Pearson correlation (**`research/results/_02e_correlations.csv`**; reproducible via **`research/notebooks/_diag_02e_correlations.py`**) between each 02e candidate and the **21** pre-02e PASS feature columns (**11,416** triggers; non-null pairwise intersection). The **02d** sidecar precedent treats **|ρ| ≥ 0.6** as the bar for tagging **`redundant_with`** (vs the weaker “meaningful overlap” **0.3–0.6** band reserved for advisory L1 penalties).

Applied (**6 CSV rows** — three walk-forward folds × two features):

- **`fav_red_zone_trips`** → **`redundant_with = plays_so_far`** at **ρ = +0.781** (maximum **|ρ|** against validated columns).
- **`fav_red_zone_tds`** → **`redundant_with = plays_so_far`** at **ρ = +0.650** (**`dog_points_from_explosives`** is next at **+0.635**; **`plays_so_far`** clears **0.6** and denotes the coarsest cumulative clock underlying both metrics).

Companion highs **0.6–0.65** (**`dog_explosive_play_count`**, **`dog_points_from_explosives`**) remain in **`_02e_correlations.csv`** for **N03** weight-awareness; **`plays_so_far`** is designated as redundant partner per maximal clarity.

Within-matrix coherence (not substituted for tagging): **`fav_red_zone_trips` ↔ `fav_red_zone_tds`** **ρ = +0.772** (*n* = **11,416**).

**`fav_yards_per_point`:** every validated-column Pearson ρ satisfies **|ρ| < 0.6**. **Largest magnitude:** **`fav_def_epa_after_first_drive`**, ρ = **−0.210** (*n* = **4,892**). **Largest positive ρ:** **`fav_turnovers_so_far`**, **≈ +0.185** (*n* = **4,902**). **`redundant_with`** remains **empty** on all **`fav_yards_per_point`** rows.

### Redundancy discoveries (02e plan-time audit)

Plan-time **structural** audit: zero byte-identical duplicate among three 02e
candidates. **Execution** attaches **Pearson-derived** redundancy tags (**above**) on top of that baseline — **distinct** from purely algebraic inclusion (**trips ⊇ TDs**) or correlation **below** **0.6**.

Three **conditional** relationships (orthogonal to **`plays_so_far`** tagging):

1. `fav_red_zone_trips` ⊇ `fav_red_zone_tds`: inclusion (every TD requires
   a trip). The (trips, tds) basis carries the same info as (tds, tds/trips)
   in linear-model space. **Independent** redundancy decision versus **`plays_so_far`**.
2. `fav_red_zone_tds` vs `fav_yards_per_point`: TDs reduce yards/point
   ratio (denominator grows by 6-8). Correlated; **below the 0.6 redundancy gate**.
3. `fav_red_zone_trips` vs `fav_yards_per_point`: more trips ~ more
   yards numerator. Correlated; **below the 0.6 redundancy gate**.

### Section provenance

- Last writer: this 02e run (timestamp + commit above).
- Splicing strategy: sentinel-delimited; re-running 02e refreshes only
  this section. Re-running 02a in its current form WILL clobber 02b's,
  02c's, 02d's, and 02e's sections -- tracked as `research/tech_debt.md`
  item 3.
"""

new_section = SECTION_BEGIN + "\\n" + section_body.rstrip() + "\\n" + SECTION_END

if FEATURE_VALIDATION_SCHEMA.exists():
    existing_text = FEATURE_VALIDATION_SCHEMA.read_text(encoding="utf-8")
    if SECTION_BEGIN in existing_text and SECTION_END in existing_text:
        start = existing_text.index(SECTION_BEGIN)
        end = existing_text.index(SECTION_END) + len(SECTION_END)
        updated = existing_text[:start] + new_section + existing_text[end:]
        print(f"[ok] spliced 02e section in place (existing markers found)")
    else:
        updated = existing_text.rstrip() + "\\n\\n" + new_section + "\\n"
        print(f"[ok] appended 02e section at end of sidecar (markers added)")
else:
    header = (
        "# feature_validation.csv -- schema sidecar\\n\\n"
        "(02a + 02b + 02c + 02d sections missing -- run 02a / 02b / 02c / 02d to regenerate.)\\n\\n"
    )
    updated = header + new_section + "\\n"
    print(f"[warn] sidecar did not exist; wrote stub header + 02e section.")

FEATURE_VALIDATION_SCHEMA.write_text(updated, encoding="utf-8")
print(f"[ok] wrote feature_validation.schema.md ({len(updated):,} chars)")
print(f"     path: {FEATURE_VALIDATION_SCHEMA}")
''')


# ---------------------------------------------------------------------------
# Cell 25 — Summary (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02e000p", """
## Phase 02e-j — Summary, headline stats, hypothesis-watch result, STOP banner
""")


# ---------------------------------------------------------------------------
# Cell 26 — Summary print code
# ---------------------------------------------------------------------------
add("code", "c02e000q", '''
print("=" * 70)
print("Notebook 02e -- red-zone failure features -- summary")
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
print(f"\\nDrive-1 trigger count: {n_drive1_final:,} "
      f"({n_drive1_final / len(feature_matrix_df) * 100:.2f}%)")

print(f"\\nD7 two-bucket NULL breakdown for fav_yards_per_point:")
print(f"  (a) no completed fav drives:    {n_a:>5,} ({n_a / len(feature_matrix_df) * 100:.2f}%)")
print(f"  (b) drives exist, 0 fav points: {n_b:>5,} ({n_b / len(feature_matrix_df) * 100:.2f}%)")
print(f"  nonnull:                         {n_nn:>5,} ({n_nn / len(feature_matrix_df) * 100:.2f}%)")

print(f"\\nDiff-vs-leaky D10 distribution:")
print(f"  fav_yards_per_point (Cat A):     byte-identical "
      f"({catA_mismatches['fav_yards_per_point']:,} mismatches)")
for feat in catB_features:
    d = catB_distributions[feat]
    print(f"  {feat} (Cat B):")
    print(f"    match: {d['match']:,}  chrono>leaky(+1/+2/+3+): "
          f"{d['p_off1']:,}/{d['p_off2']:,}/{d['p_off3plus']:,}  "
          f"leaky>chrono(-1/-2/<=-3): {d['n_off1']:,}/{d['n_off2']:,}/{d['n_off3plus']:,}  "
          f"any disagree: {d['any_disagree']:,}")

print(f"\\nRed-zone conversion (D11) bucket fractions:")
print(f"  zero trips:         {n_zero_trips:>5,} ({n_zero_trips / n_total_rz * 100:.2f}%)")
print(f"  zero conversion:    {n_zero_conv:>5,} ({n_zero_conv / n_total_rz * 100:.2f}%)")
print(f"  partial conversion: {n_partial_conv:>5,} ({n_partial_conv / n_total_rz * 100:.2f}%)")
print(f"  perfect conversion: {n_perfect_conv:>5,} ({n_perfect_conv / n_total_rz * 100:.2f}%)")
if int((trips_vals > 0).sum()) > 0:
    print(f"  median red_zone_pct (tds/trips) among non-zero-trips triggers: "
          f"{pct_50:.3f}")

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
print(f"  Cursor plan-time prediction: 3/3 pass.")
print(f"    - fav_red_zone_trips: PASS likely (3/3 or 2/3)")
print(f"    - fav_red_zone_tds:   PASS likely (3/3)")
print(f"    - fav_yards_per_point: PASS marginal (2/3, could fail 2022 fold)")
print(f"  Reviewer watch: 2/3 pass, with fav_yards_per_point as the failure.")
print(f"    - fav_red_zone_trips:  PASS magnitudes similar to fav_turnovers_so_far")
print(f"    - fav_red_zone_tds:    strongest 02e finding (>+0.010 best fold)")
print(f"    - fav_yards_per_point: weakest pass OR fail (imputation-dependent)")
n_passes = sum(1 for v in verdicts.values() if v == "PASS")
print(f"\\n  Actual: {n_passes}/3 pass.")
print(f"    fav_red_zone_trips:   {verdicts['fav_red_zone_trips']}")
print(f"    fav_red_zone_tds:     {verdicts['fav_red_zone_tds']}")
print(f"    fav_yards_per_point:  {verdicts['fav_yards_per_point']}")

ypp_pass = verdicts["fav_yards_per_point"] == "PASS"
trips_pass = verdicts["fav_red_zone_trips"] == "PASS"
tds_pass = verdicts["fav_red_zone_tds"] == "PASS"
if trips_pass and tds_pass and ypp_pass:
    print(f"  Result: Cursor prior CONFIRMED (3/3); reviewer watch FALSIFIED.")
    print(f"          Soft-prior threshold (corrections_log.md): correlation diagnostic")
    print(f"          OPTIONAL since 3/3 matches Cursor's plan-time prior. Run anyway")
    print(f"          if reviewer flagged divergence from EITHER prior.")
elif trips_pass and tds_pass and not ypp_pass:
    print(f"  Result: Reviewer watch CONFIRMED (2/3, yards_per_point fails).")
    print(f"          Cursor prior off by one feature; imputation-dependent feature")
    print(f"          was the weakness.")
    print(f"          Soft-prior threshold: correlation diagnostic warranted per the")
    print(f"          symmetric-application rule (deviation from Cursor's 3/3).")
elif not trips_pass and not tds_pass and not ypp_pass:
    print(f"  Result: 0/3 PASS -- substantial deviation from both priors.")
    print(f"          Run correlation diagnostic AND investigate extractor logic")
    print(f"          (potential bug). Per corrections_log.md soft prior,")
    print(f"          0/3 or 1/3 PASS triggers the diagnostic.")
elif n_passes == 1:
    print(f"  Result: 1/3 PASS -- substantial deviation from both priors.")
    print(f"          Run correlation diagnostic per the soft-prior threshold.")
else:
    print(f"  Result: empirical pattern doesn't match either prior cleanly.")

# --- Cumulative validated set after 02e ------------------------------------
print(f"\\nCumulative validated set after 02e:")
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
print(f"  After 02e: {len(cumul)} features (notebooks 02a/b/c/d/e done).")
print(f"  Remaining: 02f (down-and-distance efficiency, 4 candidates),")
print(f"  02g (context, 4 candidates) = 8 additional candidates projected.")

print(f"\\nDeliverables (research/results/):")
for path in [FEATURE_VALIDATION_CSV, FEATURE_VALIDATION_SCHEMA]:
    size = path.stat().st_size
    print(f"  {path.name:<40} {size:>10,} bytes")
''')


# ---------------------------------------------------------------------------
# Cell 27 — Budget print + STOP banner
# ---------------------------------------------------------------------------
add("code", "c02e000r", '''
calls_log_df = pd.read_csv(CALL_LOG)
n_total_log_rows = len(calls_log_df)
n_fresh_cfbd_total = int(((calls_log_df["service"] == "cfbd")
                          & (calls_log_df["cached"] == 0)).sum())

this_run_calls = calls_log_df.iloc[n_log_before:].copy()
n_this_run = len(this_run_calls)
n_this_run_fresh = int((this_run_calls["cached"] == 0).sum())

print("=" * 64)
print("CFBD call budget -- Notebook 02e")
print("=" * 64)
print(f"\\nThis notebook run:")
print(f"  total calls this run:     {n_this_run:>5,}  ({n_plays_lookups} /plays + {n_drives_lookups} /drives)")
print(f"  fresh (uncached) this run: {n_this_run_fresh:>5,}  (budget: 0)")

assert n_this_run_fresh == 0, (
    f"02e budget invariant violated: {n_this_run_fresh} fresh CFBD call(s) "
    f"this run. 02e is supposed to spend 0 fresh CFBD calls."
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

print(f"\\n[ok] notebook 02e complete -- STOP per R22. "
      f"Do not start Notebook 02f without approval.")
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
