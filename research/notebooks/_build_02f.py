"""
Deterministic builder for research/notebooks/02f_down_distance_efficiency.ipynb.

Scaffold emits unexecuted 02f notebook. Four DDL down-distance rates — all Category B;
D10 wired from day one. D2 enumerated play-type exclusions; D7 NULL + insufficient-sample
indicator, no median on rates (eval fillna(0)+indicators).

See notebook Cell 0 for decision locks. Not the production module.
"""

from __future__ import annotations

import json
import pathlib
import sys
import textwrap

OUT = pathlib.Path(__file__).resolve().parent / "02f_down_distance_efficiency.ipynb"

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
add("markdown", "bd02f000", """
# Phase 0 — Notebook 02f: Down-and-distance efficiency (scaffold)

Four DDL columns (BUILD_SPEC 200–204). All **Category B**. **D10** diff-vs-leaky from scaffold.

**D2 exclusions (locked):** Penalty; Timeout; End Period; End of Half; End of Game;
Two Point Conversion No Good; Two Point Conversion Failed; Two Point Pass;
Two Point Rush; placeholder; Uncategorized.

**D4 FO-style:** 1st `yardsGained >= ceil(0.50 * eff)`; 2nd `>= ceil(0.70 * eff)` with
`eff = min(distance, yardsToGoal)`. Convention only — tuning via tech_debt.

**D5 third:** yards vs eff OR offensive scoring TD where `driveResult in {'TD','END OF GAME TD'}`
for owning-team drive (**not** defender return TD).

**D7:** NULL rate iff denom zero; `{feat}_insufficient_sample` indicators; evaluator **fillna(0) + indicators**;
**no median** on rate. `imputation_value` column left empty for these four.

**Intra-02f |rho|>=0.6** early-vs-third (fav / dog pairs): post-execute `redundant_with` weaker R6 (tie: lower mean dBrier).

STOP — await **execute 02f** before running this notebook.
""")

# ---------------------------------------------------------------------------
# Cell 1 — Imports, paths, env, fail-fast (code)
# ---------------------------------------------------------------------------
add("code", "c02f0001", '''
"""
Notebook 02f -- imports, environment, path constants, fail-fast checks.
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
    f"is missing entries (none expected in 02f); load it anyway so the same "
    f"cfbd_get() helper works."
)

load_dotenv(ENV_PATH)
assert os.environ.get("CFBD_API_KEY"), (
    "CFBD_API_KEY is not set. 02f should NOT issue fresh calls, but the "
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
add("code", "c02f0002", '''
"""
HTTP helpers -- same code as Notebook 00/01/02a/02b/02c/02d, same cache directory.
02f expects ALL calls to be cache hits; the assertion in Phase 02f-b fails
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
add("markdown", "m02f0003", """
## Configuration

`WALK_FORWARD_WINDOWS` and `BASELINE_PREGAME_FEATURES` are **carried verbatim from 02a** — locked at 02a plan-approval and binding through 02g. Not re-decided here.

`CANDIDATE_FEATURES` (**4**): DDL block owning-team early-down (`down in {1,2}`) success rate vs third-down success rate — separate columns for **`fav_*`** (fav possesses) and **`dog_*`** (dog possesses). Rates are **`None`/NULL when the corresponding denominator is 0** (decision **D7**) with paired **`{feat}_insufficient_sample`** indicator columns emitted in **`build_feature_matrix`**.

**Evaluator (walk-forward cell):** for each DDL feature, **`fillna(0.0)`** on the continuous rate column only; **`imputation_value` on `feature_validation.csv` is intentionally empty** (no median fill on rates).

**Extractor structure:** **all Category B** — each rate iterates **`plays_before`** under the canonical `_chrono_key` gate; Phase **02f-e** rebuilds under the leaky **`playNumber`** filter and compares **micro-quantized** rates (**D10**).

**Third-down TD rule (decision **D5**):** TD credit only when the **owning-team offense** scored via allowed offensive **`playType`** values and **`driveResult in TD_DRIVE_RESULTS`** — excludes defensive/ST return touchdowns.

**D2 exclusions (locked `playType`s):** see `EXCLUDED_DN_PLAY_TYPES` in the code cell (Penalty, timeouts, clock meta, conversions meta, placeholders).

**Intra-02f redundancy protocol:** pairwise Pearson **early vs third** on the fav pair and dog pair; **|rho| ≥ 0.6** triggers advisory **weak R6** tie-break (prefer lower mean ΔBrier) and **`redundant_with`** tagging versus the partner rate when corroborated after execution (**Phase 02f-f**, **D11**).

`REDUNDANT_WITH`: populated after **`research/results/_02f_correlations.csv`** (Pearson **`|rho| ≥ 0.6`** vs cumulative PASS columns; see schema **02f** redundancy subsection).

`FEATURE_SET_VERSION = "v1_down_distance_efficiency"`.
""")


# ---------------------------------------------------------------------------
# Cell 4 — Configuration constants (code)
# ---------------------------------------------------------------------------
add("code", "c02f0004", '''
SEASONS: list[int] = list(range(2015, 2025))
SEASON_TYPES: list[str] = ["regular", "postseason"]

FEATURE_SET_VERSION: str = "v1_down_distance_efficiency"

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

# --- 02f-specific constants --------------------------------------------------

TD_DRIVE_RESULTS: frozenset[str] = frozenset({"TD", "END OF GAME TD"})
OFFENSIVE_TD_PLAY_TYPES: frozenset[str] = frozenset({
    "Passing Touchdown", "Rushing Touchdown", "Fumble Recovery (Own)",
})

EXCLUDED_DN_PLAY_TYPES: frozenset[str] = frozenset({
    "Penalty", "Timeout", "End Period", "End of Half", "End of Game",
    "Two Point Conversion No Good", "Two Point Conversion Failed",
    "Two Point Pass", "Two Point Rush", "placeholder", "Uncategorized",
})

CANDIDATE_FEATURES: list[str] = [
    "fav_early_down_success_rate",
    "fav_third_down_success_rate",
    "dog_early_down_success_rate",
    "dog_third_down_success_rate",
]

INSUFFICIENT_SAMPLE_COLS: dict[str, str] = {
    "fav_early_down_success_rate": "fav_early_down_success_rate_insufficient_sample",
    "fav_third_down_success_rate": "fav_third_down_success_rate_insufficient_sample",
    "dog_early_down_success_rate": "dog_early_down_success_rate_insufficient_sample",
    "dog_third_down_success_rate": "dog_third_down_success_rate_insufficient_sample",
}

REDUNDANT_WITH: dict[str, str] = {
    # Cross-notebook diagnostic (2026-05-14): ρ(dog_third, dog_avg_drive_yards)=+0.647 on 22f trigger matrix — partner wins per protocol.
    "dog_third_down_success_rate": "dog_avg_drive_yards",
}

EXTRACTOR_CATEGORY: dict[str, str] = {f: "B" for f in CANDIDATE_FEATURES}

RANDOM_STATE: int = 42

print(f"seasons: {SEASONS}")
print(f"season types: {SEASON_TYPES}")
print(f"feature_set_version: {FEATURE_SET_VERSION}")
print(f"walk-forward windows:")
for w in WALK_FORWARD_WINDOWS:
    print(f"  train={w['train_window_label']}  val={w['val_season']}  test={w['test_season']}")
print(f"baseline pre-game ({len(BASELINE_PREGAME_FEATURES)}): {BASELINE_PREGAME_FEATURES}")
print(f"candidates ({len(CANDIDATE_FEATURES)}) all Category B:")
for f in CANDIDATE_FEATURES:
    print(f"  - {f}")
print(f"excluded_dn_play_types ({len(EXCLUDED_DN_PLAY_TYPES)}): {sorted(EXCLUDED_DN_PLAY_TYPES)!r}")
''')


# ---------------------------------------------------------------------------
# Cell 5 — Load triggers (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02f0005", """
## Phase 02f-a — Load trigger artifacts

Identical join setup to 02a / 02b / 02c / 02d: read `trigger_events.csv` and `trigger_outcomes.csv`, inner-join on `(game_id, fav_deficit)`, drop rows with `final_fav_won is NaN`. The label only enters as the model target in the walk-forward eval cell; it does NOT enter any feature extractor.

Print the drive-1 trigger count — on those triggers there are **no completed owning-team drives** before the snapshot, so every DDL denominator is typically **zero** (**insufficient-sample** flags fire; rates are **NULL**).
""")


# ---------------------------------------------------------------------------
# Cell 6 — Load triggers code
# ---------------------------------------------------------------------------
add("code", "c02f0006", '''
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
print(f"In-scope rows for 02f: {len(trigger_full_df):,}")

# Drive-1 trigger count — no completed drives before trig => DDL denominators zero.
n_drive1 = int((trigger_full_df["drive_number_in_game"] == 1).sum())
n_drive2plus = int((trigger_full_df["drive_number_in_game"] >= 2).sum())
print(f"\\nDrive-1 scale:")
print(f"  drive_number_in_game == 1 (no pre-trigger drives; DDL rates NULL / insufficient): "
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
add("markdown", "m02f0007", """
## Phase 02f-b — Re-load cached `/plays` AND `/drives` (zero fresh calls)

Identical setup to 02b / 02c / 02d / 02e / 02f: iterate the (season, season_type, week) tuples for `/plays` and (season, season_type) tuples for `/drives`. **Assert every call is a cache hit.** Budget: **0** fresh CFBD calls.

`/plays` supports **DDL** Category **B** iteration on completed owning-team `{1,2,3}`-down snaps (`effective_distance`, exclusions). **`/drives`** supplies `driveResult` for **D5**. Phase **02f-e** reruns **`build_feature_matrix`** under the leaky `playNumber` filter for **D10** magnitude tables.
""")


# ---------------------------------------------------------------------------
# Cell 8 — Cache re-load code (mirrors 02d structure, includes chrono_key)
# ---------------------------------------------------------------------------
add("code", "c02f0008", '''
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
    f"02f budget invariant violated: {n_fresh_this_cell} non-cached CFBD call(s) "
    f"issued in this cell. 02f is supposed to spend 0 fresh CFBD calls; the "
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

_DN_KEYS = frozenset({"distance", "yardsToGoal", "down", "yardsGained", "offense", "driveNumber", "playType"})
_n_aud = 0
_n_aud_skip_neg_eff = 0
_cap_aud = min(250_000, sum(len(px) for px in plays_by_game.values()))
for _px in plays_by_game.values():
    for _pj in _px:
        if _n_aud >= _cap_aud:
            break
        if _pj.get("down") not in (1, 2, 3):
            continue
        ms = sorted(k for k in _DN_KEYS if k not in _pj)
        assert not ms, f"Dn-audit missing {ms} game={_pj.get('gameId')}"
        assert _pj.get("yardsGained") is not None, "yardsGained required on dn snaps"
        int(_pj["distance"])
        int(_pj["yardsToGoal"])
        if min(int(_pj["distance"]), int(_pj["yardsToGoal"])) < 0:
            _n_aud_skip_neg_eff += 1
            continue
        _n_aud += 1
    if _n_aud >= _cap_aud:
        break
print(f"[ok] dn field audit {_n_aud:,} snaps effective_distance=max(0,min(distance,yardsToGoal))")
if _n_aud_skip_neg_eff:
    print(f"[warn] audit skipped {_n_aud_skip_neg_eff:,} DN snaps with negative raw min(distance,yardsToGoal)")
''')


# ---------------------------------------------------------------------------
# Cell 9 — assert_no_lookahead + extractors (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02f0009", """
## Phase 02f-c — `assert_no_lookahead` (R3 hard gate) + feature extractors

`assert_no_lookahead` is the per-row R3 gate -- same definition as 02a / 02b / 02c / 02d (composite chrono_key gate).

Accumulator **`_accumulate_owning_dn_rates`** walks **completed** owning-team drives before each trigger (`driveNumber < trig_drive`). On each **`down ∈ {1,2,3}`** snap with **`offense == owning_team`**, **`playType ∉ EXCLUDED_DN_PLAY_TYPES`**, and required **`distance`, `yardsToGoal`, `yardsGained`**:

- **Early (1–2):** success iff `yardsGained >= ceil(0.50*eff)` on 1st, `>= ceil(0.70*eff)` on 2nd (`eff=min(distance,yardsToGoal)`).
- **Third:** yards vs **`eff`** **or** (**D5**) offensive scoring-touchdown semantics tied to **`TD_DRIVE_RESULTS`**, **`playType ∈ OFFENSIVE_TD_PLAY_TYPES`**, owning drive.

Emitted rates = successes / denominators; **`None` + insufficient flag** when denominator is **0**.
""")


# ---------------------------------------------------------------------------
# Cell 10 — assert_no_lookahead code (verbatim from 02d)
# ---------------------------------------------------------------------------
add("code", "c02f000a", '''
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
add("code", "c02f000b", '''
# --- Helpers (down-distance) ------------------------------------------------

import math


def _completed_drives_before_trigger(
    drives_for_game: list[dict], trig_drive_in_game: int,
) -> list[dict]:
    out: list[dict] = []
    for d in drives_for_game:
        dn = d.get("driveNumber")
        if dn is None:
            continue
        if int(dn) >= trig_drive_in_game:
            continue
        out.append(d)
    return out


def _eligible_dn_snap(p: dict, owning_team: str) -> bool:
    if p.get("offense") != owning_team:
        return False
    if str(p.get("playType") or "") in EXCLUDED_DN_PLAY_TYPES:
        return False
    dwn = p.get("down")
    if dwn is None:
        return False
    try:
        di = int(dwn)
    except (TypeError, ValueError):
        return False
    return di in (1, 2, 3)


def _effective_distance(p: dict) -> int:
    """FO effective distance; clamp at 0 — rare CFBD rows carry negative distance/YTG."""
    return max(0, min(int(p["distance"]), int(p["yardsToGoal"])))


def _yards_gained_int(p: dict) -> int:
    return int(p["yardsGained"])


def _early_need_yards(effective: int, down: int) -> int:
    if down == 1:
        return math.ceil(effective * 0.50 - 1e-12)
    if down == 2:
        return math.ceil(effective * 0.70 - 1e-12)
    raise ValueError("_early_need_yards: down must be 1 or 2")


def _third_down_success(p: dict, drive: dict, owning_team: str) -> bool:
    eff = _effective_distance(p)
    yg = _yards_gained_int(p)
    if yg >= eff:
        return True
    if drive.get("offense") != owning_team:
        return False
    if str(drive.get("driveResult") or "") not in TD_DRIVE_RESULTS:
        return False
    if not p.get("scoring"):
        return False
    pt = str(p.get("playType") or "")
    return pt in OFFENSIVE_TD_PLAY_TYPES


def _accumulate_owning_dn_rates(
    plays_before: list[dict],
    drives_for_game: list[dict],
    trig_drive_in_game: int,
    owning_team: str,
) -> tuple[float | None, float | None, int, int]:
    """Return (early_rate, third_rate, early_den, third_den).

    Rates are ``None`` when denominator is zero (D7).
    """
    early_succ = early_den = 0
    third_succ = third_den = 0
    for dr in _completed_drives_before_trigger(drives_for_game, trig_drive_in_game):
        if dr.get("offense") != owning_team:
            continue
        dn_drive = dr.get("driveNumber")
        if dn_drive is None:
            continue
        dn_int = int(dn_drive)
        team = dr.get("offense")
        for p in plays_before:
            if p.get("driveNumber") != dn_int:
                continue
            if p.get("offense") != team:
                continue
            if not _eligible_dn_snap(p, owning_team):
                continue
            dwn = int(p["down"])
            eff = _effective_distance(p)
            yg = _yards_gained_int(p)
            if dwn in (1, 2):
                early_den += 1
                if yg >= _early_need_yards(eff, dwn):
                    early_succ += 1
            elif dwn == 3:
                third_den += 1
                if _third_down_success(p, dr, owning_team):
                    third_succ += 1
    er: float | None = None if early_den == 0 else early_succ / early_den
    tr: float | None = None if third_den == 0 else third_succ / third_den
    return er, tr, early_den, third_den


print("[ok] down-distance extractors initialized (Category B -- play iteration)")
''')


# ---------------------------------------------------------------------------
# Cell 12 — Build matrix (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02f000c", """
## Phase 02f-d — Build feature matrix (canonical `_chrono_key` filter)

All **four DDL rates + four `*_insufficient_sample` indicators** are emitted here.
Filters and `assert_no_lookahead` identical to predecessors. **Phase 02f-e** rebuilds under
the leaky filter and compares **quantized micro-rates** (see D10 diagnostic cell).

Insufficient-sample indicator = 1 when that rate's denominator is zero **before**
NULL coercion in the dataframe.
""")


# ---------------------------------------------------------------------------
# Cell 13 — Build matrix code (canonical pass)
# ---------------------------------------------------------------------------
add("code", "c02f000d", '''
ID_COLS = ["game_id", "fav_deficit", "trigger_sequence", "season", "season_type",
           "week", "fav_team", "dog_team", "play_number", "quarter",
           "drive_number_in_game", "dog_score_at_trigger",
           "seconds_remaining_in_regulation"]
LABEL_COL = "final_fav_won"

MICRO_NAN_SENT = -(2 ** 30)


def build_feature_matrix(
    triggers: pd.DataFrame,
    plays_by_game: dict[int, list[dict]],
    drives_by_game: dict[int, list[dict]],
    plays_before_filter: str,  # "chrono_key" or "leaky_playnumber"
) -> tuple[pd.DataFrame, int]:
    """Build per-trigger matrix: four DDL rates + four insufficient-sample flags.

    All four features are Category B; leaky filter can change rate numerators/denominators.
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

        if plays_before_filter == "chrono_key":
            plays_before = [p for p in plays if _chrono_key(p) < trig_chrono_key]
        else:
            plays_before = [
                p for p in plays
                if p.get("playNumber") is not None and int(p["playNumber"]) < trig_pn
            ]

        if plays_before_filter == "chrono_key":
            assert_no_lookahead(plays_before, trig_chrono_key, "<02f-dn>", gid)

        drives_for_game = drives_by_game.get(gid, [])

        row: dict[str, Any] = {col: trig[col] for col in ID_COLS}
        for col in ALWAYS_PRESENT_PREGAME_COLS:
            row[col] = trig[col]
        sm_raw = trig["spread_movement"]
        sm_is_null = bool(pd.isna(sm_raw))
        row["spread_movement"] = 0.0 if sm_is_null else float(sm_raw)
        row["spread_movement_is_null"] = int(sm_is_null)
        row[LABEL_COL] = bool(trig[LABEL_COL])

        fe, ft, fe_d, ft_d = _accumulate_owning_dn_rates(
            plays_before, drives_for_game, trig_drive_in_game, fav,
        )
        de, dt, de_d, dt_d = _accumulate_owning_dn_rates(
            plays_before, drives_for_game, trig_drive_in_game, dog,
        )

        row["fav_early_down_success_rate"] = np.nan if fe is None else float(fe)
        row["fav_third_down_success_rate"] = np.nan if ft is None else float(ft)
        row["dog_early_down_success_rate"] = np.nan if de is None else float(de)
        row["dog_third_down_success_rate"] = np.nan if dt is None else float(dt)

        row[INSUFFICIENT_SAMPLE_COLS["fav_early_down_success_rate"]] = int(fe_d == 0)
        row[INSUFFICIENT_SAMPLE_COLS["fav_third_down_success_rate"]] = int(ft_d == 0)
        row[INSUFFICIENT_SAMPLE_COLS["dog_early_down_success_rate"]] = int(de_d == 0)
        row[INSUFFICIENT_SAMPLE_COLS["dog_third_down_success_rate"]] = int(dt_d == 0)

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

print(f"\\nNull counts per DDL rate (D7 denom == 0 -> NaN):")
print(f"  total in-scope triggers: {len(feature_matrix_df):,}")
null_counts: dict[str, int] = {}
for feat in CANDIDATE_FEATURES:
    n_null = int(feature_matrix_df[feat].isna().sum())
    pct = (n_null / len(feature_matrix_df) * 100) if len(feature_matrix_df) else 0
    null_counts[feat] = n_null
    ic = INSUFFICIENT_SAMPLE_COLS[feat]
    n_ins = int(feature_matrix_df[ic].sum())
    assert n_ins == n_null, f"{feat}: indicator {n_ins} vs null {n_null}"
    print(f"  {feat:<36} {n_null:>5,} null / insum={n_ins:>5,} ({pct:5.2f}%)")

sm_null_after = int(feature_matrix_df["spread_movement"].isna().sum())
sm_indicator_sum = int(feature_matrix_df["spread_movement_is_null"].sum())
print(f"\\nR16 baseline spread_movement:")
print(f"  nulls after impute: {sm_null_after:,}  indicator sum: {sm_indicator_sum:,}")
assert sm_null_after == 0

print(f"\\nSummary statistics per candidate (non-null rows):")
for feat in CANDIDATE_FEATURES:
    ser = feature_matrix_df[feat].dropna()
    if len(ser) == 0:
        print(f"  {feat:<36} (all null)")
        continue
    s_num = ser.astype(float)
    print(f"  {feat:<36} n={len(ser):>5,}  mean={s_num.mean():>6.3f}  "
          f"median={s_num.median():>6.3f}  min={s_num.min():>5.2f}  max={s_num.max():>5.2f}")
''')


# ---------------------------------------------------------------------------
# Cell 14 — Diff-vs-leaky verification (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02f000e", """
## Phase 02f-e — Diff-vs-leaky (**D10**, all Category B)

Rebuild under the leaky **`playNumber`** filter. Every DDL rate can move because each
iterates `plays_before`.

Disagreement is quantified on **`micro_rate = round(rate * 1e6)`** with a shared **NaN sentinel**
so NULL-vs-NULL counts as **match**. Buckets mirror the prior **02b/02e**
integer micro-diff convention.

**No Cat-A byte-identical assertion** (02f is Cat-B-only).
""")


# ---------------------------------------------------------------------------
# Cell 15 — Diff-vs-leaky verification code
# ---------------------------------------------------------------------------
add("code", "c02f000f", '''
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


def _rate_to_micro_series(sr: pd.Series) -> pd.Series:
    arr = sr.astype(float)
    return pd.Series(
        np.where(np.isnan(arr), MICRO_NAN_SENT, np.rint(arr * 1_000_000.0)),
        index=sr.index,
        dtype=np.int64,
    )


# --- Category B (all four features): micro-rate diff -----------------------
print(f"\\n--- Category B disagreement-magnitude distribution (D10) ---")
catB_features = list(CANDIDATE_FEATURES)
catB_diffs: dict[str, list[int]] = {}
catB_distributions: dict[str, dict[str, int]] = {}

for feat in catB_features:
    left = _rate_to_micro_series(feature_matrix_df[feat])
    right = _rate_to_micro_series(feature_matrix_df_leaky[feat])
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

print(f"\\nInterpretation (D10 bidirectional micro-rates):")
print(f"  Positive diff -- canonical micro-rate lower than leaky micro-rate (mixed causes).")
print(f"  Negative diff -- leaky micro-rate higher than canonical.")
print(f"  Canonical `_chrono_key` filter is the leakage-safe reference path.")

for _bf in catB_features:
    _nn = catB_distributions[_bf]["n_negative_triggers"]
    if _nn > 0:
        print(f"[info] {_bf}: {_nn} triggers where leaky > chrono (forward contamination; expected).")

# Discard the leaky matrix.
del feature_matrix_df_leaky
print(f"\\n[ok] leaky matrix discarded; canonical matrix retained for eval.")
''')


# ---------------------------------------------------------------------------
# Cell 16 — Early/third correlation diagnostic (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02f000g", """
## Phase 02f-f — Intra-02f early vs third correlation (D11)

Pairwise-complete Pearson **rho** between **`fav_early_down_success_rate`** and **`fav_third_down_success_rate`**, and analogously on the **`dog_*`** pair. High **|rho|** on the fav-down-7+ trigger slice implies the early and third DDL signals co-move — the approved plan gates **weak R6** / **`redundant_with`** handling at **|rho| ≥ 0.6** for these intra-02f pairs (tie-break: lower mean ΔBrier vs partner).

Separate from **D10** leakage mechanics (**semantic** redundancy within the four DDL rates).
""")


# ---------------------------------------------------------------------------
# Cell 17 — Early/third correlation diagnostic (code)
# ---------------------------------------------------------------------------
add("code", "c02f000h", '''
_fe = feature_matrix_df["fav_early_down_success_rate"]
_ft = feature_matrix_df["fav_third_down_success_rate"]
mask_fav_pair = _fe.notna() & _ft.notna()
n_pair_fav_early_third = int(mask_fav_pair.sum())
rho_fav_early_third = (
    float(_fe[mask_fav_pair].corr(_ft[mask_fav_pair]))
    if n_pair_fav_early_third > 1 else float("nan")
)

_de = feature_matrix_df["dog_early_down_success_rate"]
_dt = feature_matrix_df["dog_third_down_success_rate"]
mask_dog_pair = _de.notna() & _dt.notna()
n_pair_dog_early_third = int(mask_dog_pair.sum())
rho_dog_early_third = (
    float(_de[mask_dog_pair].corr(_dt[mask_dog_pair]))
    if n_pair_dog_early_third > 1 else float("nan")
)

print(f"\\nIntra-notebook DDL correlation (D11) — canonical matrix, {len(feature_matrix_df):,} triggers:")
print(f"  fav_early vs fav_third:  rho={rho_fav_early_third:+.4f}  n_pairwise={n_pair_fav_early_third:,}")
print(f"  dog_early vs dog_third:  rho={rho_dog_early_third:+.4f}  n_pairwise={n_pair_dog_early_third:,}")

for lbl, rho in [
    ("fav_early_vs_fav_third", rho_fav_early_third),
    ("dog_early_vs_dog_third", rho_dog_early_third),
]:
    if not pd.isna(rho) and abs(rho) >= 0.6:
        print(f"  [advisory] |rho|>=0.6 on {lbl} — intra-02f redundancy / weak-R6 tie-break territory")
''')


# ---------------------------------------------------------------------------
# Cell 18 — Walk-forward eval (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02f000i", """
## Phase 02f-g — Walk-forward per-feature evaluation

Each DDL feature is evaluated with **`BASELINE_PREGAME_FEATURES`**, the **continuous rate** column, and the paired **`{feat}_insufficient_sample`** flag from **`INSUFFICIENT_SAMPLE_COLS`** — **four** modeling columns atop baseline.

Before fitting: **`feat.fillna(0.0)`** on train/val/test (NULL encodes denominator zero upstream). **`imputation_value` is intentionally empty** for each of the four DDL rows (**no median**).

Pipeline: `StandardScaler` → L1 logistic (`C=1.0`) → **`CalibratedClassifierCV(..., cv="prefit")`** fitted on validation; metrics on held-out **test**.

**12** eval rows: **4 candidates × 3 walk-forward windows**.
""")


# ---------------------------------------------------------------------------
# Cell 19 — ECE + fit helper code
# ---------------------------------------------------------------------------
add("code", "c02f000j", '''
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
add("code", "c02f000k", '''
eval_rows: list[dict] = []
t_start = time.perf_counter()

for window in WALK_FORWARD_WINDOWS:
    train_seasons = window["train_seasons"]
    val_season = window["val_season"]
    test_season = window["test_season"]
    win_label = window["train_window_label"]

    for feat in CANDIDATE_FEATURES:
        ind_col = INSUFFICIENT_SAMPLE_COLS[feat]

        train_sub = feature_matrix_df[
            feature_matrix_df["season"].isin(train_seasons)
        ].copy()
        val_sub = feature_matrix_df[feature_matrix_df["season"] == val_season].copy()
        test_sub = feature_matrix_df[feature_matrix_df["season"] == test_season].copy()

        train_sub[feat] = train_sub[feat].fillna(0.0).astype(float)
        val_sub[feat] = val_sub[feat].fillna(0.0).astype(float)
        test_sub[feat] = test_sub[feat].fillna(0.0).astype(float)

        cand_cols = BASELINE_PREGAME_FEATURES + [feat, ind_col]
        imputation_value_str = ""

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
            "imputation_value": imputation_value_str,
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
    print(f"  {feat:<42} {verdict}  "
          f"({n_pos}/3 brier-improving, {n_pos_ece}/3 ece-improving)")
''')


# ---------------------------------------------------------------------------
# Cell 21 — CSV write (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02f000l", """
## Phase 02f-h — Write `feature_validation.csv` (defensive append)

Same defensive-append pattern as 02a / 02b / 02c / 02d:

1. Read existing CSV with `keep_default_na=False` so the `redundant_with` empty-string convention round-trips.
2. Drop rows matching this run's `(feature, train_window, test_season)` keys.
3. Concatenate this run's **12** new rows. `pd.concat` unions columns.
4. Sort by `(feature_set_version, feature, train_window, test_season)`.
5. Write.

02a / 02b / 02c / 02d rows are preserved (their keys don't overlap with 02f's). Natural-key uniqueness is asserted after the write.
""")


# ---------------------------------------------------------------------------
# Cell 22 — CSV write code
# ---------------------------------------------------------------------------
add("code", "c02f000m", '''
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
add("markdown", "m02f000n", """
## Phase 02f-i — Splice `feature_validation.schema.md` sidecar

02f owns a sentinel-delimited block (same pattern as 02b / 02c / 02d / 02e). Includes DDL decisions **D2 / D5 / D7**, **D10** micro-diff table, **D11** pairwise early-vs-third correlations from this run, per-feature **NULL** counts, insufficient-sample totals, stability table, **D12** cumulative validated-set rollup after **`v1_down_distance_efficiency`**.

02a / … / **02e** sections stay intact upstream of splicing. Known limitation from **02a** writer (**tech_debt** item **3**) still applies.
""")


# ---------------------------------------------------------------------------
# Cell 24 — Schema sidecar code
# ---------------------------------------------------------------------------
add("code", "c02f000o", '''
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

# Per-trigger corpus size reused in tables.
_nl_tbl = len(feature_matrix_df)

# Per-feature NULL counts + insufficient-sample totals.
null_rows = []
for feat in CANDIDATE_FEATURES:
    n_null = null_counts[feat]
    pct = (n_null / _nl_tbl * 100) if _nl_tbl else 0.0
    null_rows.append(f"| `{feat}` | {n_null:,} | {pct:.2f}% |")

insuff_rows = []
for feat in CANDIDATE_FEATURES:
    ic = INSUFFICIENT_SAMPLE_COLS[feat]
    ni = int(feature_matrix_df[ic].sum())
    pct = (ni / _nl_tbl * 100) if _nl_tbl else 0.0
    insuff_rows.append(f"| `{ic}` | {ni:,} | {pct:.2f}% |")

# D10 micro-rate disagreement table (quantize diff on micro ints).
catB_dist_rows = []
for feat in catB_features:
    d = catB_distributions[feat]
    catB_dist_rows.append(
        f"| `{feat}` | {d['match']:,} ({d['match'] / _nl_tbl * 100:.2f}%) | "
        f"{d['p_off1']:,} | {d['p_off2']:,} | {d['p_off3plus']:,} | {d['p_total']:,} | "
        f"{d['n_off1']:,} | {d['n_off2']:,} | {d['n_off3plus']:,} | {d['n_total']:,} | "
        f"{d['any_disagree']:,} |"
    )

rho_f_disp = (
    f"{rho_fav_early_third:+.4f}" if not pd.isna(rho_fav_early_third) else "nan"
)
rho_d_disp = (
    f"{rho_dog_early_third:+.4f}" if not pd.isna(rho_dog_early_third) else "nan"
)

# Cumulative PASS snapshot after this CSV write (includes current 02f rows).
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

# Sentinel-delimited 02f-authored sidecar subsection.
SECTION_BEGIN = "<!-- BEGIN: 02f down_distance_efficiency -->"
SECTION_END = "<!-- END: 02f down_distance_efficiency -->"

section_body = (
    f"""
## 02f — Down-distance efficiency (`{FEATURE_SET_VERSION}`)

**Section last writer:** `research/notebooks/02f_down_distance_efficiency.ipynb`
**Last writer commit:** `{commit_hash}`
**Last writer generation timestamp:** {now_text}

### Candidate DDL rates (Category B)

- `{'`, `'.join(CANDIDATE_FEATURES)}`

Each rate scans **`plays_before`** under **`_chrono_key`**. **`NULL`** when either **early** or **third** denominator is zero for that owning-team column within completed drives before each trigger **(D7)**. Walk-forward evaluator: **`rate.fillna(0.0)`** + paired **`*_insufficient_sample`**; **`imputation_value`** remains **blank**.

### Locked extraction keys

**D2 exclusions — `playType`:** `{sorted(EXCLUDED_DN_PLAY_TYPES)!r}`

**D5 TD gate — **`driveResult`:** `{sorted(TD_DRIVE_RESULTS)!r}` **`playType`:** `{sorted(OFFENSIVE_TD_PLAY_TYPES)!r}`

### D7 NULL + insufficient-sample (this execution)

Triggers: **`{_nl_tbl:,}`** (`final_fav_won` non-null subset). Drive-1 triggers (**no prior completed drives**) — **`{n_drive1:,}`**.

**Numeric rate NaNs (denominator-free states):**

| Feature | Null rows | % |
|---|---:|---:|
"""
    + "\\n".join(null_rows)
    + """

**Insufficient-sample (=1):**

| Column | Rows | % |
|---|---:|---:|
"""
    + "\\n".join(insuff_rows)
    + """

### D10 (`playNumber` leak diagnostic — micro quantization)

Integer buckets tally **`micro_chrono - micro_leaky`** on quantized rates (**`NaN`** rows share **`MICRO_NAN_SENT`** so NULL-vs-NULL pairs count as matches).

| Feature | Match | chr>lck +1 | +2 | +3+ | sub | lck>chr -1 | -2 | <=-3 | sub | any diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    + "\\n".join(catB_dist_rows)
    + """

Positive / negative tails recover the **02b-class** truncation vs cross-drive contamination story generalized to DDL rates.

### D11 pairwise early vs third (intra-notebook)

| Comparison | Pearson rho | Pairwise n |
|---|---:|---:|
"""
    + f"| fav_early_vs_fav_third | {rho_f_disp} | {n_pair_fav_early_third:,} |\\n"
    + f"| dog_early_vs_dog_third | {rho_d_disp} | {n_pair_dog_early_third:,} |\\n"
    + f"""

Approved advisory bar (**|rho| ≥ 0.6**) earmarks **`redundant_with` / weaker R6** tie-break turf between correlated partners (**CSV tag starts empty**, fill only after corr corroboration).

### Stability table per walk-forward folds (`feature_validation.csv`)

| Feature | Train window → test season | Δ Brier test | Δ ECE test | R6 stab |
|---|---|---:|---:|---|
"""
    + "\\n".join(verdict_rows)
    + f"""

`R6 stab` echoes **`passed_stability`**: **`≥2`** folds with **Δ Brier > 0**.

### D12 Consolidated **`feature_validation`** PASS rollup (post-append)

| Feature | Feature set version | Brier-positive folds | ECE-positive folds |
|---|---|---:|---:|"""
    + "\\n".join(cumul_rows)
    + f"""

**Rows in rollup:** **`{len(cumulative_validated):,}`** (distinct `(feature_set_version, feature)` pairs surviving the **PASS** heuristic above).

Historical cross-notebook identities / redundancy anecdotes stay in **02a–02e** sidecars unless this run adds new overlaps (correlate against validated PASS columns separately per **`research/notebooks/_diag_*` harnesses**).

### Section provenance

- Notebook **02f** splice updates only the sentinel-delimited block.
- Re-running stale **02a** writers can clobber sibling sections (**`research/tech_debt.md` item 3**).
"""
)

new_section = SECTION_BEGIN + "\\n" + section_body.rstrip() + "\\n" + SECTION_END

if FEATURE_VALIDATION_SCHEMA.exists():
    existing_text = FEATURE_VALIDATION_SCHEMA.read_text(encoding="utf-8")
    if SECTION_BEGIN in existing_text and SECTION_END in existing_text:
        start = existing_text.index(SECTION_BEGIN)
        end = existing_text.index(SECTION_END) + len(SECTION_END)
        updated = existing_text[:start] + new_section + existing_text[end:]
        print(f"[ok] spliced 02f section in place (existing markers found)")
    else:
        updated = existing_text.rstrip() + "\\n\\n" + new_section + "\\n"
        print(f"[ok] appended 02f section at end of sidecar (markers added)")
else:
    header = (
        "# feature_validation.csv -- schema sidecar\\n\\n"
        "(Prior notebook sections unavailable -- rerun 02a+ writers for full prose.)\\n\\n"
    )
    updated = header + new_section + "\\n"
    print(f"[warn] schema sidecar absent; seeded minimal header + 02f sentinel block.")

FEATURE_VALIDATION_SCHEMA.write_text(updated, encoding="utf-8")
print(f"[ok] wrote feature_validation.schema.md ({len(updated):,} chars)")
print(f"     path: {FEATURE_VALIDATION_SCHEMA}")
''')


# ---------------------------------------------------------------------------
# Cell 25 — Summary (markdown)
# ---------------------------------------------------------------------------
add("markdown", "m02f000p", """
## Phase 02f-j — Summary, headline stats, hypothesis-watch result, STOP banner
""")


# ---------------------------------------------------------------------------
# Cell 26 — Summary print code
# ---------------------------------------------------------------------------
add("code", "c02f000q", '''
print("=" * 70)
print("Notebook 02f -- down-distance efficiency -- summary")
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
print(f"\\nDrive-1 triggers: {n_drive1_final:,} "
      f"({n_drive1_final / len(feature_matrix_df) * 100:.2f}% of matrix)")

print(f"\\nD7 NULL + insufficient-sample totals:")
for feat in CANDIDATE_FEATURES:
    nc = null_counts[feat]
    pct = nc / len(feature_matrix_df) * 100 if len(feature_matrix_df) else 0
    ic = INSUFFICIENT_SAMPLE_COLS[feat]
    ni = int(feature_matrix_df[ic].sum())
    print(f"  {feat:<42} NaN={nc:>6,} ({pct:5.2f}%)  {ic}={ni:>6,}")

print(f"\\nD10 diff-vs-leaky (micro buckets; all four candidates):")
for feat in catB_features:
    d = catB_distributions[feat]
    print(f"  {feat}: match={d['match']:,}  any_disagree={d['any_disagree']:,}  "
          f"leaky>chrono_triggers={d['n_negative_triggers']:,}")

print(f"\\nD11 correlations (pairwise complete rows):")
print(f"  fav_early vs fav_third: rho={rho_fav_early_third:+.4f} "
      f"n={n_pair_fav_early_third:,}")
print(f"  dog_early vs dog_third: rho={rho_dog_early_third:+.4f} "
      f"n={n_pair_dog_early_third:,}")

print(f"\\nWalk-forward grids ({FEATURE_SET_VERSION}):")
print(f"  {'feature':<42} {'window->test':<18} "
      f"{'d_brier':>10} {'d_ece':>10} {'stab':>6}")
for _, r in eval_df.sort_values(["feature", "test_season"]).iterrows():
    win = f"{r['train_window']}->{int(r['test_season'])}"
    print(f"  {r['feature']:<42} {win:<18} "
          f"{r['brier_improvement']:>+10.5f} {r['calibration_improvement']:>+10.5f} "
          f"{'PASS' if r['passed_stability'] else 'FAIL':>6}")

verdicts = {}
print(f"\\nR6 stability summary:")
for feat in CANDIDATE_FEATURES:
    n_pos = int((eval_df[eval_df["feature"] == feat]["brier_improvement"] > 0).sum())
    n_pos_cal = int((eval_df[eval_df["feature"] == feat]["calibration_improvement"] > 0).sum())
    verdict = "PASS" if stability_decision[feat] else "FAIL"
    verdicts[feat] = verdict
    print(f"  {feat:<42} {verdict:<5} "
          f"({n_pos}/3 d_brier+, {n_pos_cal}/3 d_ece+)")

pass_ct = sum(1 for v in verdicts.values() if v == "PASS")
print(f"\\nHypothesis watch: scaffold emits structure only; PASS count this run: {pass_ct}/"
      f"{len(CANDIDATE_FEATURES)}.")

print(f"\\nPlanned notebook after execute-02f: 02g (context block).")

fv = pd.read_csv(FEATURE_VALIDATION_CSV, keep_default_na=False)
fv["brier_improvement"] = fv["brier_improvement"].astype(float)
cumul_groups = []
for (fsv, feat), grp in fv.groupby(["feature_set_version", "feature"]):
    if int((grp["brier_improvement"] > 0).sum()) >= 2:
        cumul_groups.append((fsv, feat))
cumul_groups.sort()
print(f"\\nCumulative validated PASS groups in feature_validation.csv: {len(cumul_groups)}")

print(f"\\nDeliverables (research/results/):")
for path in [FEATURE_VALIDATION_CSV, FEATURE_VALIDATION_SCHEMA]:
    sz = path.stat().st_size
    print(f"  {path.name:<40} {sz:>10,} bytes")
''')


# ---------------------------------------------------------------------------
# Cell 27 — Budget print + STOP banner
# ---------------------------------------------------------------------------
add("code", "c02f000r", '''
calls_log_df = pd.read_csv(CALL_LOG)
n_total_log_rows = len(calls_log_df)
n_fresh_cfbd_total = int(((calls_log_df["service"] == "cfbd")
                          & (calls_log_df["cached"] == 0)).sum())

this_run_calls = calls_log_df.iloc[n_log_before:].copy()
n_this_run = len(this_run_calls)
n_this_run_fresh = int((this_run_calls["cached"] == 0).sum())

print("=" * 64)
print("CFBD call budget -- Notebook 02f")
print("=" * 64)
print(f"\\nThis notebook run:")
print(f"  total calls this run:     {n_this_run:>5,}  ({n_plays_lookups} /plays + {n_drives_lookups} /drives)")
print(f"  fresh (uncached) this run: {n_this_run_fresh:>5,}  (budget: 0)")

assert n_this_run_fresh == 0, (
    f"02f budget invariant violated: {n_this_run_fresh} fresh CFBD call(s) "
    f"this run. Notebook 02f is supposed to spend 0 fresh CFBD calls."
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

print(f"\\n[ok] notebook 02f scaffold complete — STOP.")
print("Do not execute this notebook until explicit execute-02f approval (R24 run-state protocol).")
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
