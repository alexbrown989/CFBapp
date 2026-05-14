"""02f correlation diagnostic: four DDL PASS features vs cumulative 24-notebook
PASS numeric columns (Pearson rho on pairwise non-null intersection).

Validated side = all columns in the shared 02e-style matrix **except** the
four 02f DDL rates themselves (24 = 28 total - 4 new).

Output: research/results/_02f_correlations.csv (untracked). Cache-only /
no fresh CFBD calls expected.

Builds on extractor bundle from `_diag_02e_correlations.py` plus DDL helpers
from `_build_02f.py`.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pathlib
import sys
import time
from collections import Counter
from typing import Any

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RESEARCH_DIR = REPO_ROOT / "research"
DATA_DIR = RESEARCH_DIR / "data"
RESULTS_DIR = RESEARCH_DIR / "results"
CACHE_DIR = DATA_DIR / "cache"
CALL_LOG = CACHE_DIR / "cfbd_call_log.csv"
ENV_PATH = REPO_ROOT / "backend" / ".env"

TRIGGER_EVENTS_CSV = RESULTS_DIR / "trigger_events.csv"
TRIGGER_OUTCOMES_CSV = RESULTS_DIR / "trigger_outcomes.csv"
OUT_CSV = RESULTS_DIR / "_02f_correlations.csv"

load_dotenv(ENV_PATH)

# -----------------------------------------------------------------------------
# HTTP helper (same as the N02 notebooks). Cache-only is expected.
# -----------------------------------------------------------------------------
CFBD_BASE = "https://apinext.collegefootballdata.com"


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


# -----------------------------------------------------------------------------
# Shared chrono key
# -----------------------------------------------------------------------------
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _lib_chrono import _chrono_key  # type: ignore  # noqa: E402

# -----------------------------------------------------------------------------
# Constants reused from 02b/02c
# -----------------------------------------------------------------------------
EXPLOSIVE_PASS_YARDS: int = 20
EXPLOSIVE_RUSH_YARDS: int = 12
EXPLOSIVE_PASS_PLAY_TYPES = frozenset({"Pass Reception", "Passing Touchdown"})
EXPLOSIVE_RUSH_PLAY_TYPES = frozenset({"Rush", "Rushing Touchdown"})

# 02c scoring registry (verbatim).
SCORING_PLAYTYPE_REGISTRY: dict[str, str] = {
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

# 02d constants.
TURNOVER_DRIVE_RESULTS = frozenset({"INT", "INT TD", "FUMBLE", "FUMBLE TD", "FUMBLE RETURN TD"})
NON_RETURN_TURNOVERS = frozenset({"INT", "FUMBLE"})
SHORT_FIELD_THRESHOLD = 40

# 02e red-zone extraction (mirror _build_02e.ipynb).
RED_ZONE_THRESHOLD = 20
FAV_TD_DRIVE_RESULTS = frozenset({"TD", "END OF GAME TD"})

# --- 02f DDL constants (mirror `_build_02f.py`) --------------------------------
TD_DRIVE_RESULTS: frozenset[str] = frozenset({"TD", "END OF GAME TD"})
OFFENSIVE_TD_PLAY_TYPES: frozenset[str] = frozenset({
    "Passing Touchdown", "Rushing Touchdown", "Fumble Recovery (Own)",
})
EXCLUDED_DN_PLAY_TYPES: frozenset[str] = frozenset({
    "Penalty", "Timeout", "End Period", "End of Half", "End of Game",
    "Two Point Conversion No Good", "Two Point Conversion Failed",
    "Two Point Pass", "Two Point Rush", "placeholder", "Uncategorized",
})


# -----------------------------------------------------------------------------
# Helpers (verbatim from each build script)
# -----------------------------------------------------------------------------

def _mean_ppa(plays: list[dict]) -> float | None:
    vals = [p["ppa"] for p in plays if p.get("ppa") is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


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


def _play_game_seconds(play: dict) -> int | None:
    period = play.get("period")
    clock = play.get("clock") or {}
    m = clock.get("minutes")
    s = clock.get("seconds")
    if period is None or m is None or s is None:
        return None
    if int(period) > 4:
        return None
    return (int(period) - 1) * 900 + (900 - int(m) * 60 - int(s))


def _find_drive1(drives_for_game: list[dict]) -> dict | None:
    for d in drives_for_game:
        if d.get("driveNumber") == 1:
            return d
    return None


# 02a extractors

def feat_fav_def_epa_per_play(plays_before, fav, dog):
    return _mean_ppa([p for p in plays_before if p.get("defense") == fav])


def feat_dog_off_epa_per_play(plays_before, fav, dog):
    """Dog offense EPA / play average (paired with fav_def EPA in 02a)."""
    return _mean_ppa([p for p in plays_before if p.get("offense") == dog])


def feat_epa_divergence(plays_before, fav, dog):
    fav_off = _mean_ppa([p for p in plays_before if p.get("offense") == fav])
    dog_off = _mean_ppa([p for p in plays_before if p.get("offense") == dog])
    if fav_off is None or dog_off is None:
        return None
    return fav_off - dog_off


def feat_plays_so_far(plays_before, fav, dog):
    return len(plays_before)


# 02b extractors

def feat_dog_received_opening_kickoff(plays_before, drives_for_game, trig_dn, fav, dog):
    for p in plays_before:
        if p.get("driveNumber") == 1:
            return int(p.get("offense") == dog)
    if trig_dn >= 1:
        d1 = _find_drive1(drives_for_game)
        if d1 is not None:
            return int(d1.get("offense") == dog)
    return None


def feat_dog_scored_on_opening_drive(plays_before, drives_for_game, trig_dn, fav, dog):
    if trig_dn <= 1:
        return None
    d1 = _find_drive1(drives_for_game)
    if d1 is None:
        return None
    return int(bool(d1.get("scoring")) and d1.get("offense") == dog)


def feat_opening_drive_was_td(plays_before, drives_for_game, trig_dn, fav, dog):
    if trig_dn <= 1:
        return None
    d1 = _find_drive1(drives_for_game)
    if d1 is None:
        return None
    return int(d1.get("driveResult") == "TD")


def feat_opening_drive_was_explosive_td(plays_before, drives_for_game, trig_dn, fav, dog):
    if trig_dn <= 1:
        return None
    d1 = _find_drive1(drives_for_game)
    if d1 is None:
        return None
    if d1.get("driveResult") != "TD":
        return 0
    d1_plays = [p for p in plays_before if p.get("driveNumber") == 1]
    return int(any(_is_explosive(p) for p in d1_plays))


def feat_fav_def_epa_first_drive(plays_before, drives_for_game, trig_dn, fav, dog):
    sub = [p for p in plays_before if p.get("driveNumber") == 1 and p.get("defense") == fav]
    return _mean_ppa(sub)


def feat_fav_def_epa_after_first_drive(plays_before, drives_for_game, trig_dn, fav, dog):
    sub = [p for p in plays_before
           if p.get("driveNumber") is not None
           and p.get("driveNumber") > 1
           and p.get("defense") == fav]
    return _mean_ppa(sub)


def feat_defense_stabilized_flag(plays_before, drives_for_game, trig_dn, fav, dog):
    a = feat_fav_def_epa_first_drive(plays_before, drives_for_game, trig_dn, fav, dog)
    b = feat_fav_def_epa_after_first_drive(plays_before, drives_for_game, trig_dn, fav, dog)
    if a is None or b is None:
        return None
    return int(b < a)


# 02c extractors

def _completed_dog_drives(drives_for_game, trig_dn, dog):
    return [d for d in drives_for_game
            if d.get("driveNumber") is not None
            and int(d["driveNumber"]) < trig_dn
            and d.get("offense") == dog]


def feat_dog_avg_drive_yards(plays_before, drives_for_game, trig_dn, fav, dog):
    drives = _completed_dog_drives(drives_for_game, trig_dn, dog)
    vals = [int(d["yards"]) for d in drives if d.get("yards") is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def feat_dog_avg_drive_plays(plays_before, drives_for_game, trig_dn, fav, dog):
    drives = _completed_dog_drives(drives_for_game, trig_dn, dog)
    vals = [int(d["plays"]) for d in drives if d.get("plays") is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def feat_seconds_since_last_dog_explosive_play(plays_before, drives_for_game, trig_dn, fav, dog, trigger_secs):
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


def compute_points_buckets(plays_before, fav, dog):
    """02c's single-pass drive-attribution helper. Returns
    (points_explosives, points_sustained, points_returns, dog_explosive_play_count)."""
    drive_had_dog_explosive: dict[int, bool] = {}
    for p in plays_before:
        dn = p.get("driveNumber")
        if dn is None:
            continue
        if p.get("offense") == dog and _is_explosive(p):
            drive_had_dog_explosive[int(dn)] = True
    dog_explosive_play_count = sum(
        1 for p in plays_before
        if p.get("offense") == dog and _is_explosive(p)
    )
    points_explosives = 0
    points_sustained = 0
    points_returns = 0
    last_dog_td_bucket: str | None = None
    for p in plays_before:
        if not p.get("scoring"):
            continue
        pt = p.get("playType", "")
        cat = SCORING_PLAYTYPE_REGISTRY.get(pt)
        if cat is None or cat == "exclude":
            continue
        play_offense = p.get("offense")
        play_dn = p.get("driveNumber")
        if cat == "offensive_td":
            if play_offense != dog:
                continue
            had_explosive = bool(drive_had_dog_explosive.get(int(play_dn), False)) if play_dn is not None else False
            if had_explosive:
                points_explosives += 6
                last_dog_td_bucket = "explosives"
            else:
                points_sustained += 6
                last_dog_td_bucket = "sustained"
        elif cat == "fg":
            if play_offense != dog:
                continue
            had_explosive = bool(drive_had_dog_explosive.get(int(play_dn), False)) if play_dn is not None else False
            if had_explosive:
                points_explosives += 3
            else:
                points_sustained += 3
        elif cat == "return_td":
            scoring_is_dog = (play_offense != dog)
            if scoring_is_dog:
                points_returns += 6
                last_dog_td_bucket = "returns"
        elif cat == "pat_1pt":
            if play_offense == dog and last_dog_td_bucket is not None:
                if last_dog_td_bucket == "explosives":
                    points_explosives += 1
                elif last_dog_td_bucket == "sustained":
                    points_sustained += 1
                else:
                    points_returns += 1
        elif cat == "pat_2pt":
            if play_offense == dog and last_dog_td_bucket is not None:
                if last_dog_td_bucket == "explosives":
                    points_explosives += 2
                elif last_dog_td_bucket == "sustained":
                    points_sustained += 2
                else:
                    points_returns += 2
        elif cat == "safety_def":
            scoring_is_dog = (play_offense != dog)
            if scoring_is_dog:
                points_returns += 2
        elif cat == "pat_def_ret":
            scoring_is_dog = (play_offense != dog)
            if scoring_is_dog:
                points_returns += 2
    return points_explosives, points_sustained, points_returns, dog_explosive_play_count


# 02d extractors

def _completed_drives_before_trigger(drives_for_game, trig_dn):
    return [d for d in drives_for_game
            if d.get("driveNumber") is not None
            and int(d["driveNumber"]) < trig_dn]


def _trigger_drive(drives_for_game, trig_dn):
    for d in drives_for_game:
        if d.get("driveNumber") is not None and int(d["driveNumber"]) == trig_dn:
            return d
    return None


def _drive_points_for_offense(drive):
    try:
        return max(0, int(drive.get("endOffenseScore") or 0) - int(drive.get("startOffenseScore") or 0))
    except (TypeError, ValueError):
        return 0


def feat_fav_turnovers_so_far(drives_for_game, trig_dn, fav):
    n = 0
    for dr in _completed_drives_before_trigger(drives_for_game, trig_dn):
        if dr.get("offense") != fav:
            continue
        if str(dr.get("driveResult", "")) in TURNOVER_DRIVE_RESULTS:
            n += 1
    return n


def feat_dog_points_off_turnovers(drives_for_game, trig_dn, fav, dog, dog_score_at_trigger):
    completed = _completed_drives_before_trigger(drives_for_game, trig_dn)
    if not any(d.get("offense") == dog for d in completed):
        return None
    total = 0
    for i, dr in enumerate(completed):
        if dr.get("offense") != fav:
            continue
        if str(dr.get("driveResult", "")) not in NON_RETURN_TURNOVERS:
            continue
        next_idx = i + 1
        if next_idx < len(completed):
            next_dr = completed[next_idx]
            if next_dr.get("offense") == dog:
                total += _drive_points_for_offense(next_dr)
        else:
            trig_dr = _trigger_drive(drives_for_game, trig_dn)
            if trig_dr is not None and trig_dr.get("offense") == dog:
                try:
                    drive_start_score = int(trig_dr.get("startOffenseScore") or 0)
                    in_progress_pts = max(0, int(dog_score_at_trigger) - drive_start_score)
                    total += in_progress_pts
                except (TypeError, ValueError):
                    pass
    return int(total)


def feat_dog_avg_starting_field_pos(drives_for_game, trig_dn, dog):
    completed_dog = [
        d for d in _completed_drives_before_trigger(drives_for_game, trig_dn)
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


def feat_short_field_tds_allowed(drives_for_game, trig_dn, dog):
    n = 0
    for dr in _completed_drives_before_trigger(drives_for_game, trig_dn):
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


# --- 02e extractors ------------------------------------------------------------

def _drive_reached_red_zone(
    drive: dict, plays_before: list[dict], threshold: int,
) -> bool:
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


def feat_fav_red_zone_trips(plays_before, drives_for_game, trig_dn, fav):
    n = 0
    for dr in _completed_drives_before_trigger(drives_for_game, trig_dn):
        if dr.get("offense") != fav:
            continue
        if _drive_reached_red_zone(dr, plays_before, RED_ZONE_THRESHOLD):
            n += 1
    return int(n)


def feat_fav_red_zone_tds(plays_before, drives_for_game, trig_dn, fav):
    n = 0
    for dr in _completed_drives_before_trigger(drives_for_game, trig_dn):
        if dr.get("offense") != fav:
            continue
        if str(dr.get("driveResult", "")) not in FAV_TD_DRIVE_RESULTS:
            continue
        if _drive_reached_red_zone(dr, plays_before, RED_ZONE_THRESHOLD):
            n += 1
    return int(n)


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


def feat_fav_yards_per_point(drives_for_game, trig_dn, fav) -> float | None:
    completed_fav = [
        d for d in _completed_drives_before_trigger(drives_for_game, trig_dn)
        if d.get("offense") == fav
    ]
    if not completed_fav:
        return None
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
        return None
    return float(yards_sum) / float(points_sum)


# -----------------------------------------------------------------------------
# Load triggers + cache
# -----------------------------------------------------------------------------

print(f"Loading triggers...")
triggers_df = pd.read_csv(TRIGGER_EVENTS_CSV)
outcomes_df = pd.read_csv(TRIGGER_OUTCOMES_CSV)
trigger_full_df = triggers_df.merge(outcomes_df, on=["game_id", "fav_deficit"], how="inner", validate="one_to_one")
trigger_full_df = trigger_full_df[trigger_full_df["final_fav_won"].notna()].copy()
print(f"  in-scope triggers: {len(trigger_full_df):,}")

_game_meta = trigger_full_df.drop_duplicates(subset=["game_id"])[["game_id", "fav_team", "dog_team"]]
game_fav = {int(r["game_id"]): str(r["fav_team"]) for _, r in _game_meta.iterrows()}
game_dog = {int(r["game_id"]): str(r["dog_team"]) for _, r in _game_meta.iterrows()}

print(f"Loading /plays + /drives from cache...")
work_tuples_df = (
    trigger_full_df[["season", "season_type", "week"]]
    .drop_duplicates()
    .sort_values(["season", "season_type", "week"])
    .reset_index(drop=True)
)
plays_by_game: dict[int, list[dict]] = {}
t0 = time.perf_counter()
for _, row in work_tuples_df.iterrows():
    plays = cfbd_get("/plays", year=int(row["season"]),
                     seasonType=str(row["season_type"]),
                     week=int(row["week"]),
                     classification="fbs")
    for p in plays:
        gid = p.get("gameId")
        if gid is None:
            continue
        plays_by_game.setdefault(gid, []).append(p)
print(f"  /plays loaded in {time.perf_counter() - t0:.1f}s; "
      f"{len(plays_by_game):,} games, {sum(len(v) for v in plays_by_game.values()):,} plays")

drives_by_game: dict[int, list[dict]] = {}
t0 = time.perf_counter()
season_type_tuples = (
    trigger_full_df[["season", "season_type"]]
    .drop_duplicates()
    .sort_values(["season", "season_type"])
    .reset_index(drop=True)
)
for _, row in season_type_tuples.iterrows():
    drives = cfbd_get("/drives", year=int(row["season"]),
                      seasonType=str(row["season_type"]),
                      classification="fbs")
    for d in drives:
        gid = d.get("gameId")
        if gid is None:
            continue
        drives_by_game.setdefault(gid, []).append(d)
print(f"  /drives loaded in {time.perf_counter() - t0:.1f}s; "
      f"{len(drives_by_game):,} games, {sum(len(v) for v in drives_by_game.values()):,} drives")

# Sort plays by chrono_key and drives by driveNumber.
for gid in plays_by_game:
    plays_by_game[gid].sort(key=_chrono_key)
for gid in drives_by_game:
    drives_by_game[gid].sort(key=lambda d: (d.get("driveNumber") if d.get("driveNumber") is not None else 10**9))


# -----------------------------------------------------------------------------
# Build per-trigger feature matrix: 24 cumulative PASS extracts (includes 02e
# red-zone trio) plus four 02f DDL rates → 28 numeric columns total.
# Correlations below: NEW (02f DDL) vs VALIDATED (the other 24).
# -----------------------------------------------------------------------------

records: list[dict] = []
t0 = time.perf_counter()
for _, trig in trigger_full_df.iterrows():
    gid = int(trig["game_id"])
    trig_pn = int(trig["play_number"])
    fav = str(trig["fav_team"])
    dog = str(trig["dog_team"])
    trig_dn = int(trig["drive_number_in_game"])
    dog_score_at_trigger = int(trig["dog_score_at_trigger"])
    trigger_secs = 3600 - int(trig["seconds_remaining_in_regulation"])
    trig_period = int(trig["quarter"])
    trig_period_elapsed = 900 - int(trig["clock_seconds_in_period_total"])
    trig_chrono_key = (trig_period, trig_period_elapsed, trig_dn, trig_pn)

    plays = plays_by_game.get(gid, [])
    plays_before = [p for p in plays if _chrono_key(p) < trig_chrono_key]
    drives_for_game = drives_by_game.get(gid, [])

    # 02a
    fav_def_epa = feat_fav_def_epa_per_play(plays_before, fav, dog)
    dog_off_epa = feat_dog_off_epa_per_play(plays_before, fav, dog)
    epa_div = feat_epa_divergence(plays_before, fav, dog)
    plays_so_far = feat_plays_so_far(plays_before, fav, dog)

    # 02b
    dog_recv_ok = feat_dog_received_opening_kickoff(plays_before, drives_for_game, trig_dn, fav, dog)
    dog_scored_od = feat_dog_scored_on_opening_drive(plays_before, drives_for_game, trig_dn, fav, dog)
    od_was_td = feat_opening_drive_was_td(plays_before, drives_for_game, trig_dn, fav, dog)
    od_was_etd = feat_opening_drive_was_explosive_td(plays_before, drives_for_game, trig_dn, fav, dog)
    fdepa_after = feat_fav_def_epa_after_first_drive(plays_before, drives_for_game, trig_dn, fav, dog)
    def_stab = feat_defense_stabilized_flag(plays_before, drives_for_game, trig_dn, fav, dog)

    # 02c
    dadp = feat_dog_avg_drive_plays(plays_before, drives_for_game, trig_dn, fav, dog)
    dady = feat_dog_avg_drive_yards(plays_before, drives_for_game, trig_dn, fav, dog)
    pts_exp, pts_sus, pts_ret, expl_count = compute_points_buckets(plays_before, fav, dog)
    has_completed_dog_drive = any(
        p.get("offense") == dog
        and p.get("driveNumber") is not None
        and int(p["driveNumber"]) < trig_dn
        for p in plays_before
    )
    if not has_completed_dog_drive:
        dpf_exp = dpf_sus = dpf_ret = None
    else:
        dpf_exp = int(pts_exp)
        dpf_sus = int(pts_sus)
        dpf_ret = int(pts_ret)
    secs_last_expl = feat_seconds_since_last_dog_explosive_play(
        plays_before, drives_for_game, trig_dn, fav, dog, trigger_secs
    )

    # 02d
    fts = feat_fav_turnovers_so_far(drives_for_game, trig_dn, fav)
    dpot = feat_dog_points_off_turnovers(drives_for_game, trig_dn, fav, dog, dog_score_at_trigger)
    dafp = feat_dog_avg_starting_field_pos(drives_for_game, trig_dn, dog)
    sfta = feat_short_field_tds_allowed(drives_for_game, trig_dn, dog)

    # 02e red-zone cluster
    rz_trips = feat_fav_red_zone_trips(plays_before, drives_for_game, trig_dn, fav)
    rz_tds = feat_fav_red_zone_tds(plays_before, drives_for_game, trig_dn, fav)
    ypp = feat_fav_yards_per_point(drives_for_game, trig_dn, fav)

    fe, ft, _, _ = _accumulate_owning_dn_rates(
        plays_before, drives_for_game, trig_dn, fav,
    )
    de, dt, _, _ = _accumulate_owning_dn_rates(
        plays_before, drives_for_game, trig_dn, dog,
    )

    records.append({
        # 02a (+ dog offense EPA / play for redundancy visibility)
        "fav_def_epa_per_play": fav_def_epa,
        "dog_off_epa_per_play": dog_off_epa,
        "epa_divergence": epa_div,
        "plays_so_far": plays_so_far,
        # 02b
        "defense_stabilized_flag": def_stab,
        "dog_received_opening_kickoff": dog_recv_ok,
        "dog_scored_on_opening_drive": dog_scored_od,
        "fav_def_epa_after_first_drive": fdepa_after,
        "opening_drive_was_explosive_td": od_was_etd,
        "opening_drive_was_td": od_was_td,
        # 02c
        "dog_avg_drive_plays": dadp,
        "dog_avg_drive_yards": dady,
        "dog_explosive_play_count": int(expl_count),
        "dog_points_from_explosives": dpf_exp,
        "dog_points_from_returns": dpf_ret,
        "dog_points_from_sustained": dpf_sus,
        "seconds_since_last_dog_explosive_play": secs_last_expl,
        # 02d
        "fav_turnovers_so_far": int(fts),
        "dog_points_off_turnovers": dpot,
        "dog_avg_starting_field_pos": dafp,
        "short_field_tds_allowed": int(sfta),
        # 02e
        "fav_red_zone_trips": rz_trips,
        "fav_red_zone_tds": rz_tds,
        "fav_yards_per_point": ypp,
        # 02f DDL (four PASS rates — same extractor as `_build_02f.build_feature_matrix`).
        "fav_early_down_success_rate": np.nan if fe is None else float(fe),
        "fav_third_down_success_rate": np.nan if ft is None else float(ft),
        "dog_early_down_success_rate": np.nan if de is None else float(de),
        "dog_third_down_success_rate": np.nan if dt is None else float(dt),
    })

df = pd.DataFrame.from_records(records)
print(f"\nBuilt 28-column feature matrix in {time.perf_counter() - t0:.1f}s: {df.shape}")

# -----------------------------------------------------------------------------
# Pearson correlations: four 02f DDL features vs 24 PASS columns (cumulative − 02f).
# -----------------------------------------------------------------------------

NEW_FEATS = [
    "fav_early_down_success_rate",
    "fav_third_down_success_rate",
    "dog_early_down_success_rate",
    "dog_third_down_success_rate",
]

VALIDATED_EX_02F = [
    "defense_stabilized_flag",
    "dog_avg_drive_plays",
    "dog_avg_drive_yards",
    "dog_avg_starting_field_pos",
    "dog_explosive_play_count",
    "dog_off_epa_per_play",
    "dog_points_from_explosives",
    "dog_points_from_returns",
    "dog_points_from_sustained",
    "dog_points_off_turnovers",
    "dog_received_opening_kickoff",
    "dog_scored_on_opening_drive",
    "epa_divergence",
    "fav_def_epa_after_first_drive",
    "fav_def_epa_per_play",
    "fav_red_zone_tds",
    "fav_red_zone_trips",
    "fav_turnovers_so_far",
    "fav_yards_per_point",
    "opening_drive_was_explosive_td",
    "opening_drive_was_td",
    "plays_so_far",
    "seconds_since_last_dog_explosive_play",
    "short_field_tds_allowed",
]
assert len(VALIDATED_EX_02F) == 24
assert sorted(VALIDATED_EX_02F) == sorted(
    c for c in df.columns if c not in set(NEW_FEATS)
)

# Structural 02f pairs (candidate-vs-candidate; not in vs-validated long-form CSV).
rho_fav_early_third = float(
    df["fav_early_down_success_rate"].astype(float).corr(
        df["fav_third_down_success_rate"].astype(float)
    ),
)
rho_dog_early_third = float(
    df["dog_early_down_success_rate"].astype(float).corr(
        df["dog_third_down_success_rate"].astype(float)
    ),
)
print(
    f"\nStructural correlation (full {len(df):,} rows): "
    f"fav_early_vs_fav_third rho={rho_fav_early_third:+.3f}  "
    f"dog_early_vs_dog_third rho={rho_dog_early_third:+.3f}"
)

rows: list[dict] = []
for new_f in NEW_FEATS:
    for val_f in VALIDATED_EX_02F:
        s_new = df[new_f].astype(float)
        s_val = df[val_f].astype(float)
        mask = s_new.notna() & s_val.notna()
        n_pair = int(mask.sum())
        if n_pair < 2:
            rho = float("nan")
        else:
            rho = float(s_new[mask].corr(s_val[mask]))
        rows.append({"new_feature": new_f, "validated_feature": val_f,
                     "n_pair_nonnull": n_pair, "pearson_rho": rho})

corr_df = pd.DataFrame(rows)
corr_df["abs_rho"] = corr_df["pearson_rho"].abs()
corr_df = corr_df.sort_values(["new_feature", "abs_rho"], ascending=[True, False]).reset_index(drop=True)

OUT_CSV.write_text(
    corr_df[["new_feature", "validated_feature", "n_pair_nonnull", "pearson_rho", "abs_rho"]]
    .to_csv(index=False),
    encoding="utf-8",
)
print(f"\n[ok] wrote {OUT_CSV}  ({OUT_CSV.stat().st_size:,} bytes)")


# -----------------------------------------------------------------------------
# Wide stdout table (rho per pair).
# -----------------------------------------------------------------------------
pivot = corr_df.pivot(index="new_feature", columns="validated_feature", values="pearson_rho")
pivot = pivot[VALIDATED_EX_02F]  # column order
pivot = pivot.loc[NEW_FEATS]     # row order
print("\nPearson correlations: 4 (02f DDL) x 24 (cumulative PASS w/o the four 02f rates)\n")
print(pivot.to_string(float_format=lambda x: f"{x:+.3f}"))

# Top correlations by |rho|.
print("\nTop |rho| pairs (descending):")
for _, r in corr_df.sort_values("abs_rho", ascending=False).head(25).iterrows():
    rho = r["pearson_rho"]
    if pd.isna(rho):
        continue
    flag = ""
    if abs(rho) >= 0.6:
        flag = "  <-- HIGH (>=0.6)"
    elif abs(rho) >= 0.4:
        flag = "  <-- notable (>=0.4)"
    elif abs(rho) >= 0.3:
        flag = "  (meaningful, 0.3-0.4)"
    print(f"  {r['new_feature']:<32} <-> {r['validated_feature']:<42} "
          f"rho={rho:+.3f}  n={int(r['n_pair_nonnull']):>6,}{flag}")

high = corr_df[corr_df["abs_rho"] >= 0.4].sort_values("abs_rho", ascending=False)
print(f"\nAll pairs with |rho| >= 0.4 ({len(high)} rows)")
for _, r in high.iterrows():
    rho = float(r["pearson_rho"])
    print(f"  {r['new_feature']:<26} vs {r['validated_feature']:<34} rho={rho:+.3f}  n={int(r['n_pair_nonnull']):>6,}")

# User-watch pairs (EPA ladder vs DDL).
print("\nUser-watch pairs (02f vs EPA / divergence):")
for new_f, val_f in [
    ("fav_early_down_success_rate", "fav_def_epa_per_play"),
    ("fav_third_down_success_rate", "fav_def_epa_per_play"),
    ("dog_early_down_success_rate", "dog_off_epa_per_play"),
    ("dog_third_down_success_rate", "dog_off_epa_per_play"),
    ("fav_early_down_success_rate", "epa_divergence"),
    ("fav_third_down_success_rate", "plays_so_far"),
]:
    row = corr_df[(corr_df.new_feature == new_f) & (corr_df.validated_feature == val_f)]
    if len(row) == 0:
        print(f"  {new_f} <-> {val_f}  MISSING")
        continue
    rho = float(row["pearson_rho"].iloc[0])
    n = int(row["n_pair_nonnull"].iloc[0])
    print(f"  {new_f:<28} <-> {val_f:<42} rho={rho:+.3f}  n={n:>6,}")
# -----------------------------------------------------------------------------
# R24 diagnostic: negative raw min(distance, yardsToGoal) on audited DN snaps (same gate as `_build_02f`).
# -----------------------------------------------------------------------------

_DN_AUD_KEYS = frozenset(
    {"distance", "yardsToGoal", "down", "yardsGained", "offense", "driveNumber", "playType"}
)

neg_snap_rows: list[dict] = []
for gid, plist in plays_by_game.items():
    gi = int(gid)
    fav_g = game_fav.get(gi)
    dog_g = game_dog.get(gi)
    if fav_g is None or dog_g is None:
        continue
    fav_g = str(fav_g)
    dog_g = str(dog_g)
    for p in plist:
        if p.get("down") not in (1, 2, 3):
            continue
        if any(k not in p for k in _DN_AUD_KEYS):
            continue
        if p.get("yardsGained") is None:
            continue
        try:
            d_i = int(p["distance"])
            ytg_i = int(p["yardsToGoal"])
        except (TypeError, ValueError):
            continue
        if min(d_i, ytg_i) >= 0:
            continue
        off = str(p.get("offense") or "")
        if off == fav_g:
            side = "fav_offense"
        elif off == dog_g:
            side = "dog_offense"
        else:
            side = "other_offense"
        neg_snap_rows.append(
            {
                "game_id": int(gid),
                "driveNumber": int(p["driveNumber"]) if p.get("driveNumber") is not None else None,
                "period": p.get("period"),
                "down": int(p["down"]),
                "distance": d_i,
                "yardsToGoal": ytg_i,
                "playType": str(p.get("playType") or ""),
                "yardsGained": int(p["yardsGained"]),
                "offense": off,
                "fav_dog_side": side,
                "distance_negative": int(d_i < 0),
                "ytg_negative": int(ytg_i < 0),
            }
        )

print(f"\n[neg-dn-scan] snaps with negative raw min(distance,yardsToGoal): {len(neg_snap_rows):,}")
if neg_snap_rows:
    pt_ctr = Counter(r["playType"] for r in neg_snap_rows)
    side_ctr = Counter(r["fav_dog_side"] for r in neg_snap_rows)
    print("  Top playTypes:")
    for pt, ct in pt_ctr.most_common(15):
        print(f"    {ct:>6,}  {pt!r}")
    print(f"  By offense vs trigger fav/dog:")
    for s, ct in sorted(side_ctr.items()):
        pct = ct / len(neg_snap_rows) * 100
        print(f"    {s}: {ct:,} ({pct:.1f}%)")
    n_pen = sum(1 for r in neg_snap_rows if r["playType"] == "Penalty")
    n_ph = sum(1 for r in neg_snap_rows if str(r["playType"]).strip().lower() == "placeholder")
    print(f"  playType == 'Penalty': {n_pen:,} ({n_pen/len(neg_snap_rows)*100:.1f}%)")
    print(f"  placeholder playType : {n_ph:,}")
    samp = sorted(neg_snap_rows, key=lambda r: r["game_id"])[:5]
    print("\n  Sample first 5 (stable sort by game_id):")
    for r in samp:
        print(f"    {r}")

print("\n[done]")
