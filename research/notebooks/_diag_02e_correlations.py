"""02e correlation diagnostic: three red-zone PASS features versus the full
21-feature pre-02e PASS set (Pearson ρ on pairwise non-null intersection).

Unlike the 02d script (which intentionally dropped dog_off_epa_per_play when
pairing fav_def_epa_per_play), this notebook includes BOTH byte-identical
pair members wherever they appear among the 21 PASS feature names —
independent columns induce ρ ≈ −1 redundancy signal.

Output: research/results/_02e_correlations.csv (untracked). Cache-only /
no fresh CFBD calls expected.

Extractors duplicated from `_diag_02d_correlations.py` plus 02e red-zone
lifted from `_build_02e.py`.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pathlib
import sys
import time
from typing import Any

import httpx
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
OUT_CSV = RESULTS_DIR / "_02e_correlations.csv"

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
# Build per-trigger feature matrix: pre-02e PASS extracts (includes dog_off EPA)
# plus three 02e red-zone columns → 21 + 3 = 24 numeric columns used below.
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
    })

df = pd.DataFrame.from_records(records)
print(f"\nBuilt 24-column feature matrix in {time.perf_counter() - t0:.1f}s: {df.shape}")


# -----------------------------------------------------------------------------
# Pearson correlations: three 02e features vs 21 pre-02e PASS columns.
# -----------------------------------------------------------------------------

NEW_FEATS = ["fav_red_zone_trips", "fav_red_zone_tds", "fav_yards_per_point"]
VALIDATED_PRE_02E = [
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
    "fav_turnovers_so_far",
    "opening_drive_was_explosive_td",
    "opening_drive_was_td",
    "plays_so_far",
    "seconds_since_last_dog_explosive_play",
    "short_field_tds_allowed",
]
assert len(VALIDATED_PRE_02E) == 21

# Structural pair: both are among newly built columns — not in long-form CSV above.
rho_trips_tds = float(df["fav_red_zone_trips"].astype(float).corr(df["fav_red_zone_tds"].astype(float)))
print(f"\nStructural correlation (full 11,416 rows): "
      f"fav_red_zone_trips vs fav_red_zone_tds  rho={rho_trips_tds:+.3f}")

rows: list[dict] = []
for new_f in NEW_FEATS:
    for val_f in VALIDATED_PRE_02E:
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
pivot = pivot[VALIDATED_PRE_02E]  # column order
pivot = pivot.loc[NEW_FEATS]       # row order
print("\nPearson correlations: 3 (02e) x 21 (pre-02e PASS)\n")
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

# User-watch pairs.
print("\nUser-watch pairs (02e):")
print(f"  {'fav_red_zone_trips':<28} <-> {'fav_red_zone_tds':<42} rho={rho_trips_tds:+.3f}  n={len(df):>6,}")
for new_f, val_f in [
    ("fav_red_zone_trips", "plays_so_far"),
    ("fav_yards_per_point", "fav_def_epa_per_play"),
    ("fav_red_zone_trips", "fav_turnovers_so_far"),
    ("fav_red_zone_trips", "dog_explosive_play_count"),
]:
    row = corr_df[(corr_df.new_feature == new_f) & (corr_df.validated_feature == val_f)]
    if len(row) == 0:
        print(f"  {new_f} <-> {val_f}  MISSING")
        continue
    rho = float(row["pearson_rho"].iloc[0])
    n = int(row["n_pair_nonnull"].iloc[0])
    print(f"  {new_f:<28} <-> {val_f:<42} rho={rho:+.3f}  n={n:>6,}")

print("\n[done]")
