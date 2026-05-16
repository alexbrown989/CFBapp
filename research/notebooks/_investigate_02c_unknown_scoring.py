"""
Untracked investigation script for the 18 unknown scoring playTypes that
halted N02c Phase-c (the SCORING_PLAYTYPE_REGISTRY gate).

Goal (pass 1, n=10 initial)
---------------------------
For each unknown scoring playType, sample <=10 plays from the SAME cache the
notebook loaded (mirror the loader exactly). Per-play extraction:

    gameId, playId, playId_is_negative, playNumber, period, clock,
    offense, defense, offenseScore, defenseScore, playText,
    prev_non_pat_playType, prev_offense, prev_offenseScore, prev_defenseScore,
    next_non_pat_playType,
    delta_team_a (signed), delta_team_b (signed), abs_delta_total

Where (team_a, team_b) = (prev play's offense, prev play's defense), and
delta_team_a / delta_team_b are score changes for those teams from the prev
non-PAT play's reported scoreboard to the current play's reported scoreboard
(team-aligned across possession changes).

Goal (pass 2, n=50 verification for 4 high-pop UNIFORM targets)
---------------------------------------------------------------
For Uncategorized (688), Punt (314), Fumble Recovery (Opponent) (256),
and Kickoff (130), sample up to 50 additional plays each and classify each
by per-target playText template. Surface any sample that does NOT match the
dominant template ("exception class"). Halt-loud is the caller's job; this
script just lists exceptions.

Goal (pass 3, drive-attribution check for Fumble Recovery (Own))
----------------------------------------------------------------
For all 34 Fumble Recovery (Own) plays, look up the dog team for that game
from trigger_events.csv, walk the play's drive history (same driveId, lower
playNumber), and count how many had a "dog explosive play" (per 02b's D1
thresholds). Output: count of dog-offense FR(Own) plays, of those how many
had >=1 prior dog explosive in the same drive. This tells us whether the
offensive_td registry mapping is enough or whether the points-bucket cell
also needs a drive-level branch.

Outputs
-------
  research/results/_investigate_02c_unknown_scoring.csv
      One row per sampled play. New column `pass_label` distinguishes
      "n10_initial" (all 18 playTypes, pass 1) from "n50_verification"
      (4 high-pop targets, pass 2). Pass-3 drive rows go into a separate
      file (different schema).
  research/results/_investigate_02c_unknown_scoring_drive_attrib.csv
      One row per Fumble Recovery (Own) play: gameId, playId, offense,
      dog_team, is_dog_offense, drive_id, n_prior_dog_explosive_pass,
      n_prior_dog_explosive_rush, had_dog_explosive_in_drive.
  research/results/_investigate_02c_unknown_scoring.summary.json
      Sections: per-playType n=10 summary (existing), n50_verification
      (new), drive_attribution (new).
  stdout: per-playType n=10 summary, n=50 verification table per target
      with exception lists, drive-attribution summary.

Conventions
-----------
- Cache-only. No fresh CFBD calls. If a tuple is missing, halt loud.
- Deterministic sample (random.seed(42), sorted by gameId then playId).
- "PAT" for skip purposes = SCORING_PLAYTYPE_REGISTRY[pt] in {pat_1pt, pat_2pt}.
- abs_delta_total is the user-spec metric: |delta_team_a| + |delta_team_b|.

Run
---
    & "<repo>/backend/.venv/Scripts/python.exe" \
        research/notebooks/_investigate_02c_unknown_scoring.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pathlib
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import httpx
import pandas as pd
from dotenv import load_dotenv

# ------------------------------------------------------------------ Paths --
SCRIPT = pathlib.Path(__file__).resolve()
NOTEBOOK_DIR = SCRIPT.parent
RESEARCH_DIR = NOTEBOOK_DIR.parent
DATA_DIR = RESEARCH_DIR / "data"
RESULTS_DIR = RESEARCH_DIR / "results"
CACHE_DIR = DATA_DIR / "cache"
CALL_LOG = CACHE_DIR / "cfbd_call_log.csv"
ENV_PATH = (RESEARCH_DIR / ".." / "backend" / ".env").resolve()

assert CACHE_DIR.exists(), f"cache dir missing: {CACHE_DIR}"
assert ENV_PATH.exists(), f".env missing: {ENV_PATH}"

load_dotenv(ENV_PATH)
assert os.environ.get("CFBD_API_KEY"), "CFBD_API_KEY not in env"

# Mirror N02c constants verbatim --------------------------------------------
SCORING_PLAYTYPE_REGISTRY: dict[str, str] = {
    "Passing Touchdown":              "offensive_td",
    "Rushing Touchdown":              "offensive_td",
    "Field Goal Good":                "fg",
    "Interception Return Touchdown":  "return_td",
    "Fumble Return Touchdown":        "return_td",
    "Kickoff Return Touchdown":       "return_td",
    "Punt Return Touchdown":          "return_td",
    "Blocked Punt Touchdown":         "return_td",
    "Blocked Field Goal Touchdown":   "return_td",
    "Missed Field Goal Return Touchdown": "return_td",
    "Fumble Recovery Touchdown":      "return_td",
    "Defensive 2pt Conversion":       "pat_def_ret",
    "Extra Point Good":               "pat_1pt",
    "PAT Good":                       "pat_1pt",
    "Two Point Pass":                 "pat_2pt",
    "Two Point Rush":                 "pat_2pt",
    "Two-Point Pass":                 "pat_2pt",
    "Two-Point Rush":                 "pat_2pt",
    "2pt Conversion Good":            "pat_2pt",
    "Safety":                         "safety_def",
}
PAT_CATEGORIES = {"pat_1pt", "pat_2pt"}

# Mirror N02c (carried verbatim from 02b D1) ------------------
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

# Pass-2 targets (the 4 high-population UNIFORM playTypes the user wants
# verified at n=50 before any registry mapping is committed).
N50_TARGETS: list[str] = [
    "Uncategorized",
    "Punt",
    "Fumble Recovery (Opponent)",
    "Kickoff",
]
N50_SAMPLE_N: int = 50


def _norm(text: str | None) -> str:
    return (text or "").lower()


def classify_uncategorized(play_text: str | None) -> tuple[str, str]:
    """Returns (template_label, exception_reason). Template labels are
    'PAT_GOOD' for the dominant template, anything else is an exception."""
    t = _norm(play_text)
    if "extra point is good" in t or "pat good" in t:
        return ("PAT_GOOD", "")
    if "extra point" in t and ("good" in t or "made" in t):
        return ("PAT_GOOD", "")
    return ("EXCEPTION", f"playText does not contain PAT-good template: {play_text!r}")


def classify_punt(play_text: str | None) -> tuple[str, str]:
    """Dominant: 'punt for X yds, [returner] returns for Y yds for a TD'."""
    t = _norm(play_text)
    has_punt = "punt for" in t or "punt blocked" in t
    has_return = "return" in t
    has_td = "for a td" in t
    if has_punt and has_return and has_td:
        return ("PUNT_RETURN_TD", "")
    if "for a safety" in t:
        return ("EXCEPTION_SAFETY",
                f"safety on punt-typed scoring play: {play_text!r}")
    if has_punt and ("downed" in t or "out of bounds" in t):
        return ("EXCEPTION_NONSCORING_PUNT",
                f"non-scoring punt marked scoring=True: {play_text!r}")
    return ("EXCEPTION", f"unrecognized punt template: {play_text!r}")


def classify_fumble_recovery_opponent(play_text: str | None) -> tuple[str, str]:
    """Dominant: '[X] fumbled, recovered by [opp team] [Y], return for Z yds
    [for a TD], (KICK)'. The (KICK) trailer confirms TD+PAT in rows where
    the 'for a TD' substring is missing or buried."""
    t = _norm(play_text)
    has_fumble = "fumbled" in t or "fumble" in t
    has_recover = "recovered by" in t or "recovery" in t
    has_td = "for a td" in t
    has_kick_trailer = "(kick" in t or "(pat" in t or "kick)" in t
    if has_fumble and has_recover and (has_td or has_kick_trailer):
        return ("FUMBLE_RETURN_TD", "")
    if "for a safety" in t:
        return ("EXCEPTION_SAFETY",
                f"safety on fumble-recovery-opp scoring play: {play_text!r}")
    return ("EXCEPTION", f"unrecognized fumble-rec-opp template: {play_text!r}")


def classify_kickoff(play_text: str | None) -> tuple[str, str]:
    """Dominant: '[kicker] kickoff for X yds, [returner] return for Y yds
    for a TD'."""
    t = _norm(play_text)
    has_kickoff = "kickoff" in t
    has_return = "return" in t
    has_td = "for a td" in t
    has_kick_trailer = "(kick" in t
    if has_kickoff and has_return and (has_td or has_kick_trailer):
        return ("KICKOFF_RETURN_TD", "")
    if "for a safety" in t:
        return ("EXCEPTION_SAFETY",
                f"safety on kickoff-typed scoring play: {play_text!r}")
    if has_kickoff and ("touchback" in t or "out of bounds" in t):
        return ("EXCEPTION_NONSCORING_KICKOFF",
                f"non-scoring kickoff marked scoring=True: {play_text!r}")
    return ("EXCEPTION", f"unrecognized kickoff template: {play_text!r}")


CLASSIFIERS = {
    "Uncategorized": classify_uncategorized,
    "Punt": classify_punt,
    "Fumble Recovery (Opponent)": classify_fumble_recovery_opponent,
    "Kickoff": classify_kickoff,
}

# ------------------------------------------------------------- HTTP helpers --
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


def cfbd_get_cache_only(endpoint: str, **params: Any) -> Any:
    """Cache hit or AssertionError. No fresh calls."""
    key = _cache_key(f"cfbd__{endpoint.strip('/').replace('/', '_')}", params)
    assert key.exists(), (
        f"cache miss for {endpoint} params={params} (key={key.name}). "
        "investigation script must be cache-only."
    )
    size = key.stat().st_size
    data = json.loads(key.read_text(encoding="utf-8"))
    _log("cfbd", endpoint, params, cached=True, status=200,
         bytes_=size, elapsed_ms=0)
    return data


# ----------------------------------------------- Replicate the N02c loader --
TRIGGER_EVENTS_CSV = RESULTS_DIR / "trigger_events.csv"
assert TRIGGER_EVENTS_CSV.exists(), f"missing {TRIGGER_EVENTS_CSV}; run N01 first"

trigger_full_df = pd.read_csv(TRIGGER_EVENTS_CSV)

work_tuples_df = (
    trigger_full_df[["season", "season_type", "week"]]
    .drop_duplicates()
    .sort_values(["season", "season_type", "week"])
    .reset_index(drop=True)
)
print(f"distinct (season, season_type, week) tuples to load from cache: "
      f"{len(work_tuples_df)}", flush=True)

plays_by_game: dict[int, list[dict]] = {}
t_start = time.perf_counter()
for _, row in work_tuples_df.iterrows():
    season = int(row["season"])
    season_type = str(row["season_type"])
    week = int(row["week"])
    plays = cfbd_get_cache_only(
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
        plays_by_game.setdefault(gid, []).append(p)
elapsed_plays = time.perf_counter() - t_start
n_plays = sum(len(v) for v in plays_by_game.values())
print(f"[ok] /plays loaded from cache in {elapsed_plays:.1f}s -- "
      f"{len(plays_by_game):,} games, {n_plays:,} plays", flush=True)

# Sort by playNumber the same way N02c does so prev/next semantics match.
for gid in plays_by_game:
    plays_by_game[gid].sort(
        key=lambda p: (p.get("playNumber") if p.get("playNumber") is not None else 10**9)
    )


# ------------------------------------------------ Identify unknown scoring --
unknown_pt_to_plays: dict[str, list[tuple[int, dict, int]]] = defaultdict(list)
# value entries are (gameId, play_dict, index_within_game)

for gid, plays in plays_by_game.items():
    for idx, p in enumerate(plays):
        if not p.get("scoring"):
            continue
        pt = p.get("playType", "")
        if pt in SCORING_PLAYTYPE_REGISTRY:
            continue
        unknown_pt_to_plays[pt].append((gid, p, idx))

print(f"\n[ok] {len(unknown_pt_to_plays)} unknown scoring playType(s) found, "
      f"{sum(len(v) for v in unknown_pt_to_plays.values()):,} plays total",
      flush=True)


# ------------------------------------------------ Sample <=10 deterministically
SAMPLE_N = 10
random.seed(42)

samples: dict[str, list[tuple[int, dict, int]]] = {}
for pt, occurrences in unknown_pt_to_plays.items():
    occurrences_sorted = sorted(
        occurrences,
        key=lambda x: (x[0], x[1].get("id") or x[1].get("playId") or 0),
    )
    if len(occurrences_sorted) <= SAMPLE_N:
        samples[pt] = occurrences_sorted
    else:
        samples[pt] = sorted(
            random.sample(occurrences_sorted, SAMPLE_N),
            key=lambda x: (x[0], x[1].get("id") or x[1].get("playId") or 0),
        )


# ------------------------------------------------ Helpers for prev/next walk
def find_prev_non_pat(plays: list[dict], idx: int) -> dict | None:
    for j in range(idx - 1, -1, -1):
        cand = plays[j]
        cat = SCORING_PLAYTYPE_REGISTRY.get(cand.get("playType", ""))
        if cat in PAT_CATEGORIES:
            continue
        return cand
    return None


def find_next_non_pat(plays: list[dict], idx: int) -> dict | None:
    for j in range(idx + 1, len(plays)):
        cand = plays[j]
        cat = SCORING_PLAYTYPE_REGISTRY.get(cand.get("playType", ""))
        if cat in PAT_CATEGORIES:
            continue
        return cand
    return None


def team_aligned_deltas(prev_p: dict | None, curr_p: dict) -> tuple[int | None, int | None]:
    """Return (delta_team_a, delta_team_b) where team_a = prev offense team,
    team_b = prev defense team. Aligns across possession changes by matching
    team strings. Returns (None, None) if alignment fails."""
    if prev_p is None:
        return (None, None)
    prev_off_team = prev_p.get("offense")
    prev_def_team = prev_p.get("defense")
    prev_off_score = prev_p.get("offenseScore")
    prev_def_score = prev_p.get("defenseScore")
    curr_off_team = curr_p.get("offense")
    curr_def_team = curr_p.get("defense")
    curr_off_score = curr_p.get("offenseScore")
    curr_def_score = curr_p.get("defenseScore")
    if None in (prev_off_score, prev_def_score, curr_off_score, curr_def_score):
        return (None, None)
    if curr_off_team == prev_off_team and curr_def_team == prev_def_team:
        delta_a = int(curr_off_score) - int(prev_off_score)
        delta_b = int(curr_def_score) - int(prev_def_score)
    elif curr_off_team == prev_def_team and curr_def_team == prev_off_team:
        delta_a = int(curr_def_score) - int(prev_off_score)
        delta_b = int(curr_off_score) - int(prev_def_score)
    else:
        # Team strings don't reconcile -- shouldn't happen within one game.
        return (None, None)
    return (delta_a, delta_b)


# -------------------------------------------------------- Build CSV rows --
out_rows: list[dict] = []
for pt, occs in samples.items():
    for gid, p, idx in occs:
        plays = plays_by_game[gid]
        prev_p = find_prev_non_pat(plays, idx)
        next_p = find_next_non_pat(plays, idx)
        delta_a, delta_b = team_aligned_deltas(prev_p, p)
        abs_total = (
            None if delta_a is None or delta_b is None
            else abs(delta_a) + abs(delta_b)
        )
        play_id = p.get("id") or p.get("playId")
        play_id_neg = (
            int(play_id) < 0 if isinstance(play_id, (int, float)) else False
        )
        play_text = (p.get("playText") or "").replace("\n", " ").strip()
        if len(play_text) > 200:
            play_text = play_text[:197] + "..."

        out_rows.append({
            "pass_label": "n10_initial",
            "playType_unknown": pt,
            "gameId": gid,
            "playId": play_id,
            "playId_is_negative": int(bool(play_id_neg)),
            "playNumber": p.get("playNumber"),
            "period": p.get("period"),
            "clock": (
                f"{(p.get('clock') or {}).get('minutes', '')}:"
                f"{(p.get('clock') or {}).get('seconds', '')}"
                if isinstance(p.get("clock"), dict) else p.get("clock")
            ),
            "offense": p.get("offense"),
            "defense": p.get("defense"),
            "play_offenseScore": p.get("offenseScore"),
            "play_defenseScore": p.get("defenseScore"),
            "prev_non_pat_playType":
                None if prev_p is None else prev_p.get("playType"),
            "prev_offense":
                None if prev_p is None else prev_p.get("offense"),
            "prev_offenseScore":
                None if prev_p is None else prev_p.get("offenseScore"),
            "prev_defenseScore":
                None if prev_p is None else prev_p.get("defenseScore"),
            "next_non_pat_playType":
                None if next_p is None else next_p.get("playType"),
            "delta_team_a_signed": delta_a,
            "delta_team_b_signed": delta_b,
            "abs_delta_total": abs_total,
            "playText": play_text,
            "template_label": "",
            "exception_reason": "",
        })

# =====================================================================
# Pass 2 -- n=50 verification for the four high-pop UNIFORM playTypes
# =====================================================================
print(f"\n{'='*78}\nPASS 2: n={N50_SAMPLE_N} verification for "
      f"{N50_TARGETS}\n{'='*78}", flush=True)

random.seed(43)  # different seed from pass-1 to maximize fresh coverage

n50_summary: dict[str, dict[str, Any]] = {}

for pt in N50_TARGETS:
    occurrences = unknown_pt_to_plays.get(pt, [])
    if not occurrences:
        print(f"\n[skip] {pt}: 0 plays in cache", flush=True)
        continue

    occurrences_sorted = sorted(
        occurrences,
        key=lambda x: (x[0], x[1].get("id") or x[1].get("playId") or 0),
    )
    pop = len(occurrences_sorted)
    if pop <= N50_SAMPLE_N:
        sample = occurrences_sorted
    else:
        sample = sorted(
            random.sample(occurrences_sorted, N50_SAMPLE_N),
            key=lambda x: (x[0], x[1].get("id") or x[1].get("playId") or 0),
        )

    classifier = CLASSIFIERS[pt]
    template_counter: Counter = Counter()
    exceptions: list[dict] = []

    for gid, p, idx in sample:
        plays = plays_by_game[gid]
        prev_p = find_prev_non_pat(plays, idx)
        next_p = find_next_non_pat(plays, idx)
        delta_a, delta_b = team_aligned_deltas(prev_p, p)
        abs_total = (
            None if delta_a is None or delta_b is None
            else abs(delta_a) + abs(delta_b)
        )
        play_id = p.get("id") or p.get("playId")
        play_id_neg = (
            int(play_id) < 0 if isinstance(play_id, (int, float)) else False
        )
        play_text = (p.get("playText") or "").replace("\n", " ").strip()
        play_text_short = (
            play_text[:197] + "..." if len(play_text) > 200 else play_text
        )
        template_label, exception_reason = classifier(play_text)
        template_counter[template_label] += 1
        if template_label.startswith("EXCEPTION"):
            exceptions.append({
                "gameId": gid,
                "playId": play_id,
                "playText": play_text,
                "reason": exception_reason,
                "template_label": template_label,
            })

        out_rows.append({
            "pass_label": "n50_verification",
            "playType_unknown": pt,
            "gameId": gid,
            "playId": play_id,
            "playId_is_negative": int(bool(play_id_neg)),
            "playNumber": p.get("playNumber"),
            "period": p.get("period"),
            "clock": (
                f"{(p.get('clock') or {}).get('minutes', '')}:"
                f"{(p.get('clock') or {}).get('seconds', '')}"
                if isinstance(p.get("clock"), dict) else p.get("clock")
            ),
            "offense": p.get("offense"),
            "defense": p.get("defense"),
            "play_offenseScore": p.get("offenseScore"),
            "play_defenseScore": p.get("defenseScore"),
            "prev_non_pat_playType":
                None if prev_p is None else prev_p.get("playType"),
            "prev_offense":
                None if prev_p is None else prev_p.get("offense"),
            "prev_offenseScore":
                None if prev_p is None else prev_p.get("offenseScore"),
            "prev_defenseScore":
                None if prev_p is None else prev_p.get("defenseScore"),
            "next_non_pat_playType":
                None if next_p is None else next_p.get("playType"),
            "delta_team_a_signed": delta_a,
            "delta_team_b_signed": delta_b,
            "abs_delta_total": abs_total,
            "playText": play_text_short,
            "template_label": template_label,
            "exception_reason": exception_reason,
        })

    n_exceptions = sum(v for k, v in template_counter.items()
                       if k.startswith("EXCEPTION"))
    n_dominant = sum(v for k, v in template_counter.items()
                     if not k.startswith("EXCEPTION"))
    print(f"\n--- {pt} (pop={pop}, sampled={len(sample)}) ---", flush=True)
    for label, ct in sorted(template_counter.items(),
                            key=lambda x: (-x[1], x[0])):
        marker = " <-- EXCEPTION" if label.startswith("EXCEPTION") else ""
        print(f"    {label:<32} {ct:>3}{marker}")
    print(f"    {'dominant template hits':<32} {n_dominant:>3}")
    print(f"    {'total exceptions':<32} {n_exceptions:>3}")
    if exceptions:
        print(f"    Exception details:")
        for ex in exceptions:
            print(f"      gameId={ex['gameId']} playId={ex['playId']}: "
                  f"{ex['template_label']}")
            print(f"        playText: {ex['playText'][:160]!r}")

    n50_summary[pt] = {
        "population": pop,
        "samples": len(sample),
        "template_counts": dict(template_counter),
        "n_dominant_template": n_dominant,
        "n_exceptions": n_exceptions,
        "exception_details": exceptions,
    }

# =====================================================================
# Pass 3 -- drive-attribution check for Fumble Recovery (Own)
# =====================================================================
print(f"\n{'='*78}\nPASS 3: drive-attribution check for "
      f"Fumble Recovery (Own)\n{'='*78}", flush=True)

# Build game_id -> dog_team / fav_team map from trigger_events.csv
needed_cols_present = {"game_id", "dog_team", "fav_team"}.issubset(
    trigger_full_df.columns
)
assert needed_cols_present, (
    "trigger_events.csv missing one of game_id/dog_team/fav_team -- check N01 "
    "deliverable schema."
)
game_to_dog: dict[int, str] = {}
game_to_fav: dict[int, str] = {}
for _, r in trigger_full_df.iterrows():
    game_to_dog[int(r["game_id"])] = str(r["dog_team"])
    game_to_fav[int(r["game_id"])] = str(r["fav_team"])

drive_attrib_rows: list[dict] = []
for gid, p, idx in unknown_pt_to_plays.get("Fumble Recovery (Own)", []):
    plays = plays_by_game[gid]
    play_id = p.get("id") or p.get("playId")
    offense = p.get("offense")
    defense = p.get("defense")
    play_number = p.get("playNumber")
    drive_id = p.get("driveId") or p.get("driveNumber")
    dog_team = game_to_dog.get(gid)
    fav_team = game_to_fav.get(gid)
    is_dog_offense = (offense == dog_team) if dog_team is not None else None

    # Walk same-drive prior plays where the dog team was on offense
    n_prior_pass = 0
    n_prior_rush = 0
    n_prior_dog_explosive_pass = 0
    n_prior_dog_explosive_rush = 0
    if drive_id is not None and is_dog_offense:
        for q in plays:
            q_drive = q.get("driveId") or q.get("driveNumber")
            q_pn = q.get("playNumber")
            if q_drive != drive_id:
                continue
            if q_pn is None or play_number is None:
                continue
            if q_pn >= play_number:
                continue
            if q.get("offense") != dog_team:
                continue
            q_pt = q.get("playType", "")
            q_yards = q.get("yardsGained")
            if q_pt in EXPLOSIVE_PASS_PLAY_TYPES:
                n_prior_pass += 1
                if q_yards is not None and q_yards >= EXPLOSIVE_PASS_YARDS:
                    n_prior_dog_explosive_pass += 1
            elif q_pt in EXPLOSIVE_RUSH_PLAY_TYPES:
                n_prior_rush += 1
                if q_yards is not None and q_yards >= EXPLOSIVE_RUSH_YARDS:
                    n_prior_dog_explosive_rush += 1

    had_dog_explosive_in_drive = (
        (n_prior_dog_explosive_pass + n_prior_dog_explosive_rush) > 0
        if is_dog_offense else None
    )

    drive_attrib_rows.append({
        "gameId": gid,
        "playId": play_id,
        "playNumber": play_number,
        "drive_id": drive_id,
        "offense": offense,
        "defense": defense,
        "fav_team": fav_team,
        "dog_team": dog_team,
        "is_dog_offense": is_dog_offense,
        "n_prior_dog_pass_plays": n_prior_pass,
        "n_prior_dog_rush_plays": n_prior_rush,
        "n_prior_dog_explosive_pass": n_prior_dog_explosive_pass,
        "n_prior_dog_explosive_rush": n_prior_dog_explosive_rush,
        "had_dog_explosive_in_drive": had_dog_explosive_in_drive,
        "playText": (p.get("playText") or "")[:200],
    })

drive_attrib_df = pd.DataFrame(drive_attrib_rows)
drive_attrib_path = RESULTS_DIR / "_investigate_02c_unknown_scoring_drive_attrib.csv"
drive_attrib_df.to_csv(drive_attrib_path, index=False, encoding="utf-8")
print(f"\n[ok] wrote {drive_attrib_path} ({len(drive_attrib_df)} rows)",
      flush=True)

n_total = len(drive_attrib_rows)
n_in_trigger_game = sum(1 for r in drive_attrib_rows if r["dog_team"] is not None)
n_dog_offense = sum(1 for r in drive_attrib_rows if r["is_dog_offense"] is True)
n_fav_offense = sum(1 for r in drive_attrib_rows
                    if r["is_dog_offense"] is False)
n_no_trigger = sum(1 for r in drive_attrib_rows if r["dog_team"] is None)
n_dog_with_explosive = sum(
    1 for r in drive_attrib_rows
    if r["is_dog_offense"] is True and r["had_dog_explosive_in_drive"] is True
)
n_dog_without_explosive = sum(
    1 for r in drive_attrib_rows
    if r["is_dog_offense"] is True and r["had_dog_explosive_in_drive"] is False
)

print(f"\n  Fumble Recovery (Own) plays in cache:           {n_total}")
print(f"    in a trigger game (game has dog/fav):         {n_in_trigger_game}")
print(f"    not in any trigger game:                      {n_no_trigger}")
print(f"    dog team on offense:                          {n_dog_offense}")
print(f"    fav team on offense:                          {n_fav_offense}")
print(f"  of dog-offense FR(Own) plays:")
print(f"    drive had >=1 prior dog explosive (pass/rush):"
      f"{n_dog_with_explosive:>4}")
print(f"    drive had 0 prior dog explosives:             "
      f"{n_dog_without_explosive:>4}")

drive_attrib_summary = {
    "n_total": n_total,
    "n_in_trigger_game": n_in_trigger_game,
    "n_no_trigger_game": n_no_trigger,
    "n_dog_offense": n_dog_offense,
    "n_fav_offense": n_fav_offense,
    "n_dog_offense_with_prior_explosive": n_dog_with_explosive,
    "n_dog_offense_without_prior_explosive": n_dog_without_explosive,
}

# =====================================================================
# Write the unified per-row CSV (n=10 + n=50 rows together)
# =====================================================================
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
csv_path = RESULTS_DIR / "_investigate_02c_unknown_scoring.csv"
out_df = pd.DataFrame(out_rows)
out_df.to_csv(csv_path, index=False, encoding="utf-8")
print(f"\n[ok] wrote {csv_path} ({len(out_df)} rows; "
      f"n10={int((out_df['pass_label']=='n10_initial').sum())}, "
      f"n50={int((out_df['pass_label']=='n50_verification').sum())})",
      flush=True)


# ---------------------------------------------------- Summary table per pt --
print("\n" + "=" * 78)
print("SUMMARY: per unknown playType (count = total population in cache; "
      "buckets = sampled rows)")
print("=" * 78)
header = (
    f"{'playType':<40} {'pop':>5} {'samp':>5} "
    f"{'d=0':>4} {'d=2':>4} {'d=6':>4} {'d=7':>4} {'d=8':>4} "
    f"{'d=oth':>5} {'negID':>5}"
)
print(header)
print("-" * len(header))

summary_rows: list[dict] = []
for pt in sorted(unknown_pt_to_plays.keys(),
                 key=lambda x: (-len(unknown_pt_to_plays[x]), x)):
    pop = len(unknown_pt_to_plays[pt])
    rows = [r for r in out_rows if r["playType_unknown"] == pt]
    samp = len(rows)
    delta_counter: Counter = Counter()
    other_examples: list[Any] = []
    neg_id = 0
    for r in rows:
        d = r["abs_delta_total"]
        if d in (0, 2, 6, 7, 8):
            delta_counter[d] += 1
        else:
            delta_counter["other"] += 1
            other_examples.append(d)
        if r["playId_is_negative"]:
            neg_id += 1
    print(
        f"{pt:<40} {pop:>5} {samp:>5} "
        f"{delta_counter.get(0,0):>4} {delta_counter.get(2,0):>4} "
        f"{delta_counter.get(6,0):>4} {delta_counter.get(7,0):>4} "
        f"{delta_counter.get(8,0):>4} "
        f"{delta_counter.get('other',0):>5} {neg_id:>5}"
    )
    summary_rows.append({
        "playType": pt,
        "population_count": pop,
        "samples_examined": samp,
        "delta_0": delta_counter.get(0, 0),
        "delta_2": delta_counter.get(2, 0),
        "delta_6": delta_counter.get(6, 0),
        "delta_7": delta_counter.get(7, 0),
        "delta_8": delta_counter.get(8, 0),
        "delta_other": delta_counter.get("other", 0),
        "delta_other_examples": other_examples,
        "negative_playId_count": neg_id,
    })

# Save summary as JSON for easy programmatic re-read
summary_path = RESULTS_DIR / "_investigate_02c_unknown_scoring.summary.json"
full_summary = {
    "n10_initial": summary_rows,
    "n50_verification": n50_summary,
    "drive_attribution_fumble_recovery_own": drive_attrib_summary,
}
summary_path.write_text(
    json.dumps(full_summary, indent=2, default=str), encoding="utf-8"
)
print(f"\n[ok] wrote summary JSON: {summary_path}", flush=True)

print("\n[done]")
