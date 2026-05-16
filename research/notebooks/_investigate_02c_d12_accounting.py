"""Investigate the 2,346 triggers (26.2% of 8,940 with completed dog drives) where
the three dog points buckets sum to MORE than dog_score_at_trigger (negative
D12 accounting delta).

The N02c notebook's compute_points_buckets emitted [WARN] on this at execution
time. The stability verdicts (8/8 PASS) are unaffected because train and test
see the same convention, but if buckets attribute more dog points than the
scoreboard shows then either:
  (a) Buckets are over-attributing -- some scoring playType / drive flag is
      double-counting or routing wrong-team plays to the dog.
  (b) dog_score_at_trigger is under-reporting -- maybe excludes certain
      scoring types or has a CFBD-side off-by-one.
  (c) Both, or some combination.

Cache-only; mirrors the N02c loader and compute_points_buckets verbatim.

Outputs
-------
  research/results/_investigate_02c_d12_accounting.stdout.txt
      Sections (1) distribution by delta magnitude, (2) distribution by
      deficit threshold, (3) 10 random sampled negative-delta triggers with
      full attribution detail.
  research/results/_investigate_02c_d12_accounting.csv
      One row per investigated trigger (negative-delta only): trigger
      identifiers + bucket values + delta + per-bucket-explanation.
  research/results/_investigate_02c_d12_accounting.summary.json
      Top-level aggregated counts for quick consumption.
"""
from __future__ import annotations
import csv
import hashlib
import json
import pathlib
import random
import time
from collections import Counter, defaultdict
from typing import Any

import pandas as pd

SCRIPT = pathlib.Path(__file__).resolve()
NOTEBOOK_DIR = SCRIPT.parent
RESEARCH_DIR = NOTEBOOK_DIR.parent
DATA_DIR = RESEARCH_DIR / "data"
RESULTS_DIR = RESEARCH_DIR / "results"
CACHE_DIR = DATA_DIR / "cache"

assert CACHE_DIR.exists(), f"cache dir missing: {CACHE_DIR}"

STDOUT_PATH = RESULTS_DIR / "_investigate_02c_d12_accounting.stdout.txt"
CSV_PATH = RESULTS_DIR / "_investigate_02c_d12_accounting.csv"
SUMMARY_PATH = RESULTS_DIR / "_investigate_02c_d12_accounting.summary.json"

# --- Constants mirror current N02c (registry post-D12 extension) -----------

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
    "Timeout":                        "exclude",
    "placeholder":                    "exclude",
}

EXPLOSIVE_PASS_YARDS: int = 20
EXPLOSIVE_RUSH_YARDS: int = 12
EXPLOSIVE_PASS_PLAY_TYPES: frozenset[str] = frozenset({"Pass Reception", "Passing Touchdown"})
EXPLOSIVE_RUSH_PLAY_TYPES: frozenset[str] = frozenset({"Rush", "Rushing Touchdown"})


def is_explosive(play: dict) -> bool:
    pt = play.get("playType", "")
    yg = play.get("yardsGained")
    if yg is None:
        return False
    if pt in EXPLOSIVE_PASS_PLAY_TYPES:
        return int(yg) >= EXPLOSIVE_PASS_YARDS
    if pt in EXPLOSIVE_RUSH_PLAY_TYPES:
        return int(yg) >= EXPLOSIVE_RUSH_YARDS
    return False


def compute_points_buckets_traced(
    plays_before: list[dict], fav: str, dog: str
) -> tuple[int | None, int | None, int | None, int, list[dict]]:
    """Same as N02c's compute_points_buckets but also returns a TRACE list:
    one dict per scoring-play attribution decision."""
    drive_had_dog_explosive: dict[int, bool] = {}
    for p in plays_before:
        dn = p.get("driveNumber")
        if dn is None:
            continue
        if p.get("offense") == dog and is_explosive(p):
            drive_had_dog_explosive[int(dn)] = True

    dog_explosive_play_count = sum(
        1 for p in plays_before if p.get("offense") == dog and is_explosive(p)
    )

    points_explosives = 0
    points_sustained = 0
    points_returns = 0
    last_dog_td_bucket: str | None = None

    trace: list[dict] = []
    for p in plays_before:
        if not p.get("scoring"):
            continue
        pt = p.get("playType", "")
        cat = SCORING_PLAYTYPE_REGISTRY.get(pt)
        play_offense = p.get("offense")
        play_drive_no = p.get("driveNumber")
        attr_bucket = None
        attr_points = 0
        attr_reason = ""

        if cat is None:
            attr_reason = "UNKNOWN_PLAYTYPE_DEFENSIVE_NOOP"
        elif cat == "exclude":
            attr_reason = "exclude category"
        elif cat == "offensive_td":
            if play_offense != dog:
                attr_reason = f"offensive_td but offense={play_offense!r}, not dog"
            else:
                had_exp = bool(
                    drive_had_dog_explosive.get(int(play_drive_no), False)
                ) if play_drive_no is not None else False
                if had_exp:
                    points_explosives += 6
                    attr_bucket = "explosives"
                    attr_points = 6
                    last_dog_td_bucket = "explosives"
                    attr_reason = "dog offensive TD on drive WITH dog explosive"
                else:
                    points_sustained += 6
                    attr_bucket = "sustained"
                    attr_points = 6
                    last_dog_td_bucket = "sustained"
                    attr_reason = "dog offensive TD on drive WITHOUT dog explosive"
        elif cat == "fg":
            if play_offense != dog:
                attr_reason = f"fg but offense={play_offense!r}, not dog"
            else:
                had_exp = bool(
                    drive_had_dog_explosive.get(int(play_drive_no), False)
                ) if play_drive_no is not None else False
                if had_exp:
                    points_explosives += 3
                    attr_bucket = "explosives"
                    attr_points = 3
                    attr_reason = "dog FG on drive WITH dog explosive"
                else:
                    points_sustained += 3
                    attr_bucket = "sustained"
                    attr_points = 3
                    attr_reason = "dog FG on drive WITHOUT dog explosive"
        elif cat == "return_td":
            scoring_team = fav if play_offense == dog else dog
            if scoring_team == dog:
                points_returns += 6
                attr_bucket = "returns"
                attr_points = 6
                last_dog_td_bucket = "returns"
                attr_reason = f"return_td: offense={play_offense!r} -> scoring_team={scoring_team!r}=dog"
            else:
                attr_reason = f"return_td: offense={play_offense!r} -> scoring_team={scoring_team!r}=fav (no attribution)"
        elif cat == "pat_1pt":
            if play_offense == dog and last_dog_td_bucket is not None:
                if last_dog_td_bucket == "explosives":
                    points_explosives += 1
                    attr_bucket = "explosives"
                elif last_dog_td_bucket == "sustained":
                    points_sustained += 1
                    attr_bucket = "sustained"
                else:
                    points_returns += 1
                    attr_bucket = "returns"
                attr_points = 1
                attr_reason = f"pat_1pt to {last_dog_td_bucket!r} (last_dog_td_bucket)"
            else:
                attr_reason = (
                    f"pat_1pt: offense={play_offense!r}, last_dog_td_bucket="
                    f"{last_dog_td_bucket!r} -> no attribution"
                )
        elif cat == "pat_2pt":
            if play_offense == dog and last_dog_td_bucket is not None:
                if last_dog_td_bucket == "explosives":
                    points_explosives += 2
                    attr_bucket = "explosives"
                elif last_dog_td_bucket == "sustained":
                    points_sustained += 2
                    attr_bucket = "sustained"
                else:
                    points_returns += 2
                    attr_bucket = "returns"
                attr_points = 2
                attr_reason = f"pat_2pt to {last_dog_td_bucket!r}"
            else:
                attr_reason = (
                    f"pat_2pt: offense={play_offense!r}, last_dog_td_bucket="
                    f"{last_dog_td_bucket!r} -> no attribution"
                )
        elif cat == "safety_def":
            scoring_team = fav if play_offense == dog else dog
            if scoring_team == dog:
                points_returns += 2
                attr_bucket = "returns"
                attr_points = 2
                attr_reason = f"safety: offense={play_offense!r} -> dog gets +2"
            else:
                attr_reason = f"safety: offense={play_offense!r} -> fav gets +2 (no dog attribution)"
        elif cat == "pat_def_ret":
            scoring_team = fav if play_offense == dog else dog
            if scoring_team == dog:
                points_returns += 2
                attr_bucket = "returns"
                attr_points = 2
                attr_reason = "pat_def_ret to dog"
            else:
                attr_reason = "pat_def_ret to fav"

        trace.append({
            "play_id": p.get("id") or p.get("playId"),
            "play_number": p.get("playNumber"),
            "drive_number": p.get("driveNumber"),
            "period": p.get("period"),
            "offense": p.get("offense"),
            "defense": p.get("defense"),
            "playType": pt,
            "category": cat or "<unknown>",
            "scoring": p.get("scoring"),
            "yardsGained": p.get("yardsGained"),
            "offenseScore": p.get("offenseScore"),
            "defenseScore": p.get("defenseScore"),
            "attr_bucket": attr_bucket,
            "attr_points": attr_points,
            "reason": attr_reason,
            "playText": (p.get("playText") or "")[:150],
        })

    return (
        points_explosives,
        points_sustained,
        points_returns,
        dog_explosive_play_count,
        trace,
    )


# --- Cache helper (cache-only, hash-based filenames; mirrors N02c) ----------

def _params_hash(params: dict) -> str:
    return hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def _cache_key(prefix: str, params: dict) -> pathlib.Path:
    return CACHE_DIR / f"{prefix}__{_params_hash(params)}.json"


def cfbd_get_cache_only(endpoint: str, **params: Any) -> Any:
    key = _cache_key(f"cfbd__{endpoint.strip('/').replace('/', '_')}", params)
    assert key.exists(), f"cache miss for {endpoint} params={params}"
    return json.loads(key.read_text(encoding="utf-8"))


# --- Load triggers ----------------------------------------------------------

TRIGGER_EVENTS_CSV = RESULTS_DIR / "trigger_events.csv"
TRIGGER_OUTCOMES_CSV = RESULTS_DIR / "trigger_outcomes.csv"
te = pd.read_csv(TRIGGER_EVENTS_CSV)
to = pd.read_csv(TRIGGER_OUTCOMES_CSV)

# Merge same as N02c: on (game_id, fav_deficit). Both columns appear in both CSVs.
merged = te.merge(to, on=["game_id", "fav_deficit"], how="inner")
n_pre = len(merged)
merged = merged.dropna(subset=["final_fav_won"]).reset_index(drop=True)
print(
    f"merged: {len(merged):,} rows on ['game_id','fav_deficit'] "
    f"(dropped {n_pre - len(merged)} with NaN final_fav_won)"
)
assert len(merged) == 11416, (
    f"merge sanity check: expected 11,416 rows post-NaN-drop (matches N02c), got {len(merged)}"
)

# Load /plays cache for every distinct (season, season_type, week) tuple.
work_tuples = (
    merged[["season", "season_type", "week"]]
    .drop_duplicates()
    .sort_values(["season", "season_type", "week"])
    .reset_index(drop=True)
)
print(f"distinct cache tuples: {len(work_tuples)}")

plays_by_game: dict[int, list[dict]] = {}
t0 = time.perf_counter()
for _, row in work_tuples.iterrows():
    plays = cfbd_get_cache_only(
        "/plays",
        year=int(row["season"]),
        seasonType=str(row["season_type"]),
        week=int(row["week"]),
        classification="fbs",
    )
    for p in plays:
        gid = p.get("gameId")
        if gid is None:
            continue
        plays_by_game.setdefault(gid, []).append(p)
elapsed = time.perf_counter() - t0
n_plays = sum(len(v) for v in plays_by_game.values())
print(f"[ok] /plays cache load: {elapsed:.1f}s, {len(plays_by_game):,} games, {n_plays:,} plays")
for gid in plays_by_game:
    plays_by_game[gid].sort(
        key=lambda p: (p.get("playNumber") if p.get("playNumber") is not None else 10**9)
    )

# --- Iterate triggers; collect negative-delta rows --------------------------

negative_rows: list[dict] = []
positive_rows_count = 0
zero_rows_count = 0
nullbucket_rows_count = 0
for _, trig in merged.iterrows():
    gid = int(trig["game_id"])
    fav = str(trig["fav_team"])
    dog = str(trig["dog_team"])
    trig_pn = int(trig["play_number"])
    trig_drive_no = int(trig["drive_number_in_game"])
    dog_score = int(trig["dog_score_at_trigger"])
    fav_deficit = int(trig["fav_deficit"])
    plays_all = plays_by_game.get(gid, [])
    plays_before = [
        p for p in plays_all
        if (p.get("playNumber") or 0) < trig_pn
    ]
    # Mirror N02c's D7 "completed dog drive" test
    has_completed_dog_drive = any(
        p.get("offense") == dog
        and p.get("driveNumber") is not None
        and int(p["driveNumber"]) < trig_drive_no
        for p in plays_before
    )
    if not has_completed_dog_drive:
        nullbucket_rows_count += 1
        continue
    pe, ps, pr, n_exp, trace = compute_points_buckets_traced(plays_before, fav, dog)
    total = pe + ps + pr
    delta = dog_score - total
    if delta > 0:
        positive_rows_count += 1
    elif delta == 0:
        zero_rows_count += 1
    else:
        negative_rows.append({
            "game_id": gid,
            "season": int(trig["season"]),
            "season_type": str(trig["season_type"]),
            "week": int(trig["week"]),
            "fav_team": fav,
            "dog_team": dog,
            "trigger_play_id": int(trig["trigger_play_id"]),
            "play_number": trig_pn,
            "trigger_play_type": str(trig["play_type"]),
            "quarter": int(trig["quarter"]),
            "drive_number_in_game": trig_drive_no,
            "fav_deficit": fav_deficit,
            "dog_score_at_trigger": dog_score,
            "fav_score_at_trigger": int(trig["fav_score_at_trigger"]),
            "dog_points_from_explosives": pe,
            "dog_points_from_sustained": ps,
            "dog_points_from_returns": pr,
            "bucket_sum": total,
            "delta": delta,
            "dog_explosive_play_count": n_exp,
            "trace": trace,
        })

print(
    f"\ntriggers iterated: {len(merged):,}\n"
    f"  with completed dog drive (D7 non-null buckets): "
    f"{len(merged) - nullbucket_rows_count:,}\n"
    f"  positive delta (typical: trigger play is dog scoring): {positive_rows_count:,}\n"
    f"  zero delta (clean attribution): {zero_rows_count:,}\n"
    f"  NEGATIVE delta (over-attribution candidate): {len(negative_rows):,}\n"
)

# --- Distribution by delta magnitude ----------------------------------------

delta_counter = Counter(r["delta"] for r in negative_rows)
print("=== Distribution by delta magnitude (most-common negative deltas) ===")
for d, ct in sorted(delta_counter.items(), key=lambda x: -x[1])[:30]:
    pct = 100.0 * ct / len(negative_rows)
    print(f"  delta={d:>5}  count={ct:>5}  ({pct:>5.2f}%)")
print()

# --- Distribution by deficit threshold --------------------------------------

print("=== Distribution by deficit threshold (fav_deficit) ===")
deficit_counter = Counter(r["fav_deficit"] for r in negative_rows)
deficit_totals = Counter(int(t["fav_deficit"]) for _, t in merged.iterrows())
for d in sorted(set(deficit_counter) | set(deficit_totals)):
    n_neg = deficit_counter.get(d, 0)
    n_all = deficit_totals.get(d, 0)
    pct = 100.0 * n_neg / n_all if n_all else 0.0
    print(f"  deficit={d:>4}  triggers_total={n_all:>5}  negative_delta={n_neg:>4}  ({pct:>5.2f}%)")
print()

# --- 10 random sampled negative-delta triggers ------------------------------

rng = random.Random(42)
sample = rng.sample(negative_rows, min(10, len(negative_rows)))
print("=== 10 random sampled negative-delta triggers (with attribution traces) ===")
for i, r in enumerate(sample):
    print(f"\n--- sample {i+1}/{len(sample)} ---")
    print(
        f"  game_id={r['game_id']}  fav={r['fav_team']!r}  dog={r['dog_team']!r}  "
        f"season={r['season']} week={r['week']}"
    )
    print(
        f"  trigger: play_id={r['trigger_play_id']}  play_number={r['play_number']}  "
        f"play_type={r['trigger_play_type']!r}  q={r['quarter']}  "
        f"drive_in_game={r['drive_number_in_game']}  deficit={r['fav_deficit']}"
    )
    print(
        f"  scoreboard: dog_score={r['dog_score_at_trigger']}  fav_score={r['fav_score_at_trigger']}"
    )
    print(
        f"  buckets: explosives={r['dog_points_from_explosives']}  "
        f"sustained={r['dog_points_from_sustained']}  "
        f"returns={r['dog_points_from_returns']}  "
        f"SUM={r['bucket_sum']}  DELTA={r['delta']}  "
        f"dog_explosive_play_count={r['dog_explosive_play_count']}"
    )
    print(f"  attribution trace (scoring plays in plays_before, in play order):")
    print(
        f"    {'pn':>4} {'drv':>3} {'q':>2} {'offense':<20} "
        f"{'playType':<32} {'cat':<14} {'bkt':<11} {'pts':>3}  reason"
    )
    for ev in r["trace"]:
        if ev["attr_bucket"] is None and ev["category"] in ("exclude", "<unknown>"):
            continue  # skip noise
        print(
            f"    {(ev['play_number'] or 0):>4} "
            f"{(ev['drive_number'] or 0):>3} "
            f"{(ev['period'] or 0):>2} "
            f"{(ev['offense'] or '?'):<20} "
            f"{(ev['playType'] or '?'):<32} "
            f"{ev['category']:<14} "
            f"{(ev['attr_bucket'] or '-'):<11} "
            f"{ev['attr_points']:>3}  {ev['reason']}"
        )

# --- Write CSV / JSON outputs -----------------------------------------------

CSV_COLS = [
    "game_id", "season", "season_type", "week", "fav_team", "dog_team",
    "trigger_play_id", "play_number", "trigger_play_type", "quarter",
    "drive_number_in_game", "fav_deficit", "dog_score_at_trigger",
    "fav_score_at_trigger", "dog_points_from_explosives",
    "dog_points_from_sustained", "dog_points_from_returns", "bucket_sum",
    "delta", "dog_explosive_play_count",
]
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=CSV_COLS)
    w.writeheader()
    for r in negative_rows:
        w.writerow({k: r[k] for k in CSV_COLS})

with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    json.dump({
        "triggers_total": int(len(merged)),
        "completed_dog_drive_count": int(len(merged) - nullbucket_rows_count),
        "positive_delta_count": positive_rows_count,
        "zero_delta_count": zero_rows_count,
        "negative_delta_count": len(negative_rows),
        "delta_histogram_top20": [
            {"delta": int(d), "count": int(c)}
            for d, c in sorted(delta_counter.items(), key=lambda x: -x[1])[:20]
        ],
        "by_deficit_threshold": [
            {
                "deficit": int(d),
                "total_triggers": int(deficit_totals.get(d, 0)),
                "negative_delta": int(deficit_counter.get(d, 0)),
                "pct": (
                    100.0 * deficit_counter.get(d, 0) / deficit_totals.get(d, 1)
                    if deficit_totals.get(d, 0) else 0.0
                ),
            }
            for d in sorted(set(deficit_counter) | set(deficit_totals))
        ],
    }, f, indent=2)

print(f"\n[ok] wrote {CSV_PATH.name} ({len(negative_rows)} rows)")
print(f"[ok] wrote {SUMMARY_PATH.name}")
