"""
In-memory simulation of the proposed N01 fix. NO CFBD calls.

For every game that produced triggers in the broken run we have
(game_id, fav_team, dog_team) in trigger_events.csv. For each of those
games we:
  1. Load cached /plays.
  2. Sort with _sorted_plays_for_game (period, driveNumber, playNumber).
  3. Walk plays, computing fav_score / dog_score via the new
     _post_play_scores_for_team (per-subtype POST_PLAY).
  4. Emit deficit-threshold-cross rows.

Then we diff against the broken trigger_events.csv:
  - row count
  - distribution by fav_deficit
  - distribution by quarter
  - distribution by play_type
  - sample of per-game per-D shifts (where the new run lands on a
    different play_id than the broken run did, but the same (game_id,
    fav_deficit) row exists)

Bound on the "new triggers in non-triggering games" caveat:
  The broken sort's artifact mostly lifts fav_score at low-playNumber
  positions (period-4 dup rows). For that to systematically prevent
  trigger detection in a game where fav genuinely trailed, the artifact
  would have to inflate fav_score at EVERY play through the trailing
  window -- which would require dup-row collisions across the entire
  walk, not just the leading positions. Empirically the dup-row inflation
  is concentrated at playNumber=1..3 of each period, so the caveat's
  contribution to the diff is bounded above by ~tens of games.
"""
from __future__ import annotations

import csv
import glob
import json
import pathlib
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"

DEFICIT_THRESHOLDS = [3, 7, 10, 14, 21]
PAT_KEYWORDS = {"point after", "two point conversion", "extra point"}
SCORING_REGISTRY = [
    ("rushing touchdown",                  "offense", 6, "TD_OFFENSIVE"),
    ("passing touchdown",                  "offense", 6, "TD_OFFENSIVE"),
    ("interception return touchdown",      "defense", 6, "TD_DEFENSIVE_RETURN"),
    ("fumble return touchdown",            "defense", 6, "TD_DEFENSIVE_RETURN"),
    ("punt return touchdown",              "defense", 6, "TD_SPECIAL_RETURN"),
    ("blocked punt touchdown",             "defense", 6, "TD_SPECIAL_RETURN"),
    ("blocked field goal touchdown",       "defense", 6, "TD_SPECIAL_RETURN"),
    ("missed field goal return touchdown", "defense", 6, "TD_SPECIAL_RETURN"),
    ("kickoff return touchdown",           "offense", 6, "TD_KICKOFF_RETURN"),
    ("field goal good",                    "offense", 3, "FIELD_GOAL"),
    ("safety",                             "defense", 2, "SAFETY"),
]
KO_RETURN_TD_INVARIANT_TOLERANCE = 2


def _classify_scoring_play(play):
    pt = (play.get("playType") or "").strip().lower()
    for kw, side, points, label in SCORING_REGISTRY:
        if kw in pt:
            return side, points, label
    return None


def _read_team_score(play, team):
    if play.get("offense") == team:
        v = play.get("offenseScore")
        return int(v) if v is not None else None
    if play.get("defense") == team:
        v = play.get("defenseScore")
        return int(v) if v is not None else None
    return None


def _find_clean_next(plays_sorted, idx, team):
    for j in range(idx + 1, len(plays_sorted)):
        nxt = plays_sorted[j]
        ptype = (nxt.get("playType") or "").strip().lower()
        if any(kw in ptype for kw in PAT_KEYWORDS):
            continue
        ns = _read_team_score(nxt, team)
        if ns is None:
            return None
        return nxt
    return None


class ScoringPlayAmbiguityError(ValueError):
    pass


# Counters that the cell-19 summary will print
KO_RETURN_TD_VERIFIED_BY_INVARIANT = 0
KO_RETURN_TD_RAISED_BY_INVARIANT = 0
SORT_DEDUP_LOG = []


def _post_play_scores_for_team(plays_sorted, idx, team):
    global KO_RETURN_TD_VERIFIED_BY_INVARIANT, KO_RETURN_TD_RAISED_BY_INVARIANT
    p = plays_sorted[idx]
    direct = _read_team_score(p, team)
    if not bool(p.get("scoring")):
        return direct
    cls = _classify_scoring_play(p)
    if cls is None:
        raise ScoringPlayAmbiguityError(
            f"unknown scoring playType={p.get('playType')!r} "
            f"play_id={p.get('id')} game_id={p.get('gameId')}"
        )
    _, _, label = cls
    if label == "TD_KICKOFF_RETURN":
        nxt = _find_clean_next(plays_sorted, idx, team)
        if nxt is None:
            return direct
        ns = _read_team_score(nxt, team)
        if ns is None or direct is None:
            return direct if direct is not None else ns
        if abs(int(ns) - int(direct)) > KO_RETURN_TD_INVARIANT_TOLERANCE:
            KO_RETURN_TD_RAISED_BY_INVARIANT += 1
            raise ScoringPlayAmbiguityError(
                f"KO Return TD ±{KO_RETURN_TD_INVARIANT_TOLERANCE} invariant "
                f"failed: play_id={p.get('id')} delta={ns - direct}"
            )
        KO_RETURN_TD_VERIFIED_BY_INVARIANT += 1
        return direct
    return direct


def _sorted_plays_for_game(plays, *, game_id):
    valid = [p for p in plays
             if p.get("period") is not None
             and p.get("driveNumber") is not None
             and p.get("playNumber") is not None]
    sk = lambda p: (int(p["period"]), int(p["driveNumber"]), int(p["playNumber"]))
    sp = sorted(valid, key=sk)
    seen = {}
    discarded = []
    for p in sp:
        k = sk(p)
        if k in seen:
            existing = seen[k]
            challenger_id = int(p.get("id") or 1 << 62)
            existing_id = int(existing.get("id") or 1 << 62)
            if challenger_id < existing_id:
                discarded.append(existing)
                seen[k] = p
            else:
                discarded.append(p)
        else:
            seen[k] = p
    if len(discarded) > 3:
        raise ValueError(
            f"game_id={game_id}: {len(discarded)} duplicate-key discards"
        )
    if discarded:
        SORT_DEDUP_LOG.append({
            "game_id": game_id,
            "n_discarded": len(discarded),
            "discarded_ids": [d.get("id") for d in discarded],
        })
    return sorted(seen.values(), key=sk)


def _clock_to_int(value):
    if value is None:
        return 0
    if isinstance(value, int):
        return int(value)
    s = str(value).strip()
    if s == "":
        return 0
    try:
        return int(s)
    except ValueError:
        return 0  # mirrors current behavior for the simulation


def detect_triggers_for_game(plays, fav_team, dog_team, game_id):
    sp = _sorted_plays_for_game(plays, game_id=game_id)
    if not sp:
        return []
    triggers_seen = set()
    rows = []
    for idx, p in enumerate(sp):
        period = int(p.get("period") or 0)
        if period >= 5:
            continue  # OT
        try:
            fav_score = _post_play_scores_for_team(sp, idx, fav_team)
            dog_score = _post_play_scores_for_team(sp, idx, dog_team)
        except ScoringPlayAmbiguityError:
            continue
        if fav_score is None or dog_score is None:
            continue
        deficit = int(dog_score) - int(fav_score)
        if deficit <= 0:
            continue
        for D in DEFICIT_THRESHOLDS:
            if D in triggers_seen:
                continue
            if deficit < D:
                continue
            triggers_seen.add(D)

            clock = p.get("clock") or {}
            cm = _clock_to_int(clock.get("minutes"))
            cs = _clock_to_int(clock.get("seconds"))
            clock_secs_in_period = cm * 60 + cs

            rows.append({
                "game_id": int(game_id),
                "fav_deficit": int(D),
                "play_number": int(p["playNumber"]),
                "trigger_play_id": int(p["id"]) if p.get("id") is not None else None,
                "play_type": p.get("playType"),
                "quarter": period,
                "fav_score_at_trigger": int(fav_score),
                "dog_score_at_trigger": int(dog_score),
                "actual_deficit_at_trigger": int(deficit),
                "drive_number_in_game": int(p["driveNumber"]) if p.get("driveNumber") is not None else None,
            })
    rows.sort(key=lambda r: (r["play_number"], r["fav_deficit"]))
    for k, r in enumerate(rows, start=1):
        r["trigger_sequence"] = k
    return rows


# --- Load -----------------------------------------------------------------
plays_by_game: dict[int, list[dict]] = {}
for fp in sorted(glob.glob(str(CACHE / "cfbd__plays__*.json"))):
    with open(fp, encoding="utf-8") as f:
        plays = json.load(f)
    if not isinstance(plays, list):
        continue
    for p in plays:
        gid = p.get("gameId")
        if gid is None:
            continue
        plays_by_game.setdefault(gid, []).append(p)
print(f"plays_by_game: {len(plays_by_game):,} games")


# Load broken trigger events
broken_rows: list[dict] = []
with open(RESULTS / "trigger_events.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        broken_rows.append(row)
print(f"broken triggers: {len(broken_rows):,} rows")

# game_id -> (fav_team, dog_team)
fav_dog_by_game: dict[int, tuple[str, str]] = {}
for r in broken_rows:
    gid = int(r["game_id"])
    if gid not in fav_dog_by_game:
        fav_dog_by_game[gid] = (r["fav_team"], r["dog_team"])
broken_triggering_games = set(fav_dog_by_game.keys())
print(f"games triggering in broken run: {len(broken_triggering_games):,}")


# --- Simulate fix --------------------------------------------------------
fixed_rows: list[dict] = []
games_with_dedup = 0
games_failing_dedup_threshold = 0
games_skipped_no_plays = 0
games_with_unknown_scoring = 0
games_with_ko_invariant_raise = 0

for gid, (fav, dog) in fav_dog_by_game.items():
    plays = plays_by_game.get(gid)
    if not plays:
        games_skipped_no_plays += 1
        continue
    try:
        rows = detect_triggers_for_game(plays, fav, dog, gid)
    except ValueError as e:
        # Either dedup-threshold raise or a real ambiguity that the inner
        # try/except didn't catch -- shouldn't happen for our wrap, but
        # surface for diagnosis.
        if "duplicate-key discards" in str(e):
            games_failing_dedup_threshold += 1
            continue
        raise
    fixed_rows.extend(rows)

print(f"fixed triggers (broken-game subset): {len(fixed_rows):,}")
print(f"games skipped (no cached plays)    : {games_skipped_no_plays:,}")
print(f"games failing dedup threshold     : {games_failing_dedup_threshold:,}")
print(f"sort dedup log entries             : {len(SORT_DEDUP_LOG):,}")
print(f"KO Return TD verified by ±2        : {KO_RETURN_TD_VERIFIED_BY_INVARIANT:,}")
print(f"KO Return TD raised by ±2          : {KO_RETURN_TD_RAISED_BY_INVARIANT:,}")


# --- Diff stats ----------------------------------------------------------
def cnt(rows, key):
    return Counter((r.get(key) if isinstance(r.get(key), str) else int(r[key])) for r in rows)


print("\n" + "=" * 72)
print(f"ROW COUNT  broken={len(broken_rows):,}  fixed={len(fixed_rows):,}  "
      f"delta={len(fixed_rows) - len(broken_rows):+,}  "
      f"({100*(len(fixed_rows) - len(broken_rows))/len(broken_rows):+.2f}%)")
print("=" * 72)


def show_dist(name, broken_dist, fixed_dist, sort_key=None):
    keys = sorted(set(broken_dist) | set(fixed_dist),
                  key=sort_key if sort_key else (lambda x: (-broken_dist.get(x, 0), str(x))))
    print(f"\nDistribution by {name}:")
    print(f"  {'value':<35} {'broken':>8} {'fixed':>8} {'delta':>8}  {'b%':>6}  {'f%':>6}")
    for k in keys:
        b = broken_dist.get(k, 0)
        f = fixed_dist.get(k, 0)
        bp = 100 * b / max(len(broken_rows), 1)
        fp = 100 * f / max(len(fixed_rows), 1)
        print(f"  {str(k):<35} {b:>8,} {f:>8,} {f-b:>+8,}  {bp:>5.1f}%  {fp:>5.1f}%")


show_dist("fav_deficit", cnt(broken_rows, "fav_deficit"), cnt(fixed_rows, "fav_deficit"),
          sort_key=lambda x: int(x))
show_dist("quarter", cnt(broken_rows, "quarter"), cnt(fixed_rows, "quarter"),
          sort_key=lambda x: int(x))
show_dist("play_type", cnt(broken_rows, "play_type"), cnt(fixed_rows, "play_type"))


# --- Per-(game, deficit) shift analysis ----------------------------------
broken_keyed = {(int(r["game_id"]), int(r["fav_deficit"])): r for r in broken_rows}
fixed_keyed = {(int(r["game_id"]), int(r["fav_deficit"])): r for r in fixed_rows}

both = set(broken_keyed) & set(fixed_keyed)
only_broken = set(broken_keyed) - set(fixed_keyed)
only_fixed = set(fixed_keyed) - set(broken_keyed)
print(f"\n(game_id, fav_deficit) overlap:")
print(f"  in both                 : {len(both):,}")
print(f"  only in broken (deleted): {len(only_broken):,}")
print(f"  only in fixed (added)   : {len(only_fixed):,}")

# Of "in both": how many have a different trigger_play_id?
moved_play_id = 0
moved_quarter = 0
moved_actual_deficit = 0
same_everything = 0
for k in both:
    b = broken_keyed[k]
    f = fixed_keyed[k]
    if int(b["trigger_play_id"]) != int(f["trigger_play_id"]):
        moved_play_id += 1
    if int(b["quarter"]) != int(f["quarter"]):
        moved_quarter += 1
    if int(b["actual_deficit_at_trigger"]) != int(f["actual_deficit_at_trigger"]):
        moved_actual_deficit += 1
    if (int(b["trigger_play_id"]) == int(f["trigger_play_id"])
            and int(b["actual_deficit_at_trigger"]) == int(f["actual_deficit_at_trigger"])):
        same_everything += 1
print(f"  of overlapping rows: moved trigger_play_id : {moved_play_id:,}")
print(f"                       moved quarter         : {moved_quarter:,}")
print(f"                       moved actual_deficit  : {moved_actual_deficit:,}")
print(f"                       identical everything  : {same_everything:,}")


# --- Sample of moved rows -----------------------------------------------
print("\nSample of 10 moved (game_id, fav_deficit) rows:")
moved_examples = [k for k in sorted(both) if int(broken_keyed[k]["trigger_play_id"]) != int(fixed_keyed[k]["trigger_play_id"])][:10]
for k in moved_examples:
    b = broken_keyed[k]
    f = fixed_keyed[k]
    print(f"  game={k[0]} D={k[1]}: "
          f"broken pNum={b['play_number']} type={b['play_type']!r:32}  "
          f"fav={b['fav_score_at_trigger']} dog={b['dog_score_at_trigger']} actD={b['actual_deficit_at_trigger']}")
    print(f"    {' ' * 20}fixed  pNum={f['play_number']} type={f['play_type']!r:32}  "
          f"fav={f['fav_score_at_trigger']} dog={f['dog_score_at_trigger']} actD={f['actual_deficit_at_trigger']}")


print("\nSample of 10 'only in broken' (deleted by fix) rows:")
for k in sorted(only_broken)[:10]:
    b = broken_keyed[k]
    print(f"  game={k[0]} D={k[1]} pNum={b['play_number']} type={b['play_type']!r:32}  "
          f"fav={b['fav_score_at_trigger']} dog={b['dog_score_at_trigger']} actD={b['actual_deficit_at_trigger']}")

print("\nSample of 10 'only in fixed' (added by fix) rows:")
for k in sorted(only_fixed)[:10]:
    f = fixed_keyed[k]
    print(f"  game={k[0]} D={k[1]} pNum={f['play_number']} type={f['play_type']!r:32}  "
          f"fav={f['fav_score_at_trigger']} dog={f['dog_score_at_trigger']} actD={f['actual_deficit_at_trigger']}")
