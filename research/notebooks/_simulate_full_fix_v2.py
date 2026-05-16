"""
v2 of the simulation. The v1 hardcoded registry missed 17 scoring=true
playTypes (Sack-with-scoring, Fumble Recovery (Opponent), Punt-with-scoring,
Kickoff-with-scoring, Pass Reception-with-scoring, Rush-with-scoring,
Interception, Defensive 2pt Conversion, Two Point Pass/Rush, etc.). The
v1 simulation raised on those, leading to one game (401301021) losing 2
triggers because its D=10/D=14 trigger lived on a Sack-with-scoring row.

v2 fix: cell 14 now ENUMERATES every distinct scoring=true playType in
the cached corpus and empirically verifies each one. The scorer's
registry is data-driven rather than hand-curated.

Verification per (playType_lower, attribution_side):
  same triangulated d_pre/d_next test as before, parameterized by points
  in {6, 3, 2}. We pick the (side, points) that gives the highest
  VERIFIED_POST share. PAT-like subtypes are detected via playText
  ("extra point" / "two-point" / etc.) and classified separately as
  PAT_NO_TRIGGER (the scorer reads them direct, they do not need to
  participate in trigger detection because the prior TD already credited
  the points under POST_PLAY).
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
PAT_KEYWORDS_PT = {"point after", "two point conversion", "extra point"}
PAT_TEXT_KEYWORDS = ("extra point", "two-point", "two point", "defensive pat")


def _is_pat_like(play: dict) -> bool:
    """A PAT/2pt/defensive-PAT-conversion play. These don't fire triggers
    on their own; their points are already credited on the prior TD's
    POST_PLAY scoreboard (or via the +1/+2 increment)."""
    pt = (play.get("playType") or "").strip().lower()
    if any(kw in pt for kw in PAT_KEYWORDS_PT):
        return True
    if pt == "defensive 2pt conversion":
        return True
    if pt == "two point pass" or pt == "two point rush":
        return True
    txt = (play.get("playText") or "").strip().lower()
    if any(kw in txt for kw in PAT_TEXT_KEYWORDS):
        return True
    return False


def _read_team_score(play, team):
    if play.get("offense") == team:
        v = play.get("offenseScore")
        return int(v) if v is not None else None
    if play.get("defense") == team:
        v = play.get("defenseScore")
        return int(v) if v is not None else None
    return None


def _drv_sort_key(p):
    return (int(p["period"]), int(p["driveNumber"]), int(p["playNumber"]))


# --- Load all plays ------------------------------------------------------
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


# --- Cell-14-equivalent: enumerate scoring playTypes, verify per subtype --
# For each scoring=true play (excluding PAT-like), try both attribution
# rules and points in {6, 3, 2}. Pick the (side, points) maximizing the
# VERIFIED_POST share. Build SCORING_PLAY_REGISTRY mapping
# playType_lower -> dict(side, points, verdict, post_share, n_samples).
def _find_clean_prev(plays_sorted, i, team, this_score):
    for j in range(i - 1, -1, -1):
        prev = plays_sorted[j]
        if _is_pat_like(prev):
            continue
        ps = _read_team_score(prev, team)
        if ps is None:
            return None
        if ps == this_score:
            continue
        return prev
    return None


def _find_clean_next(plays_sorted, i, team):
    for j in range(i + 1, len(plays_sorted)):
        nxt = plays_sorted[j]
        if _is_pat_like(nxt):
            continue
        ns = _read_team_score(nxt, team)
        if ns is None:
            return None
        return nxt
    return None


# Pre-build sorted plays for every game (we'll use them in trigger detection too)
sorted_plays_cache: dict[int, list[dict]] = {}
for gid, plays in plays_by_game.items():
    valid = [p for p in plays
             if p.get("period") is not None
             and p.get("driveNumber") is not None
             and p.get("playNumber") is not None]
    sp = sorted(valid, key=_drv_sort_key)
    # dedup (period, driveNumber, playNumber) keeping min-id
    seen: dict[tuple[int, int, int], dict] = {}
    for p in sp:
        k = _drv_sort_key(p)
        if k in seen:
            challenger_id = int(p.get("id") or 1 << 62)
            existing_id = int(seen[k].get("id") or 1 << 62)
            if challenger_id < existing_id:
                seen[k] = p
        else:
            seen[k] = p
    sorted_plays_cache[gid] = sorted(seen.values(), key=_drv_sort_key)


# Empirical verification per (playType_lower, side, points)
SUBTYPE_TRIAL_RESULTS: dict[str, dict[tuple[str, int], Counter]] = defaultdict(
    lambda: defaultdict(Counter)
)

for gid, sp in sorted_plays_cache.items():
    for i, p in enumerate(sp):
        if not bool(p.get("scoring")):
            continue
        if _is_pat_like(p):
            continue
        if int(p.get("period") or 0) >= 5:
            continue
        pt_l = (p.get("playType") or "").strip().lower()
        if not pt_l:
            continue

        for side in ("offense", "defense"):
            team = p.get(side)
            if not team:
                continue
            ts = _read_team_score(p, team)
            if ts is None:
                continue
            prev = _find_clean_prev(sp, i, team, ts)
            if prev is None:
                continue
            ps = _read_team_score(prev, team)
            nxt = _find_clean_next(sp, i, team)
            if nxt is None:
                continue
            ns = _read_team_score(nxt, team)
            d_pre = ts - ps
            d_next = ns - ts

            for points in (6, 3, 2):
                if d_pre >= points and d_next <= 2:
                    SUBTYPE_TRIAL_RESULTS[pt_l][(side, points)]["VERIFIED_POST"] += 1
                elif d_pre == 0 and d_next >= points:
                    SUBTYPE_TRIAL_RESULTS[pt_l][(side, points)]["VERIFIED_PRE"] += 1


# Pick the best (side, points) per subtype
SCORING_PLAY_REGISTRY: dict[str, dict] = {}
VERIFIED_POST_THRESHOLD = 0.95
KO_RETURN_TD_THRESHOLD = 0.65
MIN_SAMPLES = 20

for pt_l, trials in SUBTYPE_TRIAL_RESULTS.items():
    best_key = None
    best_share = -1.0
    best_n = 0
    for (side, points), c in trials.items():
        n = c["VERIFIED_POST"] + c["VERIFIED_PRE"]
        if n < 1:
            continue
        share = c["VERIFIED_POST"] / n
        if share > best_share:
            best_share = share
            best_key = (side, points)
            best_n = n
    if best_key is None:
        verdict = "FAIL_NO_SAMPLES"
        side = points = None
    else:
        side, points = best_key
        if best_n < MIN_SAMPLES:
            verdict = "BEST_EFFORT_LOW_N"
        elif "kickoff return touchdown" in pt_l:
            verdict = "POST_BEST_EFFORT" if best_share >= KO_RETURN_TD_THRESHOLD else "FAIL"
        else:
            verdict = "POST_PLAY" if best_share >= VERIFIED_POST_THRESHOLD else "FAIL"
    SCORING_PLAY_REGISTRY[pt_l] = {
        "side": side, "points": points,
        "verdict": verdict, "post_share": best_share,
        "n_samples": best_n,
    }

print(f"\nSCORING_PLAY_REGISTRY entries: {len(SCORING_PLAY_REGISTRY)}")
print(f"\n{'playType':<40} {'side':<8} {'pts':>3} {'n':>6}  POST%   verdict")
for pt_l in sorted(SCORING_PLAY_REGISTRY.keys()):
    r = SCORING_PLAY_REGISTRY[pt_l]
    s = r['side'] or '-'
    p = r['points'] if r['points'] is not None else '-'
    n = r['n_samples']
    pct = (100 * r['post_share']) if r['post_share'] >= 0 else float('nan')
    print(f"  {pt_l:<40} {s:<8} {p!s:>3} {n:>6,}  {pct:5.1f}%  {r['verdict']}")


# --- Cell-16-equivalent: scorer + trigger detection ----------------------
class ScoringPlayAmbiguityError(ValueError):
    pass


KO_RETURN_TD_VERIFIED_BY_INVARIANT = 0
KO_RETURN_TD_RAISED_BY_INVARIANT = 0
LOW_N_VERIFIED_BY_INVARIANT = 0
LOW_N_RAISED_BY_INVARIANT = 0
SORT_DEDUP_LOG: list[dict] = []


def _post_play_scores_for_team(plays_sorted, idx, team):
    global KO_RETURN_TD_VERIFIED_BY_INVARIANT, KO_RETURN_TD_RAISED_BY_INVARIANT
    global LOW_N_VERIFIED_BY_INVARIANT, LOW_N_RAISED_BY_INVARIANT
    p = plays_sorted[idx]
    direct = _read_team_score(p, team)

    if not bool(p.get("scoring")):
        return direct
    if _is_pat_like(p):
        return direct

    pt_l = (p.get("playType") or "").strip().lower()
    if not pt_l:
        raise ScoringPlayAmbiguityError(
            f"play_id={p.get('id')} game_id={p.get('gameId')}: empty playType "
            f"with scoring=true"
        )

    entry = SCORING_PLAY_REGISTRY.get(pt_l)
    if entry is None:
        raise ScoringPlayAmbiguityError(
            f"play_id={p.get('id')} game_id={p.get('gameId')}: playType="
            f"{p.get('playType')!r} has scoring=true but is not in the "
            f"empirically-built SCORING_PLAY_REGISTRY"
        )

    if entry["verdict"] == "FAIL" or entry["verdict"] == "FAIL_NO_SAMPLES":
        raise ScoringPlayAmbiguityError(
            f"play_id={p.get('id')} game_id={p.get('gameId')}: playType="
            f"{p.get('playType')!r} verdict={entry['verdict']} "
            f"post_share={entry['post_share']:.3f}"
        )

    if entry["verdict"] in ("POST_BEST_EFFORT", "BEST_EFFORT_LOW_N"):
        nxt = _find_clean_next(plays_sorted, idx, team)
        if nxt is None:
            return direct
        ns = _read_team_score(nxt, team)
        if ns is None or direct is None:
            return direct if direct is not None else ns
        if abs(int(ns) - int(direct)) > 7:
            if entry["verdict"] == "POST_BEST_EFFORT":
                KO_RETURN_TD_RAISED_BY_INVARIANT += 1
            else:
                LOW_N_RAISED_BY_INVARIANT += 1
            raise ScoringPlayAmbiguityError(
                f"play_id={p.get('id')} game_id={p.get('gameId')}: playType="
                f"{p.get('playType')!r} cross-check delta={ns - direct} > 7"
            )
        if entry["verdict"] == "POST_BEST_EFFORT":
            KO_RETURN_TD_VERIFIED_BY_INVARIANT += 1
        else:
            LOW_N_VERIFIED_BY_INVARIANT += 1
        return direct

    return direct


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
        return 0


def detect_triggers_for_game(sp, fav_team, dog_team, game_id):
    if not sp:
        return []
    triggers_seen = set()
    rows = []
    for idx, p in enumerate(sp):
        period = int(p.get("period") or 0)
        if period >= 5:
            continue
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


# --- Run on broken-triggering games --------------------------------------
broken_rows: list[dict] = list(csv.DictReader(open(RESULTS / "trigger_events.csv", encoding="utf-8")))
fav_dog_by_game: dict[int, tuple[str, str]] = {}
for r in broken_rows:
    gid = int(r["game_id"])
    if gid not in fav_dog_by_game:
        fav_dog_by_game[gid] = (r["fav_team"], r["dog_team"])

print(f"\nbroken triggers: {len(broken_rows):,}")
print(f"games triggering in broken run: {len(fav_dog_by_game):,}")

fixed_rows: list[dict] = []
for gid, (fav, dog) in fav_dog_by_game.items():
    sp = sorted_plays_cache.get(gid, [])
    rows = detect_triggers_for_game(sp, fav, dog, gid)
    fixed_rows.extend(rows)

print(f"fixed triggers (broken-game subset): {len(fixed_rows):,}")
print(f"KO Return TD verified by ±2 / raised: "
      f"{KO_RETURN_TD_VERIFIED_BY_INVARIANT:,} / {KO_RETURN_TD_RAISED_BY_INVARIANT:,}")
print(f"LOW_N subtype verified by ±7 / raised: "
      f"{LOW_N_VERIFIED_BY_INVARIANT:,} / {LOW_N_RAISED_BY_INVARIANT:,}")


# --- Diff --------------------------------------------------------------
def cnt(rows, key, key_int=False):
    if key_int:
        return Counter(int(r[key]) for r in rows)
    return Counter(r.get(key) or "" for r in rows)


print("\n" + "=" * 72)
print(f"ROW COUNT broken={len(broken_rows):,}  fixed={len(fixed_rows):,}  "
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


show_dist("fav_deficit", cnt(broken_rows, "fav_deficit", True), cnt(fixed_rows, "fav_deficit", True),
          sort_key=lambda x: int(x))
show_dist("quarter", cnt(broken_rows, "quarter", True), cnt(fixed_rows, "quarter", True),
          sort_key=lambda x: int(x))
show_dist("play_type", cnt(broken_rows, "play_type"), cnt(fixed_rows, "play_type"))


broken_keyed = {(int(r["game_id"]), int(r["fav_deficit"])): r for r in broken_rows}
fixed_keyed = {(int(r["game_id"]), int(r["fav_deficit"])): r for r in fixed_rows}
both = set(broken_keyed) & set(fixed_keyed)
only_broken = set(broken_keyed) - set(fixed_keyed)
only_fixed = set(fixed_keyed) - set(broken_keyed)
print(f"\n(game_id, fav_deficit) overlap:")
print(f"  in both                 : {len(both):,}")
print(f"  only in broken (deleted): {len(only_broken):,}")
print(f"  only in fixed (added)   : {len(only_fixed):,}")

if only_broken:
    print(f"\nFirst 10 'only in broken' (would be lost):")
    for k in sorted(only_broken)[:10]:
        b = broken_keyed[k]
        print(f"  game={k[0]} D={k[1]} pNum={b['play_number']} type={b['play_type']!r:32}  "
              f"fav={b['fav_score_at_trigger']} dog={b['dog_score_at_trigger']} actD={b['actual_deficit_at_trigger']}")

if only_fixed:
    print(f"\nFirst 10 'only in fixed' (would be added):")
    for k in sorted(only_fixed)[:10]:
        f = fixed_keyed[k]
        print(f"  game={k[0]} D={k[1]} pNum={f['play_number']} type={f['play_type']!r:32}  "
              f"fav={f['fav_score_at_trigger']} dog={f['dog_score_at_trigger']} actD={f['actual_deficit_at_trigger']}")
