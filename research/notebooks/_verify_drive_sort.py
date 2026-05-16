"""
Verify: (period, driveNumber, playNumber) gives strict chronological order.
Then re-run the simulated cell-14 fix using this sort.
"""
from __future__ import annotations

import glob
import json
import pathlib
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"

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


# Test: (period, driveNumber, playNumber) tuple uniqueness within game,
# and id monotonicity under this sort.
unique_key_games = 0
key_dup_games = 0
key_dup_examples = []
id_mono_games = 0

for gid, plays in plays_by_game.items():
    valid = [p for p in plays
             if p.get("period") is not None
             and p.get("driveNumber") is not None
             and p.get("playNumber") is not None
             and p.get("id") is not None]
    if not valid:
        continue
    keys = [(int(p["period"]), int(p["driveNumber"]), int(p["playNumber"]))
            for p in valid]
    if len(set(keys)) == len(keys):
        unique_key_games += 1
    else:
        key_dup_games += 1
        if len(key_dup_examples) < 3:
            key_dup_examples.append((gid, keys))

    sp = sorted(valid, key=lambda p: (int(p["period"]), int(p["driveNumber"]),
                                       int(p["playNumber"])))
    ids = [int(p["id"]) for p in sp]
    # ids should be non-decreasing for a chronological sort to be id-monotonic
    if all(ids[i] >= ids[i - 1] for i in range(1, len(ids))):
        id_mono_games += 1

print(f"games total : {len(plays_by_game):,}")
print(f"  (period, driveNumber, playNumber) unique within game : {unique_key_games:,}")
print(f"  (period, driveNumber, playNumber) has duplicates     : {key_dup_games:,}")
print(f"  id monotonically non-decreasing under this sort      : {id_mono_games:,}")

if key_dup_examples:
    print()
    print("Sample games with key duplicates (likely CFBD edge cases):")
    for gid, keys in key_dup_examples:
        c = Counter(keys)
        dups = {k: n for k, n in c.items() if n > 1}
        print(f"  game={gid}  duplicated_keys={list(dups.items())[:5]}")

# Now: re-run the simulation using (period, driveNumber, playNumber) sort,
# AND treat (period, driveNumber, playNumber) as the unique chronological key
# for collision detection.
PAT = {"point after", "two point conversion", "extra point"}

SCORING_REGISTRY = [
    ("rushing touchdown", "offense", 6, "TD_OFFENSIVE"),
    ("passing touchdown", "offense", 6, "TD_OFFENSIVE"),
    ("interception return touchdown", "defense", 6, "TD_DEFENSIVE_RETURN"),
    ("fumble return touchdown",       "defense", 6, "TD_DEFENSIVE_RETURN"),
    ("punt return touchdown",         "defense", 6, "TD_SPECIAL_RETURN"),
    ("blocked punt touchdown",        "defense", 6, "TD_SPECIAL_RETURN"),
    ("blocked field goal touchdown",  "defense", 6, "TD_SPECIAL_RETURN"),
    ("missed field goal return touchdown", "defense", 6, "TD_SPECIAL_RETURN"),
    ("kickoff return touchdown", "offense", 6, "TD_KICKOFF_RETURN"),
    ("field goal good", "offense", 3, "FIELD_GOAL"),
    ("safety", "defense", 2, "SAFETY"),
]


def classify_subtype(playtype_str):
    pt = (playtype_str or "").strip().lower()
    for kw, side, points, label in SCORING_REGISTRY:
        if kw in pt:
            return side, points, label
    return None


def score_for_team(play, team):
    if play.get("offense") == team:
        v = play.get("offenseScore")
        return int(v) if v is not None else None
    if play.get("defense") == team:
        v = play.get("defenseScore")
        return int(v) if v is not None else None
    return None


def find_clean_prev(plays, i, team, this_score):
    skipped_pat = 0
    skipped_paired = 0
    for j in range(i - 1, -1, -1):
        prev = plays[j]
        ptype = (prev.get("playType") or "").strip().lower()
        if any(kw in ptype for kw in PAT):
            skipped_pat += 1
            continue
        ps = score_for_team(prev, team)
        if ps is None:
            return None, "team_mismatch"
        if ps == this_score:
            skipped_paired += 1
            continue
        return prev, "ok"
    return None, "no_prev"


def find_clean_next(plays, i, team):
    for j in range(i + 1, len(plays)):
        nxt = plays[j]
        ptype = (nxt.get("playType") or "").strip().lower()
        if any(kw in ptype for kw in PAT):
            continue
        ns = score_for_team(nxt, team)
        if ns is None:
            return None, "team_mismatch"
        return nxt, "ok"
    return None, "no_next"


per_label = defaultdict(Counter)
skip_reasons = defaultdict(Counter)

for gid, plays in plays_by_game.items():
    valid = [p for p in plays
             if p.get("period") is not None
             and p.get("driveNumber") is not None
             and p.get("playNumber") is not None]
    sp = sorted(valid, key=lambda p: (int(p["period"]),
                                       int(p["driveNumber"]),
                                       int(p["playNumber"])))

    for i, p in enumerate(sp):
        cls = classify_subtype(p.get("playType") or "")
        if cls is None:
            continue
        side, points, label = cls

        if int(p.get("period") or 0) >= 5:
            skip_reasons[label]["ot_excluded"] += 1
            continue

        team = p.get(side)
        if not team:
            skip_reasons[label]["no_team_field"] += 1
            continue
        ts = score_for_team(p, team)
        if ts is None:
            skip_reasons[label]["no_this_score"] += 1
            continue

        prev, reason = find_clean_prev(sp, i, team, ts)
        if prev is None:
            skip_reasons[label][f"no_clean_prev_{reason}"] += 1
            continue
        ps = score_for_team(prev, team)

        nxt, nreason = find_clean_next(sp, i, team)
        if nxt is None:
            skip_reasons[label][f"no_next_{nreason}"] += 1
            continue
        ns = score_for_team(nxt, team)

        d_pre = ts - ps
        d_next = ns - ts

        # POST convention: this play's scoreboard already credits the
        # >= points scored. The next non-PAT play adds at most +1 (PAT)
        # or +2 (2pt) for the same scoring team.
        # PRE convention: this play's scoreboard equals prev's; the post
        # state lives on the next non-PAT play.
        if d_pre >= points and d_next <= 2:
            per_label[label]["VERIFIED_POST"] += 1
        elif d_pre == 0 and d_next >= points:
            per_label[label]["VERIFIED_PRE"] += 1
        elif d_pre >= points and d_next >= points:
            per_label[label]["AMBIGUOUS_DOUBLE_JUMP"] += 1
        else:
            per_label[label]["AMBIGUOUS_OTHER"] += 1


print()
print("=" * 80)
print("Re-simulated cell 14 with (period,driveNumber,playNumber) sort")
print("=" * 80)
for lab in sorted(per_label.keys()):
    c = per_label[lab]
    t = sum(c.values())
    print(f"\n{lab}  verified samples = {t:,}")
    for k in ("VERIFIED_POST", "VERIFIED_PRE",
              "AMBIGUOUS_DOUBLE_JUMP", "AMBIGUOUS_OTHER"):
        n = c[k]
        if t:
            print(f"  {k:<25} {n:>7,}  ({100*n/t:6.2f}%)")

print()
print("=" * 80)
print("Skip-reason audit")
print("=" * 80)
for lab in sorted(per_label.keys()):
    sr = skip_reasons[lab]
    if not sr:
        continue
    total = sum(sr.values())
    print(f"\n{lab}  total skipped = {total:,}")
    for k, n in sr.most_common():
        print(f"  {k:<45} {n:>7,}")
