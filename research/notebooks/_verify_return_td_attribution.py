"""
Verify hypothesis: for return TDs (INT/Fumble/Punt/Blocked Punt/Blocked FG/
Missed FG return), the play's `offense` field is the team that LOST
possession during the play, not the team that scored the TD. So treating
`offense` as the scoring team gives diff=0 (PRE bucket) -- not because
of a score-state convention, but because we're tracking the wrong team.
"""
from __future__ import annotations

import glob
import json
import pathlib
from collections import Counter

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


def sk(p):
    return (int(p["period"]), int(p["playNumber"]),
            int(p["id"]) if p.get("id") is not None else 0)


PAT = {"point after", "two point conversion", "extra point"}
RETURN_TD_KWS = {
    "interception return touchdown",
    "fumble return touchdown",
    "punt return touchdown",
    "blocked punt touchdown",
    "blocked field goal touchdown",
    "missed field goal return touchdown",
}


def find_prev(plays, i):
    for j in range(i - 1, -1, -1):
        pt = (plays[j].get("playType") or "").strip().lower()
        if any(kw in pt for kw in PAT):
            continue
        return plays[j]
    return None


def score_for_team(play, team):
    if play.get("offense") == team:
        v = play.get("offenseScore")
        return int(v) if v is not None else None
    if play.get("defense") == team:
        v = play.get("defenseScore")
        return int(v) if v is not None else None
    return None


# For each return TD, classify under TWO attribution rules:
#   A) scoring_team = play.offense  (cell 14 current rule)
#   B) scoring_team = play.defense  (rule under hypothesis: defense returned)
buckets_A = Counter()
buckets_B = Counter()
samples_B = []

for gid, plays in plays_by_game.items():
    sp = sorted([p for p in plays if p.get("playNumber") is not None
                 and p.get("period") is not None], key=sk)
    for i, p in enumerate(sp):
        pt = (p.get("playType") or "").strip().lower()
        if not any(kw in pt for kw in RETURN_TD_KWS):
            continue
        prev = find_prev(sp, i)
        if prev is None:
            continue
        for label, team_field in [("A_offense", "offense"), ("B_defense", "defense")]:
            team = p.get(team_field)
            if not team:
                continue
            ps = score_for_team(prev, team)
            ts = score_for_team(p, team)
            if ps is None or ts is None:
                continue
            d = ts - ps
            if d >= 6:
                bucket = "POST"
            elif d == 0:
                bucket = "PRE"
            else:
                bucket = "OTHER"
            if label == "A_offense":
                buckets_A[bucket] += 1
            else:
                buckets_B[bucket] += 1
                if len(samples_B) < 5 and bucket == "POST":
                    samples_B.append((gid, p, prev, ps, ts, d))


def fmt(n, c):
    t = sum(c.values())
    if t == 0:
        return f"{n}: total=0"
    return (f"{n}: total={t:,}  "
            f"POST={c['POST']:,} ({100*c['POST']/t:5.2f}%)  "
            f"PRE={c['PRE']:,} ({100*c['PRE']/t:5.2f}%)  "
            f"OTHER={c['OTHER']:,} ({100*c['OTHER']/t:5.2f}%)")


print("Return TDs reclassified under two team-attribution rules:")
print(" ", fmt("A) scoring_team = play.offense (cell 14 current)", buckets_A))
print(" ", fmt("B) scoring_team = play.defense (proposed: returner = defense)", buckets_B))
print()
print("First 5 POST samples under rule B:")
for gid, p, prev, ps, ts, d in samples_B:
    print(f"  game={gid} play_id={p.get('id')} type={p.get('playType')}")
    print(f"    offense={p.get('offense')} defense={p.get('defense')}  "
          f"offScore={p.get('offenseScore')} defScore={p.get('defenseScore')}")
    print(f"    prev offense={prev.get('offense')} defense={prev.get('defense')}  "
          f"offScore={prev.get('offenseScore')} defScore={prev.get('defenseScore')}")
    print(f"    rule-B team prev_score={ps} this_score={ts} diff={d}")
    print(f"    playText: {(p.get('playText') or '')[:140]}")
