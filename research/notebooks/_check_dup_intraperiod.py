"""
How common are duplicate (period, playNumber) keys (flavor B) within the
same game? Show distribution and a couple of concrete examples.
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

# Distribution of (period, playNumber) collision counts
pp_collision_dist = Counter()
games_with_collision = 0
games_total = 0
for gid, plays in plays_by_game.items():
    games_total += 1
    pp = Counter()
    for p in plays:
        if p.get("playNumber") is None or p.get("period") is None:
            continue
        pp[(int(p["period"]), int(p["playNumber"]))] += 1
    has_col = False
    for k, n in pp.items():
        if n > 1:
            pp_collision_dist[n] += 1
            has_col = True
    if has_col:
        games_with_collision += 1

print(f"games total                : {games_total:,}")
print(f"games with any (period,playNumber) collision : {games_with_collision:,}")
print(f"distribution of collision sizes (collisions, not games):")
for n, c in sorted(pp_collision_dist.items()):
    print(f"  {n} rows share a (period,playNumber) : {c:,} such collisions")

# Sample one collision and dump the rows
print()
print("Sample collision: game_id=401416616 (Akron @ Ohio 2022 wk 6)")
g = plays_by_game.get(401416616, [])
g_pp = Counter((int(p["period"]), int(p["playNumber"])) for p in g
               if p.get("period") is not None and p.get("playNumber") is not None)
sample_keys = [k for k, n in g_pp.most_common(3)]
for key in sample_keys:
    period, pn = key
    rows = [p for p in g if int(p.get("period") or -1) == period
            and int(p.get("playNumber") or -1) == pn]
    print(f"\n  ({period},{pn}) has {len(rows)} rows:")
    for p in rows[:6]:
        print(f"    id={p.get('id')}  type={p.get('playType')!r:30}  "
              f"off={p.get('offense')!r:20} def={p.get('defense')!r:20}  "
              f"offS={p.get('offenseScore')} defS={p.get('defenseScore')}")
