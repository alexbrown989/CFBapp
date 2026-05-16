"""
If the OTHER bucket is a duplicate-playNumber artifact (not a 3rd score
state convention), then sorting plays by (period, playNumber, id) instead
of playNumber alone -- i.e., disambiguating same-playNumber rows -- should
collapse OTHER to near zero for all three play types.

Also: if cell 14's `find_prev_non_pat` is the only weak link, an even
stricter check is to require prev's period == trigger's period (or the
immediately preceding period) AND prev.playNumber < trigger.playNumber.
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

PAT_KEYWORDS = {"point after", "two point conversion", "extra point"}
TARGETS = [
    ("TOUCHDOWN", 6, "offense", {"touchdown"}),
    ("FIELD_GOAL", 3, "offense", {"field goal good"}),
    ("SAFETY", 2, "defense", {"safety"}),
]


def score_for_team(play: dict, team: str) -> int | None:
    if play.get("offense") == team:
        v = play.get("offenseScore")
        return int(v) if v is not None else None
    if play.get("defense") == team:
        v = play.get("defenseScore")
        return int(v) if v is not None else None
    return None


def classify_with_sort(sort_key_name: str, sort_key):
    overall = {}
    for label, points, side, kws in TARGETS:
        b = Counter()
        for gid, plays in plays_by_game.items():
            sorted_plays = sorted(
                [p for p in plays if p.get("playNumber") is not None
                 and p.get("period") is not None],
                key=sort_key,
            )
            for i, p in enumerate(sorted_plays):
                ptype = (p.get("playType") or "").strip().lower()
                if not any(kw in ptype for kw in kws):
                    continue
                team = p.get("offense") if side == "offense" else p.get("defense")
                if not team:
                    continue
                # Walk back to first non-PAT
                prev = None
                for j in range(i - 1, -1, -1):
                    if any(kw in (sorted_plays[j].get("playType") or "").strip().lower()
                           for kw in PAT_KEYWORDS):
                        continue
                    prev = sorted_plays[j]
                    break
                if prev is None:
                    continue
                ps = score_for_team(prev, team)
                ts = score_for_team(p, team)
                if ps is None or ts is None:
                    continue
                d = ts - ps
                if d >= points:
                    b["POST"] += 1
                elif d == 0:
                    b["PRE"] += 1
                else:
                    b["OTHER"] += 1
        total = sum(b.values())
        overall[label] = (b, total)
    return overall


def fmt(name, results):
    print(f"\n--- sort = {name} ---")
    for label, (b, total) in results.items():
        post = b['POST']; pre = b['PRE']; other = b['OTHER']
        if total == 0:
            print(f"  {label:<10}: total=0")
            continue
        print(f"  {label:<10}: total={total:<6,} "
              f"POST={post:>6,} ({100*post/total:5.2f}%)  "
              f"PRE={pre:>5,} ({100*pre/total:5.2f}%)  "
              f"OTHER={other:>5,} ({100*other/total:5.2f}%)")


# Sort A (cell 14 baseline): playNumber only
res_A = classify_with_sort(
    "playNumber only (cell 14 current)",
    lambda p: int(p["playNumber"]),
)
fmt("playNumber only (cell 14 current)", res_A)

# Sort B: (period, playNumber)
res_B = classify_with_sort(
    "(period, playNumber)",
    lambda p: (int(p["period"]), int(p["playNumber"])),
)
fmt("(period, playNumber)", res_B)

# Sort C: (period, playNumber, id)
res_C = classify_with_sort(
    "(period, playNumber, id)",
    lambda p: (int(p["period"]), int(p["playNumber"]),
               int(p["id"]) if p.get("id") is not None else 0),
)
fmt("(period, playNumber, id)", res_C)
