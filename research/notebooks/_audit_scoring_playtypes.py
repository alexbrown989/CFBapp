"""Audit: every distinct playType where scoring=true across the full
cached corpus. Report counts and a representative example. Specifically
flag playTypes NOT in the proposed SCORING_REGISTRY -- those are the
gaps.
"""
from __future__ import annotations
import glob
import json
import pathlib
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"

REGISTRY_KWS = {
    "rushing touchdown", "passing touchdown",
    "interception return touchdown", "fumble return touchdown",
    "punt return touchdown", "blocked punt touchdown",
    "blocked field goal touchdown", "missed field goal return touchdown",
    "kickoff return touchdown",
    "field goal good", "safety",
}


def in_registry(pt):
    pt_l = (pt or "").strip().lower()
    return any(kw in pt_l for kw in REGISTRY_KWS)


type_counts = Counter()
type_examples = {}

for fp in sorted(glob.glob(str(CACHE / "cfbd__plays__*.json"))):
    with open(fp, encoding="utf-8") as f:
        plays = json.load(f)
    if not isinstance(plays, list):
        continue
    for p in plays:
        if p.get("scoring") is True:
            pt = p.get("playType") or "<None>"
            type_counts[pt] += 1
            if pt not in type_examples:
                type_examples[pt] = p

print(f"distinct playTypes with scoring=true: {len(type_counts)}\n")
print(f"{'playType':<40} {'count':>8}  {'in_registry'}")
for pt, n in type_counts.most_common():
    flag = "YES" if in_registry(pt) else "**NO**"
    print(f"  {pt:<40} {n:>8,}  {flag}")

print("\n--- Representative samples for playTypes NOT in registry ---")
for pt, n in type_counts.most_common():
    if not in_registry(pt):
        ex = type_examples[pt]
        print(f"\n{pt}  (n={n:,})  example play_id={ex.get('id')} game_id={ex.get('gameId')}")
        print(f"  off={ex.get('offense')} def={ex.get('defense')}  "
              f"oS={ex.get('offenseScore')} dS={ex.get('defenseScore')}")
        print(f"  playText: {(ex.get('playText') or '')[:200]}")
