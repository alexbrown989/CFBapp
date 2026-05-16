"""Find the missing scoring play in game 401301021 between i=40 and i=59."""
from __future__ import annotations
import glob
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"

GID = 401301021
plays = []
for fp in sorted(glob.glob(str(CACHE / "cfbd__plays__*.json"))):
    with open(fp, encoding="utf-8") as f:
        ps = json.load(f)
    if not isinstance(ps, list):
        continue
    for p in ps:
        if p.get("gameId") == GID:
            plays.append(p)

sp = sorted(
    [p for p in plays if p.get("playNumber") is not None
     and p.get("driveNumber") is not None and p.get("period") is not None],
    key=lambda p: (int(p["period"]), int(p["driveNumber"]), int(p["playNumber"])),
)

# Print everything with scoring=true OR a meaningful score change between
# index 35 and index 65 (zoom in on the missing TD).
print("Plays from i=35 to i=65 (chronological / fixed sort):")
for i in range(35, 66):
    if i >= len(sp):
        break
    p = sp[i]
    print(f"  i={i:3d} per={p['period']} drv={p['driveNumber']:>2} pNum={p['playNumber']:>3} "
          f"id={p.get('id')} type={(p.get('playType') or '')[:34]:34}  "
          f"off={p.get('offense'):<10} def={p.get('defense'):<10} "
          f"oS={p.get('offenseScore')} dS={p.get('defenseScore')}  "
          f"scoring={p.get('scoring')}")

# All scoring=true plays in this game
print("\nAll scoring=true plays in this game (chronological):")
for i, p in enumerate(sp):
    if p.get("scoring") is True:
        print(f"  i={i:3d} type={(p.get('playType') or '')[:40]:40}  "
              f"off={p.get('offense')} oS={p.get('offenseScore')} dS={p.get('defenseScore')}  "
              f"id={p.get('id')}")
