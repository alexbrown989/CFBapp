"""Why did game 401301021 produce D=10 and D=14 triggers in broken run
but none in the fixed run? Walk both runs by hand for this game."""
from __future__ import annotations
import csv
import glob
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"

GID = 401301021

# Load plays for this game
plays = []
for fp in sorted(glob.glob(str(CACHE / "cfbd__plays__*.json"))):
    with open(fp, encoding="utf-8") as f:
        ps = json.load(f)
    if not isinstance(ps, list):
        continue
    for p in ps:
        if p.get("gameId") == GID:
            plays.append(p)

print(f"plays for game {GID}: {len(plays)}")

# Get fav/dog from broken trigger CSV
fav = dog = None
for r in csv.DictReader(open(ROOT / "results" / "trigger_events.csv", encoding="utf-8")):
    if int(r["game_id"]) == GID:
        fav, dog = r["fav_team"], r["dog_team"]
        break
print(f"fav={fav}  dog={dog}")

# Sort under broken rule (playNumber only) and dump score progression
plays_pnum_sorted = sorted(
    [p for p in plays if p.get("playNumber") is not None],
    key=lambda p: int(p["playNumber"]),
)


def _sf(p, team):
    if p.get("offense") == team:
        return p.get("offenseScore")
    if p.get("defense") == team:
        return p.get("defenseScore")
    return None


print("\n--- BROKEN sort (playNumber only) -- first 30 plays ---")
prev_fav = prev_dog = None
for i, p in enumerate(plays_pnum_sorted[:30]):
    fs = _sf(p, fav)
    ds = _sf(p, dog)
    df = (int(ds) - int(fs)) if fs is not None and ds is not None else None
    print(f"  i={i:3d} pNum={p.get('playNumber'):>3} per={p.get('period')} drv={p.get('driveNumber'):>2} "
          f"id={p.get('id'):<22} type={(p.get('playType') or '')[:30]:30}  "
          f"fav({fav})={fs} dog({dog})={ds}  deficit={df}")


print("\n--- FIXED sort (period, driveNumber, playNumber) -- first 30 plays ---")
plays_drv_sorted = sorted(
    [p for p in plays if p.get("playNumber") is not None
     and p.get("driveNumber") is not None and p.get("period") is not None],
    key=lambda p: (int(p["period"]), int(p["driveNumber"]), int(p["playNumber"])),
)
for i, p in enumerate(plays_drv_sorted[:30]):
    fs = _sf(p, fav)
    ds = _sf(p, dog)
    df = (int(ds) - int(fs)) if fs is not None and ds is not None else None
    print(f"  i={i:3d} pNum={p.get('playNumber'):>3} per={p.get('period')} drv={p.get('driveNumber'):>2} "
          f"id={p.get('id'):<22} type={(p.get('playType') or '')[:30]:30}  "
          f"fav={fs} dog={ds}  deficit={df}")


# Walk full FIXED-sort game and trace deficit history through period<5
max_deficit_fixed = -1000
prev_d = None
print("\nFIXED-sort: chronological deficit history (changes only, regulation):")
for i, p in enumerate(plays_drv_sorted):
    if int(p.get("period") or 0) >= 5:
        continue
    fs = _sf(p, fav)
    ds = _sf(p, dog)
    if fs is None or ds is None:
        continue
    d = int(ds) - int(fs)
    if d != prev_d:
        print(f"  i={i:3d} per={p.get('period')} drv={p.get('driveNumber'):>2} pNum={p.get('playNumber'):>3} "
              f"type={(p.get('playType') or '')[:32]:32}  fav={fs} dog={ds} deficit={d:>+3d}")
        prev_d = d
    if d > max_deficit_fixed:
        max_deficit_fixed = d
print(f"\nFIXED-sort: max fav deficit at any regulation play = {max_deficit_fixed}")


# Walk full BROKEN-sort game and see max deficit ever
max_deficit_broken = -1000
broken_first_d10 = None
broken_first_d14 = None
for i, p in enumerate(plays_pnum_sorted):
    fs = _sf(p, fav)
    ds = _sf(p, dog)
    if fs is not None and ds is not None:
        d = int(ds) - int(fs)
        if d > max_deficit_broken:
            max_deficit_broken = d
        if broken_first_d10 is None and d >= 10:
            broken_first_d10 = (i, p, d)
        if broken_first_d14 is None and d >= 14:
            broken_first_d14 = (i, p, d)
print(f"BROKEN-sort: max fav deficit at any play = {max_deficit_broken}")
if broken_first_d10:
    i, p, d = broken_first_d10
    print(f"BROKEN: first D>=10 at i={i} pNum={p['playNumber']} per={p['period']} type={p['playType']} deficit={d}")
if broken_first_d14:
    i, p, d = broken_first_d14
    print(f"BROKEN: first D>=14 at i={i} pNum={p['playNumber']} per={p['period']} type={p['playType']} deficit={d}")
