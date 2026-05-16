"""
Verify: int(play["id"]) is strictly monotonically increasing in
chronological order within a game.

If true, the right global sort is simply by id, and the playNumber-based
sort schemes we tried are all moot. The "(period, playNumber)" tuple is
not a unique key by design -- playNumber is per-drive in CFBD.
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

# Test 1: within each game, sort by id ascending; check that period
# is non-decreasing (chronologically meaningful).
games_with_period_violation = 0
violation_examples = []
games_with_dup_id = 0

# Test 2: id should be unique within game
all_clean = 0

for gid, plays in plays_by_game.items():
    valid = [p for p in plays if p.get("id") is not None
             and p.get("period") is not None and p.get("playNumber") is not None]
    if not valid:
        continue
    sorted_by_id = sorted(valid, key=lambda p: int(p["id"]))
    ids = [int(p["id"]) for p in sorted_by_id]
    if len(set(ids)) != len(ids):
        games_with_dup_id += 1
    periods = [int(p["period"]) for p in sorted_by_id]
    if any(periods[i] < periods[i - 1] for i in range(1, len(periods))):
        games_with_period_violation += 1
        if len(violation_examples) < 3:
            for i in range(1, len(periods)):
                if periods[i] < periods[i - 1]:
                    violation_examples.append((
                        gid, sorted_by_id[i - 1], sorted_by_id[i]
                    ))
                    break
    else:
        all_clean += 1

print(f"games checked                          : {len(plays_by_game):,}")
print(f"games with id sort -> period strictly non-decreasing : {all_clean:,}")
print(f"games with id sort -> some period dropped (violation): {games_with_period_violation:,}")
print(f"games with duplicate id values         : {games_with_dup_id:,}")
print()
if violation_examples:
    print("First 3 period-violation examples:")
    for gid, p_prev, p_curr in violation_examples:
        print(f"  game={gid}")
        print(f"    prev id={p_prev['id']} period={p_prev['period']} playNumber={p_prev['playNumber']} type={p_prev['playType']}")
        print(f"    curr id={p_curr['id']} period={p_curr['period']} playNumber={p_curr['playNumber']} type={p_curr['playType']}")

# Test 3: within a single drive, playNumber should be monotonic (verifies
# that playNumber is per-drive)
print()
print("Spot-check: within driveId, playNumber should be monotonic.")
sample_game = next(iter(plays_by_game))
plays = plays_by_game[sample_game]
from collections import defaultdict
drives = defaultdict(list)
for p in plays:
    did = p.get("driveId")
    if did is not None and p.get("playNumber") is not None:
        drives[did].append(p)
for did in list(drives.keys())[:3]:
    ds = sorted(drives[did], key=lambda p: int(p["id"]))
    pns = [p["playNumber"] for p in ds]
    print(f"  driveId={did} playNumbers in id order: {pns[:15]}{'...' if len(pns)>15 else ''}")
