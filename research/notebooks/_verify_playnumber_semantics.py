"""Verify whether CFBD's play.playNumber is globally ordered within a game or per-drive.

If global: max(playNumber) over a game roughly equals total play count.
If per-drive: playNumber resets to 1 at the start of each drive, so values
              are bounded by the longest drive's play count (~20 ish) and
              repeat heavily across drives.

Sample game: 400869254 (Wyoming @ New Mexico, 2016 wk 13), the first sample
from the D12 accounting investigation.
"""
from __future__ import annotations
import json, pathlib, hashlib
from collections import Counter

CACHE_DIR = pathlib.Path(
    r"C:\Users\Alexander\Documents\CFB\CFBapp\research\data\cache"
)

def _params_hash(params: dict) -> str:
    return hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]

# 2016 week 13 regular season
params = {"year": 2016, "seasonType": "regular", "week": 13, "classification": "fbs"}
key = CACHE_DIR / f"cfbd__plays__{_params_hash(params)}.json"
assert key.exists(), key
plays_all = json.loads(key.read_text(encoding="utf-8"))

# Find plays for our target game
TARGET_GAME = 400869254
plays = [p for p in plays_all if p.get("gameId") == TARGET_GAME]
print(f"target game {TARGET_GAME}: {len(plays)} plays in cache")

# Print the available numeric/order fields on a sample play
sample = plays[0]
print(f"\nsample play (index 0) keys (sorted):")
for k in sorted(sample.keys()):
    v = sample[k]
    if isinstance(v, (str, int, float, bool, type(None))):
        print(f"  {k:<22} = {v!r}")
    else:
        print(f"  {k:<22} = <complex: {type(v).__name__}>")

# Distribution of playNumber values within the game
pn_counts = Counter(p.get("playNumber") for p in plays)
print(f"\nplayNumber distribution in game (top 20 most common):")
for pn, ct in sorted(pn_counts.items(), key=lambda x: -x[1])[:20]:
    print(f"  playNumber={pn!r:<8}  count={ct}")
print(f"unique playNumber values: {len(pn_counts)}")
print(f"max playNumber: {max(pn for pn in pn_counts if pn is not None)}")
print(f"total plays in game: {len(plays)}")

# Distribution of (driveNumber, playNumber) — should be unique pairs if playNumber resets per drive
dpn_counts = Counter((p.get("driveNumber"), p.get("playNumber")) for p in plays)
dups = [(k, v) for k, v in dpn_counts.items() if v > 1]
print(f"\n(driveNumber, playNumber) duplicate pairs: {len(dups)} (should be 0 if both fields combined uniquely identify a play)")
for k, v in dups[:5]:
    print(f"  {k}: count={v}")

# Show first 3 plays of each of drives 1, 5, 9, 21 — if playNumber resets per drive,
# we expect drive 1's playNumber values to be 1,2,3,... drive 5's also 1,2,3,..., etc.
for target_drive in [1, 5, 9, 21]:
    drive_plays = [p for p in plays if p.get("driveNumber") == target_drive]
    drive_plays.sort(key=lambda p: (p.get("playNumber") if p.get("playNumber") is not None else 1e9))
    print(f"\ndrive {target_drive}: {len(drive_plays)} plays")
    for p in drive_plays[:3]:
        print(
            f"  playNumber={p.get('playNumber')!r}  period={p.get('period')!r}  "
            f"clock={p.get('clock')}  playType={p.get('playType')!r}  "
            f"offense={p.get('offense')!r}"
        )

# Look for a globally-ordered field. Candidates: 'id' (playId), some kind of
# sequence number, gamePlayNumber, etc.
print(f"\nAll integer-looking fields seen on plays (and their min/max in this game):")
int_fields: dict[str, list[int]] = {}
for p in plays:
    for k, v in p.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            int_fields.setdefault(k, []).append(v)
for k in sorted(int_fields):
    vs = [v for v in int_fields[k] if v is not None]
    if not vs:
        continue
    print(f"  {k:<22} min={min(vs)}  max={max(vs)}  unique={len(set(vs))}  n={len(vs)}")
