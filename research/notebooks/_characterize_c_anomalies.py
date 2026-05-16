"""Characterize the 749 category (c) anomalies surfaced by
_verify_chrono_key_composite.py. Re-runs the same classification but
groups (c) cases by structural pattern so we can decide whether the
composite chrono_key is good-enough or needs further refinement.

Specifically:
  - Negative play.id count (CFBD placeholder rows)
  - Distribution by A.playType and B.playType
  - Distribution by clock value (15:00 / 0:00 / other)
  - Drive-number gap distribution (B.drive - A.drive)
  - Same-game concentration (are anomalies clustered in a few "bad" games?)
  - For triggers specifically: how many trigger rows would be affected
    if the (c) anomalies leaked into plays_before? (Bounds the actual
    impact on N02a/b/c.)
"""
from __future__ import annotations
import json
import pathlib
from collections import Counter, defaultdict

import pandas as pd

CACHE_DIR = pathlib.Path(
    r"C:\Users\Alexander\Documents\CFB\CFBapp\research\data\cache"
)
TRIGGER_EVENTS = pathlib.Path(
    r"C:\Users\Alexander\Documents\CFB\CFBapp\research\results\trigger_events.csv"
)


def clock_key(p: dict) -> tuple[int, int]:
    period = int(p.get("period") or 0)
    clock = p.get("clock") or {}
    m, s = clock.get("minutes"), clock.get("seconds")
    elapsed = 900 - 60 * int(m) - int(s) if m is not None and s is not None else 0
    return (period, elapsed)


def chrono_key(p: dict) -> tuple[int, int, int, int]:
    ck = clock_key(p)
    return (ck[0], ck[1], int(p.get("driveNumber") or 0), int(p.get("playNumber") or 0))


# Load all /plays cache shards
all_plays_by_game: dict[int, list[dict]] = defaultdict(list)
for cf in sorted(CACHE_DIR.glob("cfbd__plays__*.json")):
    data = json.loads(cf.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        continue
    for p in data:
        gid = p.get("gameId")
        if gid is not None:
            all_plays_by_game[int(gid)].append(p)

# Collect every (c) case
c_cases: list[tuple[int, dict, dict, str]] = []
for gid, plays in all_plays_by_game.items():
    if not plays:
        continue
    chrono_sorted = sorted(plays, key=chrono_key)
    id_sorted = sorted(plays, key=lambda x: int(x.get("id") or x.get("playId") or 0))
    if [p.get("id") for p in chrono_sorted] == [p.get("id") for p in id_sorted]:
        continue
    drive_min_pn: dict[int, int] = {}
    drive_max_pn: dict[int, int] = {}
    for p in plays:
        dn, pn = p.get("driveNumber"), p.get("playNumber")
        if dn is None or pn is None:
            continue
        di, pi = int(dn), int(pn)
        drive_min_pn[di] = min(drive_min_pn.get(di, pi), pi)
        drive_max_pn[di] = max(drive_max_pn.get(di, pi), pi)
    for i in range(len(chrono_sorted) - 1):
        A = chrono_sorted[i]
        B = chrono_sorted[i + 1]
        try:
            id_A = int(A.get("id") or A.get("playId") or 0)
            id_B = int(B.get("id") or B.get("playId") or 0)
        except (ValueError, TypeError):
            c_cases.append((gid, A, B, "non_integer_play_id"))
            continue
        if id_A <= id_B:
            continue
        ck_A, ck_B = clock_key(A), clock_key(B)
        if ck_A < ck_B:
            continue
        if ck_A > ck_B:
            c_cases.append((gid, A, B, "primary_disagrees_with_clock_IMPOSSIBLE"))
            continue
        A_dn, B_dn = int(A.get("driveNumber") or 0), int(B.get("driveNumber") or 0)
        A_pn, B_pn = int(A.get("playNumber") or 0), int(B.get("playNumber") or 0)
        if A_dn == B_dn:
            continue
        a_is_last = (A_pn == drive_max_pn.get(A_dn))
        b_is_first = (B_pn == drive_min_pn.get(B_dn))
        if a_is_last and b_is_first:
            continue
        c_cases.append(
            (gid, A, B,
             f"same_clock_diff_drive_not_transition "
             f"(A_is_last={a_is_last}, B_is_first={b_is_first})")
        )

print(f"category (c) cases: {len(c_cases)}\n")

# 1. Negative play IDs
neg_id_cases = [c for c in c_cases if (int(c[1].get("id") or 0) < 0 or int(c[2].get("id") or 0) < 0)]
print(f"1. Negative play.id (CFBD placeholder rows): {len(neg_id_cases)} / {len(c_cases)}")

# 2. PlayType distribution
type_pair_counter: Counter = Counter()
for gid, A, B, reason in c_cases:
    type_pair_counter[(A.get("playType"), B.get("playType"))] += 1
print(f"\n2. Top 15 (A.playType, B.playType) combinations:")
for (at, bt), n in type_pair_counter.most_common(15):
    print(f"   {n:>5}  A={at!r:30}  B={bt!r}")

# 3. Clock distribution (period_seconds_elapsed)
clock_counter: Counter = Counter()
for gid, A, B, reason in c_cases:
    ck = clock_key(A)
    bucket = "0:00 (end of period)" if ck[1] == 900 else \
             "15:00 (start of period)" if ck[1] == 0 else "other"
    clock_counter[bucket] += 1
print(f"\n3. Clock-bucket distribution:")
for bucket, n in clock_counter.most_common():
    print(f"   {n:>5}  {bucket}")

# 4. Drive gap distribution
gap_counter: Counter = Counter()
for gid, A, B, reason in c_cases:
    gap = int(B.get("driveNumber") or 0) - int(A.get("driveNumber") or 0)
    bucket = ("1 (adjacent drives)" if gap == 1 else
              "2-5" if 2 <= gap <= 5 else
              "6-10" if 6 <= gap <= 10 else
              ">10" if gap > 10 else f"non-positive ({gap})")
    gap_counter[bucket] += 1
print(f"\n4. Drive-number gap (B.drive - A.drive) distribution:")
for bucket, n in sorted(gap_counter.items(), key=lambda x: -x[1]):
    print(f"   {n:>5}  {bucket}")

# 5. Same-game concentration
games_with_c: Counter = Counter()
for gid, _, _, _ in c_cases:
    games_with_c[gid] += 1
print(f"\n5. Per-game (c) anomaly concentration:")
print(f"   unique games with (c) anomalies: {len(games_with_c)}")
print(f"   median anomalies per affected game: {pd.Series(list(games_with_c.values())).median()}")
print(f"   max anomalies in one game: {max(games_with_c.values())}")
print(f"   top 5 'anomaly-rich' games:")
for gid, n in games_with_c.most_common(5):
    print(f"     game_id={gid}  {n} anomalies")

# 6. Trigger-impact bound: how many triggers share an exact (period, elapsed)
#    with at least one (c) anomaly? (Only those triggers could be affected.)
te = pd.read_csv(TRIGGER_EVENTS)
te["clock_elapsed"] = 900 - te["clock_seconds_in_period_total"].astype(int)
trig_keys = set(zip(te["game_id"].astype(int), te["quarter"].astype(int), te["clock_elapsed"].astype(int)))

c_clock_keys = set()
for gid, A, _, _ in c_cases:
    ck = clock_key(A)
    c_clock_keys.add((gid, ck[0], ck[1]))

triggers_potentially_affected = trig_keys & c_clock_keys
print(f"\n6. Trigger-impact upper bound:")
print(f"   total trigger rows: {len(te):,}")
print(f"   unique (game, period, elapsed) trigger keys: {len(trig_keys):,}")
print(f"   triggers sharing a clock with a (c) anomaly: {len(triggers_potentially_affected):,}")
print(f"   percentage potentially affected: {100.0 * len(triggers_potentially_affected) / len(te):.3f}%")

# 7. Pattern detail: when A_is_last AND NOT B_is_first, what's going on?
ail_nbf = [c for c in c_cases if "A_is_last=True, B_is_first=False" in c[3]]
nail_bf = [c for c in c_cases if "A_is_last=False, B_is_first=True" in c[3]]
nail_nbf = [c for c in c_cases if "A_is_last=False, B_is_first=False" in c[3]]
print(f"\n7. (c) sub-pattern by transition-shape:")
print(f"   A is last play of its drive, B is NOT first play of its drive:  {len(ail_nbf)}")
print(f"     -> B's drive has playNumbers < B.pn elsewhere; B isn't drive entry")
print(f"   A is NOT last play of its drive, B IS first play of its drive:  {len(nail_bf)}")
print(f"     -> A's drive continues past A.pn but at LATER clock(s); A isn't drive exit")
print(f"   Neither A_is_last nor B_is_first:                                 {len(nail_nbf)}")
print(f"     -> both plays are interior to their respective drives")
