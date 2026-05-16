"""Pre-flight verification before applying the (driveNumber, playNumber)
lexicographic plays_before filter to fix the R3 lookahead-bias bug.

Requirement: within each game, the sequence `driveNumber` must be
monotonically non-decreasing when plays are sorted in true chronological
order. We can't trust playNumber as the chronological key (it resets per
drive), so we use the clock fields as the chronological ground truth:

    chrono_key = (period ASC, 900 - 60*clock.minutes - clock.seconds ASC)
                 = (period, period_seconds_elapsed)

This is the actual time-axis of the game — period 1 starts at 15:00 and
ends at 0:00; period 2 picks up at 15:00 again; etc. Clock counts DOWN
within a period.

Tests per game (all must pass):
    T1. driveNumber is non-decreasing when plays sorted by chrono_key.
    T2. driveNumber values form a contiguous sequence {1, 2, ..., max}
        with no gaps. (A gap would mean we lose plays when sorting by
        driveNumber, which would silently corrupt the lex filter.)
    T3. Within each drive, playNumber is non-decreasing AND starts at 1.
        Confirms playNumber resets per drive and is per-drive-monotonic.
    T4. Sorting all plays by (driveNumber, playNumber) lex produces the
        same ordering as sorting by chrono_key. This is the operational
        test: the proposed filter is correct iff this holds.

Sample: 50 random games (random.seed(42)) from the unique game_ids in
trigger_events.csv. Cache-only.

Output: pass/fail per test, with the first few violating games' details
if any fail. If ALL 50 games pass all 4 tests, the (driveNumber,
playNumber) lex filter is safe to apply.
"""
from __future__ import annotations
import csv
import hashlib
import json
import pathlib
import random
from collections import defaultdict
from typing import Any

import pandas as pd

CACHE_DIR = pathlib.Path(
    r"C:\Users\Alexander\Documents\CFB\CFBapp\research\data\cache"
)
TRIGGER_EVENTS = pathlib.Path(
    r"C:\Users\Alexander\Documents\CFB\CFBapp\research\results\trigger_events.csv"
)

def _params_hash(params: dict) -> str:
    return hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]

def cfbd_get_cache_only(endpoint: str, **params: Any) -> Any:
    key = CACHE_DIR / f"cfbd__{endpoint.strip('/').replace('/', '_')}__{_params_hash(params)}.json"
    assert key.exists(), f"cache miss for {endpoint} {params}"
    return json.loads(key.read_text(encoding="utf-8"))


def play_id_key(p: dict) -> int:
    """CFBD's globally-unique numeric play ID. Established as chronological
    ground truth by T0 below (clock fields monotonic when sorted by play.id)."""
    pid = p.get("id") or p.get("playId")
    if pid is None:
        return -1
    return int(pid)


def clock_key(p: dict) -> tuple[int, int]:
    """(period, period_seconds_elapsed). Clock counts DOWN within a period
    so chronological order within a period is ascending period_seconds_elapsed,
    i.e. ascending (900 - 60*m - s). NO playNumber tiebreaker -- playNumber
    resets per drive and cannot be used as a chronological tiebreaker (that
    was the bug in the prior version of this script)."""
    period = int(p.get("period") or 0)
    clock = p.get("clock") or {}
    m = clock.get("minutes")
    s = clock.get("seconds")
    elapsed = 0 if (m is None or s is None) else 900 - 60 * int(m) - int(s)
    return (period, elapsed)


def drive_play_key(p: dict) -> tuple[int, int]:
    return (int(p.get("driveNumber") or 0), int(p.get("playNumber") or 0))


te = pd.read_csv(TRIGGER_EVENTS)
all_game_ids = sorted(te["game_id"].unique().tolist())
print(f"trigger_events covers {len(all_game_ids):,} unique game_ids")

rng = random.Random(42)
sample_gids = rng.sample(all_game_ids, k=50)
print(f"sampling {len(sample_gids)} games (random.seed=42):")
print(f"  {sample_gids[:10]} ... (showing first 10)\n")

# We need to know each game's (season, season_type, week) tuple to load
# the right /plays cache shard.
gid_to_tuple = {
    int(r["game_id"]): (int(r["season"]), str(r["season_type"]), int(r["week"]))
    for _, r in te[["game_id", "season", "season_type", "week"]].drop_duplicates("game_id").iterrows()
}

# Load only the cache shards we need.
needed_tuples = sorted({gid_to_tuple[gid] for gid in sample_gids})
print(f"loading {len(needed_tuples)} cache shards for the sample")

plays_by_game: dict[int, list[dict]] = defaultdict(list)
for season, st, week in needed_tuples:
    shard = cfbd_get_cache_only(
        "/plays", year=season, seasonType=st, week=week, classification="fbs"
    )
    for p in shard:
        gid = p.get("gameId")
        if gid in set(sample_gids):
            plays_by_game[gid].append(p)
print(f"loaded {sum(len(v) for v in plays_by_game.values()):,} plays across "
      f"{len(plays_by_game)} games in the sample\n")

# --- Tests ------------------------------------------------------------------
# Strategy: establish play.id as chronological ground truth via T0 (clock-
# fields monotone when sorted by id), then test driveNumber monotonicity and
# lex-vs-id equivalence against id-order.

t0_fail: list[tuple[int, str]] = []  # play.id monotonic with clock (independent ground truth)
t1_fail: list[tuple[int, str]] = []  # driveNumber non-decreasing along play.id order
t2_fail: list[tuple[int, str]] = []  # driveNumber values contiguous {1..max}
t3_fail: list[tuple[int, str]] = []  # playNumber per-drive starts at 1 and is monotone
t4_fail: list[tuple[int, str]] = []  # sort by lex (driveNumber, playNumber) == sort by play.id

for gid in sample_gids:
    plays = plays_by_game.get(gid, [])
    if not plays:
        t0_fail.append((gid, "empty (no plays in cache for this game)"))
        continue
    by_id = sorted(plays, key=play_id_key)

    # T0: play.id order is chronologically consistent with the clock fields.
    # For each adjacent pair (a, b) in by_id, b's clock should be at or after
    # a's. Equivalently: clock_key(b) >= clock_key(a) componentwise, but with
    # the wrinkle that PERIOD must be non-decreasing while period_seconds_elapsed
    # is non-decreasing within a period (and resets when period increments).
    t0_violation: str | None = None
    prev_pid, prev_ck = None, None
    for p in by_id:
        ck = clock_key(p)
        if prev_ck is not None:
            # period must be non-decreasing
            if ck[0] < prev_ck[0]:
                t0_violation = (
                    f"play.id={p.get('id')} period={ck[0]} comes after play.id={prev_pid} "
                    f"period={prev_ck[0]} (period went backwards along play.id order)"
                )
                break
            # within same period, period_seconds_elapsed must be non-decreasing
            if ck[0] == prev_ck[0] and ck[1] < prev_ck[1]:
                t0_violation = (
                    f"play.id={p.get('id')} period={ck[0]} clock_elapsed={ck[1]} "
                    f"comes after play.id={prev_pid} period={prev_ck[0]} clock_elapsed={prev_ck[1]} "
                    f"(clock went backwards within period)"
                )
                break
        prev_pid, prev_ck = p.get("id"), ck
    if t0_violation:
        t0_fail.append((gid, t0_violation))

    # T1: driveNumber non-decreasing along play.id order
    t1_violation = None
    last_dn = -1
    for p in by_id:
        dn = p.get("driveNumber")
        if dn is None:
            continue
        if int(dn) < last_dn:
            t1_violation = (
                f"play.id={p.get('id')} has driveNumber={dn}, prior play in id-order had "
                f"driveNumber={last_dn} (drive went backwards)"
            )
            break
        last_dn = int(dn)
    if t1_violation:
        t1_fail.append((gid, t1_violation))

    # T2: driveNumber values form contiguous 1..max
    dns = sorted({int(p["driveNumber"]) for p in plays if p.get("driveNumber") is not None})
    if dns:
        expected = list(range(1, max(dns) + 1))
        if dns != expected:
            missing = sorted(set(expected) - set(dns))
            extra = sorted(set(dns) - set(expected))
            t2_fail.append((gid, f"non-contiguous: missing={missing}, extra={extra}, observed={dns}"))

    # T3: within each drive, playNumber starts at 1 and is non-decreasing in id-order
    by_drive: dict[int, list[dict]] = defaultdict(list)
    for p in plays:
        dn = p.get("driveNumber")
        if dn is not None:
            by_drive[int(dn)].append(p)
    t3_violation = None
    for dn, ps in by_drive.items():
        ps_sorted = sorted(ps, key=play_id_key)
        pn_seq = [int(p["playNumber"]) for p in ps_sorted if p.get("playNumber") is not None]
        if not pn_seq:
            continue
        if pn_seq[0] != 1:
            t3_violation = f"drive {dn} first play (by id) has playNumber={pn_seq[0]} (expected 1); seq={pn_seq[:10]}"
            break
        if any(pn_seq[i] < pn_seq[i - 1] for i in range(1, len(pn_seq))):
            t3_violation = f"drive {dn} playNumber not monotone in id-order: {pn_seq[:15]}"
            break
    if t3_violation:
        t3_fail.append((gid, t3_violation))

    # T4: sort by (driveNumber, playNumber) lex == sort by play.id
    id_order = [p.get("id") for p in by_id]
    lex_sorted = sorted(plays, key=drive_play_key)
    lex_order = [p.get("id") for p in lex_sorted]
    if id_order != lex_order:
        for i, (a, b) in enumerate(zip(id_order, lex_order)):
            if a != b:
                pa = next(p for p in plays if p.get("id") == a)
                pb = next(p for p in plays if p.get("id") == b)
                t4_fail.append((gid,
                    f"first divergence at index {i}:\n"
                    f"      id-order[{i}]: id={a} drive={pa.get('driveNumber')} pn={pa.get('playNumber')} "
                    f"period={pa.get('period')} clock={pa.get('clock')}\n"
                    f"      lex-order[{i}]: id={b} drive={pb.get('driveNumber')} pn={pb.get('playNumber')} "
                    f"period={pb.get('period')} clock={pb.get('clock')}"
                ))
                break
        else:
            t4_fail.append((gid, "lengths differ"))


# --- Report -----------------------------------------------------------------

def report(name: str, fails: list[tuple[int, str]]) -> None:
    n_pass = len(sample_gids) - len(fails)
    flag = "PASS" if not fails else "FAIL"
    print(f"[{flag}] {name}  ({n_pass}/{len(sample_gids)} games pass)")
    for gid, detail in fails[:5]:
        print(f"      game_id={gid}")
        for line in detail.splitlines():
            print(f"        {line}")
    if len(fails) > 5:
        print(f"      ... and {len(fails) - 5} more")

print("=== Test results (50 random games, random.seed=42) ===\n")
report("T0. play.id order is chronologically consistent with clock fields", t0_fail)
report("T1. driveNumber non-decreasing along play.id order", t1_fail)
report("T2. driveNumber values form contiguous {1..max} per game", t2_fail)
report("T3. playNumber per-drive starts at 1 and non-decreasing in id-order", t3_fail)
report("T4. sort by (driveNumber, playNumber) lex == sort by play.id", t4_fail)

any_fail = bool(t0_fail or t1_fail or t2_fail or t3_fail or t4_fail)
print()
if not any_fail:
    print("[ok] all 5 tests PASS on all 50 sampled games.")
    print("     play.id is chronological ground truth (T0).")
    print("     driveNumber is monotone along play.id order (T1).")
    print("     (driveNumber, playNumber) lex order EQUALS play.id order (T4).")
    print("     Therefore the (driveNumber, playNumber) lex filter is a safe chronological filter.")
    print("     Option (A) approved.")
else:
    print("[fail] one or more tests failed.")
    if t0_fail:
        print("       T0 failed: play.id is NOT a reliable chronological ground truth.")
        print("       Need a different verification approach.")
    if t1_fail and not t0_fail:
        print("       T1 failed: driveNumber is NOT monotone along chronological order.")
        print("       Cannot use (driveNumber, playNumber) as a filter.")
    if t4_fail and not (t0_fail or t1_fail):
        print("       T4 failed: lex order diverges from chronological order.")
        print("       Fall back to Option (B): filter by play.id directly.")
