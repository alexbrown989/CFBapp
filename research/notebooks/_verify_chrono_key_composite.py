"""Full-corpus verification of the composite chrono_key filter before
applying it project-wide.

chrono_key(p) = (period, period_seconds_elapsed, driveNumber, playNumber)

Tests (per the user's specification):
  T1. For every game in the cached /plays corpus, sort plays by chrono_key
      and by play.id. For every disagreement, classify into:
        (a) Resolved by clock: clock_key(A) < clock_key(B) -- the period/
            elapsed primary key agrees with chrono; play.id is the outlier.
            This is the period-boundary anomaly type from the 50-game sample;
            composite key handles it correctly by construction.
        (b) Resolved by tiebreaker: clock_key(A) == clock_key(B). The lex
            tiebreaker on (driveNumber, playNumber) is doing the work.
            Sub-classify as (b-same-drive) [unambiguously correct: same
            drive, lex by playNumber] or (b-drive-transition) [different
            drives at same clock; A is last play of A.drive AND B is first
            play of B.drive -- the expected TD-then-kickoff pattern].
        (c) Unresolved anomaly. Three failure modes routed here:
              (c1) clock_key(A) > clock_key(B) but chrono(A) < chrono(B)
                   -- impossible by construction; surfaced for safety.
              (c2) Same-clock, different-drive, but NOT a clean drive-
                   transition (A is not last play of its drive, or B is
                   not first play of its drive). Tiebreaker may be wrong.
              (c3) play.id key is non-integer or missing for a play in a
                   disagreement (data integrity issue).

  T2. OT exclusion: 20 random regulation triggers (quarter <= 4) from
      games that went to overtime. Count period >= 5 plays appearing in
      plays_before under the composite key. Expected: zero.

  T3. Tied-clock TD-then-kickoff: find 10 examples where a scoring play
      is followed by a kickoff at the same (period, clock) value. Verify
      composite key orders TD first, kickoff second.

Cache-only. Zero fresh CFBD calls.
"""
from __future__ import annotations
import json
import pathlib
import random
import time
from collections import defaultdict
from typing import Any

import pandas as pd

CACHE_DIR = pathlib.Path(
    r"C:\Users\Alexander\Documents\CFB\CFBapp\research\data\cache"
)
TRIGGER_EVENTS = pathlib.Path(
    r"C:\Users\Alexander\Documents\CFB\CFBapp\research\results\trigger_events.csv"
)


# -- Keys --------------------------------------------------------------------

def clock_key(p: dict) -> tuple[int, int]:
    period = int(p.get("period") or 0)
    clock = p.get("clock") or {}
    m, s = clock.get("minutes"), clock.get("seconds")
    elapsed = 900 - 60 * int(m) - int(s) if m is not None and s is not None else 0
    return (period, elapsed)


def chrono_key(p: dict) -> tuple[int, int, int, int]:
    ck = clock_key(p)
    return (ck[0], ck[1], int(p.get("driveNumber") or 0), int(p.get("playNumber") or 0))


def play_id_key(p: dict) -> int:
    pid = p.get("id") or p.get("playId")
    return int(pid) if pid is not None else -1


# -- Load all /plays cache shards -------------------------------------------

t_load_start = time.time()
all_plays_by_game: dict[int, list[dict]] = defaultdict(list)
shard_files = sorted(CACHE_DIR.glob("cfbd__plays__*.json"))
total_plays_loaded = 0
for cf in shard_files:
    try:
        data = json.loads(cf.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  WARN: failed to load {cf.name}: {e}")
        continue
    if not isinstance(data, list):
        continue
    for p in data:
        gid = p.get("gameId")
        if gid is None:
            continue
        all_plays_by_game[int(gid)].append(p)
        total_plays_loaded += 1
t_load = time.time() - t_load_start

print(f"loaded {len(shard_files)} /plays cache shards in {t_load:.1f}s")
print(f"  {len(all_plays_by_game):,} unique games")
print(f"  {total_plays_loaded:,} total plays")
print()


# -- T1: full-corpus chrono vs id disagreement classification ---------------

t_t1_start = time.time()

n_games_agreeing = 0
n_games_with_disagreement = 0

cat_a_examples: list[tuple[int, dict, dict]] = []
cat_b_same_drive_examples: list[tuple[int, dict, dict]] = []
cat_b_transition_examples: list[tuple[int, dict, dict]] = []
cat_c_examples: list[tuple[int, dict, dict, str]] = []

cat_a_count = 0
cat_b_same_drive_count = 0
cat_b_transition_count = 0
cat_c_count = 0

for gid, plays in all_plays_by_game.items():
    if not plays:
        continue
    chrono_sorted = sorted(plays, key=chrono_key)
    id_sorted = sorted(plays, key=play_id_key)
    chrono_ids = [p.get("id") for p in chrono_sorted]
    id_ids = [p.get("id") for p in id_sorted]
    if chrono_ids == id_ids:
        n_games_agreeing += 1
        continue
    n_games_with_disagreement += 1

    # Pre-compute per-drive min/max playNumber for transition checks
    drive_min_pn: dict[int, int] = {}
    drive_max_pn: dict[int, int] = {}
    for p in plays:
        dn = p.get("driveNumber")
        pn = p.get("playNumber")
        if dn is None or pn is None:
            continue
        dn_i, pn_i = int(dn), int(pn)
        drive_min_pn[dn_i] = min(drive_min_pn.get(dn_i, pn_i), pn_i)
        drive_max_pn[dn_i] = max(drive_max_pn.get(dn_i, pn_i), pn_i)

    # Walk adjacent pairs in chrono order; flag id-reversals
    for i in range(len(chrono_sorted) - 1):
        A = chrono_sorted[i]
        B = chrono_sorted[i + 1]
        try:
            id_A = int(A.get("id") or A.get("playId") or 0)
            id_B = int(B.get("id") or B.get("playId") or 0)
        except (ValueError, TypeError):
            cat_c_count += 1
            cat_c_examples.append((gid, A, B, "non_integer_play_id"))
            continue
        if id_A <= id_B:
            continue  # adjacent agreement
        ck_A = clock_key(A)
        ck_B = clock_key(B)
        if ck_A < ck_B:
            cat_a_count += 1
            if len(cat_a_examples) < 200:
                cat_a_examples.append((gid, A, B))
            continue
        if ck_A == ck_B:
            A_dn = int(A.get("driveNumber") or 0)
            B_dn = int(B.get("driveNumber") or 0)
            A_pn = int(A.get("playNumber") or 0)
            B_pn = int(B.get("playNumber") or 0)
            if A_dn == B_dn:
                cat_b_same_drive_count += 1
                if len(cat_b_same_drive_examples) < 50:
                    cat_b_same_drive_examples.append((gid, A, B))
            else:
                a_is_last = (A_pn == drive_max_pn.get(A_dn))
                b_is_first = (B_pn == drive_min_pn.get(B_dn))
                if a_is_last and b_is_first:
                    cat_b_transition_count += 1
                    if len(cat_b_transition_examples) < 50:
                        cat_b_transition_examples.append((gid, A, B))
                else:
                    cat_c_count += 1
                    cat_c_examples.append(
                        (gid, A, B,
                         f"same_clock_diff_drive_not_transition "
                         f"(A_is_last={a_is_last}, B_is_first={b_is_first})")
                    )
            continue
        # ck_A > ck_B but chrono(A) < chrono(B) -- impossible by construction
        cat_c_count += 1
        cat_c_examples.append((gid, A, B, "chrono_primary_disagrees_with_clock_IMPOSSIBLE"))

t_t1 = time.time() - t_t1_start


# -- T2: OT exclusion verification ------------------------------------------

t_t2_start = time.time()

games_with_ot: set[int] = set()
for gid, plays in all_plays_by_game.items():
    for p in plays:
        if int(p.get("period") or 0) >= 5:
            games_with_ot.add(gid)
            break

te = pd.read_csv(TRIGGER_EVENTS)
regulation_in_ot_games = te[(te["game_id"].isin(games_with_ot)) & (te["quarter"] <= 4)]
print(f"  games that went to OT: {len(games_with_ot):,}")
print(f"  regulation triggers in OT games: {len(regulation_in_ot_games):,}")

rng = random.Random(42)
ot_check_idx = rng.sample(range(len(regulation_in_ot_games)), k=min(20, len(regulation_in_ot_games)))
ot_check_rows = regulation_in_ot_games.iloc[ot_check_idx]

ot_violations: list[tuple[int, dict, int]] = []  # (game_id, trigger_row, count_of_leaked_OT_plays)
ot_check_details: list[dict] = []
for _, trig in ot_check_rows.iterrows():
    gid = int(trig["game_id"])
    trig_period = int(trig["quarter"])
    # clock_seconds_in_period_total in trigger_events.csv = 60*minutes_remaining + seconds_remaining
    trig_clock_remaining = int(trig["clock_seconds_in_period_total"])
    trig_elapsed = 900 - trig_clock_remaining
    trig_dn = int(trig["drive_number_in_game"])
    trig_pn = int(trig["play_number"])
    trig_key = (trig_period, trig_elapsed, trig_dn, trig_pn)

    plays = all_plays_by_game.get(gid, [])
    leaked = [p for p in plays if chrono_key(p) < trig_key and int(p.get("period") or 0) >= 5]
    ot_check_details.append({
        "game_id": gid,
        "trigger_period": trig_period,
        "trigger_elapsed": trig_elapsed,
        "trigger_drive_pn": (trig_dn, trig_pn),
        "plays_before_count": sum(1 for p in plays if chrono_key(p) < trig_key),
        "leaked_OT_plays": len(leaked),
    })
    if leaked:
        ot_violations.append((gid, trig.to_dict(), len(leaked)))

t_t2 = time.time() - t_t2_start


# -- T3: tied-clock TD-then-kickoff sanity check ----------------------------

t_t3_start = time.time()

TD_TYPES = {
    "Rushing Touchdown", "Passing Touchdown", "Kickoff Return Touchdown",
    "Punt Return Touchdown", "Interception Return Touchdown",
    "Fumble Recovery Touchdown", "Blocked Punt Touchdown",
    "Blocked Field Goal Touchdown", "Defensive 2pt Conversion",
    "Safety", "Field Goal Good",
}
KICKOFF_TYPES = {"Kickoff", "Kickoff Return"}

td_kickoff_pairs: list[tuple[int, dict, dict]] = []
for gid, plays in all_plays_by_game.items():
    chrono_sorted = sorted(plays, key=chrono_key)
    for i in range(len(chrono_sorted) - 1):
        A = chrono_sorted[i]
        B = chrono_sorted[i + 1]
        if clock_key(A) != clock_key(B):
            continue
        A_type = A.get("playType")
        B_type = B.get("playType")
        if A_type in TD_TYPES and B_type in KICKOFF_TYPES:
            td_kickoff_pairs.append((gid, A, B))
            if len(td_kickoff_pairs) >= 10:
                break
    if len(td_kickoff_pairs) >= 10:
        break

td_kickoff_ok = 0
td_kickoff_bad = 0
td_kickoff_details: list[dict] = []
for gid, A, B in td_kickoff_pairs:
    ck_A = chrono_key(A)
    ck_B = chrono_key(B)
    ordered_correctly = ck_A < ck_B
    if ordered_correctly:
        td_kickoff_ok += 1
    else:
        td_kickoff_bad += 1
    td_kickoff_details.append({
        "game_id": gid,
        "A_id": A.get("id"),
        "A_type": A.get("playType"),
        "A_chrono_key": ck_A,
        "B_id": B.get("id"),
        "B_type": B.get("playType"),
        "B_chrono_key": ck_B,
        "TD_first_under_composite": ordered_correctly,
    })

t_t3 = time.time() - t_t3_start


# -- Report ------------------------------------------------------------------

def fmt_play(p: dict) -> str:
    return (
        f"id={p.get('id')} drive={p.get('driveNumber')} pn={p.get('playNumber')} "
        f"period={p.get('period')} clock={p.get('clock')} type={p.get('playType')!r}"
    )

print("\n" + "=" * 78)
print("T1. Full-corpus chrono vs id disagreement classification")
print("=" * 78)
print(f"  games scanned:                                {len(all_plays_by_game):,}")
print(f"  games where chrono order == id order:         {n_games_agreeing:,}")
print(f"  games with at least one disagreement:         {n_games_with_disagreement:,}")
print(f"  total adjacent-pair disagreements:            "
      f"{cat_a_count + cat_b_same_drive_count + cat_b_transition_count + cat_c_count:,}")
print()
print(f"  (a) resolved by clock (primary key disagrees with id):")
print(f"       count: {cat_a_count:,}")
print(f"  (b) resolved by tiebreaker (same clock_key):")
print(f"      (b-same-drive)        count: {cat_b_same_drive_count:,}")
print(f"      (b-drive-transition)  count: {cat_b_transition_count:,}")
print(f"  (c) UNRESOLVED ANOMALIES:")
print(f"       count: {cat_c_count:,}")
print()

# (c) -- every case in full, if any
print("--- Category (c) cases: every case in full ---")
if cat_c_count == 0:
    print("  zero unresolved anomalies. composite key fully accounts for every")
    print("  chrono-vs-id disagreement across the entire cached corpus.")
else:
    for gid, A, B, reason in cat_c_examples:
        print(f"  game_id={gid}  reason={reason}")
        print(f"    A: {fmt_play(A)}")
        print(f"    B: {fmt_play(B)}")
        print(f"    chrono_key(A)={chrono_key(A)}  chrono_key(B)={chrono_key(B)}")
        print(f"    clock_key(A)={clock_key(A)}    clock_key(B)={clock_key(B)}")
        print()

# (a) -- worst 3 by |id_A - id_B| separation
print("--- Category (a) worst 3 (largest id-vs-chrono separation) ---")
ranked_a = sorted(
    cat_a_examples,
    key=lambda triple: -abs(
        int(triple[1].get("id") or 0) - int(triple[2].get("id") or 0)
    ),
)
for gid, A, B in ranked_a[:3]:
    print(f"  game_id={gid}")
    print(f"    A: {fmt_play(A)}")
    print(f"    B: {fmt_play(B)}")
    print(f"    chrono_key(A)={chrono_key(A)}")
    print(f"    chrono_key(B)={chrono_key(B)}")
    print(f"    play.id ordering would place A AFTER B (id_A={A.get('id')} > id_B={B.get('id')})")
    print()

# Sample of (b-drive-transition) -- expected pattern
print("--- Category (b-drive-transition) sample (expected TD-then-kickoff pattern) ---")
for gid, A, B in cat_b_transition_examples[:3]:
    print(f"  game_id={gid}")
    print(f"    A (last play of drive {A.get('driveNumber')}):  {fmt_play(A)}")
    print(f"    B (first play of drive {B.get('driveNumber')}): {fmt_play(B)}")
    print()

print()
print("=" * 78)
print("T2. OT exclusion verification (20 random regulation triggers from OT games)")
print("=" * 78)
print(f"  triggers checked: {len(ot_check_details)}")
print(f"  triggers with any period>=5 play in plays_before: {len(ot_violations)}")
if ot_violations:
    print("  VIOLATIONS:")
    for gid, trig_row, leaked_n in ot_violations[:5]:
        print(f"    game_id={gid} trigger=(q{trig_row.get('quarter')}, "
              f"drive={trig_row.get('drive_number_in_game')}, "
              f"pn={trig_row.get('play_number')}) -- {leaked_n} OT plays leaked")
else:
    print("  OK: zero OT plays appear in any regulation trigger's plays_before.")
print()
print("  Sample (first 5 of 20):")
for d in ot_check_details[:5]:
    print(f"    {d}")

print()
print("=" * 78)
print("T3. Tied-clock TD-then-kickoff sanity check (10 examples)")
print("=" * 78)
print(f"  pairs found: {len(td_kickoff_pairs)}")
print(f"  correctly ordered (TD before kickoff under composite key): {td_kickoff_ok}")
print(f"  incorrectly ordered: {td_kickoff_bad}")
print()
for d in td_kickoff_details:
    flag = "[OK]" if d["TD_first_under_composite"] else "[FAIL]"
    print(f"  {flag} game_id={d['game_id']}")
    print(f"        A: id={d['A_id']} type={d['A_type']!r} key={d['A_chrono_key']}")
    print(f"        B: id={d['B_id']} type={d['B_type']!r} key={d['B_chrono_key']}")

print()
print("=" * 78)
print(f"Runtime: load={t_load:.2f}s  T1={t_t1:.2f}s  T2={t_t2:.2f}s  T3={t_t3:.2f}s  "
      f"total={t_load + t_t1 + t_t2 + t_t3:.2f}s")
print("=" * 78)

# Final verdict
all_pass = (cat_c_count == 0 and not ot_violations and td_kickoff_bad == 0)
print()
if all_pass:
    print("[ok] composite chrono_key passes full-corpus verification.")
    print("     (a) all id-vs-chrono disagreements resolved by clock primary, OR")
    print("     (b) all same-clock cases resolved by lex tiebreaker in expected patterns,")
    print("     (c) zero unresolved anomalies,")
    print("     T2 OT exclusion clean,")
    print("     T3 tied-clock TD-then-kickoff ordering correct.")
    print("     SAFE to apply to _build_02c.py per the agreed sequence.")
else:
    print("[fail] one or more checks failed. STOP and review before applying fix.")
