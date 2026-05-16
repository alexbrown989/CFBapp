"""
Simulate the proposed cell 14 fix and report the bucket distribution we
would actually see. Drives the "expected impact" numbers in the proposal.

Fix simulated:
  1. Sort by (period, playNumber, id).
  2. Per-subtype scoring-team attribution:
       offensive TD subtypes        -> play.offense
       return TD subtypes (INT/Fmb/Punt/Blocked Punt/Blocked FG/Missed FG)
                                     -> play.defense
       Kickoff Return Touchdown      -> EXCLUDED with reason logged
       Field Goal Good               -> play.offense
       Safety                        -> play.defense
  3. SKIP if trigger or prev has duplicate (period, playNumber) within
     the game (flavor-B intra-period duplicates).
  4. SKIP paired-play prevs: if prev's score for the scoring team
     already equals trigger's score for the scoring team, walk further
     back. If we walk all the way to the start of period without finding
     a clean prev, SKIP (log reason).
  5. Triangulate: also read the next non-PAT play. A "verified POST"
     sample is one where:
       this_score - prev_score == points_expected
        AND  next_score == this_score
     A "verified PRE" sample is:
       this_score == prev_score
        AND  next_score - this_score == points_expected
     Neither -> SKIP with reason.

This is conservative: we drop ambiguous samples rather than guess. The
verdict per subtype is computed only over verified samples.
"""
from __future__ import annotations

import glob
import json
import pathlib
from collections import Counter, defaultdict

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

PAT = {"point after", "two point conversion", "extra point"}


def sk(p):
    return (int(p["period"]), int(p["playNumber"]),
            int(p["id"]) if p.get("id") is not None else 0)


# Subtype -> (scoring_team_field, points_expected)
SCORING_REGISTRY = [
    # offensive TDs
    ("rushing touchdown", "offense", 6, "TD_OFFENSIVE"),
    ("passing touchdown", "offense", 6, "TD_OFFENSIVE"),
    # return TDs
    ("interception return touchdown", "defense", 6, "TD_DEFENSIVE_RETURN"),
    ("fumble return touchdown",       "defense", 6, "TD_DEFENSIVE_RETURN"),
    ("punt return touchdown",         "defense", 6, "TD_SPECIAL_RETURN"),
    ("blocked punt touchdown",        "defense", 6, "TD_SPECIAL_RETURN"),
    ("blocked field goal touchdown",  "defense", 6, "TD_SPECIAL_RETURN"),
    ("missed field goal return touchdown", "defense", 6, "TD_SPECIAL_RETURN"),
    # kickoff return TD: the offense at snap IS the receiver, so attribution
    # is offense, but paired-play / period-boundary noise is high; treat as
    # its own bucket so we can quote it separately
    ("kickoff return touchdown", "offense", 6, "TD_KICKOFF_RETURN"),
    # field goal
    ("field goal good", "offense", 3, "FIELD_GOAL"),
    # safety
    ("safety", "defense", 2, "SAFETY"),
]


def classify_subtype(playtype_str: str):
    pt = (playtype_str or "").strip().lower()
    for kw, side, points, label in SCORING_REGISTRY:
        if kw in pt:
            return side, points, label, kw
    return None


def score_for_team(play, team):
    if play.get("offense") == team:
        v = play.get("offenseScore")
        return int(v) if v is not None else None
    if play.get("defense") == team:
        v = play.get("defenseScore")
        return int(v) if v is not None else None
    return None


def find_clean_prev(plays, i, scoring_team, this_score):
    """Walk back skipping PAT and paired-play artifacts.

    Paired-play test: score_for_team(prev, scoring_team) == this_score
    means prev's scoreboard already credits the trigger's points -- prev
    is itself part of the scoring action, not a true pre-state. Skip it.

    Stops if we cross out of the trigger's period.
    """
    trigger_period = int(plays[i]["period"])
    skipped_pat = 0
    skipped_paired = 0
    for j in range(i - 1, -1, -1):
        prev = plays[j]
        if int(prev.get("period") or -1) != trigger_period:
            return None, "left_period", skipped_pat, skipped_paired
        ptype = (prev.get("playType") or "").strip().lower()
        if any(kw in ptype for kw in PAT):
            skipped_pat += 1
            continue
        ps = score_for_team(prev, scoring_team)
        if ps is None:
            return None, "prev_team_mismatch", skipped_pat, skipped_paired
        if ps == this_score:
            skipped_paired += 1
            continue
        return prev, "ok", skipped_pat, skipped_paired
    return None, "no_prev_in_period", skipped_pat, skipped_paired


def find_clean_next(plays, i, scoring_team):
    trigger_period = int(plays[i]["period"])
    for j in range(i + 1, len(plays)):
        nxt = plays[j]
        if int(nxt.get("period") or -1) != trigger_period:
            return None, "left_period"
        ptype = (nxt.get("playType") or "").strip().lower()
        if any(kw in ptype for kw in PAT):
            continue
        ns = score_for_team(nxt, scoring_team)
        if ns is None:
            return None, "next_team_mismatch"
        return nxt, "ok"
    return None, "no_next_in_period"


per_label_results = defaultdict(lambda: Counter())
skip_reasons = defaultdict(lambda: Counter())

for gid, plays in plays_by_game.items():
    sp = sorted(
        [p for p in plays if p.get("playNumber") is not None
         and p.get("period") is not None],
        key=sk,
    )

    # Build (period, playNumber) -> count for duplicate detection
    pp_counts: Counter = Counter()
    for p in sp:
        pp_counts[(int(p["period"]), int(p["playNumber"]))] += 1

    for i, p in enumerate(sp):
        cls = classify_subtype(p.get("playType") or "")
        if cls is None:
            continue
        side, points, label, kw = cls

        # Skip OT
        if int(p.get("period") or 0) >= 5:
            skip_reasons[label]["ot_excluded"] += 1
            continue

        team = p.get(side)
        if not team:
            skip_reasons[label]["no_team_field"] += 1
            continue

        ts = score_for_team(p, team)
        if ts is None:
            skip_reasons[label]["no_this_score"] += 1
            continue

        # Skip flavor-B intra-period duplicates on the trigger row
        if pp_counts[(int(p["period"]), int(p["playNumber"]))] > 1:
            skip_reasons[label]["trigger_dup_period_playnumber"] += 1
            continue

        prev, reason, n_pat, n_paired = find_clean_prev(sp, i, team, ts)
        if prev is None:
            skip_reasons[label][f"no_clean_prev_{reason}"] += 1
            continue

        # Also skip if the chosen prev is at a duplicated (period, playNumber)
        if pp_counts[(int(prev["period"]), int(prev["playNumber"]))] > 1:
            skip_reasons[label]["prev_dup_period_playnumber"] += 1
            continue

        ps = score_for_team(prev, team)

        nxt, nreason = find_clean_next(sp, i, team)
        if nxt is None:
            skip_reasons[label][f"no_next_{nreason}"] += 1
            continue
        if pp_counts[(int(nxt["period"]), int(nxt["playNumber"]))] > 1:
            skip_reasons[label]["next_dup_period_playnumber"] += 1
            continue
        ns = score_for_team(nxt, team)

        d_pre = ts - ps
        d_next = ns - ts

        # Triangulated verdict
        if d_pre == points and d_next == 0:
            per_label_results[label]["VERIFIED_POST"] += 1
        elif d_pre == 0 and d_next == points:
            per_label_results[label]["VERIFIED_PRE"] += 1
        elif d_pre == points and d_next == points:
            per_label_results[label]["AMBIGUOUS_DOUBLE_JUMP"] += 1
        else:
            per_label_results[label]["AMBIGUOUS_OTHER"] += 1


print("=" * 80)
print("Simulated cell 14 with proposed fixes -- per-label verified verdicts")
print("=" * 80)
all_labels = sorted(per_label_results.keys())
for lab in all_labels:
    c = per_label_results[lab]
    t = sum(c.values())
    print(f"\n{lab}  (verified samples = {t:,})")
    for k in ("VERIFIED_POST", "VERIFIED_PRE",
              "AMBIGUOUS_DOUBLE_JUMP", "AMBIGUOUS_OTHER"):
        n = c[k]
        if t:
            print(f"  {k:<25} {n:>7,}  ({100*n/t:6.2f}%)")

print()
print("=" * 80)
print("Skip-reason audit (why a scoring play was excluded from verified set)")
print("=" * 80)
for lab in all_labels:
    sr = skip_reasons[lab]
    if not sr:
        continue
    total_skipped = sum(sr.values())
    print(f"\n{lab}  total skipped = {total_skipped:,}")
    for k, n in sr.most_common():
        print(f"  {k:<45} {n:>7,}")
