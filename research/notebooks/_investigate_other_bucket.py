"""
Reproduces cell 14's empirical score-state convention check, then drills
into the "other" bucket by classifying EVERY TD/FG/Safety scoring play in
the cached corpus and dumping raw CFBD fields for a sample of "other" rows.

Output: research/results/_investigate_other_bucket.txt + .csv
NOT committed. NOT a fix. Diagnostic only.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import pathlib
import sys
from collections import Counter, defaultdict
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
OUT_TXT = ROOT / "results" / "_investigate_other_bucket.txt"
OUT_CSV = ROOT / "results" / "_investigate_other_bucket.csv"

# ----- Load every cached /plays response into plays_by_game --------------
plays_files = sorted(glob.glob(str(CACHE / "cfbd__plays__*.json")))
print(f"[info] plays cache files: {len(plays_files)}")

plays_by_game: dict[int, list[dict]] = {}
total_plays = 0
for fp in plays_files:
    with open(fp, encoding="utf-8") as f:
        plays = json.load(f)
    if not isinstance(plays, list):
        continue
    total_plays += len(plays)
    for p in plays:
        gid = p.get("gameId")
        if gid is None:
            continue
        plays_by_game.setdefault(gid, []).append(p)
print(f"[info] total plays loaded: {total_plays:,}; games: {len(plays_by_game):,}")


# ----- Load games to attach season/week context -------------------------
games_files = sorted(glob.glob(str(CACHE / "cfbd__games__*.json")))
games_by_id: dict[int, dict] = {}
for fp in games_files:
    with open(fp, encoding="utf-8") as f:
        games = json.load(f)
    if not isinstance(games, list):
        continue
    for g in games:
        gid = g.get("id")
        if gid is None:
            continue
        games_by_id[gid] = g
print(f"[info] games_by_id entries: {len(games_by_id):,}")


# ----- Reproduce cell 14 logic, but classify ALL plays -------------------
PAT_KEYWORDS = {"point after", "two point conversion", "extra point"}
TARGET_KEYWORDS = {
    ("offense", 6): {"touchdown"},
    ("offense", 3): {"field goal good"},
    ("defense", 2): {"safety"},
}


def find_prev_non_pat(sorted_plays: list[dict], i: int) -> dict | None:
    for j in range(i - 1, -1, -1):
        ptype_prev = (sorted_plays[j].get("playType") or "").strip().lower()
        if any(kw in ptype_prev for kw in PAT_KEYWORDS):
            continue
        return sorted_plays[j]
    return None


def find_next_non_pat(sorted_plays: list[dict], i: int) -> dict | None:
    for j in range(i + 1, len(sorted_plays)):
        ptype_next = (sorted_plays[j].get("playType") or "").strip().lower()
        if any(kw in ptype_next for kw in PAT_KEYWORDS):
            continue
        return sorted_plays[j]
    return None


def score_for_team(play: dict, team: str) -> int | None:
    if play.get("offense") == team:
        v = play.get("offenseScore")
        return int(v) if v is not None else None
    if play.get("defense") == team:
        v = play.get("defenseScore")
        return int(v) if v is not None else None
    return None


def classify(label: str, points_expected: int, side: str):
    target_keywords = TARGET_KEYWORDS[(side, points_expected)]

    samples: list[dict] = []
    skipped_team_mismatch = 0
    skipped_no_prev = 0
    skipped_missing_score = 0

    for gid, plays in plays_by_game.items():
        sorted_plays = sorted(
            [p for p in plays if p.get("playNumber") is not None],
            key=lambda p: int(p["playNumber"]),
        )
        for i, p in enumerate(sorted_plays):
            ptype = (p.get("playType") or "").strip().lower()
            if not any(kw in ptype for kw in target_keywords):
                continue
            scoring_team = p.get("offense") if side == "offense" else p.get("defense")
            if not scoring_team:
                continue
            prev = find_prev_non_pat(sorted_plays, i)
            if prev is None:
                skipped_no_prev += 1
                continue
            prev_score = score_for_team(prev, scoring_team)
            if prev_score is None:
                # scoring_team not on either side of prev play
                skipped_team_mismatch += 1
                continue
            this_score = score_for_team(p, scoring_team)
            if this_score is None:
                skipped_team_mismatch += 1
                continue

            diff = this_score - prev_score

            if diff >= points_expected:
                bucket = "POST"
            elif diff == 0:
                bucket = "PRE"
            else:
                bucket = "OTHER"

            # Detect duplicate playNumber within game.
            dup_pns = [q for q in sorted_plays
                       if int(q.get("playNumber") or -1) == int(p.get("playNumber"))]
            n_dup_for_this_pn = len(dup_pns)
            dup_pns_prev = [q for q in sorted_plays
                            if int(q.get("playNumber") or -1) == int(prev.get("playNumber"))]
            n_dup_for_prev_pn = len(dup_pns_prev)

            samples.append({
                "game_id": gid,
                "play_id": p.get("id"),
                "play_number": p.get("playNumber"),
                "period": p.get("period"),
                "play_type": p.get("playType"),
                "scoring_team": scoring_team,
                "offense": p.get("offense"),
                "defense": p.get("defense"),
                "offenseScore": p.get("offenseScore"),
                "defenseScore": p.get("defenseScore"),
                "n_dup_for_this_pn": n_dup_for_this_pn,
                "prev_play_id": prev.get("id"),
                "prev_play_number": prev.get("playNumber"),
                "prev_period": prev.get("period"),
                "prev_play_type": prev.get("playType"),
                "prev_offense": prev.get("offense"),
                "prev_defense": prev.get("defense"),
                "prev_offenseScore": prev.get("offenseScore"),
                "prev_defenseScore": prev.get("defenseScore"),
                "n_dup_for_prev_pn": n_dup_for_prev_pn,
                "prev_team_score": prev_score,
                "this_team_score": this_score,
                "diff": diff,
                "bucket": bucket,
                "scoring_flag": p.get("scoring"),
                "yardsGained": p.get("yardsGained"),
                "playText": (p.get("playText") or "")[:200],
                "prev_playText": (prev.get("playText") or "")[:200],
            })

    return samples, dict(
        skipped_team_mismatch=skipped_team_mismatch,
        skipped_no_prev=skipped_no_prev,
        skipped_missing_score=skipped_missing_score,
    )


lines: list[str] = []
all_other: list[dict] = []

for label, points, side in [
    ("TOUCHDOWN", 6, "offense"),
    ("FIELD_GOAL", 3, "offense"),
    ("SAFETY", 2, "defense"),
]:
    samples, skips = classify(label, points, side)
    bucket_counts = Counter(s["bucket"] for s in samples)
    diff_counts_post = Counter(s["diff"] for s in samples if s["bucket"] == "POST")
    diff_counts_other = Counter(s["diff"] for s in samples if s["bucket"] == "OTHER")
    lines.append("")
    lines.append("=" * 72)
    lines.append(f"{label}  (points_expected={points}, side={side})")
    lines.append("=" * 72)
    lines.append(f"total_classified : {len(samples):,}")
    lines.append(f"  POST  : {bucket_counts['POST']:>6,}  (diff >= {points})")
    lines.append(f"  PRE   : {bucket_counts['PRE']:>6,}  (diff == 0)")
    lines.append(f"  OTHER : {bucket_counts['OTHER']:>6,}  (everything else)")
    lines.append(f"skipped_team_mismatch : {skips['skipped_team_mismatch']:,}")
    lines.append(f"skipped_no_prev       : {skips['skipped_no_prev']:,}")
    lines.append("")
    lines.append("POST-bucket diff distribution (top 10):")
    for d, n in diff_counts_post.most_common(10):
        lines.append(f"    diff={d:>4}  n={n:,}")
    lines.append("")
    lines.append("OTHER-bucket diff distribution (full):")
    for d, n in sorted(diff_counts_other.items()):
        lines.append(f"    diff={d:>4}  n={n:,}")
    lines.append("")

    others = [s for s in samples if s["bucket"] == "OTHER"]

    # Cross-tab: how often is "OTHER" caused by a duplicate-playNumber
    # row landing as the "prev" play? Period mismatch between trigger and
    # prev is the smoking gun.
    period_mismatch = sum(
        1 for s in others
        if s["period"] is not None and s["prev_period"] is not None
        and int(s["period"]) != int(s["prev_period"])
    )
    prev_pn_has_dups = sum(1 for s in others if (s["n_dup_for_prev_pn"] or 0) > 1)
    this_pn_has_dups = sum(1 for s in others if (s["n_dup_for_this_pn"] or 0) > 1)
    lines.append(f"OTHER rows where period(prev) != period(trigger) : "
                 f"{period_mismatch:,} / {len(others):,}")
    lines.append(f"OTHER rows where prev's playNumber has duplicates  : "
                 f"{prev_pn_has_dups:,} / {len(others):,}")
    lines.append(f"OTHER rows where trigger's playNumber has duplicates: "
                 f"{this_pn_has_dups:,} / {len(others):,}")

    # Same cross-tab restricted to POST-bucket as a sanity baseline.
    post_rows = [s for s in samples if s["bucket"] == "POST"]
    post_period_mismatch = sum(
        1 for s in post_rows
        if s["period"] is not None and s["prev_period"] is not None
        and int(s["period"]) != int(s["prev_period"])
    )
    lines.append(f"  (baseline, POST bucket: period mismatch "
                 f"{post_period_mismatch:,} / {len(post_rows):,})")
    lines.append("")

    # Sample 10 deterministically: take a stride across the list to span
    # different games / seasons.
    if others:
        stride = max(1, len(others) // 10)
        sample_others = others[::stride][:10]
    else:
        sample_others = []

    lines.append(f"-- 10 OTHER rows for {label} (raw CFBD fields) --")
    for k, s in enumerate(sample_others, 1):
        g = games_by_id.get(s["game_id"], {})
        s["_season"] = g.get("season")
        s["_week"] = g.get("week")
        s["_homeTeam"] = g.get("homeTeam")
        s["_awayTeam"] = g.get("awayTeam")
        s["_homePoints"] = g.get("homePoints")
        s["_awayPoints"] = g.get("awayPoints")
        s["_play_type_label"] = label
        all_other.append(s)
        lines.append(f"  [{k:02d}] game_id={s['game_id']} season={s['_season']} week={s['_week']}")
        lines.append(f"        {s['_awayTeam']} @ {s['_homeTeam']}  final={s['_awayPoints']}-{s['_homePoints']}")
        lines.append(f"        play_id={s['play_id']} playNumber={s['play_number']} period={s['period']}  (n_dup_for_this_pn={s['n_dup_for_this_pn']})")
        lines.append(f"        playType='{s['play_type']}' scoring={s['scoring_flag']} yardsGained={s['yardsGained']}")
        lines.append(f"        offense='{s['offense']}'  defense='{s['defense']}'")
        lines.append(f"        offenseScore={s['offenseScore']} defenseScore={s['defenseScore']}")
        lines.append(f"        scoring_team='{s['scoring_team']}'")
        lines.append(f"        prev_play_id={s['prev_play_id']} prev_playNumber={s['prev_play_number']} prev_period={s['prev_period']}  (n_dup_for_prev_pn={s['n_dup_for_prev_pn']})")
        lines.append(f"        prev_playType='{s['prev_play_type']}'")
        lines.append(f"        prev_offense='{s['prev_offense']}'  prev_defense='{s['prev_defense']}'")
        lines.append(f"        prev_offenseScore={s['prev_offenseScore']} prev_defenseScore={s['prev_defenseScore']}")
        lines.append(f"        prev_team_score={s['prev_team_score']}  this_team_score={s['this_team_score']}  diff={s['diff']}")
        lines.append(f"        playText: {s['playText']}")
        lines.append(f"        prev_playText: {s['prev_playText']}")
        lines.append("")

OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")

# CSV with all "other" samples (full population, not just the 10)
if all_other:
    fieldnames = list(all_other[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_other:
            w.writerow(r)

# Print summary tail to stdout for the user
print("\n".join(lines))
print(f"\n[ok] wrote {OUT_TXT}")
print(f"[ok] wrote {OUT_CSV}")
