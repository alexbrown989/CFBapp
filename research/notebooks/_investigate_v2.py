"""
v2: Use the correct sort key (period, playNumber, id). PRE is ~19% for
TD/FG with this sort, so the convention really is MIXED at the play level.

Goals:
1. Cross-tab playType sub-string vs bucket -> does sub-type drive convention?
2. Sample OTHER + PRE rows with raw fields.
3. Look at the play AFTER the trigger (next non-PAT) -- if cell 14's "POST"
   classification reads end-of-play scoreboard, then "PRE" plays have
   their post-state available on the NEXT play. Confirm.
"""
from __future__ import annotations

import csv
import glob
import json
import pathlib
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
OUT_TXT = ROOT / "results" / "_investigate_other_bucket_v2.txt"

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

games_by_id: dict[int, dict] = {}
for fp in sorted(glob.glob(str(CACHE / "cfbd__games__*.json"))):
    with open(fp, encoding="utf-8") as f:
        games = json.load(f)
    if not isinstance(games, list):
        continue
    for g in games:
        gid = g.get("id")
        if gid is None:
            continue
        games_by_id[gid] = g

PAT_KEYWORDS = {"point after", "two point conversion", "extra point"}
TARGETS = [
    ("TOUCHDOWN", 6, "offense", {"touchdown"}),
    ("FIELD_GOAL", 3, "offense", {"field goal good"}),
    ("SAFETY", 2, "defense", {"safety"}),
]


def sort_key(p):
    return (int(p["period"]), int(p["playNumber"]),
            int(p["id"]) if p.get("id") is not None else 0)


def score_for_team(play, team):
    if play.get("offense") == team:
        v = play.get("offenseScore")
        return int(v) if v is not None else None
    if play.get("defense") == team:
        v = play.get("defenseScore")
        return int(v) if v is not None else None
    return None


def find_prev_non_pat(plays, i):
    for j in range(i - 1, -1, -1):
        ptype = (plays[j].get("playType") or "").strip().lower()
        if any(kw in ptype for kw in PAT_KEYWORDS):
            continue
        return plays[j], j
    return None, None


def find_next_non_pat(plays, i):
    for j in range(i + 1, len(plays)):
        ptype = (plays[j].get("playType") or "").strip().lower()
        if any(kw in ptype for kw in PAT_KEYWORDS):
            continue
        return plays[j], j
    return None, None


def collect(label, points, side, kws):
    rows = []
    for gid, plays in plays_by_game.items():
        sorted_plays = sorted(
            [p for p in plays if p.get("playNumber") is not None
             and p.get("period") is not None],
            key=sort_key,
        )
        for i, p in enumerate(sorted_plays):
            ptype = (p.get("playType") or "").strip().lower()
            if not any(kw in ptype for kw in kws):
                continue
            team = p.get("offense") if side == "offense" else p.get("defense")
            if not team:
                continue
            prev, pj = find_prev_non_pat(sorted_plays, i)
            nxt, nj = find_next_non_pat(sorted_plays, i)
            if prev is None:
                continue
            ps = score_for_team(prev, team)
            ts = score_for_team(p, team)
            ns = score_for_team(nxt, team) if nxt is not None else None
            if ps is None or ts is None:
                continue
            d_pre = ts - ps
            d_next = (ns - ts) if ns is not None else None
            if d_pre >= points:
                bucket = "POST"
            elif d_pre == 0:
                bucket = "PRE"
            else:
                bucket = "OTHER"
            rows.append(dict(
                game_id=gid,
                play_id=p.get("id"),
                play_type_full=p.get("playType"),
                period=p.get("period"),
                play_number=p.get("playNumber"),
                offense=p.get("offense"),
                defense=p.get("defense"),
                team=team,
                offenseScore=p.get("offenseScore"),
                defenseScore=p.get("defenseScore"),
                prev_play_id=prev.get("id"),
                prev_period=prev.get("period"),
                prev_play_number=prev.get("playNumber"),
                prev_play_type=prev.get("playType"),
                prev_offense=prev.get("offense"),
                prev_defense=prev.get("defense"),
                prev_offenseScore=prev.get("offenseScore"),
                prev_defenseScore=prev.get("defenseScore"),
                prev_team_score=ps,
                this_team_score=ts,
                next_play_id=nxt.get("id") if nxt else None,
                next_period=nxt.get("period") if nxt else None,
                next_play_number=nxt.get("playNumber") if nxt else None,
                next_play_type=nxt.get("playType") if nxt else None,
                next_offenseScore=nxt.get("offenseScore") if nxt else None,
                next_defenseScore=nxt.get("defenseScore") if nxt else None,
                next_team_score=ns,
                d_pre=d_pre,
                d_next=d_next,
                bucket=bucket,
                playText=(p.get("playText") or "")[:160],
            ))
    return rows


lines = []

for label, points, side, kws in TARGETS:
    rows = collect(label, points, side, kws)
    bucket_counts = Counter(r["bucket"] for r in rows)
    total = len(rows)

    # Cross-tab: full playType vs bucket. This will show whether sub-type
    # (Rushing TD vs Passing TD vs Interception Return TD etc.) drives
    # convention.
    by_type = defaultdict(Counter)
    for r in rows:
        by_type[r["play_type_full"]][r["bucket"]] += 1

    # For PRE rows, check: does the next play show the post-score?
    # If yes, then "PRE" really means "this play's offenseScore is the
    # state going INTO the play, and the next play has the state coming
    # OUT" -- a true pre-play stamp.
    next_post_count = 0
    pre_rows = [r for r in rows if r["bucket"] == "PRE"]
    for r in pre_rows:
        if r["next_team_score"] is not None and r["d_next"] is not None:
            if r["d_next"] >= points:
                next_post_count += 1

    # For POST rows, the next play should show same score (no further jump)
    # or another scoring play.
    post_rows = [r for r in rows if r["bucket"] == "POST"]
    post_next_zero = sum(1 for r in post_rows
                          if r["d_next"] is not None and r["d_next"] == 0)

    # For OTHER rows, what does d_next say?
    other_rows = [r for r in rows if r["bucket"] == "OTHER"]
    other_d_next_dist = Counter(r["d_next"] for r in other_rows
                                 if r["d_next"] is not None)

    lines.append("")
    lines.append("=" * 80)
    lines.append(f"{label}  total={total:,}  POST={bucket_counts['POST']:,} "
                 f"PRE={bucket_counts['PRE']:,}  OTHER={bucket_counts['OTHER']:,}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Cross-tab playType sub-string -> bucket counts:")
    lines.append(f"  {'sub-type':<40} {'POST':>7} {'PRE':>7} {'OTHER':>7}  POST%   PRE%")
    for st, bc in sorted(by_type.items(), key=lambda kv: -sum(kv[1].values())):
        st_total = sum(bc.values())
        if st_total < 5:
            continue
        post_pct = 100 * bc['POST'] / st_total
        pre_pct = 100 * bc['PRE'] / st_total
        lines.append(f"  {st:<40} {bc['POST']:>7,} {bc['PRE']:>7,} {bc['OTHER']:>7,}  "
                     f"{post_pct:5.1f}%  {pre_pct:5.1f}%")
    lines.append("")
    if pre_rows:
        lines.append(f"PRE rows where next play d_next >= points_expected ({points}): "
                     f"{next_post_count:,}/{len(pre_rows):,} ({100*next_post_count/len(pre_rows):.1f}%)")
        lines.append("  -> if high, confirms PRE truly is pre-play and the post-state lives on the next play")
    lines.append(f"POST rows where next play's d_next == 0 (no further jump): "
                 f"{post_next_zero:,}/{len(post_rows):,} ({100*post_next_zero/max(1,len(post_rows)):.1f}%)")
    lines.append("")
    lines.append("OTHER rows: distribution of d_next (next-play minus this-play, scoring team):")
    for d, n in sorted(other_d_next_dist.items())[:30]:
        lines.append(f"    d_next={d:>4}  n={n:,}")
    lines.append("")

    # Sample 10 PRE rows
    if pre_rows:
        stride = max(1, len(pre_rows) // 10)
        sample = pre_rows[::stride][:10]
        lines.append(f"-- 10 PRE samples for {label} --")
        for k, s in enumerate(sample, 1):
            g = games_by_id.get(s["game_id"], {})
            lines.append(f"  [{k:02d}] game_id={s['game_id']} season={g.get('season')} week={g.get('week')} "
                         f"final={g.get('awayTeam')} {g.get('awayPoints')}-{g.get('homePoints')} {g.get('homeTeam')}")
            lines.append(f"        play_id={s['play_id']} pNum={s['play_number']} per={s['period']} type='{s['play_type_full']}'")
            lines.append(f"        scoring_team='{s['team']}'  off='{s['offense']}' def='{s['defense']}'")
            lines.append(f"        offenseScore={s['offenseScore']} defenseScore={s['defenseScore']}")
            lines.append(f"        prev_score={s['prev_team_score']} this_score={s['this_team_score']} next_score={s['next_team_score']}  "
                         f"d_pre={s['d_pre']} d_next={s['d_next']}")
            lines.append(f"        prev: id={s['prev_play_id']} per={s['prev_period']} pNum={s['prev_play_number']} type='{s['prev_play_type']}'")
            lines.append(f"        next: id={s['next_play_id']} per={s['next_period']} pNum={s['next_play_number']} type='{s['next_play_type']}'  "
                         f"offScore={s['next_offenseScore']} defScore={s['next_defenseScore']}")
            lines.append(f"        playText: {s['playText']}")
            lines.append("")

    # Sample 10 OTHER rows
    if other_rows:
        stride = max(1, len(other_rows) // 10)
        sample = other_rows[::stride][:10]
        lines.append(f"-- 10 OTHER samples for {label} --")
        for k, s in enumerate(sample, 1):
            g = games_by_id.get(s["game_id"], {})
            lines.append(f"  [{k:02d}] game_id={s['game_id']} season={g.get('season')} week={g.get('week')} "
                         f"final={g.get('awayTeam')} {g.get('awayPoints')}-{g.get('homePoints')} {g.get('homeTeam')}")
            lines.append(f"        play_id={s['play_id']} pNum={s['play_number']} per={s['period']} type='{s['play_type_full']}'")
            lines.append(f"        scoring_team='{s['team']}'  off='{s['offense']}' def='{s['defense']}'")
            lines.append(f"        offenseScore={s['offenseScore']} defenseScore={s['defenseScore']}")
            lines.append(f"        prev_score={s['prev_team_score']} this_score={s['this_team_score']} next_score={s['next_team_score']}  "
                         f"d_pre={s['d_pre']} d_next={s['d_next']}")
            lines.append(f"        prev: id={s['prev_play_id']} per={s['prev_period']} pNum={s['prev_play_number']} type='{s['prev_play_type']}'")
            lines.append(f"        next: id={s['next_play_id']} per={s['next_period']} pNum={s['next_play_number']} type='{s['next_play_type']}'  "
                         f"offScore={s['next_offenseScore']} defScore={s['next_defenseScore']}")
            lines.append(f"        playText: {s['playText']}")
            lines.append("")


OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
print(f"[ok] wrote {OUT_TXT}")
