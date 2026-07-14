from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .data_source import ScoreboardGameState, ScoreboardStub
from .logger import JSONLTriggerLogger, REQUIRED_TRIGGER_FIELDS
from .trigger_detect import TriggerDetector
from .watchlist import WatchGame, build_watchlist


REPO_ROOT = Path(__file__).resolve().parents[1]
GAME_ID = "401628374"
PLAYS_CACHE = REPO_ROOT / "research/data/cache/cfbd__plays__50fb856363e865e5.json"
GAMES_CACHE = REPO_ROOT / "research/data/cache/cfbd__games__9d0a751d3eb65363.json"
LINES_CACHE = REPO_ROOT / "research/data/cache/cfbd__lines__9d0a751d3eb65363.json"
RANKINGS_CACHE = REPO_ROOT / "research/data/cache/cfbd__rankings__2024.json"
LOG_PATH = REPO_ROOT / "live/logs/replay_verification.jsonl"
REPORT_PATH = REPO_ROOT / "research/results/n13_stage1_replay_verification.md"


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def ap_teams_for_week(records: list[dict[str, object]], week: int) -> set[str]:
    candidates = [row for row in records if row.get("seasonType") == "regular" and int(row.get("week", -1)) == week]
    if not candidates:
        raise AssertionError(f"no 2024 regular ranking record for week {week}")
    teams: set[str] = set()
    for poll in candidates[0].get("polls", []):
        if poll.get("poll") == "AP Top 25":
            teams = {str(rank["school"]) for rank in poll.get("ranks", [])}
            break
    if len(teams) != 25:
        raise AssertionError(f"expected 25 AP teams for week {week}, got {len(teams)}")
    return teams


def chronological_key(play: dict[str, object]) -> tuple[int, int, int, int]:
    clock = play.get("clock") or {}
    remaining = int(clock.get("minutes", 0)) * 60 + int(clock.get("seconds", 0))
    elapsed = 900 - remaining
    return (
        int(play.get("period") or 0),
        elapsed,
        int(play.get("driveNumber") or 0),
        int(play.get("playNumber") or 0),
    )


def score_state(play: dict[str, object], poll_number: int) -> ScoreboardGameState:
    offense = str(play["offense"])
    defense = str(play["defense"])
    offense_score = int(play["offenseScore"])
    defense_score = int(play["defenseScore"])
    home = str(play["home"])
    away = str(play["away"])
    scores = {offense: offense_score, defense: defense_score}
    clock = play.get("clock") or {}
    return ScoreboardGameState(
        game_id=GAME_ID,
        season=2024,
        week=5,
        home_team=home,
        away_team=away,
        home_score=scores[home],
        away_score=scores[away],
        period=int(play["period"]),
        clock=f"{int(clock.get('minutes', 0))}:{int(clock.get('seconds', 0)):02d}",
        status="in_progress",
        possession=offense,
        data_source="stub",
        poll_number=poll_number,
        observed_at=str(play.get("wallclock") or datetime.now(timezone.utc).isoformat()),
    )


def verify_watchlist() -> WatchGame:
    games = [row for row in load_json(GAMES_CACHE) if str(row.get("id")) == GAME_ID]
    lines = [row for row in load_json(LINES_CACHE) if str(row.get("id")) == GAME_ID]
    rankings = load_json(RANKINGS_CACHE)
    ap_teams = ap_teams_for_week(rankings, 5)
    assert "Georgia" in ap_teams
    watchlist = build_watchlist(games, lines, ranked_teams=ap_teams)
    game = watchlist.get(GAME_ID)
    assert game is not None, "cached 2024 top-25 game did not enter watch list"
    assert game.favorite == "Georgia" and game.dog == "Alabama"
    assert game.pregame_spread == -2.0
    return game


def verify_replay(game: WatchGame) -> tuple[list[dict[str, object]], int]:
    plays = [row for row in load_json(PLAYS_CACHE) if str(row.get("gameId")) == GAME_ID]
    scoring_plays = sorted((row for row in plays if row.get("scoring") is True), key=chronological_key)
    batches = [[score_state(play, index)] for index, play in enumerate(scoring_plays, start=1)]
    source = ScoreboardStub(batches)
    detector = TriggerDetector()
    logger = JSONLTriggerLogger(LOG_PATH)
    LOG_PATH.unlink(missing_ok=True)
    events = []
    while True:
        states = source.poll()
        if not states:
            break
        for state in states:
            new_events = detector.process(state, game)
            logger.append_many(new_events)
            events.extend(new_events)

    actual = [(event.threshold_crossed, event.period, event.clock, event.fav_score, event.dog_score) for event in events]
    expected = [
        (3, 1, "10:11", 0, 7),
        (7, 1, "10:11", 0, 7),
        (10, 1, "4:39", 0, 14),
        (14, 1, "4:39", 0, 14),
        (21, 1, "2:21", 0, 21),
        (3, 4, "2:18", 34, 41),
        (7, 4, "2:18", 34, 41),
    ]
    assert actual == expected, {"actual": actual, "expected": expected}
    records = [json.loads(line) for line in LOG_PATH.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 7
    assert all(tuple(record.keys()) == tuple(sorted(REQUIRED_TRIGGER_FIELDS)) for record in records)
    return records, len(scoring_plays)


def verify_refire() -> list[int]:
    game = WatchGame("refire", 2024, 1, "Favorite", "Dog", -7.0, "Favorite", "Dog", "test")
    detector = TriggerDetector()
    states = [
        ScoreboardGameState("refire", 2024, 1, "Favorite", "Dog", 0, 7, 1, "12:00", "in_progress", poll_number=1),
        ScoreboardGameState("refire", 2024, 1, "Favorite", "Dog", 10, 7, 2, "10:00", "in_progress", poll_number=2),
        ScoreboardGameState("refire", 2024, 1, "Favorite", "Dog", 10, 17, 3, "8:00", "in_progress", poll_number=3),
    ]
    thresholds = [event.threshold_crossed for state in states for event in detector.process(state, game)]
    assert thresholds == [3, 7, 3, 7], thresholds
    return thresholds


def write_report(records: list[dict[str, object]], scoring_play_count: int, refire: list[int]) -> None:
    table_rows = "\n".join(
        f"| {row['threshold_crossed']} | Q{row['period']} {row['clock']} | {row['fav_score']}-{row['dog_score']} | {row['poll_number']} |"
        for row in records
    )
    report = f"""# N13 Stage 1 Replay Verification

Date: 2026-07-14

## Result

PASS. The source-agnostic Stage 1 detector replayed cached 2024 game `{GAME_ID}` (No. 2 Georgia at Alabama), built its watch-list entry from cached games, lines, and the week-5 AP Top 25 poll, and emitted the expected trigger sequence.

CFBD applies some completed-drive score values to earlier plays in a drive. The replay therefore advances the scoreboard only on records marked `scoring=true`, matching actual scoring state transitions rather than treating those backward-stamped values as live score changes.

| Threshold | Observed state | Favorite-dog score | Poll |
|---:|---|---:|---:|
{table_rows}

- Scoring states replayed: {scoring_play_count}
- Initial-descent thresholds emitted once: 3, 7, 10, 14, 21
- Multi-threshold crossings: 0-7 emitted D=3 and D=7; 0-14 emitted D=10 and D=14
- Real-game re-fire after Georgia recovered to a 34-33 lead: D=3 and D=7 at Q4 2:18
- Synthetic re-fire test after recovery: {refire}
- JSONL records: {len(records)}
- JSONL schema: PASS, all {len(REQUIRED_TRIGGER_FIELDS)} required fields present
- Data source recorded as `stub`
- Network/API calls: 0

## Acceptance

Trigger timing, one-fire deduplication, recovery-based re-arming, multi-threshold crossing, watch-list construction, and append-only local log schema all pass. `ScoreboardLive` was not activated.
"""
    REPORT_PATH.write_text(report, encoding="utf-8", newline="\n")


def main() -> int:
    game = verify_watchlist()
    records, scoring_play_count = verify_replay(game)
    refire = verify_refire()
    write_report(records, scoring_play_count, refire)
    print(f"PASS: {REPORT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
