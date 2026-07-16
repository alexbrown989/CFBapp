from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import ensure_n12_lookup_import_path
from .data_source import ScoreboardGameState, ScoreboardStub
from .logger import (
    JSONLMarketSeriesLogger,
    JSONLTriggerLogger,
    MARKET_SERIES_FIELDS,
    MARKET_TRIGGER_FIELDS,
    REQUIRED_TRIGGER_FIELDS,
    SCORING_FIELDS,
)
from .main import LiveMonitor
from .markets.base import MarketRef, Quote, utc_now
from .markets.service import MarketService
from .scoring import ScoringContext, score_trigger
from .trigger_detect import TriggerDetector
from .watchlist import WatchGame, build_watchlist

ensure_n12_lookup_import_path()
import _lib_lookup  # type: ignore  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
GAME_ID = "401628374"
PLAYS_CACHE = REPO_ROOT / "research/data/cache/cfbd__plays__50fb856363e865e5.json"
GAMES_CACHE = REPO_ROOT / "research/data/cache/cfbd__games__9d0a751d3eb65363.json"
LINES_CACHE = REPO_ROOT / "research/data/cache/cfbd__lines__9d0a751d3eb65363.json"
RANKINGS_CACHE = REPO_ROOT / "research/data/cache/cfbd__rankings__2024.json"
LOG_PATH = REPO_ROOT / "live/logs/replay_verification.jsonl"
MARKET_SERIES_PATH = REPO_ROOT / "live/logs/replay_market_series_verification.jsonl"
REPORT_PATH = REPO_ROOT / "research/results/n13_stage1_replay_verification.md"
N11_PATH = REPO_ROOT / "research/results/n11_ranking_stratification.parquet"
N06_PATH = REPO_ROOT / "research/results/n06_calibrated_predictions.parquet"


class RecordedKalshiClient:
    """Deterministic read-only quote source for the end-to-end replay."""

    venue = "kalshi"

    def find_market(self, _):
        return None

    def get_quote(self, market_ref: MarketRef) -> Quote:
        return Quote(
            venue=self.venue,
            market_id=market_ref.market_id,
            favorite_outcome_id=market_ref.favorite_outcome_id,
            best_bid=0.40,
            best_ask=0.42,
            mid=0.41,
            spread=0.02,
            implied_prob_raw=0.42,
            implied_prob_no_vig=0.42 / (0.42 + 0.60),
            depth_top_levels=(
                {"level": 1.0, "bid_price": 0.40, "bid_size": 100.0, "ask_price": 0.42, "ask_size": 80.0},
            ),
            volume_since_last_poll=0.0,
            timestamp=utc_now(),
            source_timestamp=None,
            is_stale=False,
            dog_best_ask=0.60,
            no_vig_method="two_sided_derived_ask_normalization_from_yes_no_bids",
        )


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def ap_ranks_for_week(records: list[dict[str, object]], week: int) -> dict[str, int]:
    candidates = [row for row in records if row.get("seasonType") == "regular" and int(row.get("week", -1)) == week]
    if not candidates:
        raise AssertionError(f"no 2024 regular ranking record for week {week}")
    ranks: dict[str, int] = {}
    for poll in candidates[0].get("polls", []):
        if poll.get("poll") == "AP Top 25":
            ranks = {str(rank["school"]): int(rank["rank"]) for rank in poll.get("ranks", [])}
            break
    if len(ranks) != 25:
        raise AssertionError(f"expected 25 AP teams for week {week}, got {len(ranks)}")
    return ranks


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
    ap_ranks = ap_ranks_for_week(rankings, 5)
    assert ap_ranks["Georgia"] == 2
    watchlist = build_watchlist(games, lines, rank_by_team=ap_ranks)
    game = watchlist.get(GAME_ID)
    assert game is not None, "cached 2024 top-25 game did not enter watch list"
    assert game.favorite == "Georgia" and game.dog == "Alabama"
    assert game.pregame_spread == -2.0
    assert game.spread_bucket == "small_favorite"
    assert game.ranking_bucket == "top_5"
    return game


def verify_replay(game: WatchGame) -> tuple[list[dict[str, object]], int, int]:
    plays = [row for row in load_json(PLAYS_CACHE) if str(row.get("gameId")) == GAME_ID]
    scoring_plays = sorted((row for row in plays if row.get("scoring") is True), key=chronological_key)
    batches = [[score_state(play, index)] for index, play in enumerate(scoring_plays, start=1)]
    expected_detector = TriggerDetector()
    expected_trigger_polls: list[int] = []
    for poll_number, batch in enumerate(batches, start=1):
        state = replace(batch[0], poll_number=poll_number)
        if expected_detector.process(state, game):
            expected_trigger_polls.append(poll_number)
    source = ScoreboardStub(batches)
    detector = TriggerDetector()
    logger = JSONLTriggerLogger(LOG_PATH)
    LOG_PATH.unlink(missing_ok=True)
    MARKET_SERIES_PATH.unlink(missing_ok=True)
    market_client = RecordedKalshiClient()
    market_service = MarketService(
        [market_client],
        market_series_logger=JSONLMarketSeriesLogger(MARKET_SERIES_PATH),
    )
    market_service.set_mapping(
        GAME_ID,
        MarketRef(
            venue="kalshi",
            market_id="RECORDED-GEORGIA-ALABAMA",
            favorite_outcome_id="RECORDED-GEORGIA-ALABAMA:yes",
            favorite_side="yes",
            dog_outcome_id="RECORDED-GEORGIA-ALABAMA:no",
            favorite_team="Georgia",
            dog_team="Alabama",
            mapping_confidence="exact",
            mapping_reason="recorded replay fixture with explicit Georgia YES outcome",
        ),
    )
    context_provider = _replay_context_provider()
    monitor = LiveMonitor(
        source=source,
        watchlist={GAME_ID: game},
        detector=detector,
        logger=logger,
        poll_interval_seconds=0,
        scorer=score_trigger,
        context_provider=context_provider,
        market_service=market_service,
    )
    for _ in batches:
        monitor.poll_once()

    records = [json.loads(line) for line in LOG_PATH.read_text(encoding="utf-8").splitlines()]
    actual = [
        (record["threshold_crossed"], record["period"], record["clock"], record["fav_score"], record["dog_score"])
        for record in records
    ]
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
    assert len(records) == 7
    actual_trigger_polls = sorted({int(record["poll_number"]) for record in records})
    assert actual_trigger_polls == expected_trigger_polls
    grouped_states: dict[int, tuple[object, ...]] = {}
    for record in records:
        poll_number = int(record["poll_number"])
        state = (
            record["fav_score"],
            record["dog_score"],
            record["period"],
            record["clock"],
            record["timestamp"],
        )
        if poll_number in grouped_states:
            assert grouped_states[poll_number] == state, {
                "poll_number": poll_number,
                "first_state": grouped_states[poll_number],
                "later_state": state,
            }
        else:
            grouped_states[poll_number] = state
    required = set(REQUIRED_TRIGGER_FIELDS) | set(SCORING_FIELDS) | set(MARKET_TRIGGER_FIELDS)
    assert all(set(record) == required for record in records)
    assert [record["tier_used"] for record in records] == [3, 3, 3, 3, 3, 2, 2]
    assert all(record["n06_calibrated_prob"] is not None for record in records[:5])
    assert all(record["n06_unavailable_reason"] == "unavailable - replay re-fire has no committed feature snapshot" for record in records[5:])
    assert all(record["market_status"] == "OK" for record in records)
    assert all(record["kalshi_market_id"] == "RECORDED-GEORGIA-ALABAMA" for record in records)
    assert all(record["kalshi_gap"] is not None for record in records)
    assert all(record["polymarket_gap"] is None for record in records)
    series = [json.loads(line) for line in MARKET_SERIES_PATH.read_text(encoding="utf-8").splitlines()]
    assert len(series) == len(scoring_plays)
    assert all(set(row) == set(MARKET_SERIES_FIELDS) for row in series)
    assert sum(bool(row["is_triggered"]) for row in series) == len(expected_trigger_polls)
    return records, len(scoring_plays), len(series)


def _replay_context_provider():
    n11 = pd.read_parquet(N11_PATH)
    n11 = n11[(n11["game_id"] == int(GAME_ID)) & (n11["season"] == 2024)]
    n06 = pd.read_parquet(N06_PATH)
    n06 = n06[(n06["game_id"] == int(GAME_ID)) & (n06["scheme"] == "U") & (n06["fold"] == 2024)]
    rows = n11.merge(
        n06,
        on=["game_id", "trigger_play_id", "trigger_sequence", "fav_deficit", "season"],
        suffixes=("_n11", "_n06"),
        validate="one_to_one",
    )
    by_deficit = {int(row["fav_deficit"]): row for row in rows.to_dict(orient="records")}
    fit = next(
        item
        for item in _lib_lookup.load_scoring_spec()["n06_fitted_state"]["fits"]
        if item["scheme"] == "U" and int(item["fold"]) == 2024
    )
    seen: Counter[int] = Counter()

    def provider(_: WatchGame, __: ScoreboardGameState, event) -> ScoringContext:
        seen[event.threshold_crossed] += 1
        if seen[event.threshold_crossed] > 1:
            return ScoringContext(
                spread_bucket="small_favorite",
                ranking_bucket="top_5",
                fluke_bucket=None,
                tier3_features=None,
                tier3_certified=False,
                tier3_unavailable_reason="unavailable - replay re-fire has no committed feature snapshot",
            )
        row = by_deficit[event.threshold_crossed]
        return ScoringContext(
            spread_bucket=str(row["spread_bucket"]),
            ranking_bucket=str(row["ranking_bucket"]),
            fluke_bucket=str(row["fluke_bucket"]),
            time_bucket=str(row["time_bucket_n11"]),
            tier3_features={feature: row[feature] for feature in fit["core_features"]},
            tier3_certified=True,
            tier3_feature_source="cached_historical",
            tier3_scheme="U",
            tier3_fold=2024,
            tier3_unavailable_reason=None,
        )

    return provider


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


def write_report(
    records: list[dict[str, object]], scoring_play_count: int, market_series_count: int, refire: list[int]
) -> None:
    table_rows = "\n".join(
        f"| {row['threshold_crossed']} | Q{row['period']} {row['clock']} | {row['fav_score']}-{row['dog_score']} | {row['poll_number']} |"
        for row in records
    )
    report = f"""# N13 Stage 1 + Stage 2 + Stage 3 Replay Verification

Date: 2026-07-15

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
- JSONL schema: PASS, all {len(REQUIRED_TRIGGER_FIELDS)} Stage 1 fields, {len(SCORING_FIELDS)} Stage 2 fields, and {len(MARKET_TRIGGER_FIELDS)} additive Stage 3 fields present
- Scoring tiers: first five committed trigger snapshots reached Tier 3; the two Q4 re-fire events reached Tier 2 and explicitly suppressed N06 because no committed feature snapshot exists
- Every scoring read includes both-label baseline_C, historical descriptive context where available, tier reasons, sample sizes, reliability, and conformal bounds whenever N06 is shown
- Recorded Kalshi quote: favorite ask 0.42, dog ask 0.60, no-vig favorite probability {0.42 / 1.02:.6f}
- End-to-end trigger gap: baseline_C `favorite_final_win` minus the recorded no-vig probability on all seven trigger records
- Per-poll market-series records: {market_series_count}; 7 trigger records occurred across {len({row['poll_number'] for row in records})} polls because Stage 1 multi-threshold crossings share one observed game-state (`D=3/7`, `D=10/14`, `D=21`, Q4 re-fire `D=3/7`)
- Every trigger record sharing a poll has identical favorite/dog score, period, clock, and timestamp
- Market label guard: the quote is compared only to `favorite_final_win`; `deficit_erased` remains descriptive context
- Data source recorded as `stub`
- Network/API calls: 0

## Acceptance

Trigger timing, one-fire deduplication, recovery-based re-arming, multi-threshold crossing, watch-list construction, tiered scoring, market gap computation, per-poll quote logging, and the additive local log schema all pass. `ScoreboardLive` was not activated and the replay made no network calls.
"""
    REPORT_PATH.write_text(report, encoding="utf-8", newline="\n")


def main() -> int:
    game = verify_watchlist()
    records, scoring_play_count, market_series_count = verify_replay(game)
    refire = verify_refire()
    write_report(records, scoring_play_count, market_series_count, refire)
    print(f"PASS: {REPORT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
