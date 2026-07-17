from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
from threading import RLock
from typing import Callable, Mapping

from .config import RiskConfig, RuntimeConfigStore, Settings
from .data_source import ScoreboardGameState, ScoreboardLive, ScoreboardSource, ScoreboardStub
from .logger import (
    JSONLMarketSeriesLogger,
    JSONLRecordLogger,
    JSONLTriggerLogger,
    read_trigger_records,
)
from .markets import GameMarket, KalshiClient, PolymarketClient
from .markets.base import Quote
from .markets.service import MarketPollResult, MarketService
from .risk import (
    WIN_MARKET_LABEL,
    comfort_stake_fraction,
    expected_losing_streaks,
    expected_value,
    fractional_kelly,
    kelly_fraction,
    losing_streak_probability,
    risk_of_ruin,
)
from .scoring import ScoringContext, ScoringResult, score_trigger
from .trigger_detect import TriggerDetector, TriggerEvent
from .watchlist import WatchGame


Scorer = Callable[[TriggerEvent, ScoringContext], ScoringResult]
ContextProvider = Callable[[WatchGame, ScoreboardGameState, TriggerEvent], ScoringContext]
DASHBOARD_PATH = Path(__file__).parent / "static" / "dashboard.html"
LOCALHOST = "127.0.0.1"


class LiveMonitor:
    """Polling coordinator and in-memory read model for the local dashboard."""

    def __init__(
        self,
        source: ScoreboardSource,
        watchlist: dict[str, WatchGame],
        detector: TriggerDetector,
        logger: JSONLTriggerLogger,
        poll_interval_seconds: float,
        scorer: Scorer | None = None,
        context_provider: ContextProvider | None = None,
        market_service: MarketService | None = None,
    ) -> None:
        self.source = source
        self.watchlist = watchlist
        self.detector = detector
        self.logger = logger
        self.poll_interval_seconds = poll_interval_seconds
        self.scorer = scorer
        self.context_provider = context_provider
        self.market_service = market_service
        self.current_states: dict[str, ScoreboardGameState] = {}
        self.latest_scoring: dict[str, ScoringResult] = {}
        self.latest_markets: dict[str, MarketPollResult] = {}
        self.latest_triggers: dict[str, TriggerEvent] = {}
        self.trigger_history: list[
            tuple[WatchGame, ScoreboardGameState, TriggerEvent, ScoringResult | None, MarketPollResult]
        ] = []
        self._state_lock = RLock()

    @property
    def mode(self) -> str:
        return "stub" if isinstance(self.source, ScoreboardStub) else "live"

    def configure_watchlist(self, watchlist: dict[str, WatchGame]) -> None:
        """Install the daily watch list and cache exact market mappings once per game."""
        self.watchlist = dict(watchlist)
        if self.market_service is None:
            return
        for game in self.watchlist.values():
            self.market_service.discover_game(
                GameMarket(
                    game_id=game.game_id,
                    season=game.season,
                    week=game.week,
                    home_team=game.home_team,
                    away_team=game.away_team,
                    favorite=game.favorite,
                    dog=game.dog,
                    pregame_spread=game.pregame_spread,
                    kickoff=game.kickoff,
                )
            )

    def poll_once(self) -> int:
        event_count = 0
        for state in self.source.poll():
            game = self.watchlist.get(state.game_id)
            if game is None:
                continue
            events = self.detector.process(state, game)
            market_poll = (
                self.market_service.poll_game(game, state, is_triggered=bool(events))
                if self.market_service is not None
                else MarketPollResult(str(game.game_id), {}, "NO_MARKET", {})
            )
            with self._state_lock:
                self.current_states[str(game.game_id)] = state
                self.latest_markets[str(game.game_id)] = market_poll
            for event in events:
                scoring_result = None
                if self.scorer is not None:
                    if self.context_provider is None:
                        raise RuntimeError("a scoring context provider is required when scoring is enabled")
                    context = self.context_provider(game, state, event)
                    scoring_result = self.scorer(event, context)
                market_fields = (
                    None
                    if scoring_result is None or self.market_service is None
                    else self.market_service.trigger_fields(scoring_result, market_poll)
                )
                self.logger.append(
                    event,
                    None if scoring_result is None else scoring_result.as_log_fields(),
                    market_fields,
                )
                with self._state_lock:
                    self.latest_triggers[str(game.game_id)] = event
                    if scoring_result is not None:
                        self.latest_scoring[str(game.game_id)] = scoring_result
                    self.trigger_history.append((game, state, event, scoring_result, market_poll))
                print(
                    f"TRIGGER {event.game_id} {event.favorite} D={event.threshold_crossed} "
                    f"score={event.fav_score}-{event.dog_score} Q{event.period} {event.clock}"
                    + ("" if scoring_result is None else f" tier={scoring_result.tier_used}")
                )
                if scoring_result is not None:
                    print("SCORING " + json.dumps(scoring_result.as_dict(), sort_keys=True))
            event_count += len(events)
        return event_count

    async def poll_forever(self) -> None:
        while True:
            self.poll_once()
            await asyncio.sleep(self.poll_interval_seconds)

    def dashboard_snapshot(self, risk_config: RiskConfig) -> dict[str, object]:
        with self._state_lock:
            games = [
                _serialize_game(
                    game,
                    self.current_states.get(str(game.game_id)),
                    self.latest_triggers.get(str(game.game_id)),
                    self.latest_scoring.get(str(game.game_id)),
                    self.latest_markets.get(str(game.game_id)),
                    risk_config,
                )
                for game in self.watchlist.values()
            ]
        return {
            "mode": self.mode,
            "poll_interval_seconds": self.poll_interval_seconds,
            "games": games,
            "game_count": len(games),
            "methodology_notice": (
                "Market gaps and all financial math use Tier 1 favorite_final_win only. "
                "Tier 3 and its conformal band estimate deficit_erased."
            ),
        }

    def recent_trigger_snapshots(self, risk_config: RiskConfig, limit: int = 100) -> list[dict[str, object]]:
        with self._state_lock:
            selected = self.trigger_history[-max(1, int(limit)):]
            return [
                _serialize_game(game, state, trigger, scoring, market, risk_config)
                for game, state, trigger, scoring, market in reversed(selected)
            ]


def create_source(settings: Settings) -> ScoreboardSource:
    if settings.data_source == "stub":
        return ScoreboardStub([])
    if settings.data_source == "live":
        return ScoreboardLive(base_url=settings.cfbd_base_url)
    raise ValueError(f"unsupported N13_DATA_SOURCE={settings.data_source!r}")


def create_monitor(settings: Settings | None = None) -> LiveMonitor:
    settings = settings or Settings.from_env()
    log_dir = Path(__file__).parent / "logs"
    market_service = MarketService(
        [KalshiClient(), PolymarketClient()],
        market_series_logger=JSONLMarketSeriesLogger(log_dir / "market_series.jsonl"),
        mapping_logger=JSONLRecordLogger(
            log_dir / "market_mappings.jsonl",
            (
                "timestamp", "game_id", "venue", "status", "market_id", "favorite", "dog",
                "favorite_outcome_id", "favorite_side", "mapping_reason", "inversion_guard_passed",
            ),
        ),
        error_logger=JSONLRecordLogger(
            log_dir / "market_errors.jsonl",
            ("timestamp", "game_id", "venue", "operation", "error_type", "message"),
        ),
    )
    return LiveMonitor(
        source=create_source(settings),
        watchlist={},
        detector=TriggerDetector(settings.deficit_thresholds),
        logger=JSONLTriggerLogger(log_dir / "triggers.jsonl"),
        poll_interval_seconds=settings.poll_interval_seconds,
        scorer=score_trigger,
        context_provider=default_scoring_context,
        market_service=market_service,
    )


def default_scoring_context(
    game: WatchGame,
    _: ScoreboardGameState,
    __: TriggerEvent,
) -> ScoringContext:
    """Stage 2 context available without /live/plays."""
    return ScoringContext(
        spread_bucket=game.spread_bucket,
        ranking_bucket=game.ranking_bucket,
        fluke_bucket=None,
        tier3_features=None,
        tier3_certified=False,
        tier3_unavailable_reason="unavailable - no live play feed",
    )


def _serialize_game(
    game: WatchGame,
    state: ScoreboardGameState | None,
    trigger: TriggerEvent | None,
    scoring: ScoringResult | None,
    market: MarketPollResult | None,
    risk_config: RiskConfig,
) -> dict[str, object]:
    state_payload = None if state is None else asdict(state)
    scoring_payload = None if scoring is None else scoring.as_dict()
    market_payload = _serialize_market(market)
    risk_payload = _serialize_risk(scoring, market, risk_config)
    status = "MONITORING"
    if state is not None and state.status.lower() in {"final", "completed", "post"}:
        status = "FINAL"
    elif trigger is not None:
        status = "TRIGGERED"
    elif market is not None and market.status in {"NO_MARKET", "MAPPING_FAILED"}:
        status = market.status
    fav_score = dog_score = deficit = None
    if state is not None:
        fav_score, dog_score = game.favorite_scores(state)
        deficit = fav_score - dog_score
    return {
        "game_id": str(game.game_id),
        "season": game.season,
        "week": game.week,
        "favorite": game.favorite,
        "dog": game.dog,
        "favorite_ap_rank": game.favorite_ap_rank,
        "ranking_bucket": game.ranking_bucket,
        "pregame_spread": game.pregame_spread,
        "home_team": game.home_team,
        "away_team": game.away_team,
        "kickoff": game.kickoff,
        "state": state_payload,
        "favorite_score": fav_score,
        "dog_score": dog_score,
        "deficit": deficit,
        "status": status,
        "latest_trigger": None if trigger is None else trigger.as_dict(),
        "scoring": scoring_payload,
        "market": market_payload,
        "risk": risk_payload,
    }


def _serialize_market(market: MarketPollResult | None) -> dict[str, object]:
    if market is None:
        return {"status": "NO_MARKET", "errors": {}, "quotes": {}, "best_venue": None}
    quotes = {venue: quote.as_dict() for venue, quote in market.quotes.items()}
    fresh_gaps = {
        venue: quote
        for venue, quote in market.quotes.items()
        if not quote.is_stale
    }
    best_venue = None
    if fresh_gaps:
        # Actual best venue depends on engine probability and is filled by risk serialization.
        best_venue = min(fresh_gaps, key=lambda name: fresh_gaps[name].implied_prob_no_vig)
    return {
        "status": market.status,
        "errors": dict(market.errors),
        "quotes": quotes,
        "best_venue": best_venue,
    }


def _serialize_risk(
    scoring: ScoringResult | None,
    market: MarketPollResult | None,
    config: RiskConfig,
) -> dict[str, object]:
    notice = (
        "No label-matched conformal interval is available for favorite_final_win - "
        "uncertainty is not quantified for this estimate."
    )
    if scoring is None:
        return {"status": "NO_ENGINE_ESTIMATE", "label": WIN_MARKET_LABEL, "notice": notice, "venues": {}}
    estimate = scoring.tier_1[WIN_MARKET_LABEL]
    if market is None or not market.quotes:
        return {"status": "NO_MARKET", "label": WIN_MARKET_LABEL, "notice": notice, "venues": {}}
    venue_reads: dict[str, object] = {}
    for venue, quote in market.quotes.items():
        if quote.is_stale:
            venue_reads[venue] = {"status": "STALE", "market_id": quote.market_id}
            continue
        venue_reads[venue] = _risk_for_quote(
            probability=estimate.value,
            reliability=estimate.reliability_flag,
            tier=scoring.tier_used,
            quote=quote,
            config=config,
        )
    return {"status": "OK", "label": WIN_MARKET_LABEL, "notice": notice, "venues": venue_reads}


def _risk_for_quote(
    *,
    probability: float,
    reliability: str,
    tier: int,
    quote: Quote,
    config: RiskConfig,
) -> dict[str, object]:
    decimal_odds = 1.0 / quote.implied_prob_raw
    ev = expected_value(probability, decimal_odds, label=WIN_MARKET_LABEL)
    full_kelly = kelly_fraction(probability, decimal_odds, label=WIN_MARKET_LABEL)
    suggested_uncapped = fractional_kelly(
        probability,
        decimal_odds,
        config.kelly_fraction,
        tier,
        None,
        reliability,
        label=WIN_MARKET_LABEL,
    )
    suggested = min(suggested_uncapped, config.max_bankroll_fraction_per_bet)
    ruin = risk_of_ruin(
        config.bankroll,
        suggested,
        probability,
        decimal_odds,
        config.season_bets,
        config.drawdown_floor,
        label=WIN_MARKET_LABEL,
    )
    comfort = comfort_stake_fraction(
        config.bankroll,
        suggested,
        probability,
        decimal_odds,
        config.season_bets,
        config.drawdown_floor,
        config.ruin_comfort_threshold,
        label=WIN_MARKET_LABEL,
    )
    gap = probability - quote.implied_prob_no_vig
    streaks = {
        str(length): {
            "probability": losing_streak_probability(
                probability, length, config.season_bets, label=WIN_MARKET_LABEL
            ),
            "expected_count": expected_losing_streaks(
                probability, length, config.season_bets, label=WIN_MARKET_LABEL
            ),
        }
        for length in (3, 5, 7, 10)
    }
    return {
        "status": "OK",
        "market_id": quote.market_id,
        "engine_probability": probability,
        "decimal_odds": decimal_odds,
        "ev_per_dollar": ev,
        "positive_ev": ev > 0.0,
        "full_kelly_fraction": full_kelly,
        "suggested_fraction_uncapped": suggested_uncapped,
        "suggested_fraction": suggested,
        "suggested_dollars": config.bankroll * suggested,
        "cap_applied": suggested < suggested_uncapped,
        "reliability": reliability,
        "reliability_factor": _reliability_factor(reliability),
        "tier_factor": 1.0,
        "ruin_probability": ruin,
        "ruin_comfort_threshold": config.ruin_comfort_threshold,
        "comfort_fraction": comfort,
        "comfort_dollars": config.bankroll * comfort,
        "ruin_warning": ruin > config.ruin_comfort_threshold,
        "drawdown_floor": config.drawdown_floor,
        "season_bets": config.season_bets,
        "streaks": streaks,
        "gap_no_vig": gap,
        "spread": quote.spread,
        "survives_friction": gap > quote.spread,
        "favorite_longshot_bias_note": quote.implied_prob_raw < 0.10,
        "sizing_formula": (
            "full Kelly x configured fractional Kelly x reliability factor; "
            "tier factor is 1.00 and no conformal-width multiplier is applied"
        ),
    }


def _reliability_factor(reliability: str) -> float:
    return {"reliable": 1.0, "thin": 0.5, "unreliable": 0.25}.get(
        str(reliability).lower(), 0.25
    )


def create_app(
    monitor: LiveMonitor | None = None,
    config_store: RuntimeConfigStore | None = None,
    settings: Settings | None = None,
):
    """Create the localhost-only FastAPI presentation layer."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("FastAPI is required to serve the Stage 4 dashboard") from exc

    resolved_settings = settings or Settings.from_env()
    if resolved_settings.dashboard_host != LOCALHOST:
        raise ValueError("Stage 4 dashboard must bind to 127.0.0.1")
    resolved_monitor = monitor or create_monitor(resolved_settings)
    resolved_store = config_store or RuntimeConfigStore(resolved_settings.runtime_config_path)
    poll_task: asyncio.Task[None] | None = None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal poll_task
        if os.getenv("N13_AUTO_POLL", "0") == "1":
            poll_task = asyncio.create_task(resolved_monitor.poll_forever())
        try:
            yield
        finally:
            if poll_task is not None:
                poll_task.cancel()
                try:
                    await poll_task
                except asyncio.CancelledError:
                    pass

    dashboard = FastAPI(title="N13 Live Monitor", version="0.4.0", lifespan=lifespan)
    dashboard.state.monitor = resolved_monitor
    dashboard.state.config_store = resolved_store
    dashboard.state.bind_host = LOCALHOST

    @dashboard.get("/")
    def dashboard_page():
        if not DASHBOARD_PATH.exists():
            raise HTTPException(status_code=500, detail="dashboard asset is missing")
        return FileResponse(DASHBOARD_PATH, media_type="text/html")

    @dashboard.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "data_source": type(resolved_monitor.source).__name__,
            "mode": resolved_monitor.mode,
            "watchlist_games": len(resolved_monitor.watchlist),
            "read_only": True,
            "bind_host": LOCALHOST,
        }

    @dashboard.get("/api/state")
    def api_state() -> dict[str, object]:
        return resolved_monitor.dashboard_snapshot(resolved_store.get())

    @dashboard.get("/api/triggers")
    def api_triggers(limit: int = 100) -> dict[str, object]:
        bounded = min(500, max(1, int(limit)))
        records = read_trigger_records(resolved_monitor.logger.path)
        return {
            "triggers": list(reversed(records[-bounded:])),
            "snapshots": resolved_monitor.recent_trigger_snapshots(resolved_store.get(), bounded),
            "count": min(len(records), bounded),
        }

    @dashboard.get("/api/game/{game_id}")
    def api_game(game_id: str) -> dict[str, object]:
        snapshot = resolved_monitor.dashboard_snapshot(resolved_store.get())
        for game in snapshot["games"]:
            if game["game_id"] == str(game_id):
                return game
        raise HTTPException(status_code=404, detail="game is not in the current watch list")

    @dashboard.get("/api/config")
    def api_config() -> dict[str, object]:
        return {
            **resolved_store.get().as_dict(),
            "bind_host": LOCALHOST,
            "mode": resolved_monitor.mode,
        }

    @dashboard.post("/api/config")
    def update_config(values: Mapping[str, object]) -> dict[str, object]:
        try:
            updated = resolved_store.update(values)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return updated.as_dict()

    @dashboard.post("/poll-once")
    def poll_once() -> dict[str, int]:
        return {"triggers_detected": resolved_monitor.poll_once()}

    return dashboard


try:
    app = create_app()
except RuntimeError:
    app = None


def main() -> int:
    parser = argparse.ArgumentParser(description="N13 localhost live monitor")
    parser.add_argument("--once", action="store_true", help="run one poll and exit")
    parser.add_argument("--serve", action="store_true", help="serve the dashboard on localhost")
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.serve:
        import uvicorn

        uvicorn.run(
            "live.main:app",
            host=LOCALHOST,
            port=settings.dashboard_port,
            reload=False,
        )
        return 0
    monitor = create_monitor(settings)
    if args.once:
        print(f"poll complete: {monitor.poll_once()} trigger(s)")
        return 0
    asyncio.run(monitor.poll_forever())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
