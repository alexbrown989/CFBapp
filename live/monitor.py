from __future__ import annotations

import asyncio
import json
from threading import RLock
from typing import Callable

from .config import RiskConfig
from .data_source import ScoreboardGameState, ScoreboardSource, ScoreboardStub
from .logger import JSONLTriggerLogger
from .markets import GameMarket
from .markets.service import MarketPollResult, MarketService
from .presentation import serialize_game
from .scoring import ScoringContext, ScoringResult
from .trigger_detect import TriggerDetector, TriggerEvent
from .watchlist import WatchGame

Scorer = Callable[[TriggerEvent, ScoringContext], ScoringResult]
ContextProvider = Callable[[WatchGame, ScoreboardGameState, TriggerEvent], ScoringContext]


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
                serialize_game(
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
                serialize_game(game, state, trigger, scoring, market, risk_config)
                for game, state, trigger, scoring, market in reversed(selected)
            ]



def default_scoring_context(
    game: WatchGame,
    _state: ScoreboardGameState,
    _event: TriggerEvent,
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
