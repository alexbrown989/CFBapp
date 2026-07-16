"""Failure-isolated market discovery, polling, logging, and gap orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..data_source import ScoreboardGameState
from ..scoring import ScoringResult
from ..watchlist import WatchGame
from .base import (
    GameMarket,
    MappingDecision,
    MarketAuthenticationError,
    MarketClient,
    MarketRef,
    Quote,
    utc_now,
)
from .gap import compute_gap, validate_favorite_mapping


TRIGGER_MARKET_FIELDS = (
    "kalshi_implied_no_vig",
    "kalshi_implied_raw",
    "kalshi_gap",
    "kalshi_spread",
    "kalshi_depth",
    "kalshi_market_id",
    "polymarket_implied_no_vig",
    "polymarket_implied_raw",
    "polymarket_gap",
    "polymarket_spread",
    "polymarket_depth",
    "polymarket_market_id",
    "best_venue",
    "market_status",
    "market_errors",
)


@dataclass(frozen=True)
class MarketPollResult:
    game_id: str
    quotes: Mapping[str, Quote]
    status: str
    errors: Mapping[str, str]


class MarketService:
    """Public read-only market layer; venue errors never escape poll_game()."""

    def __init__(
        self,
        clients: Sequence[MarketClient],
        *,
        market_series_logger: object | None = None,
        mapping_logger: object | None = None,
        error_logger: object | None = None,
    ) -> None:
        self.clients = {client.venue: client for client in clients}
        self.market_series_logger = market_series_logger
        self.mapping_logger = mapping_logger
        self.error_logger = error_logger
        self._refs: dict[str, dict[str, MarketRef]] = {}
        self._mapping_failures: dict[str, dict[str, str]] = {}

    def discover_game(self, game: GameMarket) -> Mapping[str, MarketRef]:
        refs: dict[str, MarketRef] = {}
        failures: dict[str, str] = {}
        for venue, client in self.clients.items():
            try:
                market_ref = client.find_market(game)
                if market_ref is None:
                    failures[venue] = "NO_MARKET: no unique exact team/date mapping"
                    self._log_mapping(game, venue, None, failures[venue])
                    continue
                pregame_quote = client.get_quote(market_ref)
                if pregame_quote is None or pregame_quote.is_stale:
                    raise ValueError("mapping inversion guard requires a fresh pregame quote")
                validate_favorite_mapping(
                    game,
                    market_ref,
                    pregame_favorite_probability=pregame_quote.implied_prob_no_vig,
                )
                refs[venue] = market_ref
                self._log_mapping(game, venue, market_ref, market_ref.mapping_reason)
            except MarketAuthenticationError:
                # Stage 3 is public-data-only. An auth challenge is a mandatory
                # halt condition, not an ordinary mapping miss.
                raise
            except Exception as exc:
                failures[venue] = f"MAPPING_FAILED: {type(exc).__name__}: {exc}"
                self._log_mapping(game, venue, None, failures[venue])
                self._log_error(game.game_id, venue, "discovery", exc)
        self._refs[game.game_id] = refs
        self._mapping_failures[game.game_id] = failures
        return dict(refs)

    def set_mapping(self, game_id: str, market_ref: MarketRef) -> None:
        """Inject a previously audited mapping, used by replay and daily cache restore."""
        if market_ref.venue not in self.clients:
            raise ValueError(f"no configured client for venue={market_ref.venue!r}")
        self._refs.setdefault(str(game_id), {})[market_ref.venue] = market_ref

    def poll_game(
        self,
        game: WatchGame,
        state: ScoreboardGameState,
        *,
        is_triggered: bool,
    ) -> MarketPollResult:
        refs = self._refs.get(str(game.game_id), {})
        mapping_failures = self._mapping_failures.get(str(game.game_id), {})
        if not refs:
            status = "MAPPING_FAILED" if any(value.startswith("MAPPING_FAILED") for value in mapping_failures.values()) else "NO_MARKET"
            return MarketPollResult(str(game.game_id), {}, status, mapping_failures)
        quotes: dict[str, Quote] = {}
        errors: dict[str, str] = {}
        for venue, market_ref in refs.items():
            client = self.clients[venue]
            try:
                quote = client.get_quote(market_ref)
                if quote is None:
                    errors[venue] = "quote unavailable"
                    continue
                quotes[venue] = quote
                self._log_series(game, state, quote, is_triggered)
            except MarketAuthenticationError:
                raise
            except Exception as exc:
                errors[venue] = f"{type(exc).__name__}: {exc}"
                self._log_error(str(game.game_id), venue, "quote", exc)
        fresh = [quote for quote in quotes.values() if not quote.is_stale]
        if fresh:
            status = "OK"
        elif quotes:
            status = "STALE"
        elif errors:
            status = "NO_MARKET"
        else:
            status = "NO_MARKET"
        return MarketPollResult(str(game.game_id), quotes, status, errors)

    @staticmethod
    def trigger_fields(scoring: ScoringResult, market_poll: MarketPollResult) -> dict[str, object]:
        fields: dict[str, object] = {field: None for field in TRIGGER_MARKET_FIELDS}
        fields["market_status"] = market_poll.status
        fields["market_errors"] = dict(market_poll.errors) or None
        engine_estimate = scoring.tier_1["favorite_final_win"].value
        gaps: dict[str, float] = {}
        for venue, quote in market_poll.quotes.items():
            prefix = venue
            fields[f"{prefix}_market_id"] = quote.market_id
            fields[f"{prefix}_implied_raw"] = quote.implied_prob_raw
            fields[f"{prefix}_implied_no_vig"] = quote.implied_prob_no_vig
            fields[f"{prefix}_spread"] = quote.spread
            fields[f"{prefix}_depth"] = _top_ask_depth(quote)
            if quote.is_stale:
                continue
            result = compute_gap(engine_estimate, quote, label="favorite_final_win")
            fields[f"{prefix}_gap"] = result.gap_no_vig
            gaps[venue] = result.gap_no_vig
        fields["best_venue"] = max(gaps, key=gaps.get) if gaps else None
        return fields

    def _log_series(
        self, game: WatchGame, state: ScoreboardGameState, quote: Quote, is_triggered: bool
    ) -> None:
        if self.market_series_logger is None:
            return
        fav_score, dog_score = _favorite_scores(game, state)
        self.market_series_logger.append(
            {
                "timestamp": quote.timestamp,
                "game_id": str(game.game_id),
                "venue": quote.venue,
                "market_id": quote.market_id,
                "best_bid": quote.best_bid,
                "best_ask": quote.best_ask,
                "mid": quote.mid,
                "spread": quote.spread,
                "implied_prob_raw": quote.implied_prob_raw,
                "implied_prob_no_vig": quote.implied_prob_no_vig,
                "depth": list(quote.depth_top_levels),
                "volume": quote.volume_since_last_poll,
                "fav_score": fav_score,
                "dog_score": dog_score,
                "period": state.period,
                "clock": state.clock,
                "is_triggered": bool(is_triggered),
                "is_stale": quote.is_stale,
            }
        )

    def _log_mapping(
        self, game: GameMarket, venue: str, market_ref: MarketRef | None, reason: str
    ) -> None:
        if self.mapping_logger is None:
            return
        self.mapping_logger.append(
            MappingDecision(
                timestamp=utc_now(),
                game_id=game.game_id,
                venue=venue,
                status="OK" if market_ref is not None else ("MAPPING_FAILED" if reason.startswith("MAPPING_FAILED") else "NO_MARKET"),
                market_id=None if market_ref is None else market_ref.market_id,
                favorite=game.favorite,
                dog=game.dog,
                favorite_outcome_id=None if market_ref is None else market_ref.favorite_outcome_id,
                favorite_side=None if market_ref is None else market_ref.favorite_side,
                mapping_reason=reason,
                inversion_guard_passed=market_ref is not None,
            ).as_dict()
        )

    def _log_error(self, game_id: str, venue: str, operation: str, exc: Exception) -> None:
        if self.error_logger is None:
            return
        self.error_logger.append(
            {
                "timestamp": utc_now(),
                "game_id": str(game_id),
                "venue": venue,
                "operation": operation,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        )


def _favorite_scores(game: WatchGame, state: ScoreboardGameState) -> tuple[int, int]:
    if game.favorite == state.home_team:
        return state.home_score, state.away_score
    if game.favorite == state.away_team:
        return state.away_score, state.home_score
    raise ValueError("watch-list favorite does not match scoreboard teams")


def _top_ask_depth(quote: Quote) -> float:
    return sum(float(level.get("ask_size", 0.0)) for level in quote.depth_top_levels)
