"""Public Polymarket Gamma discovery and CLOB quote adapter."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from .base import (
    GameMarket,
    MarketDataError,
    MarketRef,
    PublicHTTPClient,
    Quote,
    is_timestamp_stale,
    parse_json_array,
    parse_timestamp,
    team_in_text,
    utc_now,
)
from .gap import two_sided_no_vig, validate_favorite_mapping


class PolymarketClient:
    venue = "polymarket"

    def __init__(
        self,
        *,
        http: PublicHTTPClient | None = None,
        gamma_base_url: str = "https://gamma-api.polymarket.com",
        clob_base_url: str = "https://clob.polymarket.com",
        stale_after_seconds: float = 120.0,
    ) -> None:
        self.http = http or PublicHTTPClient()
        self.gamma_base_url = gamma_base_url.rstrip("/")
        self.clob_base_url = clob_base_url.rstrip("/")
        self.stale_after_seconds = float(stale_after_seconds)
        self._last_volume: dict[str, float] = {}

    def find_market(self, game: GameMarket) -> MarketRef | None:
        payload = self.http.get_json(
            self.gamma_base_url,
            "/public-search",
            {
                "q": f"{game.away_team} {game.home_team}",
                "events_status": "active",
                "limit_per_type": 25,
                "search_profiles": "false",
                "search_tags": "false",
            },
        )
        if not isinstance(payload, Mapping):
            raise MarketDataError("Polymarket public-search returned a non-object payload")
        candidates: list[MarketRef] = []
        for event in payload.get("events") or []:
            if not isinstance(event, Mapping):
                continue
            event_text = " ".join(str(event.get(key) or "") for key in ("title", "subtitle", "slug"))
            if not _contains_game_teams(game, event_text):
                continue
            for market in event.get("markets") or []:
                if not isinstance(market, Mapping) or not market.get("enableOrderBook"):
                    continue
                if not _date_matches(game.kickoff, market, event):
                    continue
                mapped = _map_polymarket_outcomes(game, market, event_text)
                if mapped is not None:
                    candidates.append(mapped)
        unique = {candidate.market_id: candidate for candidate in candidates}
        if len(unique) != 1:
            return None
        result = next(iter(unique.values()))
        validate_favorite_mapping(game, result)
        return result

    def get_quote(self, market_ref: MarketRef) -> Quote | None:
        if market_ref.venue != self.venue or market_ref.favorite_side != "token":
            raise MarketDataError("Polymarket quote requested with an incompatible market reference")
        if not market_ref.dog_outcome_id:
            raise MarketDataError("Polymarket binary quote requires both token IDs")
        favorite_book = self._book(market_ref.favorite_outcome_id)
        dog_book = self._book(market_ref.dog_outcome_id)
        favorite_bid, favorite_ask = _best_prices(favorite_book)
        _, dog_ask = _best_prices(dog_book)
        observed_at = utc_now()
        source_timestamp = str(favorite_book.get("timestamp") or "") or None
        volume_delta = self._volume_delta(market_ref)
        return Quote(
            venue=self.venue,
            market_id=market_ref.market_id,
            favorite_outcome_id=market_ref.favorite_outcome_id,
            best_bid=favorite_bid,
            best_ask=favorite_ask,
            mid=(favorite_bid + favorite_ask) / 2.0,
            spread=favorite_ask - favorite_bid,
            implied_prob_raw=favorite_ask,
            implied_prob_no_vig=two_sided_no_vig(favorite_ask, dog_ask),
            depth_top_levels=_favorite_depth(favorite_book),
            volume_since_last_poll=volume_delta,
            timestamp=observed_at,
            source_timestamp=source_timestamp,
            is_stale=is_timestamp_stale(source_timestamp, self.stale_after_seconds),
            dog_best_ask=dog_ask,
        )

    def find_public_smoke_market(self) -> MarketRef:
        payload = self.http.get_json(
            self.gamma_base_url,
            "/markets/keyset",
            {"limit": 100, "ascending": "false", "closed": "false"},
        )
        markets = payload.get("markets") if isinstance(payload, Mapping) else None
        if not isinstance(markets, list):
            raise MarketDataError("Polymarket keyset response is missing markets")
        viable: list[tuple[float, MarketRef]] = []
        for market in markets:
            if not isinstance(market, Mapping) or not market.get("enableOrderBook"):
                continue
            try:
                outcomes = parse_json_array(market.get("outcomes"), "outcomes")
                tokens = parse_json_array(market.get("clobTokenIds"), "clobTokenIds")
                prices = [float(value) for value in parse_json_array(market.get("outcomePrices"), "outcomePrices")]
            except (MarketDataError, TypeError, ValueError):
                continue
            if len(outcomes) != 2 or len(tokens) != 2 or len(prices) != 2 or not all(0 < p < 1 for p in prices):
                continue
            favorite_index = 0 if prices[0] >= prices[1] else 1
            dog_index = 1 - favorite_index
            ref = MarketRef(
                venue=self.venue,
                market_id=str(market.get("conditionId") or market.get("id")),
                favorite_outcome_id=tokens[favorite_index],
                favorite_side="token",
                dog_outcome_id=tokens[dog_index],
                favorite_team=outcomes[favorite_index],
                dog_team=outcomes[dog_index],
                mapping_confidence="exact",
                mapping_reason="public smoke market: higher displayed outcome probability selected",
                metadata={"gamma_market_id": str(market.get("id")), "question": str(market.get("question") or "")},
            )
            viable.append((float(market.get("liquidityNum") or market.get("liquidity") or 0), ref))
        for _, market_ref in sorted(viable, reverse=True, key=lambda item: item[0]):
            try:
                quote = self.get_quote(market_ref)
            except MarketDataError:
                continue
            if quote is not None:
                return market_ref
        raise MarketDataError("no liquid two-sided Polymarket smoke market was available")

    def _book(self, token_id: str) -> Mapping[str, Any]:
        payload = self.http.get_json(self.clob_base_url, "/book", {"token_id": token_id})
        if not isinstance(payload, Mapping):
            raise MarketDataError("Polymarket CLOB book returned a non-object payload")
        return payload

    def _volume_delta(self, market_ref: MarketRef) -> float | None:
        market_id = str(market_ref.metadata.get("gamma_market_id") or "")
        if not market_id:
            return None
        try:
            payload = self.http.get_json(self.gamma_base_url, f"/markets/{market_id}")
            total = float(payload.get("volumeNum") or payload.get("volume"))
        except (MarketDataError, TypeError, ValueError, AttributeError):
            return None
        prior = self._last_volume.get(market_ref.market_id)
        self._last_volume[market_ref.market_id] = total
        return None if prior is None else max(0.0, total - prior)


def _map_polymarket_outcomes(
    game: GameMarket, market: Mapping[str, Any], event_text: str
) -> MarketRef | None:
    try:
        outcomes = parse_json_array(market.get("outcomes"), "outcomes")
        tokens = parse_json_array(market.get("clobTokenIds"), "clobTokenIds")
    except MarketDataError:
        return None
    if len(outcomes) != 2 or len(tokens) != 2:
        return None
    favorite_index: int | None = None
    dog_index: int | None = None
    for index, outcome in enumerate(outcomes):
        if team_in_text(game.favorite, outcome):
            favorite_index = index
        if team_in_text(game.dog, outcome):
            dog_index = index
    reason = "team names mapped directly from Polymarket outcomes"
    if favorite_index is None or dog_index is None:
        labels = [outcome.lower() for outcome in outcomes]
        if set(labels) != {"yes", "no"}:
            return None
        market_text = " ".join(
            str(market.get(key) or "") for key in ("question", "groupItemTitle", "slug")
        )
        identified_favorite = team_in_text(game.favorite, market_text) and not team_in_text(game.dog, market_text)
        identified_dog = team_in_text(game.dog, market_text) and not team_in_text(game.favorite, market_text)
        if not identified_favorite and not identified_dog:
            return None
        yes_index = labels.index("yes")
        no_index = labels.index("no")
        favorite_index, dog_index = (yes_index, no_index) if identified_favorite else (no_index, yes_index)
        reason = "binary Yes/No token mapped from the uniquely named team in the market question"
    if favorite_index == dog_index:
        return None
    return MarketRef(
        venue="polymarket",
        market_id=str(market.get("conditionId") or market.get("id")),
        favorite_outcome_id=tokens[favorite_index],
        favorite_side="token",
        dog_outcome_id=tokens[dog_index],
        favorite_team=game.favorite,
        dog_team=game.dog,
        mapping_confidence="exact",
        mapping_reason=reason,
        metadata={
            "gamma_market_id": str(market.get("id")),
            "question": str(market.get("question") or event_text),
            "outcomes": outcomes,
        },
    )


def _contains_game_teams(game: GameMarket, text: str) -> bool:
    return team_in_text(game.favorite, text) and team_in_text(game.dog, text)


def _date_matches(kickoff: str | None, market: Mapping[str, Any], event: Mapping[str, Any]) -> bool:
    if not kickoff:
        return True
    game_time = parse_timestamp(kickoff)
    candidate = next(
        (
            parse_timestamp(value)
            for value in (
                market.get("gameStartTime"),
                market.get("eventStartTime"),
                event.get("startTime"),
                event.get("eventDate"),
            )
            if parse_timestamp(value) is not None
        ),
        None,
    )
    return candidate is not None and abs((candidate - game_time).total_seconds()) <= 36 * 3600


def _best_prices(book: Mapping[str, Any]) -> tuple[float, float]:
    bids = [_level(row) for row in book.get("bids") or []]
    asks = [_level(row) for row in book.get("asks") or []]
    if not bids or not asks:
        raise MarketDataError("Polymarket CLOB book is not two-sided")
    return max(price for price, _ in bids), min(price for price, _ in asks)


def _favorite_depth(book: Mapping[str, Any], levels: int = 3) -> tuple[Mapping[str, float], ...]:
    bids = sorted((_level(row) for row in book.get("bids") or []), reverse=True)[:levels]
    asks = sorted((_level(row) for row in book.get("asks") or []))[:levels]
    depth: list[Mapping[str, float]] = []
    for index in range(max(len(bids), len(asks))):
        row: dict[str, float] = {"level": float(index + 1)}
        if index < len(bids):
            row.update({"bid_price": bids[index][0], "bid_size": bids[index][1]})
        if index < len(asks):
            row.update({"ask_price": asks[index][0], "ask_size": asks[index][1]})
        depth.append(row)
    return tuple(depth)


def _level(row: object) -> tuple[float, float]:
    if isinstance(row, Mapping):
        return float(row["price"]), float(row["size"])
    if isinstance(row, (list, tuple)) and len(row) >= 2:
        return float(row[0]), float(row[1])
    raise MarketDataError(f"invalid Polymarket orderbook level: {row!r}")
