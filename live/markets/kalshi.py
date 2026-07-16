"""Public Kalshi market discovery and reciprocal-bid orderbook adapter."""

from __future__ import annotations

from typing import Any, Mapping

from .base import GameMarket, MarketDataError, MarketRef, PublicHTTPClient, Quote, team_in_text, utc_now
from .gap import two_sided_no_vig, validate_favorite_mapping


class KalshiClient:
    venue = "kalshi"

    def __init__(
        self,
        *,
        http: PublicHTTPClient | None = None,
        base_url: str = "https://external-api.kalshi.com/trade-api/v2",
    ) -> None:
        self.http = http or PublicHTTPClient()
        self.base_url = base_url.rstrip("/")
        self._last_volume: dict[str, float] = {}

    def find_market(self, game: GameMarket) -> MarketRef | None:
        payload = self.http.get_json(
            self.base_url,
            "/markets",
            {"series_ticker": "KXNCAAFGAME", "limit": 1000, "mve_filter": "exclude"},
        )
        markets = payload.get("markets") if isinstance(payload, Mapping) else None
        if not isinstance(markets, list):
            raise MarketDataError("Kalshi markets response is missing markets")
        candidates: list[MarketRef] = []
        for market in markets:
            if not isinstance(market, Mapping) or not _contains_game_teams(game, market):
                continue
            if not _date_matches(game, market):
                continue
            mapped = _map_kalshi_market(game, market)
            if mapped is not None:
                candidates.append(mapped)
        unique = {candidate.market_id: candidate for candidate in candidates}
        if len(unique) != 1:
            return None
        result = next(iter(unique.values()))
        validate_favorite_mapping(game, result)
        return result

    def get_quote(self, market_ref: MarketRef) -> Quote | None:
        if market_ref.venue != self.venue or market_ref.favorite_side not in {"yes", "no"}:
            raise MarketDataError("Kalshi quote requested with an incompatible market reference")
        market_payload = self.http.get_json(self.base_url, f"/markets/{market_ref.market_id}")
        market = market_payload.get("market") if isinstance(market_payload, Mapping) else None
        if not isinstance(market, Mapping):
            raise MarketDataError("Kalshi get-market response is missing market")
        orderbook_payload = self.http.get_json(
            self.base_url, f"/markets/{market_ref.market_id}/orderbook", {"depth": 10}
        )
        book = orderbook_payload.get("orderbook_fp") if isinstance(orderbook_payload, Mapping) else None
        if not isinstance(book, Mapping):
            raise MarketDataError("Kalshi orderbook response is missing orderbook_fp")
        yes_bids = _book_levels(book.get("yes_dollars"))
        no_bids = _book_levels(book.get("no_dollars"))
        yes_bid, no_bid = _best_reciprocal_bids(yes_bids, no_bids, market)
        yes_ask = 1.0 - no_bid
        no_ask = 1.0 - yes_bid
        _validate_direct_prices(market, yes_bid, yes_ask, no_bid, no_ask)

        if market_ref.favorite_side == "yes":
            favorite_bid, favorite_ask, dog_ask = yes_bid, yes_ask, no_ask
            favorite_levels, opposite_levels = yes_bids, no_bids
        else:
            favorite_bid, favorite_ask, dog_ask = no_bid, no_ask, yes_ask
            favorite_levels, opposite_levels = no_bids, yes_bids
        volume_delta = self._volume_delta(market_ref.market_id, market)
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
            depth_top_levels=_kalshi_depth(favorite_levels, opposite_levels),
            volume_since_last_poll=volume_delta,
            timestamp=utc_now(),
            source_timestamp=None,
            is_stale=False,
            dog_best_ask=dog_ask,
            no_vig_method="two_sided_derived_ask_normalization_from_yes_no_bids",
        )

    def find_public_smoke_market(self) -> MarketRef:
        payload = self.http.get_json(
            self.base_url,
            "/markets",
            {"status": "open", "limit": 1000, "mve_filter": "exclude"},
        )
        markets = payload.get("markets") if isinstance(payload, Mapping) else None
        if not isinstance(markets, list):
            raise MarketDataError("Kalshi smoke response is missing markets")
        candidates: list[tuple[float, MarketRef]] = []
        for market in markets:
            if not isinstance(market, Mapping):
                continue
            yes_bid = _price(market.get("yes_bid_dollars"))
            no_bid = _price(market.get("no_bid_dollars"))
            if yes_bid is None or no_bid is None or not (0 < yes_bid < 1 and 0 < no_bid < 1):
                continue
            yes_mid = (yes_bid + (1.0 - no_bid)) / 2.0
            side = "yes" if yes_mid >= 0.5 else "no"
            ticker = str(market.get("ticker") or "")
            if not ticker:
                continue
            ref = MarketRef(
                venue=self.venue,
                market_id=ticker,
                favorite_outcome_id=f"{ticker}:{side}",
                favorite_side=side,
                dog_outcome_id=f"{ticker}:{'no' if side == 'yes' else 'yes'}",
                favorite_team=str(market.get("yes_sub_title") if side == "yes" else market.get("no_sub_title") or side),
                dog_team=str(market.get("no_sub_title") if side == "yes" else market.get("yes_sub_title") or "opposite"),
                mapping_confidence="exact",
                mapping_reason="public smoke market: higher orderbook midpoint selected",
                metadata={"title": str(market.get("title") or "")},
            )
            candidates.append((float(market.get("volume_fp") or 0), ref))
        for _, market_ref in sorted(candidates, reverse=True, key=lambda item: item[0]):
            try:
                quote = self.get_quote(market_ref)
            except MarketDataError:
                continue
            if quote is not None:
                return market_ref
        raise MarketDataError("no liquid two-sided Kalshi smoke market was available")

    def _volume_delta(self, ticker: str, market: Mapping[str, Any]) -> float | None:
        try:
            total = float(market.get("volume_fp"))
        except (TypeError, ValueError):
            return None
        prior = self._last_volume.get(ticker)
        self._last_volume[ticker] = total
        return None if prior is None else max(0.0, total - prior)


def _map_kalshi_market(game: GameMarket, market: Mapping[str, Any]) -> MarketRef | None:
    yes_text = str(market.get("yes_sub_title") or "")
    no_text = str(market.get("no_sub_title") or "")
    yes_is_favorite = team_in_text(game.favorite, yes_text) and not team_in_text(game.dog, yes_text)
    yes_is_dog = team_in_text(game.dog, yes_text) and not team_in_text(game.favorite, yes_text)
    no_is_favorite = team_in_text(game.favorite, no_text) and not team_in_text(game.dog, no_text)
    no_is_dog = team_in_text(game.dog, no_text) and not team_in_text(game.favorite, no_text)
    if yes_is_favorite and (no_is_dog or not no_text):
        favorite_side = "yes"
    elif yes_is_dog and (no_is_favorite or not no_text):
        favorite_side = "no"
    else:
        return None
    ticker = str(market.get("ticker") or "")
    if not ticker:
        return None
    dog_side = "no" if favorite_side == "yes" else "yes"
    return MarketRef(
        venue="kalshi",
        market_id=ticker,
        favorite_outcome_id=f"{ticker}:{favorite_side}",
        favorite_side=favorite_side,
        dog_outcome_id=f"{ticker}:{dog_side}",
        favorite_team=game.favorite,
        dog_team=game.dog,
        mapping_confidence="exact",
        mapping_reason="Kalshi yes/no side mapped from explicit yes_sub_title/no_sub_title team labels",
        metadata={"title": str(market.get("title") or ""), "event_ticker": str(market.get("event_ticker") or "")},
    )


def _contains_game_teams(game: GameMarket, market: Mapping[str, Any]) -> bool:
    text = " ".join(
        str(market.get(key) or "")
        for key in ("title", "subtitle", "yes_sub_title", "no_sub_title", "event_ticker")
    )
    return team_in_text(game.favorite, text) and team_in_text(game.dog, text)


def _date_matches(game: GameMarket, market: Mapping[str, Any]) -> bool:
    if not game.kickoff:
        return True
    game_date = game.kickoff[:10]
    candidate = str(market.get("occurrence_datetime") or market.get("expected_expiration_time") or "")
    return bool(candidate) and candidate[:10] == game_date


def _book_levels(value: object) -> list[tuple[float, float]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise MarketDataError("Kalshi orderbook side must be a list")
    levels: list[tuple[float, float]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            raise MarketDataError(f"invalid Kalshi orderbook level: {row!r}")
        levels.append((float(row[0]), float(row[1])))
    return levels


def _best_reciprocal_bids(
    yes_bids: list[tuple[float, float]], no_bids: list[tuple[float, float]], market: Mapping[str, Any]
) -> tuple[float, float]:
    yes_bid = max((price for price, _ in yes_bids), default=_price(market.get("yes_bid_dollars")) or 0.0)
    no_bid = max((price for price, _ in no_bids), default=_price(market.get("no_bid_dollars")) or 0.0)
    if not (0 < yes_bid < 1 and 0 < no_bid < 1):
        raise MarketDataError("Kalshi orderbook is not two-sided")
    if yes_bid + no_bid > 1.0 + 1e-9:
        raise MarketDataError(f"Kalshi reciprocal bids cross: yes_bid={yes_bid}, no_bid={no_bid}")
    return yes_bid, no_bid


def _validate_direct_prices(
    market: Mapping[str, Any], yes_bid: float, yes_ask: float, no_bid: float, no_ask: float
) -> None:
    expected = {
        "yes_bid_dollars": yes_bid,
        "yes_ask_dollars": yes_ask,
        "no_bid_dollars": no_bid,
        "no_ask_dollars": no_ask,
    }
    for field, derived in expected.items():
        direct = _price(market.get(field))
        if direct is not None and direct > 0 and abs(direct - derived) > 0.011:
            raise MarketDataError(f"Kalshi {field}={direct} disagrees with reciprocal orderbook value {derived}")


def _kalshi_depth(
    favorite_bids: list[tuple[float, float]], opposite_bids: list[tuple[float, float]], levels: int = 3
) -> tuple[Mapping[str, float], ...]:
    bids = sorted(favorite_bids, reverse=True)[:levels]
    asks = sorted(((1.0 - price, size) for price, size in opposite_bids))[:levels]
    depth: list[Mapping[str, float]] = []
    for index in range(max(len(bids), len(asks))):
        row: dict[str, float] = {"level": float(index + 1)}
        if index < len(bids):
            row.update({"bid_price": bids[index][0], "bid_size": bids[index][1]})
        if index < len(asks):
            row.update({"ask_price": asks[index][0], "ask_size": asks[index][1]})
        depth.append(row)
    return tuple(depth)


def _price(value: object) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None
