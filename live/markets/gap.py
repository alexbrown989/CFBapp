"""Probability-comparison and real-price EV calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .base import GameMarket, MarketMappingError, MarketRef, Quote, normalize_team


WIN_MARKET_LABEL = "favorite_final_win"


@dataclass(frozen=True)
class GapResult:
    label: str
    engine_estimate: float
    implied_prob_raw: float
    implied_prob_no_vig: float
    gap_no_vig: float
    ev_per_dollar: float
    spread: float
    depth_top_levels: tuple[dict[str, float], ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def two_sided_no_vig(favorite_offer: float, dog_offer: float) -> float:
    """Normalize executable asks; bid-ask overround is removed symmetrically."""
    favorite = _probability(favorite_offer, "favorite_offer")
    dog = _probability(dog_offer, "dog_offer")
    total = favorite + dog
    if total <= 0:
        raise ValueError("two-sided offered probabilities must sum above zero")
    return favorite / total


def compute_gap(engine_estimate: float, quote: Quote, *, label: str = WIN_MARKET_LABEL) -> GapResult:
    if label != WIN_MARKET_LABEL:
        raise ValueError(
            f"moneyline market target is {WIN_MARKET_LABEL!r}; refusing mismatched label {label!r}"
        )
    probability = _probability(engine_estimate, "engine_estimate")
    if quote.is_stale:
        raise ValueError("cannot compute a market gap from a stale quote")
    raw = quote.implied_prob_raw
    return GapResult(
        label=label,
        engine_estimate=probability,
        implied_prob_raw=raw,
        implied_prob_no_vig=quote.implied_prob_no_vig,
        gap_no_vig=probability - quote.implied_prob_no_vig,
        ev_per_dollar=(probability / raw) - 1.0,
        spread=quote.spread,
        depth_top_levels=tuple(dict(level) for level in quote.depth_top_levels),
    )


def validate_favorite_mapping(
    game: GameMarket,
    market_ref: MarketRef,
    *,
    pregame_favorite_probability: float | None = None,
) -> None:
    if normalize_team(market_ref.favorite_team) != normalize_team(game.favorite):
        raise MarketMappingError(
            f"favorite outcome maps to {market_ref.favorite_team!r}, expected {game.favorite!r}"
        )
    if normalize_team(market_ref.dog_team) != normalize_team(game.dog):
        raise MarketMappingError(f"dog outcome maps to {market_ref.dog_team!r}, expected {game.dog!r}")
    if not market_ref.favorite_outcome_id:
        raise MarketMappingError("favorite outcome identifier is empty")
    if pregame_favorite_probability is not None:
        probability = _probability(pregame_favorite_probability, "pregame_favorite_probability")
        if game.pregame_spread < 0 and probability <= 0.5:
            raise MarketMappingError(
                "inversion guard failed: mapped favorite outcome is not above 0.5 in a normal pregame state"
            )


def _probability(value: float, name: str) -> float:
    number = float(value)
    if not 0.0 < number < 1.0:
        raise ValueError(f"{name} must be in (0,1), got {number}")
    return number
