from __future__ import annotations

from dataclasses import asdict

from .config import RiskConfig
from .data_source import ScoreboardGameState
from .markets.base import Quote
from .markets.service import MarketPollResult
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
from .scoring import ScoringResult
from .trigger_detect import TriggerEvent
from .watchlist import WatchGame

def serialize_game(
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
    expected_profit_per_dollar = expected_value(
        probability, decimal_odds, label=WIN_MARKET_LABEL
    )
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
    ruin_probability = risk_of_ruin(
        config.bankroll,
        suggested,
        probability,
        decimal_odds,
        config.season_bets,
        config.drawdown_floor,
        label=WIN_MARKET_LABEL,
    )
    comfort_fraction = comfort_stake_fraction(
        config.bankroll,
        suggested,
        probability,
        decimal_odds,
        config.season_bets,
        config.drawdown_floor,
        config.ruin_comfort_threshold,
        label=WIN_MARKET_LABEL,
    )
    gap_no_vig = probability - quote.implied_prob_no_vig
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
        "ev_per_dollar": expected_profit_per_dollar,
        "positive_ev": expected_profit_per_dollar > 0.0,
        "full_kelly_fraction": full_kelly,
        "suggested_fraction_uncapped": suggested_uncapped,
        "suggested_fraction": suggested,
        "suggested_dollars": config.bankroll * suggested,
        "cap_applied": suggested < suggested_uncapped,
        "reliability": reliability,
        "reliability_factor": _reliability_factor(reliability),
        "tier_factor": 1.0,
        "ruin_probability": ruin_probability,
        "ruin_comfort_threshold": config.ruin_comfort_threshold,
        "comfort_fraction": comfort_fraction,
        "comfort_dollars": config.bankroll * comfort_fraction,
        "ruin_warning": ruin_probability > config.ruin_comfort_threshold,
        "drawdown_floor": config.drawdown_floor,
        "season_bets": config.season_bets,
        "streaks": streaks,
        "gap_no_vig": gap_no_vig,
        "spread": quote.spread,
        "survives_friction": gap_no_vig > quote.spread,
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
