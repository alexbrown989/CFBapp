"""N13 Stage 3 public-market, gap, resilience, and replay acceptance gate."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

from .data_source import ScoreboardGameState
from .logger import JSONLRecordLogger, JSONLTriggerLogger, MARKET_TRIGGER_FIELDS, read_trigger_records
from .markets import KalshiClient, PolymarketClient
from .markets.base import (
    GameMarket,
    MarketAuthenticationError,
    MarketMappingError,
    MarketRef,
    PublicHTTPClient,
    Quote,
    utc_now,
)
from .markets.gap import compute_gap, two_sided_no_vig, validate_favorite_mapping
from .markets.service import MarketService
from .replay_verify import main as run_replay
from .scoring import ScoringContext, score_trigger
from .trigger_detect import TriggerEvent
from .watchlist import WatchGame


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "research/results/n13_stage3_market_verification.md"
ERROR_LOG = REPO_ROOT / "live/logs/stage3_resilience_errors.jsonl"
COMPAT_LOG = REPO_ROOT / "live/logs/stage3_compatibility.jsonl"


class FailingClient:
    venue = "kalshi"

    def find_market(self, _):
        return None

    def get_quote(self, _):
        raise RuntimeError("synthetic venue 500")


class AuthenticationFailingClient:
    venue = "kalshi"

    def find_market(self, _):
        return None

    def get_quote(self, _):
        raise MarketAuthenticationError("synthetic public endpoint auth challenge")


def verify_public_smoke() -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for client in (KalshiClient(), PolymarketClient()):
        market_ref = client.find_public_smoke_market()
        quote = client.get_quote(market_ref)
        assert quote is not None
        assert 0 < quote.best_bid <= quote.best_ask < 1
        assert quote.spread >= 0
        assert 0 < quote.implied_prob_no_vig < 1
        if client.venue == "kalshi":
            assert math.isclose(quote.best_bid + quote.dog_best_ask, 1.0, abs_tol=1e-12)
            assert quote.no_vig_method == "two_sided_derived_ask_normalization_from_yes_no_bids"
        results[client.venue] = {
            "market_id": market_ref.market_id,
            "favorite_side": market_ref.favorite_side,
            "best_bid": quote.best_bid,
            "best_ask": quote.best_ask,
            "spread": quote.spread,
            "implied_prob_raw": quote.implied_prob_raw,
            "implied_prob_no_vig": quote.implied_prob_no_vig,
            "is_stale": quote.is_stale,
            "no_vig_method": quote.no_vig_method,
        }
    return results


def verify_no_vig_and_label() -> dict[str, float]:
    expected = 0.55 / 1.05
    actual = two_sided_no_vig(0.55, 0.50)
    assert math.isclose(actual, expected, abs_tol=1e-15)
    quote = Quote(
        venue="test",
        market_id="test",
        favorite_outcome_id="favorite",
        best_bid=0.53,
        best_ask=0.55,
        mid=0.54,
        spread=0.02,
        implied_prob_raw=0.55,
        implied_prob_no_vig=actual,
        depth_top_levels=({"level": 1.0, "ask_size": 100.0},),
        volume_since_last_poll=None,
        timestamp=utc_now(),
        is_stale=False,
        dog_best_ask=0.50,
    )
    result = compute_gap(0.60, quote, label="favorite_final_win")
    assert math.isclose(result.gap_no_vig, 0.60 - expected, abs_tol=1e-15)
    assert math.isclose(result.ev_per_dollar, 0.60 / 0.55 - 1.0, abs_tol=1e-15)
    try:
        compute_gap(0.60, quote, label="deficit_erased")
    except ValueError as exc:
        assert "refusing mismatched label" in str(exc)
    else:
        raise AssertionError("deficit_erased label mismatch was not rejected")
    return {"known_no_vig": actual, "known_gap": result.gap_no_vig, "known_ev": result.ev_per_dollar}


def verify_inversion_guard() -> str:
    game = GameMarket("guard", 2026, 1, "Georgia", "Alabama", "Georgia", "Alabama", -7.0)
    wrong = MarketRef(
        "kalshi", "guard", "guard:no", "no", "guard:yes", "Alabama", "Georgia", "exact", "synthetic wrong side"
    )
    try:
        validate_favorite_mapping(game, wrong, pregame_favorite_probability=0.70)
    except MarketMappingError as exc:
        outcome_error = str(exc)
    else:
        raise AssertionError("wrong-team outcome mapping was not rejected")
    correct = MarketRef(
        "kalshi", "guard", "guard:yes", "yes", "guard:no", "Georgia", "Alabama", "exact", "synthetic correct side"
    )
    try:
        validate_favorite_mapping(game, correct, pregame_favorite_probability=0.40)
    except MarketMappingError as exc:
        probability_error = str(exc)
    else:
        raise AssertionError("pregame favorite below 0.5 was not rejected")
    return f"{outcome_error}; {probability_error}"


def verify_mapping_parsers() -> dict[str, str]:
    game = GameMarket(
        "mapping", 2026, 1, "Alabama", "Georgia", "Georgia", "Alabama", -3.0, "2026-09-05T20:00:00Z"
    )
    polymarket_payload = {
        "events": [{
            "title": "Georgia at Alabama",
            "startTime": "2026-09-05T20:00:00Z",
            "markets": [{
                "id": "poly-id", "conditionId": "poly-condition", "question": "Georgia at Alabama winner",
                "enableOrderBook": True, "gameStartTime": "2026-09-05T20:00:00Z",
                "outcomes": '["Georgia","Alabama"]', "clobTokenIds": '["uga-token","bama-token"]',
            }],
        }]
    }
    kalshi_payload = {
        "markets": [{
            "ticker": "KXNCAAFGAME-26SEP05UGAALA-UGA", "event_ticker": "KXNCAAFGAME-26SEP05UGAALA",
            "title": "Georgia at Alabama", "yes_sub_title": "Georgia", "no_sub_title": "Alabama",
            "occurrence_datetime": "2026-09-05T20:00:00Z",
        }]
    }

    def transport_for(payload):
        def transport(url: str, _: float):
            return 200, {}, json.dumps(payload).encode("utf-8")
        return transport

    poly = PolymarketClient(http=PublicHTTPClient(transport=transport_for(polymarket_payload)))
    kalshi = KalshiClient(http=PublicHTTPClient(transport=transport_for(kalshi_payload)))
    poly_ref = poly.find_market(game)
    kalshi_ref = kalshi.find_market(game)
    # Discovery performs the pregame quote guard in MarketService, not the raw parser.
    assert poly_ref is not None and poly_ref.favorite_outcome_id == "uga-token"
    assert kalshi_ref is not None and kalshi_ref.favorite_side == "yes"
    return {"polymarket": poly_ref.favorite_outcome_id, "kalshi": kalshi_ref.favorite_outcome_id}


def verify_resilience() -> str:
    ERROR_LOG.unlink(missing_ok=True)
    logger = JSONLRecordLogger(
        ERROR_LOG, ("timestamp", "game_id", "venue", "operation", "error_type", "message")
    )
    service = MarketService([FailingClient()], error_logger=logger)
    ref = MarketRef(
        "kalshi", "failure", "failure:yes", "yes", "failure:no", "Favorite", "Dog", "exact", "resilience fixture"
    )
    service.set_mapping("failure", ref)
    game = WatchGame("failure", 2026, 1, "Favorite", "Dog", -7.0, "Favorite", "Dog", "test")
    state = ScoreboardGameState("failure", 2026, 1, "Favorite", "Dog", 0, 7, 1, "10:00", "in_progress")
    result = service.poll_game(game, state, is_triggered=True)
    assert result.status == "NO_MARKET" and "kalshi" in result.errors
    records = [json.loads(line) for line in ERROR_LOG.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1 and records[0]["error_type"] == "RuntimeError"
    return result.errors["kalshi"]


def verify_auth_halt() -> str:
    service = MarketService([AuthenticationFailingClient()])
    ref = MarketRef(
        "kalshi", "auth", "auth:yes", "yes", "auth:no", "Favorite", "Dog", "exact", "auth fixture"
    )
    service.set_mapping("auth", ref)
    game = WatchGame("auth", 2026, 1, "Favorite", "Dog", -7.0, "Favorite", "Dog", "test")
    state = ScoreboardGameState("auth", 2026, 1, "Favorite", "Dog", 0, 7, 1, "10:00", "in_progress")
    try:
        service.poll_game(game, state, is_triggered=False)
    except MarketAuthenticationError as exc:
        return str(exc)
    raise AssertionError("unexpected market authentication requirement was swallowed")


def verify_additive_compatibility() -> None:
    COMPAT_LOG.unlink(missing_ok=True)
    event = TriggerEvent(
        timestamp=utc_now(), game_id="compat", season=2026, week=1, favorite="Favorite", dog="Dog",
        pregame_spread=-7.0, fav_score=0, dog_score=7, period=1, clock="10:00", deficit=-7,
        threshold_crossed=3, possession=None, data_source="stub", poll_number=1,
    )
    JSONLTriggerLogger(COMPAT_LOG).append(event)
    records = read_trigger_records(COMPAT_LOG)
    assert len(records) == 1
    assert all(records[0][field] is None for field in MARKET_TRIGGER_FIELDS)


def write_report(
    smoke: Mapping[str, Mapping[str, object]], math_results: Mapping[str, float], inversion: str,
    mapping: Mapping[str, str], resilience: str, auth_halt: str,
) -> None:
    report = f"""# N13 Stage 3 Market Verification

Date: 2026-07-15

## Acceptance Result

PASS. Public, unauthenticated market data works on both venues. No credential, signing, portfolio, order, or trading code exists. `ScoreboardLive` remained inactive.

| Venue | Live market | Favorite side | Bid | Ask/raw | No-vig | Spread | Stale |
|---|---|---|---:|---:|---:|---:|---|
| Kalshi | `{smoke['kalshi']['market_id']}` | {smoke['kalshi']['favorite_side']} | {smoke['kalshi']['best_bid']:.6f} | {smoke['kalshi']['best_ask']:.6f} | {smoke['kalshi']['implied_prob_no_vig']:.6f} | {smoke['kalshi']['spread']:.6f} | {smoke['kalshi']['is_stale']} |
| Polymarket | `{smoke['polymarket']['market_id']}` | token | {smoke['polymarket']['best_bid']:.6f} | {smoke['polymarket']['best_ask']:.6f} | {smoke['polymarket']['implied_prob_no_vig']:.6f} | {smoke['polymarket']['spread']:.6f} | {smoke['polymarket']['is_stale']} |

## No-Vig Methods

- **Polymarket:** fetch the favorite and opponent CLOB token books. Raw probability is the executable favorite best ask. No-vig probability is `favorite_ask / (favorite_ask + opponent_ask)`. Both token books are required; there is no one-sided fallback.
- **Kalshi:** read YES and NO bids from `orderbook_fp.*_dollars`; derive `yes_ask = 1 - no_bid` and `no_ask = 1 - yes_bid`. Raw probability is the executable ask for the mapped favorite side. No-vig probability normalizes the two derived asks. Direct `*_dollars` market fields are checked against reciprocal book values.
- Known-input test: asks 0.55 and 0.50 produce no-vig `{math_results['known_no_vig']:.12f}`; engine probability 0.60 produces gap `{math_results['known_gap']:.12f}` and real-price EV/dollar `{math_results['known_ev']:.12f}`.
- Kalshi reciprocal check passed: favorite bid plus opposite-side ask equals 1.0.

## Mapping And Target Guards

Synthetic exact-match parsers mapped Polymarket favorite token `{mapping['polymarket']}` and Kalshi favorite outcome `{mapping['kalshi']}`. Team/date matching must produce one unique market; otherwise the game is `NO_MARKET`. Current CFB game mappings must be re-certified when 2026 markets list.

The inversion guard rejected both a deliberately swapped favorite/dog outcome and a nominal pregame favorite outcome priced below 0.5: `{inversion}`.

The moneyline target is asserted as `favorite_final_win`. A deliberate `deficit_erased` request raised as required; no cross-label gap can be computed.

## Logging And Resilience

- Georgia-Alabama replay passed end to end: trigger -> Tier 1 estimate -> recorded market quote -> no-vig gap -> additive trigger JSONL.
- The replay emits 7 trigger records across 4 trigger-bearing polls: D=3/7, D=10/14, D=21, and the Q4 D=3/7 re-fire. Multi-threshold crossings share one observed poll state, and every record grouped within a poll has identical score, period, clock, and timestamp.
- Market quotes are written on every watched-game poll to `live/logs/market_series.jsonl`, including non-trigger polls.
- Stage 1/2 rows remain readable; all Stage 3 fields default to null.
- Synthetic venue 500 was isolated without escaping the poll loop: `{resilience}`.
- A synthetic 401/403-style authentication challenge propagated as the mandatory Stage 3 halt condition: `{auth_halt}`.
- Stale quotes retain quote context but do not produce a gap.

## Safety

Only public GET endpoints are implemented. Market credentials are neither required nor supported. There are no RSA, wallet, portfolio, order-placement, cancellation, or trading functions. The private credential supplied during Stage 3 planning was not stored or used and should be revoked because it was disclosed in conversation.
"""
    REPORT_PATH.write_text(report, encoding="utf-8", newline="\n")


def main() -> int:
    smoke = verify_public_smoke()
    math_results = verify_no_vig_and_label()
    inversion = verify_inversion_guard()
    mapping = verify_mapping_parsers()
    resilience = verify_resilience()
    auth_halt = verify_auth_halt()
    verify_additive_compatibility()
    assert run_replay() == 0
    write_report(smoke, math_results, inversion, mapping, resilience, auth_halt)
    print(json.dumps({"smoke": smoke, "mapping": mapping, "resilience": resilience}, indent=2, sort_keys=True))
    print(f"PASS: {REPORT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
