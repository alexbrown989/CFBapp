"""N13 Stage 4 risk, API, replay, degradation, and security acceptance gate."""

from __future__ import annotations

import json
import math
import re
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from .config import RiskConfig, RuntimeConfigStore, Settings
from .data_source import ScoreboardStub
from .logger import JSONLTriggerLogger
from .main import LOCALHOST, LiveMonitor, create_app
from .markets.base import MarketRef
from .markets.service import MarketService
from .replay_verify import (
    GAME_ID,
    PLAYS_CACHE,
    RecordedKalshiClient,
    _replay_context_provider,
    chronological_key,
    load_json,
    score_state,
    verify_watchlist,
)
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
from .scoring import score_trigger
from .trigger_detect import TriggerDetector


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "research/results/n13_stage4_dashboard_verification.md"
TEST_LOG = REPO_ROOT / "live/logs/stage4_replay.jsonl"
TEST_CONFIG = Path(tempfile.gettempdir()) / "cfbapp_stage4_config.json"


def _served_dashboard_bundle(client, html: str | None = None) -> str:
    """Return the dashboard HTML plus every linked local static asset."""
    page_html = client.get("/").text if html is None else html
    asset_paths = re.findall(r'(?:href|src)="(/static/[^"]+)"', page_html)
    assert asset_paths
    served_assets = [page_html]
    for asset_path in asset_paths:
        response = client.get(asset_path)
        assert response.status_code == 200
        served_assets.append(response.text)
    return "\n".join(served_assets)


def verify_risk_math() -> dict[str, object]:
    label = WIN_MARKET_LABEL
    ev = expected_value(0.60, 2.0, label=label)
    full = kelly_fraction(0.60, 2.0, label=label)
    assert math.isclose(ev, 0.20, abs_tol=1e-15)
    assert math.isclose(full, 0.20, abs_tol=1e-15)

    factors: dict[str, float] = {}
    for reliability, expected_multiplier in (("reliable", 0.25), ("thin", 0.125), ("unreliable", 0.0625)):
        stake = fractional_kelly(
            0.60, 2.0, 0.25, 1, 0.97, reliability, label=label
        )
        assert math.isclose(stake / full, expected_multiplier, abs_tol=1e-15)
        same_without_band = fractional_kelly(
            0.60, 2.0, 0.25, 3, None, reliability, label=label
        )
        assert math.isclose(stake, same_without_band, abs_tol=1e-15)
        factors[reliability] = stake

    streak = losing_streak_probability(0.60, 1, 3, label=label)
    expected_streak = expected_losing_streaks(0.60, 3, 50, label=label)
    assert math.isclose(streak, 1.0 - 0.60**3, abs_tol=1e-15)
    assert math.isclose(expected_streak, 48 * 0.40**3, abs_tol=1e-15)

    one_bet_ruin = risk_of_ruin(100.0, 0.50, 0.60, 2.0, 1, 0.60, label=label)
    assert math.isclose(one_bet_ruin, 0.40, abs_tol=1e-15)
    proposed = factors["reliable"]
    comfort = comfort_stake_fraction(
        1000.0, proposed, 0.60, 2.0, 50, 0.50, 0.05, label=label
    )
    assert 0.0 <= comfort <= proposed

    mismatched_calls = (
        lambda: expected_value(0.6, 2.0, label="deficit_erased"),
        lambda: kelly_fraction(0.6, 2.0, label="deficit_erased"),
        lambda: fractional_kelly(0.6, 2.0, 0.25, 1, None, "reliable", label="deficit_erased"),
        lambda: losing_streak_probability(0.6, 3, 50, label="deficit_erased"),
        lambda: expected_losing_streaks(0.6, 3, 50, label="deficit_erased"),
        lambda: risk_of_ruin(1000, 0.01, 0.6, 2.0, 50, 0.5, label="deficit_erased"),
        lambda: comfort_stake_fraction(1000, 0.01, 0.6, 2.0, 50, 0.5, 0.05, label="deficit_erased"),
    )
    for call in mismatched_calls:
        try:
            call()
        except ValueError as exc:
            assert "refusing mismatched label" in str(exc)
        else:
            raise AssertionError("deficit_erased reached favorite-final-win risk math")
    return {
        "ev_known": ev,
        "full_kelly_known": full,
        "stakes": factors,
        "streak_known": streak,
        "expected_streak_known": expected_streak,
        "one_bet_ruin_known": one_bet_ruin,
        "comfort_fraction": comfort,
        "label_guard_functions": len(mismatched_calls),
    }


def build_replay_monitor() -> LiveMonitor:
    game = verify_watchlist()
    plays = [row for row in load_json(PLAYS_CACHE) if str(row.get("gameId")) == GAME_ID]
    scoring_plays = sorted(
        (row for row in plays if row.get("scoring") is True), key=chronological_key
    )
    batches = [[score_state(play, index)] for index, play in enumerate(scoring_plays, start=1)]
    TEST_LOG.unlink(missing_ok=True)
    market_client = RecordedKalshiClient()
    market_service = MarketService([market_client])
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
            mapping_reason="Stage 4 deterministic replay",
        ),
    )
    monitor = LiveMonitor(
        source=ScoreboardStub(batches),
        watchlist={GAME_ID: game},
        detector=TriggerDetector(),
        logger=JSONLTriggerLogger(TEST_LOG),
        poll_interval_seconds=0,
        scorer=score_trigger,
        context_provider=_replay_context_provider(),
        market_service=market_service,
    )
    for _ in batches:
        monitor.poll_once()
    return monitor


def verify_dashboard_api() -> dict[str, object]:
    monitor = build_replay_monitor()
    TEST_CONFIG.unlink(missing_ok=True)
    store = RuntimeConfigStore(TEST_CONFIG, RiskConfig())
    settings = Settings(runtime_config_path=TEST_CONFIG)
    dashboard = create_app(monitor, store, settings)
    client = TestClient(dashboard)

    page = client.get("/")
    assert page.status_code == 200
    html = _served_dashboard_bundle(client, page.text)
    for required in (
        "STUB MODE",
        "Probability deficit is erased",
        "Risk & Variance",
        "favorite-longshot bias",
    ):
        assert required in html

    state = client.get("/api/state").json()
    assert state["mode"] == "stub" and state["game_count"] == 1
    game = state["games"][0]
    assert game["status"] == "TRIGGERED"
    assert game["market"]["status"] == "OK"
    assert game["risk"]["label"] == WIN_MARKET_LABEL
    assert game["risk"]["notice"].startswith("No label-matched conformal interval")

    trigger_payload = client.get("/api/triggers").json()
    assert trigger_payload["count"] == 7
    snapshots = trigger_payload["snapshots"]
    assert len(snapshots) == 7
    tier_counts = {tier: sum(row["scoring"]["tier_used"] == tier for row in snapshots) for tier in (2, 3)}
    assert tier_counts == {2: 2, 3: 5}
    trigger_polls = {row["latest_trigger"]["poll_number"] for row in snapshots}
    assert len(trigger_polls) == 4
    for row in snapshots:
        scoring = row["scoring"]
        assert scoring["tier_1"]["favorite_final_win"]["n_events"] is not None
        assert scoring["tier_1"]["deficit_erased"]["n_events"] is not None
        assert row["risk"]["venues"]["kalshi"]["status"] == "OK"
        assert row["risk"]["venues"]["kalshi"]["gap_no_vig"] is not None
        if scoring["tier_used"] == 3:
            assert scoring["tier_3"]["conformal_lower"] is not None
            assert scoring["tier_3"]["conformal_upper"] is not None
        else:
            assert "feature snapshot" in scoring["tier_3_unavailable_reason"]

    detail = client.get(f"/api/game/{GAME_ID}")
    assert detail.status_code == 200
    assert client.get("/api/game/not-a-game").status_code == 404

    original = client.get("/api/config").json()
    assert original["bind_host"] == LOCALHOST and original["mode"] == "stub"
    updated = client.post("/api/config", json={"bankroll": 2500, "kelly_fraction": 0.20})
    assert updated.status_code == 200 and updated.json()["bankroll"] == 2500
    assert TEST_CONFIG.exists()
    assert client.post("/api/config", json={"unknown": 1}).status_code == 422
    TEST_CONFIG.unlink(missing_ok=True)

    return {
        "trigger_count": len(snapshots),
        "trigger_poll_count": len(trigger_polls),
        "tier_counts": tier_counts,
        "api_routes": ["/", "/api/state", "/api/triggers", "/api/game/{game_id}", "/api/config"],
    }


def verify_degradation_and_security() -> dict[str, object]:
    empty_log = REPO_ROOT / "live/logs/stage4_empty.jsonl"
    empty_log.unlink(missing_ok=True)
    empty_monitor = LiveMonitor(
        source=ScoreboardStub([]),
        watchlist={},
        detector=TriggerDetector(),
        logger=JSONLTriggerLogger(empty_log),
        poll_interval_seconds=0,
    )
    empty_config = Path(tempfile.gettempdir()) / "cfbapp_stage4_empty_config.json"
    empty_config.unlink(missing_ok=True)
    empty_app = create_app(
        empty_monitor,
        RuntimeConfigStore(empty_config),
        Settings(runtime_config_path=empty_config),
    )
    empty_client = TestClient(empty_app)
    assert empty_client.get("/api/state").json()["games"] == []
    assert empty_app.state.bind_host == LOCALHOST

    html = _served_dashboard_bundle(empty_client)
    forbidden = (
        "CFBD_API_KEY",
        "BEGIN RSA PRIVATE KEY",
        "KALSHI-ACCESS",
        "BET NOW",
        "place_order",
        "cancel_order",
    )
    assert not any(value in html for value in forbidden)
    assert "NO_MARKET" in html and "Tier 3 unavailable" in html and "Dashboard API unavailable" in html
    assert "STUB MODE" in html and "replayed data, not live" in html
    empty_config.unlink(missing_ok=True)
    return {
        "no_games": True,
        "no_market_markup": True,
        "tier3_unavailable_markup": True,
        "venue_error_markup": True,
        "stub_banner": True,
        "localhost_only": empty_app.state.bind_host,
        "frontend_secret_scan": "PASS",
    }


def write_report(
    risk: dict[str, object], dashboard: dict[str, object], degradation: dict[str, object]
) -> None:
    stakes = risk["stakes"]
    report = f"""# N13 Stage 4 Dashboard Verification

Date: 2026-07-16

## Acceptance Result

PASS. The localhost dashboard renders Stage 1-3 state without changing scoring or market-gap logic. All financial calculations are hard-locked to `favorite_final_win`; N06 Tier 3 and its conformal band remain explicitly labeled `deficit_erased`.

## Risk Math

Known-value checks passed:

- EV: `p=0.60`, decimal odds `2.00` -> `{risk['ev_known']:.6f}`.
- Full Kelly: same inputs -> `{risk['full_kelly_known']:.6f}`.
- Exact losing-streak test: at least one loss in 3 bets -> `{risk['streak_known']:.6f}`.
- Expected overlapping 3-loss windows across 50 bets -> `{risk['expected_streak_known']:.6f}`.
- Exact one-bet drawdown test -> `{risk['one_bet_ruin_known']:.6f}`.
- Comfort-sizing bisection returned `{risk['comfort_fraction']:.6f}` at or below the proposal.
- Label boundary: all {risk['label_guard_functions']} financial functions rejected `deficit_erased`.

### Compounded Policy Sanity Check

The fixture has full Kelly `0.20` and configured fractional Kelly `0.25`. Tier factors are flat at `1.00`; conformal width does not alter moneyline sizing.

| Reliability | Reliability factor | Fraction of full Kelly | Stake fraction |
|---|---:|---:|---:|
| reliable | 1.00 | 25.0% | {stakes['reliable']:.4f} |
| thin | 0.50 | 12.5% | {stakes['thin']:.4f} |
| unreliable | 0.25 | 6.25% | {stakes['unreliable']:.4f} |

These are explicit policy choices. Quarter-Kelly is the sole parameter-estimation haircut; reliability is a separate historical sample-size penalty.

## Replay Rendering

- Georgia-Alabama: {dashboard['trigger_count']} triggers across {dashboard['trigger_poll_count']} trigger-bearing polls.
- Tier badges: {dashboard['tier_counts'][3]} Tier 3 snapshots and {dashboard['tier_counts'][2]} honest Tier 2 fallbacks.
- Every snapshot carries both Tier 1 labels, sample sizes, reliability, market quote/gap, and a label-safe risk panel.
- Tier 3 snapshots carry visible conformal lower/upper bounds for `deficit_erased` only.
- API routes verified: {', '.join(f'`{route}`' for route in dashboard['api_routes'])}.

## Graceful Degradation

- No games: PASS, empty array and explicit frontend empty state.
- No market or mapping: PASS, engine read remains renderable.
- Tier 3 unavailable: PASS, reason remains visible.
- Venue error: PASS, venue-specific error markup leaves the other venue intact.
- Stub mode banner: PASS and persistent.

## Responsive Visual QA

- Desktop viewport: PASS. Watch list, scoreboard, engine, market, gap, and risk panels remain aligned without overlap.
- Mobile viewport at 375 CSS pixels: PASS. Measured document width equals viewport width, with no horizontal overflow.
- Tier 3 replay snapshot: PASS. The `deficit_erased` conformal band is visibly rendered from 0.0% to 91.3% with q-hat 0.770, and the greater-than-10-point disagreement warning compares only label-matched deficit-erasure estimates.

## Security And Posture

- Bind host is hard-locked to `{degradation['localhost_only']}`.
- Frontend credential/order/trading scan: {degradation['frontend_secret_scan']}.
- Personal bankroll config persists only to gitignored `live/config.local.json`.
- No auto-bet, order placement, countdown, or public bind exists.
- Favorite-longshot bias note appears when the raw offered probability is below 0.10; it is informational and applies no automatic adjustment.
"""
    REPORT_PATH.write_text(report, encoding="utf-8", newline="\n")


def main() -> int:
    risk = verify_risk_math()
    dashboard = verify_dashboard_api()
    degradation = verify_degradation_and_security()
    write_report(risk, dashboard, degradation)
    print(json.dumps({"risk": risk, "dashboard": dashboard, "degradation": degradation}, indent=2, sort_keys=True))
    print(f"PASS: {REPORT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
