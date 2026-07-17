# N13 Stage 4 Dashboard Verification

Date: 2026-07-16

## Acceptance Result

PASS. The localhost dashboard renders Stage 1-3 state without changing scoring or market-gap logic. All financial calculations are hard-locked to `favorite_final_win`; N06 Tier 3 and its conformal band remain explicitly labeled `deficit_erased`.

## Risk Math

Known-value checks passed:

- EV: `p=0.60`, decimal odds `2.00` -> `0.200000`.
- Full Kelly: same inputs -> `0.200000`.
- Exact losing-streak test: at least one loss in 3 bets -> `0.784000`.
- Expected overlapping 3-loss windows across 50 bets -> `3.072000`.
- Exact one-bet drawdown test -> `0.400000`.
- Comfort-sizing bisection returned `0.050000` at or below the proposal.
- Label boundary: all 7 financial functions rejected `deficit_erased`.

### Compounded Policy Sanity Check

The fixture has full Kelly `0.20` and configured fractional Kelly `0.25`. Tier factors are flat at `1.00`; conformal width does not alter moneyline sizing.

| Reliability | Reliability factor | Fraction of full Kelly | Stake fraction |
|---|---:|---:|---:|
| reliable | 1.00 | 25.0% | 0.0500 |
| thin | 0.50 | 12.5% | 0.0250 |
| unreliable | 0.25 | 6.25% | 0.0125 |

These are explicit policy choices. Quarter-Kelly is the sole parameter-estimation haircut; reliability is a separate historical sample-size penalty.

## Replay Rendering

- Georgia-Alabama: 7 triggers across 4 trigger-bearing polls.
- Tier badges: 5 Tier 3 snapshots and 2 honest Tier 2 fallbacks.
- Every snapshot carries both Tier 1 labels, sample sizes, reliability, market quote/gap, and a label-safe risk panel.
- Tier 3 snapshots carry visible conformal lower/upper bounds for `deficit_erased` only.
- Dashboard HTML size: 29,067 bytes; no frontend build step.
- API routes verified: `/`, `/api/state`, `/api/triggers`, `/api/game/{game_id}`, `/api/config`.

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

- Bind host is hard-locked to `127.0.0.1`.
- Frontend credential/order/trading scan: PASS.
- Personal bankroll config persists only to gitignored `live/config.local.json`.
- No auto-bet, order placement, countdown, or public bind exists.
- Favorite-longshot bias note appears when the raw offered probability is below 0.10; it is informational and applies no automatic adjustment.
