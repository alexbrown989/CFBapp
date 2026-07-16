# N13 Stage 3 Market Verification

Date: 2026-07-15

## Acceptance Result

PASS. Public, unauthenticated market data works on both venues. No credential, signing, portfolio, order, or trading code exists. `ScoreboardLive` remained inactive.

| Venue | Live market | Favorite side | Bid | Ask/raw | No-vig | Spread | Stale |
|---|---|---|---:|---:|---:|---:|---|
| Kalshi | `KXLOLGAME-26JUL150940KCTS-TS` | no | 0.510000 | 0.550000 | 0.528846 | 0.040000 | False |
| Polymarket | `0x7976b8dbacf9077eb1453a62bcefd6ab2df199acd28aad276ff0d920d6992892` | token | 0.581000 | 0.582000 | 0.581419 | 0.001000 | False |

## No-Vig Methods

- **Polymarket:** fetch the favorite and opponent CLOB token books. Raw probability is the executable favorite best ask. No-vig probability is `favorite_ask / (favorite_ask + opponent_ask)`. Both token books are required; there is no one-sided fallback.
- **Kalshi:** read YES and NO bids from `orderbook_fp.*_dollars`; derive `yes_ask = 1 - no_bid` and `no_ask = 1 - yes_bid`. Raw probability is the executable ask for the mapped favorite side. No-vig probability normalizes the two derived asks. Direct `*_dollars` market fields are checked against reciprocal book values.
- Known-input test: asks 0.55 and 0.50 produce no-vig `0.523809523810`; engine probability 0.60 produces gap `0.076190476190` and real-price EV/dollar `0.090909090909`.
- Kalshi reciprocal check passed: favorite bid plus opposite-side ask equals 1.0.

## Mapping And Target Guards

Synthetic exact-match parsers mapped Polymarket favorite token `uga-token` and Kalshi favorite outcome `KXNCAAFGAME-26SEP05UGAALA-UGA:yes`. Team/date matching must produce one unique market; otherwise the game is `NO_MARKET`. Current CFB game mappings must be re-certified when 2026 markets list.

The inversion guard rejected both a deliberately swapped favorite/dog outcome and a nominal pregame favorite outcome priced below 0.5: `favorite outcome maps to 'Alabama', expected 'Georgia'; inversion guard failed: mapped favorite outcome is not above 0.5 in a normal pregame state`.

The moneyline target is asserted as `favorite_final_win`. A deliberate `deficit_erased` request raised as required; no cross-label gap can be computed.

## Logging And Resilience

- Georgia-Alabama replay passed end to end: trigger -> Tier 1 estimate -> recorded market quote -> no-vig gap -> additive trigger JSONL.
- The replay emits 7 trigger records across 4 trigger-bearing polls: D=3/7, D=10/14, D=21, and the Q4 D=3/7 re-fire. Multi-threshold crossings share one observed poll state, and every record grouped within a poll has identical score, period, clock, and timestamp.
- Market quotes are written on every watched-game poll to `live/logs/market_series.jsonl`, including non-trigger polls.
- Stage 1/2 rows remain readable; all Stage 3 fields default to null.
- Synthetic venue 500 was isolated without escaping the poll loop: `RuntimeError: synthetic venue 500`.
- A synthetic 401/403-style authentication challenge propagated as the mandatory Stage 3 halt condition: `synthetic public endpoint auth challenge`.
- Stale quotes retain quote context but do not produce a gap.

## Safety

Only public GET endpoints are implemented. Market credentials are neither required nor supported. There are no RSA, wallet, portfolio, order-placement, cancellation, or trading functions. The private credential supplied during Stage 3 planning was not stored or used and should be revoked because it was disclosed in conversation.
