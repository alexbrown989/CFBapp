# N13 Stage 4.5 Legibility Refactor Verification

Date: 2026-07-17

## Verdict

PASS. Stage 4.5 split the live service and dashboard into smaller ownership-focused modules without changing the scoring, market, risk, trigger, logging, API, or replay contracts. The complete Stage 1-4 verification sequence passed before and after the refactor. No research estimates were recomputed.

The comparison base was authorized HEAD `4c30b7f0319e3bb4f33a35f301f61ae07d1a1191`, synchronized with `origin/main` before work began.

## Verification Contract Amendment

Three Stage 4 assertions were amended because they tested file placement rather than served behavior:

1. The `dashboard.html` byte-count assertion was removed. File size is not a product contract and necessarily changes when CSS and JavaScript move to linked assets.
2. The favorite-longshot-bias assertion now searches the complete served dashboard bundle: the HTML response plus each linked CSS and JavaScript asset fetched through the FastAPI test client. The behavioral claim remains that the warning is present in the served dashboard when raw implied probability is below 0.10.
3. The degradation-state assertion now searches the same served asset bundle. The behavioral claim remains that no-games, no-market, Tier 3-unavailable, venue-error, and stub-mode states render.

No numerical, label-boundary, route, replay, or security assertion was weakened. This amendment is limited to making three checks indifferent to which served asset owns the copy.

## Before/After Gates

| Gate | Result |
|---|---|
| Stage 1 replay | PASS; stdout and verification report byte-identical |
| Stage 2 scoring | PASS; stdout and verification report byte-identical; Tier 1/2 exact lookup equality and N06 max absolute difference `0.0` unchanged |
| Stage 3 markets | PASS; deterministic output identical after normalizing the two current public smoke-market rows; reciprocal pricing, no-vig, inversion, label-match, staleness, outage, and authentication-boundary checks unchanged |
| Stage 4 dashboard/risk | PASS; structured output identical after removing only the authorized `dashboard.html_bytes` field |
| Replay integration | PASS; 7 triggers across 4 trigger-bearing polls, with 5 Tier 3 reads and 2 honest Tier 2 re-fire fallbacks |

Stage 4 known-value results remained unchanged:

- EV: `0.200000`
- Full Kelly: `0.200000`
- Losing-streak probability: `0.784000`
- Expected streak windows: `3.072000`
- One-bet drawdown-floor probability: `0.400000`
- Comfort stake fraction: `0.050000`
- Label guard: all 7 financial-risk functions reject `deficit_erased`

The existing committed Stage 1-4 verification reports were hash-checked before and after the redirected test runs. They remained unchanged.

## Refactor Map

- `live/main.py`: process entry point, CLI, global app, and backward-compatible re-exports.
- `live/bootstrap.py`: dependency composition and stub/live source selection.
- `live/monitor.py`: poll loop, trigger processing, market/scoring orchestration, and in-memory state.
- `live/presentation.py`: API-facing serialization of game, market, scoring, and risk state.
- `live/api.py`: FastAPI routes, static-asset serving, configuration endpoints, and lifespan handling.
- `live/static/dashboard.html`: dashboard structure only.
- `live/static/dashboard.css`: presentation rules and responsive layout.
- `live/static/dashboard_panels.js`: engine, market, gap, and risk panel rendering.
- `live/static/dashboard.js`: API polling, state selection, watch list, game detail, trigger scrollback, and local config interaction.
- `live/ARCHITECTURE.md`: ownership map, poll flow, immutable contracts, and the stub/live activation boundary.

Temporary Stage 4 scaffolding was removed. Internal names were expanded where they carried domain meaning; external JSON keys, API paths, logger fields, lookup keys, formulas, and constants were not renamed.

## Visual QA

The refactored FastAPI app was rendered against the cached Georgia-Alabama replay in the local browser. Verified:

- Stub-mode banner is persistent and unambiguous.
- One watch-list game and all 7 recent triggers render.
- Engine, market, gap, and risk panels render with no console warnings or errors.
- The Tier 3 replay state displays the `deficit_erased` label and a 91.3%-wide conformal band.
- The risk panel retains the explicit statement that no label-matched conformal interval exists for `favorite_final_win`.
- The desktop viewport has no horizontal overflow.

The first visual pass exposed a text-encoding artifact introduced while splitting inline JavaScript into UTF-8 assets. The affected punctuation was restored to the exact original characters (`—`, `·`, `−`, and `×`), and the Stage 4 suite plus visual checks were rerun successfully. This was a refactor defect found and corrected before review; no underlying behavior or data changed.

## Scope Confirmation

- No scoring logic changed.
- No market or no-vig logic changed.
- No risk formula or policy constant changed.
- No trigger or logging contract changed.
- No API endpoint was removed or renamed.
- No new dependency was added.
- No research notebook or committed estimate was rerun.
- Stage 4.6 remains planning-only in `research/PROJECT_STATE.md`; `/simple` was not implemented.

