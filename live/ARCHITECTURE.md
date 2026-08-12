# N13 Live Service Architecture

N13 watches selected college football games, detects when a pregame favorite crosses a deficit threshold, attaches committed historical/model estimates, reads public prediction-market prices, computes label-matched gaps and risk context, logs the observation, and exposes the current state through a localhost-only dashboard. It is read-only and never places wagers.

## Poll Flow

1. `ScoreboardSource.poll()` returns normalized game states from the stub or paid live adapter.
2. `LiveMonitor` filters states to the daily watch list.
3. `TriggerDetector` emits every newly crossed deficit threshold at the observed poll state.
4. `score_trigger()` attaches the highest certified N12 tier.
5. `MarketService` reads mapped Kalshi and Polymarket quotes.
6. Market gaps compare win contracts with `favorite_final_win` only.
7. `JSONLTriggerLogger` and the market-series logger append immutable observations.
8. `LiveMonitor` retains the latest game, trigger, scoring, and market state in memory.
9. `presentation.py` builds the dashboard's JSON-safe market and risk payloads.
10. `api.py` serves `/api/*`; the browser renders those responses without recomputing estimates.

## File Map

- `main.py`: service composition, compatibility exports, process entry point, and localhost startup.
- `bootstrap.py`: scoreboard, market, logger, detector, and scorer composition.
- `monitor.py`: poll cycle, trigger orchestration, and in-memory dashboard state.
- `presentation.py`: game, market, and label-safe risk serialization.
- `api.py`: FastAPI lifespan, dashboard assets, health endpoint, and `/api/*` routes.
- `config.py`: environment settings and gitignored personal risk configuration.
- `data_source.py`: normalized scoreboard contract, replay stub, and guarded CFBD live adapter.
- `watchlist.py`: daily game/favorite selection from games, rankings, and lines.
- `trigger_detect.py`: stateful threshold crossing, deduplication, and recovery re-arming.
- `scoring.py`: N12 Tier 1/2/3 lookup and model-scoring ladder.
- `parity_guard.py`: live-versus-cached feature drift audit for Tier 3 certification.
- `risk.py`: pure, `favorite_final_win`-guarded EV, Kelly, streak, and drawdown math.
- `logger.py`: append-only trigger, market, mapping, error, and parity records.
- `markets/`: public Kalshi/Polymarket discovery, quote normalization, inversion checks, and gap orchestration.
- `static/dashboard.html`: dashboard structure only.
- `static/dashboard.css`: responsive diagnostic-dashboard styling.
- `static/dashboard.js`: API polling, page state, watch list, detail selection, config, and trigger history.
- `static/dashboard_panels.js`: engine, market, gap, and risk panel rendering.
- `replay_verify.py`, `stage2_verify.py`, `stage3_verify.py`, `stage4_verify.py`: cumulative acceptance gates.

## Contracts That Must Not Break

1. **Log schema:** Stage 1's 16 trigger fields remain readable; later fields are additive only.
2. **N12 keys:** lookup-key construction and `_lib_lookup.py` remain the machine-precision-verified source of truth.
3. **Settlement label:** market gaps and all financial math require `favorite_final_win`; `deficit_erased` is a separate descriptive/model target.

## Stub And Live Boundary

`ScoreboardStub` and `ScoreboardLive` implement the same normalized source contract, so switching is configuration-only. Live activation remains guarded until the late-August CFBD Tier 2 subscription, a real `/scoreboard` schema check, current-season rankings, CFB market-mapping recertification, and a clean live-feature parity window. Tier 1 remains primary until those checks pass.

See [`research/PROJECT_STATE.md`](../research/PROJECT_STATE.md) for the research findings, access status, completed stages, and current next action.
