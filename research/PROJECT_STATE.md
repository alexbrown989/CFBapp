# Project State

Last updated: 2026-07-16

## Project Summary

This repo researches whether college football favorites that fall behind in-game create predictive or betting edge, then carries the validated pieces forward into a live monitoring service. The research phase built trigger-state features, calibrated probability models, market comparisons, descriptive stratifications, and a unified lookup/scoring layer. The live phase, beginning with N13, is a read-only alerting and dashboard system for 2026 games; it is designed to measure live in-game market behavior, not to auto-bet.

## Research Arc Status

- **N03 - Walk-forward validation with calibration:** built the locked calibrated trigger-state probability model and fitted-state provenance for live scoring. Key artifacts: `research/results/n06_full_fitted_state.json`, `research/results/n03_calibrated_predictions.parquet`.
- **N04 - Model vs pregame market:** found N03 probabilities beat pregame market probability on Brier, but the mechanism was current game-state calibration rather than live betting edge. Key artifacts: `research/results/n04_validation_results.parquet`, `research/results/n04_summary_report.md`.
- **N05 - Comeback descriptive/model-vs-baseline validation:** showed N03 does not beat a simple `fav_deficit x time_bucket` baseline on either `favorite_final_win` or `deficit_erased`. Key artifacts: `research/results/n05_analysis_results.json`, `research/results/n05_summary_report.md`.
- **N06 - Model fit on `deficit_erased`:** changing the label repaired calibration but still did not beat baseline_C; AUC tied the lookup baseline. Key artifacts: `research/results/n06_calibrated_predictions.parquet`, `research/results/n06_model_spec.json`.
- **N07 - Feature pool expansion:** added possession-adjusted candidates; 2 of 14 passed, but the expanded model remained marginal and not edge-grade. Key artifacts: `research/results/n07_expanded_model_spec.json`, `research/results/n07_summary_report.md`.
- **N08 - Stern-Winston + conformal diagnostic:** recommended `M3_N06_CONFORMAL` for deployment: N06 point probabilities plus split-conformal uncertainty. Key artifacts: `research/results/n08_comparison_results.json`, `research/results/n08_price_conversion_spec.json`.
- **N09 - Trigger-state stratifications and betting simulation:** confirmed baseline_C dominates structurally; always-bet favorites lost against pregame prices, while same-label edge filters were suggestive but underpowered. Key artifacts: `research/results/n09_baseline_analysis.json`, `research/results/n09_betting_summary.json`.
- **N10 - Direct fluke-deficit conditional analysis:** directly contradicted the original fluky-deficit comeback hypothesis against pregame prices. Key artifacts: `research/results/n10_conditional_analysis.json`, `research/results/n10_summary_report.md`.
- **N11 - AP ranking stratification:** closed the pregame-edge research arc; ranking behaves correctly but reveals no hidden pregame inefficiency. Key artifacts: `research/results/n11_analysis_results.json`, `research/results/n11_ranking_stratification.parquet`.
- **N12 - Unified probability lookup layer:** consolidated committed rates, probabilities, conformal intervals, and the N06 scorer into the live system's brain. `_lib_lookup.score_live_trigger()` reproduces N06 at machine precision. Key artifacts: `research/results/n12_probability_lookup.parquet`, `research/results/n12_live_scoring_spec.json`, `research/notebooks/_lib_lookup.py`.

## Core Finding

Pregame CFB markets are efficient, and often slightly favorite-rich, on favorite comeback scenarios. N04 through N11 tested pregame edge from model-vs-market, model-vs-baseline, label-change, feature-expansion, betting-simulation, direct conditional, and AP-ranking angles. None surfaced a deployable pregame betting edge. The only untested edge is **live in-game market overreaction** after a favorite falls behind. The 2026 live system exists to measure that fresh hypothesis prospectively.

## N13 Live System Architecture

N13 is a long-running service, not a research notebook.

- **Tier 1 scoring:** baseline_C (`fav_deficit x time_bucket`) is the primary live estimate. It needs only score, clock, and deficit.
- **Tier 2 enrichment:** N12 historical conditional/ranking lookups add context when the live state can build matching keys.
- **Tier 3 full N06:** full model scoring is shown only when all required features are available and live-parity-safe. N13 parity work already found attribution-heavy features are not safe for v1 without runtime guards.
- **Posture:** alert/dashboard only; no auto-bet. Kalshi and Polymarket market reads are public and require no credentials.
- **Deployment shape:** local web app, top-25 scope, 20-30 second poll when live access is active.
- **Stage 4 risk and variance panel:** completed. It shows label-safe EV, fractional-Kelly suggestion, stake as a fraction of bankroll, losing-streak probabilities, expected streak windows, and finite-season drawdown-floor risk. N06 conformal uncertainty remains attached to `deficit_erased`; moneyline risk math is hard-locked to `favorite_final_win`.
- **Build stages:** Stage 1 data loop -> Stage 2 scoring -> Stage 3 markets -> Stage 4 dashboard -> Stage 5 logging/retraining/Tailscale.

## CFBD Access Status

Keep this section current.

- `/scoreboard`: **PAID**. Requires CFBD Patreon Tier 1+. Needed for the live poll loop.
- `/live/plays`: **PAID Tier 2+**. Needed only for future Tier 3 N06 enrichment and live feature parity checks.
- **Recommendation:** subscribe to CFBD Tier 2 ($5/month) in late August 2026. It covers scoreboard, live plays, and ample call budget. It is not needed before then because the 2026 season has no live games before late August.
- `/lines`: free; 2026 data is available now. Step 0 observed 888 regular-season 2026 games with line containers.
- `/games`: free; 2026 data is available now. Step 0 observed 888 regular-season 2026 scheduled games.
- `/rankings`: 2026 returned an empty list on 2026-06-30. Expected to populate when preseason AP polling is published around August.
- Historical rankings and lines caches in repo cover 2015-2024 only.

## Build-Now vs Activate-Later Ledger

Stage 1 buildable-now components completed on 2026-07-14:

- [done] Source-agnostic scoreboard contract with `ScoreboardStub` and guarded `ScoreboardLive` implementations.
- [done] Trigger detection with multi-threshold crossing, one-fire deduplication, and recovery-based re-arming.
- [done] Watch-list construction from cached `/games` + `/lines`, with AP-ranking or manual-team input injection.
- [done] Append-only local JSONL trigger logging with a stable 16-field schema.
- [done] Cached 2024 replay verification using No. 2 Georgia at Alabama; trigger timing, deduplication, re-fire behavior, watch-list construction, and log schema all pass.
- [done] Stub-mode FastAPI service shell and configuration-only stub/live source selection.
- Verification: `research/results/n13_stage1_replay_verification.md`.

Stage 2 tiered scoring completed on 2026-07-14:

- [done] Tier 1 primary baseline_C lookup for both labels using N12's committed key and time normalization helpers.
- [done] Tier 2 historical conditional/ranking/no-vig enrichment with sample size, reliability, confidence bounds, and provenance on every estimate.
- [done] Tier 3 N06 plus conformal scoring behind an all-31-feature and certified-source runtime gate; normal runtime explicitly reports unavailable until `/live/plays` is wired and certified.
- [done] Additive scoring log schema; original Stage 1 records remain readable.
- [done] Post-game live-vs-cached feature drift comparison and append-only parity log schema.
- [done] Exact five-trigger acceptance gate: Tier 1/Tier 2 match N12 rows exactly and Tier 3 reproduces committed N06 with 0.0 maximum absolute difference.
- [done] Georgia-Alabama replay now produces full scoring reads: five committed trigger states reach Tier 3; two recovery-based re-fires stop honestly at Tier 2.
- Verification: `research/results/n13_stage2_scoring_verification.md`.

Stage 3 public-market integration completed on 2026-07-15:

- [done] Public, credential-free Kalshi and Polymarket clients behind one read-only market interface.
- [done] Exact team/date mapping with explicit favorite-outcome inversion guards and per-game daily mapping cache.
- [done] Executable raw price vs two-sided no-vig probability separation; probability gaps use `favorite_final_win` only.
- [done] Per-poll market-series logging and additive trigger-market fields.
- [done] Venue outage isolation, 429/5xx backoff, stale-quote suppression, and mandatory halt on unexpected authentication.
- [done] Public live smoke tests, Kalshi reciprocal pricing, no-vig known inputs, inversion/label guards, resilience, and Georgia-Alabama replay integration.
- Verification: `research/results/n13_stage3_market_verification.md`.
- Season-start requirement: re-certify exact game-to-market mapping when real 2026 CFB contracts list.

Stage 4 localhost dashboard completed on 2026-07-16:

- [done] Dense responsive watch list and game detail with persistent stub/live mode banner.
- [done] Tier 1/2/3 engine rendering with sample sizes, reliability, source labels, and explicit Tier 3 unavailability reasons.
- [done] Visible N06 `deficit_erased` conformal band and label-matched Tier 1/Tier 3 disagreement warning.
- [done] Per-venue raw/no-vig prices, spread, depth, staleness, gap, and friction-survival rendering.
- [done] Label-guarded EV, Kelly sizing, streak, and drawdown-floor risk panel. All seven financial functions reject `deficit_erased`.
- [done] Gitignored personal bankroll/risk configuration and localhost-only FastAPI API.
- [done] Georgia-Alabama replay, graceful degradation, security scan, and desktop/mobile browser verification.
- Verification: `research/results/n13_stage4_dashboard_verification.md`.

Stage 4.5 legibility refactor completed on 2026-07-17:

- [done] Split polling/state, presentation serialization, API routing, and process composition into focused modules without changing service contracts.
- [done] Split the no-build dashboard into structural HTML, CSS, core behavior, and panel-rendering assets.
- [done] Added `live/ARCHITECTURE.md` as the service entry point for future contributors and sessions.
- [done] Removed the external Stage 4 staging copy and one-off browser-demo server after confirming neither belonged to the service.
- [done] Preserved Stage 1-4 numerical, label, lookup, market, replay, logging, and API behavior under pre/post verification.
- Verification: `research/results/n13_stage4_5_refactor_verification.md`.

Planned Stage 4.6 - plain-language dashboard view:

- Add a `/simple` route that renders the same `/api/state` data in plain sentences without tier badges, conformal bands, raw/no-vig terminology, or reliability chips.
- Keep the diagnostic dashboard primary; `/simple` is additive and must not alter it.
- Preserve identical underlying data and label discipline: only `favorite_final_win` may feed financial statements or suggestions.
- Example framing: "Georgia is losing by 10. In 838 similar past games, teams in this spot came back and won 36% of the time. Kalshi is pricing them at 28%. Suggested bet: $45. You would lose a bet like this roughly 6 times out of 10 even when it is a good bet."

Activate later after CFBD Tier 2 subscription in late August 2026:

- Real `/scoreboard` poll loop.
- Current-season rankings watch list.
- Successful response schema verification for nested `homeTeam` / `awayTeam` score fields.
- Live latency/update-frequency measurement.
- Live-parity certification window during the first weeks of the season.
- Tier 3 `/live/plays` feature enrichment, only after parity guards pass.

## Key File Paths

- `research/notebooks/_lib_lookup.py` - N12 query helpers and committed N06 scorer.
- `research/results/n12_probability_lookup.parquet` - long-format lookup table for baseline_C, conditional/ranking rates, N06 probabilities, conformal intervals, and Stern-Winston references.
- `research/results/n12_live_scoring_spec.json` - N06 fitted state, conformal parameters, and live scoring spec.
- `research/results/n13_feature_parity_probe.md` - Tier feasibility and live feature parity recommendation.
- `research/results/n13_live_source_schema.md` - Step 0 live endpoint findings and current access blockers.
- `live/scoring.py` - tiered scoring engine backed by N12 helpers.
- `live/parity_guard.py` - post-game live-vs-cached Tier 3 drift audit.
- `research/results/n13_stage2_scoring_verification.md` - exact lookup, model reproduction, log compatibility, and parity-guard acceptance results.
- `live/markets/` - credential-free Kalshi/Polymarket discovery, quote normalization, inversion guards, and gap orchestration.
- `research/results/n13_stage3_market_verification.md` - public endpoint, pricing, mapping, label, replay, and resilience acceptance results.
- `live/risk.py` - pure, label-guarded EV, Kelly, streak, and drawdown-floor math.
- `live/static/dashboard.html` - no-build localhost dashboard.
- `research/results/n13_stage4_dashboard_verification.md` - risk math, replay, degradation, security, and responsive rendering acceptance results.
- `live/ARCHITECTURE.md` - two-minute request flow, file ownership map, and immutable service contracts.
- `research/results/n13_stage4_5_refactor_verification.md` - pre/post zero-behavior-change evidence and verifier-amendment record.
- `research/corrections_log.md` - methodology corrections and honest interpretation record.

## Methodology Discipline Reminders

- Halt-and-surface on state drift, schema mismatch, live access failure, or anything that changes trigger counting.
- No auto-bet. N13 may alert, log, and display; it must not place wagers.
- Market data uses public endpoints and needs no credentials. Do not add signing, wallet, portfolio, order, or trading capabilities. Do not log secrets or commit `.env` files.
- Preserve machine-precision reproduction gates for any code path that claims to score with committed N06/N12 state.
- Report both labels where descriptive rates are shown: `favorite_final_win` and `deficit_erased`.
- Keep predictive edge, structural edge, market edge, and betting edge terminology separate.
- Update this file at the end of every N13 stage.

## Open Decisions / Next Action

Current next action: review and commit N13 Stage 4.5. Stage 4.6 is recorded as an additive future plain-language view; Stage 5 remains logging/retraining operations and private Tailscale access. Live activation and Tier 3 certification remain pending the late-August CFBD Tier 2 subscription, current-season rankings, successful `/scoreboard` nested-field and latency certification, 2026 CFB market-mapping re-certification, and a clean first-weeks live parity window.
