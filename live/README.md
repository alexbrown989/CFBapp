# N13 Stage 1 Live Monitor

N13 Stages 1-2 provide a read-only trigger and scoring foundation. The service polls a normalized scoreboard source, filters watched games, detects favorite-deficit threshold crossings, attaches the highest certified N12 scoring tier, prints alerts, and appends local JSONL records. It does not fetch live market prices, display a UI, or place bets.

## Modes

`stub` is the default and works now. `ScoreboardStub` replays normalized cached score states through the same interface used by the service.

`live` is coded but deliberately disabled until the late-August 2026 CFBD Tier 2 subscription and live schema certification. Switching sources is configuration-only:

```text
N13_DATA_SOURCE=live
N13_ENABLE_LIVE_SCOREBOARD=1
CFBD_TIER2_CONFIRMED=1
CFBD_API_KEY=...
```

Without all three activation values, `ScoreboardLive` raises: `CFBD Tier 2 subscription required; see research/PROJECT_STATE.md`.

The live adapter contains `AUGUST VERIFICATION REQUIRED` comments at the unverified nested `homeTeam`/`awayTeam` name and score mapping. Those fields must be checked against a real paid response before activation.

## Run

From the repository root, run the acceptance replay:

```powershell
C:\Users\Alexander\AppData\Local\Programs\Python\Python312\python.exe -m live.replay_verify
```

Run the independent Stage 2 lookup/model/parity gate:

```powershell
C:\Users\Alexander\AppData\Local\Programs\Python\Python312\python.exe -m live.stage2_verify
```

Run one empty stub poll:

```powershell
C:\Users\Alexander\AppData\Local\Programs\Python\Python312\python.exe -m live.main --once
```

Run the FastAPI shell when the backend dependencies are installed:

```powershell
uvicorn live.main:app --reload
```

Automatic interval polling is off by default. Set `N13_AUTO_POLL=1` to enable the loop; the default interval is 25 seconds.

## Watch List

`build_watchlist()` accepts current games, line records, and either AP-ranked teams or a manual team list. It prefers consensus spread, then averages real-sportsbook spreads using the locked N04 provider policy. Current-season rankings will be wired in after the 2026 preseason AP poll is available.

The replay acceptance test proves this path with cached 2024 games, lines, and AP ranking data before feeding the game through `ScoreboardStub`.

## Trigger Rules

- Deficit is `favorite_score - dog_score`; negative means the favorite trails.
- Thresholds are `3, 7, 10, 14, 21` by default.
- A threshold fires once when first crossed.
- If one poll jumps across several thresholds, every newly crossed threshold fires at the observed state.
- A threshold re-arms only after the favorite recovers above it, then may fire again on a later descent.

## Tiered Scoring

- Tier 1 is the primary `baseline_C` estimate for both labels. It needs only the crossed deficit threshold and quarter, and uses N12's bucket normalizer and lookup helpers.
- Tier 2 adds N10/N11 historical descriptive conditional, ranking, and no-vig market rates when their exact keys are available. Every rate retains sample size, reliability, confidence bounds, and source provenance.
- Tier 3 applies committed N06 plus conformal bounds only when all 31 core feature keys come from an explicitly certified source. Normal Stage 2 runtime reports `unavailable - no live play feed`; it never imputes an unavailable live feature.

`live/config.py` exposes the committed `research/notebooks/_lib_lookup.py` module through one documented path shim. Key construction, baseline time normalization, lookups, and N06 scoring remain in N12 and are not copied into the service.

The conformal interval is always shown with an N06 point probability. Its deployment q-hat is about 0.770, so wide intervals are expected and must remain visible.

## Runtime Parity Guard

`live/parity_guard.py` compares features recorded at trigger time with post-game values recomputed from the completed cache. Drift records are append-only and mark `tier3_suspect=true` whenever a feature exceeds tolerance. The first weeks of 2026 are the live-parity certification window; Tier 1 remains the operational anchor until that window is clean.

## Logging And Security

Trigger records append to `live/logs/*.jsonl`; runtime logs are gitignored. Stage 2 adds scoring fields without changing the original 16-field foundation. The reader accepts older Stage 1 rows and supplies nulls for absent scoring fields. Records contain game state, estimates, and provenance only. API keys are read from environment variables, `.env` is gitignored, and secrets are never logged.

The service uses read-only public/paid data endpoints. It has no market credentials, no wager-placement integration, and no external write path. Local JSONL logging is the only persistent write.
