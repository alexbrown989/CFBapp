# N13 Stage 1 Live Monitor

Stage 1 is a read-only trigger-detection foundation. It polls a normalized scoreboard source, filters watched games, detects favorite-deficit threshold crossings, prints alerts, and appends local JSONL records. It does not score models, fetch market prices, display a UI, or place bets.

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

## Logging And Security

Trigger records append to `live/logs/*.jsonl`; runtime logs are gitignored. Records contain game state and provenance only. API keys are read from environment variables, `.env` is gitignored, and secrets are never logged.

The service uses read-only public/paid data endpoints. It has no market credentials, no wager-placement integration, and no external write path. Local JSONL logging is the only persistent write.
