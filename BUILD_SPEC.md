# CFB Live Edge Journal — Build Spec (V5 + V5.1)

> This document is the canonical spec. It is the V5 build prompt as authored by the project CTO, with the V5.1 patch appended verbatim, plus a small project-owner addendum at the bottom.
>
> The persistent agent rules live in [`.cursorrules`](./.cursorrules) at the repo root. They are reproduced inside this document for reference but the file is the source of truth.
>
> **The spec is locked.** Any further "what about adding X" idea goes to [`research/future_features.md`](./research/future_features.md) and is revisited only after Phase 0 produces a verdict.

---

## V5 — header

V5 supersedes V4. Same architecture, sharper schema and stricter validation.

**V5 changes from V4:**

- **Game-state context** in `trigger_events`: trigger sequence, drive ID, field position, possession, score pace
- **Team strength** as continuous variables (FPI / SP+) — not team identity
- **Market movement** (opening vs closing spread) as a sharp-money proxy
- **Stability rule**: features must show improvement on >=2 of 3 walk-forward test seasons, not just one
- **Calibration is a hard gate**: a model that ranks well but is badly calibrated is useless for betting. ECE thresholds added to Phase 0 acceptance.
- **Clarified uniqueness**: `trigger_events` records the *first time* a deficit is hit per game. Re-hits are ignored. Documented explicitly.

V4's research-first discipline, three-table architecture, lookahead bias rules, walk-forward validation, opening-drive shock features, and CLV-as-primary carry forward unchanged.

---

## `.cursorrules` (place at repo root)

The rules block below is the V5 author's text. Rules 14-20 are appended in the V5.1 Patch section, and rules 21-22 are appended in the Owner Addendum. The file at the repo root contains all 22 rules.

```
You are working on CFB Live Edge Journal — a personal betting research and live-edge web app.

ABSOLUTE RULES (never violate):

1. NEVER write app code (FastAPI routes, React components, database migrations) before Phase 0 (research notebooks) produces a research_findings.md showing positive CLV-equivalent edge AND acceptable calibration on a held-out test season AND a validated_filters.json file. If asked to skip ahead, refuse and explain why.

2. NEVER use lookahead data in any backtest feature. Specifically:
   - Do not use plays after the trigger play to compute features at the trigger.
   - Do not use final scores, post-game grades, or closing lines as model features (closing line is allowed only for CLV measurement, never as model input).
   - Do not optimize filter thresholds or hyperparameters on the test set.
   - When in doubt, ask: "Could this value have been computed in real time at the moment of the bet?" If no, it leaks. Add an assertion test.

3. NEVER hard-code domain assumptions. Every feature group is a hypothesis. State it, test it on training seasons, validate on held-out seasons.

4. PRIMARY METRICS:
   - CLV (closing line value) — must be positive median across test seasons
   - Calibration (ECE — Expected Calibration Error) — must be < 0.05 on test seasons
   Brier and log-loss are secondary diagnostics. A well-ranking model with bad calibration is useless for betting because the predicted probabilities feed directly into stake sizing.

5. WALK-FORWARD VALIDATION ONLY. Three rolling windows produce three test seasons. Never train and test on overlapping seasons.

6. STABILITY RULE (anti-overfitting):
   A feature is "validated" only if it shows positive held-out improvement on >=2 of 3 walk-forward test seasons. Features that work on exactly one test season are flagged unstable and excluded from validated_filters.json unless the user provides a clear football-mechanism reason and accepts the risk.

7. OVERFITTING GUARDRAILS:
   - L1-regularized logistic regression OR gradient boosting (max_depth <= 4, min_samples_leaf >= 30).
   - Feature selection happens on the validation set only, never on test.
   - n_events >= 30 x n_features in any bucket the model acts on.
   - If a (deficit, quarter) bucket has < 30 trigger events in training, the model refuses to produce a BET signal there at runtime.

8. CALIBRATION GATE: After fitting, compute reliability diagram with 10 bins. Test-season ECE must be < 0.05 (5%). If the model ranks well but is poorly calibrated, apply Platt scaling or isotonic regression — refit calibration parameters on the validation set, evaluate on test. Document calibration metrics in research_findings.md.

9. BE HONEST ABOUT DATA LIMITATIONS:
   - The Odds API historical: every 10 min before Sept 2022, every 5 min after. Not true live granularity.
   - CFBD has play-by-play + EPA + drives + lines + ratings (FPI, SP+). CFBD does NOT have PFF charting.
   - True historical live moneyline data is largely unavailable for free.

10. THIS IS A WEB APP. Stack is locked: FastAPI backend + React/Vite/Tailwind frontend + SQLite (dev) / Postgres (prod). No mobile, no native, no Next.js.

11. THREE-TABLE MODEL ARCHITECTURE:
    - trigger_events: what happened (immutable, one row per first-occurrence-of-deficit per game)
    - trigger_features: what the model saw (versioned by feature_set_version)
    - model_predictions: what the model thought (versioned by model_version, multiple per trigger over live snapshots)

12. PYTHON STYLE: black, type hints, Pydantic v2, pytest, Alembic.

13. SECRETS: Never commit .env. All API keys via os.getenv. Frontend never sees CFBD or Odds API keys.

WHEN UNCERTAIN: Ask. Do not invent assumptions about which filters work or which thresholds matter.
```

---

## Schema (V5 — Three-Table Core, Enriched)

> **Schema authority:** The DDL below is V5 SQLite-flavored. The V5.1 patch (further down) modifies the `model_predictions` table — see "Patch 1" before treating the model_predictions DDL below as final. Phase 1 Alembic migrations will emit dialect-specific DDL for SQLite (dev) and Postgres (prod).

### `trigger_events` — what happened

```sql
-- One row per (game, deficit threshold first hit).
-- "First time hit" semantics: if a favorite goes down 7, comes back, then goes
-- down 7 again, only the FIRST instance is recorded. Re-hits are ignored.
-- This is intentional — the first encounter with adversity is the cleanest
-- research event. Modeling re-collapses is a separate research question.

CREATE TABLE trigger_events (
    trigger_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id                 TEXT NOT NULL,
    season                  INTEGER NOT NULL,
    week                    INTEGER,
    fav_team                TEXT NOT NULL,
    dog_team                TEXT NOT NULL,

    -- DEFICIT CONTEXT
    fav_deficit             INTEGER NOT NULL,            -- 3, 7, 10, 14, 21
    trigger_sequence        INTEGER NOT NULL,            -- 1 = first deficit hit, 2 = next escalation, ...
    quarter                 INTEGER NOT NULL,
    seconds_remaining       INTEGER NOT NULL,            -- in regulation
    fav_score_at_trigger    INTEGER NOT NULL,
    dog_score_at_trigger    INTEGER NOT NULL,
    total_points_at_trigger INTEGER NOT NULL,            -- score pace
    points_per_minute       REAL,                        -- (total_points / minutes_elapsed)

    -- GAME-STATE CONTEXT (where on the field, who has the ball)
    trigger_play_id         TEXT,
    trigger_drive_id        TEXT,
    yardline_at_trigger     INTEGER,                     -- 0-100, 0 = own goal, 100 = opp goal
    possession_team         TEXT,                        -- 'favorite' | 'underdog'
    fav_has_ball            INTEGER,                     -- 1 if favorite has possession

    -- PRE-GAME CONTEXT
    pregame_spread          REAL NOT NULL,
    pregame_fav_ml          INTEGER,
    pregame_dog_ml          INTEGER,
    fav_pregame_rating      REAL,                        -- FPI or SP+ (continuous strength)
    dog_pregame_rating      REAL,
    rating_gap              REAL,                        -- fav - dog

    -- MARKET MOVEMENT (sharp-money signal)
    opening_spread          REAL,
    closing_spread          REAL,
    spread_movement         REAL,                        -- closing - opening; favors fav if more negative
    line_moved_against_fav  INTEGER,                     -- 1 if closing > opening (less of a fav)

    -- OUTCOME (filled when game ends)
    final_fav_won           INTEGER,
    final_fav_score         INTEGER,
    final_dog_score         INTEGER,
    margin_of_defeat        INTEGER,                     -- if fav lost, by how much

    created_at              TEXT NOT NULL,

    UNIQUE(game_id, fav_deficit)                         -- enforces first-occurrence semantics
);

CREATE INDEX idx_trigger_season_deficit ON trigger_events(season, fav_deficit);
CREATE INDEX idx_trigger_quarter ON trigger_events(quarter, fav_deficit);
CREATE INDEX idx_trigger_sequence ON trigger_events(game_id, trigger_sequence);
```

### `trigger_features` — what the model saw

```sql
CREATE TABLE trigger_features (
    feature_id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger_id                  INTEGER NOT NULL REFERENCES trigger_events(trigger_id),
    feature_set_version         TEXT NOT NULL,

    -- BASELINE EFFICIENCY
    fav_off_epa_per_play        REAL,
    fav_def_epa_per_play        REAL,
    dog_off_epa_per_play        REAL,
    epa_divergence              REAL,
    plays_so_far                INTEGER,

    -- OPENING-DRIVE SHOCK (V4's critical group)
    dog_received_opening_kickoff   INTEGER,
    dog_scored_on_opening_drive    INTEGER,
    opening_drive_was_td           INTEGER,
    opening_drive_was_explosive_td INTEGER,
    opening_drive_yards            INTEGER,
    opening_drive_plays            INTEGER,
    opening_drive_seconds          INTEGER,
    fav_def_epa_first_drive        REAL,
    fav_def_epa_after_first_drive  REAL,
    defense_stabilized_flag        INTEGER,

    -- EXPLOSIVE VS SUSTAINED
    dog_points_from_explosives  INTEGER,
    dog_points_from_sustained   INTEGER,
    dog_points_from_returns     INTEGER,
    dog_explosive_play_count    INTEGER,
    dog_avg_drive_yards         REAL,
    dog_avg_drive_plays         REAL,

    -- TURNOVER & SHORT FIELD
    fav_turnovers_so_far        INTEGER,
    dog_points_off_turnovers    INTEGER,
    dog_avg_starting_field_pos  REAL,
    short_field_tds_allowed     INTEGER,

    -- RED-ZONE FAILURE
    fav_red_zone_trips          INTEGER,
    fav_red_zone_tds            INTEGER,
    fav_yards_per_point         REAL,

    -- DOWN-AND-DISTANCE EFFICIENCY
    fav_early_down_success_rate REAL,
    fav_third_down_success_rate REAL,
    dog_early_down_success_rate REAL,
    dog_third_down_success_rate REAL,

    -- CONTEXT
    season_phase                TEXT,                    -- 'early' | 'mid' | 'late' | 'bowl'
    is_dome                     INTEGER,
    wind_mph                    REAL,
    temp_f                      REAL,

    UNIQUE(trigger_id, feature_set_version)
);

CREATE INDEX idx_features_trigger ON trigger_features(trigger_id);
CREATE INDEX idx_features_version ON trigger_features(feature_set_version);
```

### `model_predictions` — what the model thought

> **Read Patch 1 in the V5.1 section before treating this DDL as final.** V5.1 replaces `predicted_win_prob` with `raw_win_prob` + `calibrated_win_prob` (both NOT NULL), adds `calibration_version`, and adds `seconds_since_data`.

```sql
CREATE TABLE model_predictions (
    prediction_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger_id                  INTEGER NOT NULL REFERENCES trigger_events(trigger_id),
    model_version               TEXT NOT NULL,
    feature_set_version         TEXT NOT NULL,
    predicted_win_prob          REAL NOT NULL,
    calibrated_win_prob         REAL,                    -- after Platt/isotonic if applied
    fair_ml                     INTEGER,
    live_fav_ml                 INTEGER,
    live_book                   TEXT,
    devigged_market_prob        REAL,
    edge_pct                    REAL,
    confidence                  REAL,
    signal                      TEXT,                    -- 'BET' | 'WATCH' | 'AVOID' | 'INVALIDATED'
    created_at                  TEXT NOT NULL
);

CREATE INDEX idx_pred_trigger ON model_predictions(trigger_id);
CREATE INDEX idx_pred_signal ON model_predictions(signal, created_at);
```

### Supporting tables

> **Authorship note:** The V5 doc says "Supporting tables (unchanged from V4)" but V4 is not part of the project record. The DDL below is project-owner-authored from the field hints in the original build prompt and standard practice for the workflows described in V5 + V5.1. Treat it as draft-but-binding for Phase 1 Alembic migration unless explicitly revised.

#### `bankroll`

Append-only ledger of bankroll changes. The current balance is the latest row's `balance`.

```sql
CREATE TABLE bankroll (
    bankroll_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,                       -- ISO8601 of the event
    event_type      TEXT NOT NULL,                       -- 'deposit' | 'withdrawal' | 'bet_settled' | 'manual_adjust'
    delta           REAL NOT NULL,                       -- signed change applied at this row
    balance         REAL NOT NULL,                       -- balance AFTER applying delta
    bet_id          INTEGER REFERENCES bets(bet_id),     -- non-null when event_type = 'bet_settled'
    notes           TEXT,
    created_at      TEXT NOT NULL
);

CREATE INDEX idx_bankroll_timestamp ON bankroll(timestamp);
CREATE INDEX idx_bankroll_bet      ON bankroll(bet_id);
```

#### `bets`

One row per bet placed (paper or real). The schema includes every field the original prompt called out: `closing_ml`, `closing_implied_prob`, `clv_pct`, `is_paper`, `kelly_recommended_stake`, `edge_pct_at_bet`, `book_used`.

```sql
CREATE TABLE bets (
    bet_id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id               INTEGER REFERENCES model_predictions(prediction_id),
    trigger_id                  INTEGER NOT NULL REFERENCES trigger_events(trigger_id),
    placed_at                   TEXT NOT NULL,
    bet_type                    TEXT NOT NULL DEFAULT 'moneyline',  -- 'moneyline' | 'spread' | 'total' (V1 = moneyline only)
    side                        TEXT NOT NULL,                       -- 'favorite' | 'underdog'
    stake                       REAL NOT NULL,
    kelly_recommended_stake     REAL NOT NULL,
    bet_at_ml                   INTEGER NOT NULL,                    -- price at moment of bet (American odds)
    bet_at_implied_prob         REAL NOT NULL,                       -- devigged at moment of bet
    model_calibrated_prob       REAL NOT NULL,                       -- the prob that fed Kelly
    edge_pct_at_bet             REAL NOT NULL,
    book_used                   TEXT NOT NULL,
    closing_ml                  INTEGER,                             -- filled at game start (NOT used as model input)
    closing_implied_prob        REAL,
    clv_pct                     REAL,                                -- (bet_implied - closing_implied) / closing_implied
    result                      TEXT,                                -- 'win' | 'loss' | 'push' | 'pending'
    payout                      REAL,                                -- signed P/L on this bet
    is_paper                    INTEGER NOT NULL DEFAULT 1,          -- 1 = paper, 0 = real
    settled_at                  TEXT,
    notes                       TEXT,
    created_at                  TEXT NOT NULL,
    updated_at                  TEXT NOT NULL
);

CREATE INDEX idx_bets_trigger    ON bets(trigger_id);
CREATE INDEX idx_bets_placed_at  ON bets(placed_at);
CREATE INDEX idx_bets_paper      ON bets(is_paper, result);
CREATE INDEX idx_bets_prediction ON bets(prediction_id);
```

#### `snapshots`

One row per live poll of a game's state during Phase 4. Each snapshot may produce multiple `book_odds` rows and at most one `model_predictions` row.

```sql
CREATE TABLE snapshots (
    snapshot_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id             TEXT NOT NULL,
    captured_at         TEXT NOT NULL,                   -- ISO8601 wall-clock of the poll
    quarter             INTEGER,                          -- NULL between quarters or in OT
    seconds_remaining   INTEGER,                          -- regulation seconds; NULL in OT
    fav_score           INTEGER,
    dog_score           INTEGER,
    possession_team     TEXT,                             -- 'favorite' | 'underdog' | NULL
    yardline            INTEGER,                          -- 0-100
    last_play_id        TEXT,
    last_drive_id       TEXT,
    source              TEXT NOT NULL,                    -- 'cfbd_live_plays' | 'cfbd_games' | 'manual'
    raw_payload_hash    TEXT,                             -- for dedup of identical polls
    created_at          TEXT NOT NULL
);

CREATE INDEX idx_snapshots_game_time ON snapshots(game_id, captured_at);
CREATE INDEX idx_snapshots_hash      ON snapshots(raw_payload_hash);
```

#### `book_odds`

Per-book pricing observed in a snapshot. The Odds API returns multiple bookmakers per game; one row per (snapshot, book, market).

```sql
CREATE TABLE book_odds (
    book_odds_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id         INTEGER REFERENCES snapshots(snapshot_id),
    game_id             TEXT NOT NULL,                    -- denormalized for fast scans
    book                TEXT NOT NULL,                    -- 'pinnacle' | 'draftkings' | ...
    market              TEXT NOT NULL,                    -- 'h2h' | 'spreads' | 'totals'
    fav_ml              INTEGER,                           -- American odds; nullable for spreads/totals rows
    dog_ml              INTEGER,
    spread              REAL,                              -- favorite line, negative if favored
    spread_juice_fav    INTEGER,
    spread_juice_dog    INTEGER,
    total               REAL,
    over_juice          INTEGER,
    under_juice         INTEGER,
    captured_at         TEXT NOT NULL,
    created_at          TEXT NOT NULL
);

CREATE INDEX idx_book_odds_game_book_time ON book_odds(game_id, book, captured_at);
CREATE INDEX idx_book_odds_snapshot       ON book_odds(snapshot_id);
```

#### `weather`

Pre-game weather per venue/kickoff. One row per game. Sources: Open-Meteo (preferred) or CFBD `/games/weather`.

```sql
CREATE TABLE weather (
    weather_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id             TEXT NOT NULL UNIQUE,
    venue_id            TEXT,
    venue_lat           REAL,
    venue_lon           REAL,
    kickoff_at          TEXT,                              -- ISO8601
    temp_f              REAL,
    wind_mph            REAL,
    wind_dir_deg        INTEGER,
    humidity_pct        REAL,
    precipitation_in    REAL,
    conditions          TEXT,                              -- 'clear' | 'overcast' | 'snow' | ...
    is_dome             INTEGER NOT NULL DEFAULT 0,
    source              TEXT NOT NULL,                     -- 'open_meteo' | 'cfbd_games_weather'
    fetched_at          TEXT NOT NULL
);

CREATE INDEX idx_weather_game ON weather(game_id);
```

#### `api_health`

Heartbeat / latency log per external API call. Required to verify Phase 4 thresholds (latency p95 < 30s, webhook delivery > 95%).

```sql
CREATE TABLE api_health (
    api_health_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    service             TEXT NOT NULL,                     -- 'cfbd' | 'odds_api' | 'open_meteo' | 'webhook'
    endpoint            TEXT NOT NULL,
    request_at          TEXT NOT NULL,
    response_at         TEXT,
    latency_ms          INTEGER,
    status_code         INTEGER,
    success             INTEGER NOT NULL,                  -- 1 | 0
    error_class         TEXT,
    error_message       TEXT,
    quota_used          INTEGER,                           -- service-specific units consumed (e.g., Odds API quota cost)
    created_at          TEXT NOT NULL
);

CREATE INDEX idx_api_health_service_time ON api_health(service, request_at);
CREATE INDEX idx_api_health_failures     ON api_health(service, success, request_at);
```

---

## Phase 0 — Research (BLOCKING)

### Notebook 00 — Data Audit

Pull CFBD seasons 2017-2024. Document coverage of: games, lines (opening + closing), plays with EPA, drives, FPI/SP+ ratings. Confirm Open-Meteo historical weather is reachable.

### Notebook 01 — Build `trigger_events` Table

```python
# For each FBS game:
# - Identify pre-game favorite from spread sign
# - Walk plays in order, track running score
# - For each deficit threshold D in [3, 7, 10, 14, 21]:
#   * Find FIRST play where (fav_score - dog_score) reaches -D
#   * Capture: trigger_sequence (1 for first deficit hit overall, 2 for next, ...)
#              drive_id, yardline, possession, score pace, market movement
#   * Insert one trigger_events row
# - When game ends, update final_fav_won, final scores, margin_of_defeat

# DELIVERABLE: trigger_events fully populated for 2017-2024
# Sanity assertions Cursor must add:
# 1. UNIQUE(game_id, fav_deficit) holds
# 2. trigger_sequence is monotonically increasing within a game
# 3. yardline_at_trigger is 0-100
# 4. spread_movement = closing_spread - opening_spread (or NULL if missing)
```

### Notebooks 02a-g — Feature Group Hypothesis Testing

Same structure as V4: each notebook tests one feature group on training seasons, validates on holdout. Output goes to `research/results/feature_validation.csv`.

**V5 stability rule applied here**: a feature must improve held-out Brier on >=2 of 3 walk-forward test seasons. The `feature_validation.csv` tracks per-season improvement, and `validated_filters.json` only includes features that pass the stability check.

```python
# DELIVERABLE: research/results/feature_validation.csv
# Columns: feature, train_window, test_season, brier_improvement, calibration_improvement, passed_stability (bool)

# A feature passes the stability check iff:
#   sum(brier_improvement > 0 across all test seasons) >= 2
```

### Notebook 03 — Walk-Forward Validation + Calibration

```python
# Three rolling windows:
# Train 2017-2020 -> Val 2021 -> Test 2022
# Train 2017-2021 -> Val 2022 -> Test 2023
# Train 2017-2022 -> Val 2023 -> Test 2024

# For each window:
# 1. Use only features that passed Notebook 02 stability checks
# 2. Fit L1-regularized logistic regression OR gradient boosting (max_depth=3, min_samples_leaf=50)
# 3. Tune hyperparameters on validation season
# 4. Apply calibration (Platt or isotonic) — fit calibrator on validation set ONLY
# 5. Lock the model + calibrator. Evaluate ONCE on test season.
# 6. Compute and store:
#    - Brier score
#    - Log loss
#    - Expected Calibration Error (ECE) with 10 bins
#    - Reliability diagram (save as PNG)
#    - Simulated CLV

# OVERFITTING CHECK: train_brier and test_brier diverge > 15% -> reduce features or increase regularization

# DELIVERABLE: research/results/walk_forward_metrics.csv
# Columns: train_window, test_season, n_triggers, n_features_used,
#          train_brier, val_brier, test_brier,
#          test_log_loss, test_ece, test_max_calibration_error,
#          simulated_clv_median, simulated_roi
```

### Notebook 04 — CLV Simulation

Same as V3/V4. Use CFBD model WP as proxy for fair live ML. Validate against The Odds API where it exists.

### Phase 0 Acceptance Gate

`research/results/validated_filters.json`:

```json
{
  "model_version": "logreg_v1.0",
  "feature_set_version": "v1.0",
  "model_type": "logistic_regression_l1",
  "calibration_method": "isotonic",
  "trained_through_season": 2023,
  "active_features": ["..."],
  "model_coefficients": { "...": "..." },
  "calibration_params": { "...": "..." },
  "thresholds": {
    "min_confidence": 0.6,
    "min_edge_pct": 5.0,
    "min_sample_size_in_bucket": 30
  },
  "validation_metrics": {
    "test_seasons": [2022, 2023, 2024],
    "test_briers": [0.218, 0.221, 0.215],
    "test_eces": [0.038, 0.042, 0.035],
    "median_simulated_clv": 2.4,
    "calibration_passed": true,
    "stability_passed": true
  }
}
```

> See V5.1 Patch 3 for the expanded `validated_filters.json` shape that supersedes the example above.

`research_findings.md` must contain:

1. **Verdict**: edge confirmed / inconclusive / disproven
2. **Validated features list**: only features that passed stability rule (>=2 of 3 test seasons positive)
3. **Walk-forward CLV**: median > 0 across all three test seasons
4. **Calibration**: ECE < 0.05 on all three test seasons; reliability diagrams attached
5. **Sample sizes** per (deficit, quarter) bucket
6. **Honest assessment** of where the edge is real vs noise
7. **Recommendation**: build app / collect more data / reject

**Hard gates — all four must clear or do not build the app**:

- Median test-season CLV > 0
- All three test-season ECEs < 0.05
- >=1 active feature passed stability rule
- Sample size >=30 in at least one (deficit, quarter) bucket the model would act on

---

## Phase 1+ — Same as V4

Phase 1 (backend core), Phase 2 (API), Phase 3 (frontend), Phase 4 (live validation), Phase 5 (deployment) all proceed as in V4.

V5 additions to backend/services:

### `services/feature_extractor.py` updates

Must compute every field in `trigger_events` including the new ones (sequence, drive_id, yardline, possession, ratings, market movement, score pace). Shared module with research notebooks via `backend/app/ml/feature_set_v1.py`.

### `services/model_runner.py` updates

```python
# On startup, load both the model and the calibrator:
self.model = joblib.load(settings.model_path)
self.calibrator = joblib.load(settings.calibrator_path)  # NEW in V5

def predict(self, features: dict) -> float:
    raw_prob = self.model.predict_proba(self._featurize(features))[0, 1]
    calibrated_prob = self.calibrator.transform([raw_prob])[0]  # NEW
    return calibrated_prob
```

The calibrated probability is what feeds the edge calculator. The raw probability is logged separately for diagnostics.

---

## Phase 4 Validation Bar (updated for V5)

After >=4 weeks of in-season paper trading, all of these must clear before flipping to real-money mode:

| Metric                                | Required Threshold                                   |
| ------------------------------------- | ---------------------------------------------------- |
| Paper bets logged                     | >=50                                                 |
| Median CLV                            | > 0%                                                 |
| 25th percentile CLV                   | > -2%                                                |
| Live ECE (computed on paper bets)     | < 0.07 (slightly more lenient than Phase 0; smaller n) |
| Latency p95 during games              | < 30s                                                |
| Webhook delivery rate                 | > 95% within 90s                                     |
| Live Brier vs test-season Brier       | within 15%                                           |

The live ECE check is the V5 addition. If your model's predicted 40% bets actually win 25% of the time in-season, the model is miscalibrated against live data and pushing real money through it will lose. This catches calibration drift between historical training and live conditions.

---

## Updated Edge Card UI

```
┌─────────────────────────────────────────────────────────────┐
│  GEORGIA BULLDOGS                          ● LIVE  Q2 8:42 │
│  Pre-game favorite (-14.5 → close -16.5)        Sync: 4s   │
├─────────────────────────────────────────────────────────────┤
│  Score: GA 7 — TN 21    GA has ball at TN 38              │
│  Trigger #2 of game (was -7 in Q1, now -14)               │
├─────────────────────────────────────────────────────────────┤
│  WHY THIS SIGNAL                                            │
│  ✓ TN scored on opening drive via 42-yd TD pass             │
│  ✓ TN pts: 14 explosive / 7 sustained                       │
│  ✓ GA EPA/play: +0.18 (offense moving ball)                 │
│  ✓ GA has possession in opp territory                       │
│  ✓ Sharp money: spread moved -14.5 → -16.5 pre-game        │
│  ✗ GA red-zone trips: 1/2 (one stall)                       │
├─────────────────────────────────────────────────────────────┤
│  Best line       +320  (Pinnacle)        Avg market  +295  │
│  Devigged prob   24.4%                                       │
│  Model prob      38.7%   (ECE 0.04, n=87, conf 0.78)        │
│  Edge            +14.3%                                      │
│                                                              │
│  Kelly recommends:  $42  (1.4% of $3,000 bankroll)          │
├─────────────────────────────────────────────────────────────┤
│  ● BET    [ LOG PAPER → ]                                   │
└─────────────────────────────────────────────────────────────┘
```

V5 additions visible to the user:

- Spread open vs close (sharp-money proxy)
- Trigger sequence ("Trigger #2 of game") so you know if this is the first dip or escalating collapse
- Field position + possession
- Calibration diagnostic (ECE) on the model probability

> See V5.1 Patch 5 for additional UI lines (raw vs calibrated prob, data freshness).

---

## Anti-Patterns Cursor Must Refuse (V5 additions)

Carry over all V4 anti-patterns plus:

| Bad request                                                                                        | Why refuse                                                                                                                                                                                |
| -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| "Brier is fine, skip the calibration check"                                                        | A well-ranked but uncalibrated model loses money in betting because the predicted probabilities feed Kelly sizing. Calibration is non-negotiable.                                          |
| "This feature only worked on one test season but the football reason is intuitive, include it"     | Stability rule says >=2 of 3 test seasons. One-season improvements are usually overfit. Cursor should require explicit user acknowledgment of the risk before bypassing.                  |
| "Just use raw probabilities, skip Platt/isotonic"                                                  | If raw model is already well-calibrated (ECE < 0.05), fine. Otherwise apply calibration.                                                                                                  |
| "Drop trigger_sequence — it's redundant with deficit"                                              | It isn't. Sequence captures escalation pattern that deficit alone doesn't.                                                                                                                |
| "Use closing line as a model feature"                                                              | Lookahead at training time, plus you don't have closing line at the moment of a live bet anyway. Closing line is for CLV measurement only.                                                |

---

## What "Done" Looks Like (V5)

You ship V1 of the web app when:

- Phase 0 confirmed an edge with **all four hard gates clearing**: positive CLV, ECE <0.05, >=1 stable feature, >=30 events in acted-on buckets
- Phase 4 paper trading shows positive CLV across 50+ in-season bets AND live ECE <0.07
- Backend on Fly.io/Railway, frontend on Vercel, both HTTPS
- Real-money mode gated behind the validation thresholds and a 5%-bankroll-max-bet cap

If Phase 0 disproves the hypothesis or fails calibration, you ship nothing. That is still a successful outcome — you spent two weeks of research time instead of two months of build time.

---

## V5.1 Patch (applied 2026-05-06)

This is a patch on V5, not a rewrite. Apply these changes to `BUILD_SPEC.md` and `.cursorrules` before starting Phase 0.

**This is the last spec revision. After applying V5.1, Phase 0 begins.** Further "what about adding X" ideas get logged to a `future_features.md` file and revisited only after Phase 0 produces a verdict.

---

### Patch 1 — Schema additions to `model_predictions`

```sql
-- Replace the predicted_win_prob field with explicit raw + calibrated.
-- Add seconds_since_data and calibration_version for forensic auditing.

ALTER TABLE model_predictions ADD COLUMN raw_win_prob REAL NOT NULL;
ALTER TABLE model_predictions ADD COLUMN calibrated_win_prob REAL NOT NULL;
ALTER TABLE model_predictions ADD COLUMN calibration_version TEXT NOT NULL;
ALTER TABLE model_predictions ADD COLUMN seconds_since_data INTEGER;
ALTER TABLE model_predictions DROP COLUMN predicted_win_prob;  -- ambiguous, gone
```

**Rules:**

- `calibrated_win_prob` is what feeds Kelly sizing and edge calculation. Always.
- `raw_win_prob` is logged for diagnostics — drift detection, calibration recalibration triggers.
- `calibration_version` is independent of `model_version` because you may refit calibration on new data without changing the model itself.
- `seconds_since_data` records data freshness at prediction time. Lets you audit later: "did our losing bets correlate with stale data?"

---

### Patch 2 — `.cursorrules` additions

Append these to `.cursorrules`:

```
14. RATING TIMING (lookahead trap):
    FPI / SP+ / any team ratings used as features MUST be the version available
    BEFORE kickoff of that specific game. End-of-season ratings include the very
    games being predicted — using them is a silent lookahead leak that inflates
    backtest performance and breaks live.

    CFBD provides weekly ratings via /ratings/sp?year=Y&week=W. Always pull the
    rating from week W-1 (or earlier) for a game in week W.

    If only end-of-season ratings are available for some seasons, EXCLUDE
    ratings from those seasons rather than leak. Document the exclusion in
    research_findings.md.

15. NO OVERTIME IN PHASE 0:
    Trigger events that occur in overtime are excluded from Phase 0 research.
    OT in college football has fundamentally different dynamics (each team gets
    a possession from the 25, no clock, sudden-elimination rules). Modeling OT
    is a separate research question and goes in future_features.md.

    The seconds_remaining field is "regulation seconds remaining" — for any
    play in OT, mark the trigger event with quarter = NULL and exclude from
    feature_set_v1 training data.

16. MARKET MOVEMENT REQUIRES BOTH ENDPOINTS:
    The spread_movement feature requires both opening_spread and closing_spread
    to have been recorded BEFORE kickoff. If a season's CFBD lines data is
    missing one or the other, EXCLUDE market movement features for that season
    rather than reconstruct.

    Closing spread is allowed as a model feature because it was known before
    kickoff. It is NOT lookahead. Closing live moneyline IS lookahead at
    training time and is forbidden — that one is reserved for CLV measurement
    only.

17. ABSTENTION IS SUCCESS:
    A model that produces few BET signals but high CLV and calibration is
    BETTER than one that produces many signals with marginal edge. Do not
    "tune for more action." If the model returns AVOID 95% of the time and
    BET 5% with strong CLV on the BETs, that is the correct outcome. Do not
    lower thresholds to increase volume.

18. OBSERVED EDGE != EXECUTABLE EDGE:
    Live odds are polled every 60 seconds. The model sees what was true at
    that poll, not the absolute best price ever offered. Paper bets must be
    timestamped and measured against the best book price visible at that
    exact poll. Do not allow paper bets to retroactively claim better odds
    than what was actually queryable.

19. PHASE 0 OUTPUT MUST INCLUDE REJECTED FILTERS:
    research_findings.md must list both validated AND rejected feature groups,
    with the reason each one failed (e.g., "EPA divergence: improved Brier on
    2017-2020 train but regressed on 2021 val", "Wind: only 1 of 3 test seasons
    showed improvement, failed stability rule").

    Listing rejections prevents future "wait, did we try X?" cycles and keeps
    the research honest.

20. DATA QUALITY AUDIT IS PART OF NOTEBOOK 00:
    Before any feature engineering, the data audit must detect and document:
    - Duplicate game_id rows in CFBD games endpoint
    - Games with missing spreads (cannot identify favorite — exclude)
    - Games with incomplete play-by-play (< 100 plays — flag for review)
    - Neutral-site games (flag for review, may have different home/away dynamics)
    - Canceled or shortened games (exclude)
    - FCS opponents (exclude — different competitive level)
    - OT games (flagged, OT plays excluded per rule 15)

    Output: research/data/data_quality_report.md with counts and exclusion list.
```

---

### Patch 3 — Validated filters file expansion

`validated_filters.json` now includes both raw and calibrated metrics, plus action thresholds by quarter:

```json
{
  "model_version": "logreg_v1.0",
  "feature_set_version": "v1.0",
  "calibration_version": "isotonic_v1.0",
  "trained_through_season": 2023,
  "active_features": ["..."],
  "model_coefficients": { "...": "..." },
  "calibration_params": { "...": "..." },

  "action_thresholds": {
    "min_confidence": 0.6,
    "min_edge_pct": 5.0,
    "min_sample_size_in_bucket": 30,
    "max_seconds_since_data": 30,
    "allowed_quarters": [1, 2, 3, 4],
    "// Note": "allowed_quarters is set empirically based on Phase 0 results. Likely [1, 2] or [1, 2, 3] in practice."
  },

  "validation_metrics": {
    "test_seasons": [2022, 2023, 2024],
    "test_briers_raw": [0.219, 0.225, 0.218],
    "test_briers_calibrated": [0.218, 0.221, 0.215],
    "test_eces_raw": [0.082, 0.091, 0.078],
    "test_eces_calibrated": [0.038, 0.042, 0.035],
    "median_simulated_clv": 2.4,
    "n_bet_signals_per_test_season": [42, 38, 51],
    "calibration_passed": true,
    "stability_passed": true
  },

  "rejected_features": [
    {
      "feature": "wind_mph",
      "reason": "Improved val Brier in 2 seasons but degraded in 2024. Failed stability rule.",
      "test_season_results": {"2022": "+0.4%", "2023": "+0.6%", "2024": "-0.9%"}
    }
  ]
}
```

The `allowed_quarters` field is the production answer to your "should we limit to first half" question. Phase 0 fills this in based on which quarters produced positive CLV across all three test seasons. If Q3-Q4 fail stability, they get excluded automatically. Don't pre-decide; let evidence decide.

---

### Patch 4 — Decision on quarter restriction

Q: *"Should we limit to first two quarters only?"*

A: **No, not in research. Yes, probably in production — but as evidence-driven output, not a pre-imposed input.**

- Phase 0 trains and tests on all quarters (excluding OT per rule 15).
- The walk-forward + stability rule will reject quarters where edge is unstable.
- `validated_filters.json` -> `action_thresholds.allowed_quarters` records the empirical answer.
- The runtime model refuses to produce BET signals outside `allowed_quarters`.

If Phase 0 results show Q1-Q2 with strong CLV and Q3-Q4 with weak/negative CLV, `allowed_quarters` ends up `[1, 2]` and you bet only the first half. That's the right way to reach that conclusion.

If you pre-filter to Q1-Q2 in research, you'd never know whether Q3 had a real edge you're leaving on the table.

---

### Patch 5 — Edge card UI minor update

The edge card already shows the calibration metric (ECE 0.04). Add one more diagnostic line below model probability:

```
│  Model prob (raw)        37.1%                              │
│  Model prob (calibrated) 38.7%   ← used for Kelly           │
│  Calibration ECE         0.04                                │
│  Data freshness          4s ago                              │
```

This makes it visible when raw and calibrated probabilities diverge significantly — a sign the calibrator may be doing too much work and the underlying model needs a refit.

---

### Lock the spec, start Phase 0

This is the last revision before research begins. Any further "what about adding X" thinking goes in `research/future_features.md` and gets revisited only after Phase 0 produces a verdict.

Reasons to stop iterating now:

- The schema captures every feature group with a defensible football mechanism
- Calibration and stability rules prevent overfit nonsense from reaching production
- Lookahead traps are explicitly closed (rules 14, 16)
- Research output transparency is enforced (rule 19)
- Data quality is audited up front (rule 20)
- Production-vs-research filter decision is evidence-based (Patch 4)

Reasons to NOT keep adding features:

- Each new feature is another overfitting opportunity
- The marginal value of feature #26 is much smaller than the value of finishing Notebook 00
- You're not going to think of a feature in week 3 of planning that is more important than what's already in V5.1
- The point of research is to find out what works, which requires actually running it

---

## Owner addendum (2026-05-06)

A few small items that needed to be captured outside the V5/V5.1 text. None of these change the spec's intent; they reconcile concrete implementation choices.

### A.1 — Tech-stack deviation: PyJWT in place of python-jose

V5 rule 12 lists "Pydantic v2, pytest, Alembic" in the Python style block but doesn't pin a JWT library. The original build prompt's R9 listed `python-jose[cryptography]`. We instead use `PyJWT` (currently `2.12.1`) because:

- `python-jose` has been lightly maintained since 2024 and has had recent CVEs (e.g., CVE-2024-33664).
- `PyJWT` is the library used in the current FastAPI security tutorial and is actively maintained.
- The agent surface (`jwt.encode`, `jwt.decode`) is equivalent for the access-token use case in Phase 2.

This is the only deliberate deviation from the locked stack. No other dependency or framework swaps.

### A.2 — OT trigger handling: do not insert, do not store with NULL

V5 DDL says `quarter` and `seconds_remaining` are `NOT NULL` on `trigger_events`. V5.1 rule 15 says "for any play in OT, mark the trigger event with `quarter = NULL` and exclude from feature_set_v1 training data."

These two are incompatible as written. Resolution adopted by the project owner:

- **The V5 DDL stays as the schema of record (`quarter`, `seconds_remaining` NOT NULL).**
- **Notebook 01 does not insert a `trigger_events` row when the deficit is first hit during OT.** OT first-hits are simply absent from the table, not present-with-NULL.
- The intent of rule 15 ("exclude from training") is fully honored; the wording about NULL columns is treated as commentary, not a hard requirement.

Practical consequence: OT triggers cannot be recovered from the DB later. If at some future point we want to model OT (per `future_features.md`), we'll add an `ot_trigger_events` table or a `phase TEXT` column then — not retrofit NULLability into the existing schema.

### A.3 — `data_quality_report.md` location

Stays at `research/data/data_quality_report.md` as V5.1 rule 20 specifies. `research/data/` is gitignored; the report is a local audit artifact regenerated on every Notebook 00 run, which is the correct behavior for a quality report (it should reflect the current pull, not a stale committed copy).

### A.4 — Honest data limitations: CFBD v2 monthly cap

A fourth bullet to V5 rule 9, verified live against `apinext.collegefootballdata.com/api-docs.json` and the CFBD blog:

- **CFBD API v2 free tier = 1000 calls/month.** v1 was sunset before the 2025 season; both `apinext.collegefootballdata.com` and `api.collegefootballdata.com` now point to v2. There is no per-second throttle, only the monthly cap. Patreon Tier 3 ($10/mo) raises the cap to 75K calls and unlocks GraphQL. Phase 0 historical backfill of plays/drives/lines/ratings across ~10 seasons will not fit in 1K calls.

### A.5 — Open Phase 0 prerequisites (block Notebook 00, not earlier)

Decisions deferred to the start of Notebook 00:

1. **CFBD subscription tier.** 1K free vs $10/mo Tier 3 (75K calls + GraphQL). Influences whether Notebook 00 runs against all seasons in one pass or chunks across months.
2. **The Odds API plan.** Historical endpoints are paid-only and cost 10 quota per (region × market) per snapshot. Options: subscribe and pull historical NCAAF (`americanfootball_ncaaf`), or skip Odds historical entirely and use CFBD `/lines` for opening/closing spreads in research, deferring The Odds API to Phase 4 paper-trading only.

Neither decision blocks Phase 0 file scaffolding (this commit) or the dependency install that follows.

### A.6 — Repo layout

The project lives directly in the existing `CFBapp/` workspace root rather than a `cfb-edge-journal/` subdirectory. The user's git repo at `C:\Users\Alexander\Documents\CFB\CFBapp` is the project root.
