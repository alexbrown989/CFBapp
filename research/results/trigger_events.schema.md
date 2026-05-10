# trigger_events.csv — schema sidecar

**Generated:** 2026-05-09 16:16:27 Pacific Daylight Time
**Source notebook:** `research/notebooks/01_trigger_events.ipynb`
**Source commit:** `9df25b9e84b54d9d83259665b0294d4852bc35c4`
**Trigger table version:** `v1`

This file documents the columns of `trigger_events.csv`, the provider priority
used at run time, and the empirical pre/post-play score-state findings that
the trigger detection function relies on. Every downstream notebook that
reads `trigger_events.csv` should treat this file as the canonical schema
reference. Do not paraphrase from memory.

## Column dictionary

Every column with type, unit, range/sign convention, and lookahead category.

**Categories:**
- **IDENTIFIER**: primary key components and within-game counters.
- **PRE-GAME CONTEXT**: observable before kickoff. R3-safe by definition.
- **POINT-IN-TIME GAME STATE**: derivable from plays with `playNumber <= trigger.play_number` only. R3-safe by construction (verified by `assert_no_lookahead`).

Game-final **LABEL** / target columns from the V5 DDL (`final_fav_won`, `final_fav_score`,
`final_dog_score`, `margin_of_defeat`) are **not** in this CSV — they live in the sibling
file `trigger_outcomes.csv` (same row order and natural key `(game_id, fav_deficit)`) so
feature notebooks (02a–g) never parse labels alongside in-game state. Notebook 03 joins
the two files explicitly. See `trigger_outcomes.schema.md`.

| Column | Type | Unit | Range / sign | Category | Description |
|---|---|---|---|---|---|
| `game_id` | INTEGER | — | CFBD game ID | IDENTIFIER | Primary key component. Joins to trigger_features.game_id and to the games metadata. |
| `fav_deficit` | INTEGER | points | {3, 7, 10, 14, 21} | IDENTIFIER | Primary key component. Threshold value, NOT the actual deficit observed (see actual_deficit_at_trigger). |
| `trigger_sequence` | INTEGER | — | 1..N within game | IDENTIFIER | Ordinal index of this trigger within its game, sorted by (play_number ASC, fav_deficit ASC). seq=1 is the first crossed threshold. |
| `season` | INTEGER | year | 2015..2024 | PRE-GAME CONTEXT | Calendar year the season started (e.g., 2024 for the 2024-25 season). |
| `season_type` | TEXT | — | regular | postseason | PRE-GAME CONTEXT | Regular season or bowl/playoff. Excludes any 'preseason' or 'spring' values (none in our corpus). |
| `week` | INTEGER | — | 1..16 (regular), 1 (postseason) | PRE-GAME CONTEXT | Week within season. Postseason games are typically week=1. |
| `fav_team` | TEXT | — | school name | PRE-GAME CONTEXT | Pre-game favorite per pregame_spread sign. Pick'em games (spread==0) are excluded entirely. |
| `dog_team` | TEXT | — | school name | PRE-GAME CONTEXT | Pre-game underdog per pregame_spread sign. |
| `home_team` | TEXT | — | school name | PRE-GAME CONTEXT | Home team (geographic/scheduling, not favorite-related). |
| `away_team` | TEXT | — | school name | PRE-GAME CONTEXT | Away team. |
| `home_is_fav` | BOOLEAN | — | True | False | PRE-GAME CONTEXT | True iff fav_team == home_team. |
| `pregame_spread` | REAL | points | always <= 0 (favorite negative) | PRE-GAME CONTEXT | Closing spread from favorite's perspective. Sign convention: -7.0 means favorite is laying 7 points. Pick'em (0) excluded. |
| `pregame_spread_provider` | TEXT | — | provider name | PRE-GAME CONTEXT | Sportsbook actually used for pregame_spread, per priority list (see Provider priority section below). |
| `pregame_fav_ml` | REAL | American odds | negative for favorite | PRE-GAME CONTEXT | Closing moneyline price on the favorite. NULL if no real book has ML for this game (frequent in 2018-2022). |
| `pregame_dog_ml` | REAL | American odds | positive for underdog | PRE-GAME CONTEXT | Closing moneyline price on the underdog. NULL with same coverage as pregame_fav_ml. |
| `pregame_ml_provider` | TEXT | — | provider name | NULL | PRE-GAME CONTEXT | Sportsbook actually used for pregame_fav_ml/pregame_dog_ml. Real books only — consensus is excluded for ML. |
| `opening_spread` | REAL | points | <= 0 (closing-fav perspective) | PRE-GAME CONTEXT | Opening spread expressed in closing favorite's perspective. NULL if no acceptable provider has spreadOpen for this game (~30% missing rate). |
| `opening_spread_provider` | TEXT | — | provider name | NULL | PRE-GAME CONTEXT | Sportsbook actually used for opening_spread. |
| `closing_spread` | REAL | points | always <= 0 (favorite negative) | PRE-GAME CONTEXT | Same as pregame_spread (the closing spread IS the pre-game spread for purposes of trigger_events). |
| `spread_movement` | REAL | points | negative = fav got more favored | PRE-GAME CONTEXT | closing_spread - opening_spread (in closing-fav's perspective). NULL when opening_spread is NULL (R16). Negative = favorite getting more favored from open to close. |
| `line_moved_against_fav` | BOOLEAN | — | True | False | NULL | PRE-GAME CONTEXT | True iff spread_movement > 0. NULL when spread_movement is NULL. |
| `fav_pregame_rating` | REAL | Elo points | typically 1200..2100 | PRE-GAME CONTEXT | Favorite's pre-game Elo (CFBD homePregameElo / awayPregameElo). A.7 substitution for SP+/FPI which leak. |
| `dog_pregame_rating` | REAL | Elo points | typically 1200..2100 | PRE-GAME CONTEXT | Underdog's pre-game Elo. NULL only if CFBD's Elo is missing for that team in that game (rare per N00 audit, all 20 (season, type) pairs cleared 80%). |
| `rating_gap` | REAL | Elo points | always >= 0 typically | PRE-GAME CONTEXT | fav_pregame_rating - dog_pregame_rating. May be slightly negative when home-field-advantage flipped the favorite vs the rating ranking; the favorite is determined by SPREAD, not Elo. |
| `trigger_play_id` | INTEGER | — | CFBD play ID | POINT-IN-TIME GAME STATE | V5 DDL name. Unique CFBD identifier for the play that caused this trigger — distinct from any hypothetical current/next play fields. Joins to /plays. |
| `play_number` | INTEGER | — | 1..~180 typical | POINT-IN-TIME GAME STATE | Sequential play number within the game (CFBD playNumber). The trigger is the FIRST play after which fav_deficit was reached. |
| `play_type` | TEXT | — | CFBD playType string | POINT-IN-TIME GAME STATE | Type of play (e.g., 'Touchdown', 'Field Goal Good', 'Pass Reception', 'Interception Return Touchdown'). Free text per CFBD. |
| `quarter` | INTEGER | — | 1, 2, 3, 4 (regulation only) | POINT-IN-TIME GAME STATE | Quarter (CFBD period). OT (period >= 5) excluded entirely per A.2/R15. |
| `clock_minutes_remaining` | INTEGER | minutes | 0..15 | POINT-IN-TIME GAME STATE | Minutes-portion of the game clock at the start of the trigger play. |
| `clock_seconds_remaining` | INTEGER | seconds | 0..59 | POINT-IN-TIME GAME STATE | Seconds-portion of the game clock at the start of the trigger play. |
| `clock_seconds_in_period_total` | INTEGER | seconds | 0..900 | POINT-IN-TIME GAME STATE | Total seconds remaining in the current quarter (clock_minutes * 60 + clock_seconds). |
| `seconds_remaining_in_regulation` | INTEGER | seconds | 0..3600 | POINT-IN-TIME GAME STATE | Total seconds remaining until the end of regulation (Q4 0:00). |
| `minutes_elapsed_total` | REAL | minutes | 0.0..60.0 | POINT-IN-TIME GAME STATE | Minutes of game elapsed at the START of the trigger play. (15 - clock) + (period-1)*15. |
| `fav_score_at_trigger` | INTEGER | points | >=0 | POINT-IN-TIME GAME STATE | Favorite's score at the END of the trigger play. |
| `dog_score_at_trigger` | INTEGER | points | >=0 | POINT-IN-TIME GAME STATE | Underdog's score at the END of the trigger play. |
| `actual_deficit_at_trigger` | INTEGER | points | >=fav_deficit | POINT-IN-TIME GAME STATE | dog_score - fav_score at trigger. Exceeds fav_deficit when crossing happened on a multi-point play (e.g., a 7-pt TD on a 0-0 game crosses both D=3 and D=7 simultaneously). |
| `total_points_at_trigger` | INTEGER | points | fav + dog | POINT-IN-TIME GAME STATE | fav_score_at_trigger + dog_score_at_trigger. |
| `points_per_minute` | REAL | points/min | >=0 | NULL | POINT-IN-TIME GAME STATE | total_points_at_trigger / minutes_elapsed_total. NULL when minutes_elapsed_total == 0. |
| `possession_team` | TEXT | — | school name | POINT-IN-TIME GAME STATE | Team on offense at the trigger play (CFBD `offense`). |
| `fav_has_ball` | BOOLEAN | — | True | False | POINT-IN-TIME GAME STATE | True iff possession_team == fav_team. |
| `yardline_at_trigger` | INTEGER | yards | 0..100 (offense-perspective) | POINT-IN-TIME GAME STATE | CFBD `yardsToGoal`. Distance to defense's end zone from offense's perspective. 0=defense's goal line, 100=offense's own goal line. |
| `distance_to_first_down` | INTEGER | yards | >=0 | NULL | POINT-IN-TIME GAME STATE | CFBD `distance`. Yards needed for first down. NULL on certain non-snap plays. |
| `down` | INTEGER | — | 1..4 | NULL | POINT-IN-TIME GAME STATE | Down number. NULL on kickoffs, PATs, certain 2pt conversions, etc. |
| `trigger_drive_id` | INTEGER | — | CFBD drive ID | NULL | POINT-IN-TIME GAME STATE | V5 DDL name. Drive that contains the trigger play. NULL if CFBD didn't assign one. |
| `drive_number_in_game` | INTEGER | — | 1..N | NULL | POINT-IN-TIME GAME STATE | 1-indexed drive ordinal in the game (CFBD `driveNumber`). |


## Provider priority used at run time

For pregame_spread, opening_spread, and pregame_fav_ml / pregame_dog_ml,
this notebook walks providers in priority order and takes the first non-null
value.

**Real-book priority (alphabetical):**

1. `Bovada`
2. `Caesars`
3. `DraftKings`
4. `ESPN Bet`

The N00 audit printed a per-(season, season_type) provider count matrix,
but that matrix was not written to a checked-in CSV, so per-provider
coverage cannot be cited here from project artifacts. Re-evaluate the
ordering in Notebook 03 if walk-forward results show ordering affects
calibration or CLV; the priority list is a top-of-notebook constant
(`PROVIDER_PRIORITY_REAL_BOOKS`) so reordering is a one-line change.

**Spread-only fallback:** `consensus`. Used when no real book has a spread
for that game/field. **Never used for moneylines** (consensus is a synthetic
mid-market estimate, not a tradeable price; using it for ML breaks devigging
math in N04).

**Excluded entirely:** `numberfire`, `teamrankings` (rating-service projections,
not market prices). State-specific Caesars variants
(`Caesars (Pennsylvania)`, etc.) are also excluded — only the canonical
`Caesars` entry is matched.

### Closing spread provider mix actually selected

| Provider | Games |
|---|---:|
| Bovada | 3,938 |
| consensus | 2,264 |
| Caesars | 1,158 |


### Opening spread provider mix actually selected

| Provider | Games |
|---|---:|
| (NULL) | 4,241 |
| Bovada | 3,114 |
| ESPN Bet | 3 |
| DraftKings | 2 |


### Moneyline provider mix actually selected

| Provider | Games |
|---|---:|
| (NULL) | 4,371 |
| Bovada | 2,958 |
| DraftKings | 25 |
| ESPN Bet | 6 |


## Score-state confidence levels

The trigger detection scorer (cell 16 of N01) does NOT assume a uniform
POST_PLAY or PRE_PLAY convention. Instead, every scoring play type that
may carry `scoring=true` is empirically verified against the cached
corpus (cell 14 of N01) and assigned to one of three confidence tiers.
The scorer branches on the tier; cell 14 raises if any subtype fails
verification, so triggers are never produced under a `FAIL` verdict.

Per-subtype verdicts also persist to `_subtype_verdicts.json` (sibling
file in this directory) so future N01 re-runs can diff and warn on
>2pp drift in any subtype's POST share.

### Tier 1 -- VERIFIED_POST_PLAY

>= 20 samples in the cached corpus AND >= 95% verified-POST
share. The scorer reads `offenseScore` / `defenseScore` direct, with no
cross-check. Standard guarantee.

**Subtypes (17):** `blocked field goal`, `blocked field goal touchdown`, `blocked punt`, `blocked punt touchdown`, `field goal good`, `fumble recovery (opponent)`, `fumble recovery (own)`, `fumble return touchdown`, `interception`, `interception return touchdown`, `kickoff`, `passing touchdown`, `punt`, `punt return touchdown`, `rushing touchdown`, `sack`, `safety`

### Tier 2 -- POST_BEST_EFFORT (Kickoff Return TD carve-out)

>= 65% verified-POST share. The scorer reads direct, then
cross-checks against the same team's score at the next clean (non-PAT)
play with a +/-2-point tolerance. Cross-check failures are logged
to `SCORING_AMBIGUITY_LOG` and the play is skipped (no trigger emitted).
Documented carve-out for Kickoff Return TD, which has a small fraction
of pre-play recorded rows in the corpus.

**Subtypes (1):** `kickoff return touchdown`

### Tier 3 -- BEST_EFFORT_LOW_N

< 20 samples in the cached corpus. The scorer reads direct and
applies a +/-7-point cross-check at the next clean play. Soft
guarantee: with low sample counts the empirical verdict is statistically
weak. **Future N03 / N04 authors should treat trigger rows whose
`play_type` lower-cases to one of these subtypes with appropriate
skepticism**, or filter them out for the conservative variant of the
held-out test (`fav_deficit` count distribution by tier is reported in
this notebook's run summary, and `_subtype_verdicts.json` carries the
per-subtype sample sizes).

**Subtypes (9):** `kickoff return (offense)`, `missed field goal return touchdown`, `pass incompletion`, `pass interception return`, `pass reception`, `penalty`, `rush`, `timeout`, `uncategorized`

### Empirical findings -- this run

| Play type | Verdict | Verified-POST / Verified-PRE / Total | POST share |
|---|---|---|---:|
| `blocked field goal` | `POST_PLAY` | 20 / 0 / 20 | 100.0% |
| `blocked field goal touchdown` | `POST_PLAY` | 32 / 0 / 32 | 100.0% |
| `blocked punt` | `POST_PLAY` | 170 / 0 / 170 | 100.0% |
| `blocked punt touchdown` | `POST_PLAY` | 77 / 0 / 77 | 100.0% |
| `field goal good` | `POST_PLAY` | 18802 / 0 / 18802 | 100.0% |
| `fumble recovery (opponent)` | `POST_PLAY` | 246 / 0 / 246 | 100.0% |
| `fumble recovery (own)` | `POST_PLAY` | 33 / 0 / 33 | 100.0% |
| `fumble return touchdown` | `POST_PLAY` | 325 / 0 / 325 | 100.0% |
| `interception` | `POST_PLAY` | 25 / 0 / 25 | 100.0% |
| `interception return touchdown` | `POST_PLAY` | 1397 / 0 / 1397 | 100.0% |
| `kickoff` | `POST_PLAY` | 98 / 0 / 98 | 100.0% |
| `passing touchdown` | `POST_PLAY` | 25741 / 0 / 25741 | 100.0% |
| `punt` | `POST_PLAY` | 301 / 0 / 301 | 100.0% |
| `punt return touchdown` | `POST_PLAY` | 87 / 0 / 87 | 100.0% |
| `rushing touchdown` | `POST_PLAY` | 26950 / 0 / 26950 | 100.0% |
| `sack` | `POST_PLAY` | 142 / 0 / 142 | 100.0% |
| `safety` | `POST_PLAY` | 444 / 0 / 444 | 100.0% |
| `kickoff return touchdown` | `POST_BEST_EFFORT` | 300 / 0 / 300 | 100.0% |
| `kickoff return (offense)` | `BEST_EFFORT_LOW_N` | 6 / 0 / 6 | 100.0% |
| `missed field goal return touchdown` | `BEST_EFFORT_LOW_N` | 4 / 0 / 4 | 100.0% |
| `pass incompletion` | `BEST_EFFORT_LOW_N` | 2 / 0 / 2 | 100.0% |
| `pass interception return` | `BEST_EFFORT_LOW_N` | 2 / 0 / 2 | 100.0% |
| `pass reception` | `BEST_EFFORT_LOW_N` | 8 / 0 / 8 | 100.0% |
| `penalty` | `BEST_EFFORT_LOW_N` | 3 / 0 / 3 | 100.0% |
| `rush` | `BEST_EFFORT_LOW_N` | 8 / 0 / 8 | 100.0% |
| `timeout` | `BEST_EFFORT_LOW_N` | 1 / 0 / 1 | 100.0% |
| `uncategorized` | `BEST_EFFORT_LOW_N` | 4 / 0 / 4 | 100.0% |


**Convention applied to trigger detection:** `MIXED_BY_PLAY_TYPE_VERIFIED`

The scorer's behavior is invariant to the `SCORE_STATE_CONVENTION`
banner; that constant is kept only for backward compatibility with
internal logging. The branching is by registry verdict per subtype.

## Generation provenance

- Notebook: `research/notebooks/01_trigger_events.ipynb`
- Commit hash: `9df25b9e84b54d9d83259665b0294d4852bc35c4`
- Generation timestamp: 2026-05-09 16:16:27 Pacific Daylight Time
- Working-set games (post-exclusion, post-pickem): 7,360
- Trigger rows emitted: 11,416
- `trigger_events.csv` schema checks passed: 10/10
- `trigger_outcomes.csv` outcome consistency checks passed (see notebook)
- Spot-check sample size: 1,141 rows (10.0%)
- Spot-check discrepancies: 0
