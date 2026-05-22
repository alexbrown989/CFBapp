# N10 Attribution Audit

Generated: 2026-05-21

## Scope

This is a local-only audit requested after the first N10 execution halted on
`unknown_fluke` rows. No API calls were made. No N10 deliverables were written.

The question: can N10 classify fluky vs sustained dog leads from existing raw
count artifacts rather than recomputing scoring attribution from play-by-play?

## Task 1: Raw Count Availability

Required raw counts:

- `dog_points_from_turnovers`
- `dog_points_from_returns`
- `dog_points_from_explosives`
- `dog_score_at_trigger`

Availability in committed artifacts:

| Artifact | Raw count availability |
|---|---|
| `trigger_events.csv` | Has `dog_score_at_trigger`; does not have dog turnover/return/explosive point counts. |
| `n07_descriptive_features.parquet` | Has percentage features only: `dog_points_from_turnovers_pct`, `dog_points_from_returns_pct`, `dog_points_from_explosives_pct`; not raw counts. These inherit N07 modeling null semantics and caused the N10 halt. |
| `n09_trigger_state_stratifications.parquet` | Has dashboard buckets, not raw counts. |
| `n03_calibrated_predictions.parquet` | Has raw count columns, but only for held-out prediction rows, duplicated by scheme, not the full 2015-2024 descriptive corpus. |
| `n06_calibrated_predictions.parquet` | Same as N03: raw count columns exist, but only for held-out prediction rows. |
| `_investigate_02c_d12_accounting.csv` | Has 02c raw counts for 2,346 diagnostic rows, keyed by `game_id` + `fav_deficit` but not `trigger_sequence`; not a canonical full-corpus per-trigger artifact. |

Conclusion: there is no committed full-corpus per-trigger artifact with all
required raw count columns. The raw counts exist in Phase 0/N03/N06 extraction
logic and partial held-out prediction artifacts, but not in a directly usable
11,412-row N10 source table.

## Task 2: Missingness Pattern At Raw-Count Level

The initial N10 halt came from trying to classify with N07 percentage columns.
After assigning the 5 known N07 explosive-percentage edge rows to fluky status,
the remaining currently unclassified set is:

| Quantity | Count |
|---|---:|
| N07 percentage-based unknown rows before known edge handling | 2,979 |
| Known `dog_points_from_explosives_pct > 1` edge rows assigned to fluky | 5 |
| Remaining currently-unknown fluke rows | 2,974 |
| Currently-unknown rows with `dog_score_at_trigger == 0` | 0 |

Distribution of the 2,974 currently-unknown rows:

| Dimension | Breakdown |
|---|---|
| Time bucket | Q1: 2,966; Q2-first-half: 5; Q3: 1; Q4: 2 |
| Deficit | D=3: 1,816; D=7: 1,138; D=10: 13; D=14: 6; D=21: 1 |
| Season | 2015: 249; 2016: 281; 2017: 281; 2018: 268; 2019: 327; 2020: 241; 2021: 317; 2022: 325; 2023: 346; 2024: 339 |

Held-out raw-count artifact coverage check using Scheme U rows from
`n06_calibrated_predictions.parquet`:

| Check | Count |
|---|---:|
| Currently-unknown rows with a held-out N06 raw-count row | 1,010 |
| Currently-unknown rows not covered by N06 held-out artifact | 1,964 |
| N06-covered unknown rows with all four raw point columns non-null | 0 |
| N06-covered unknown rows with one or more raw point columns missing | 1,010 |

The diagnostic `_investigate_02c_d12_accounting.csv` matches 486 of the
2,974 unknown rows by `game_id` + `fav_deficit`, but it lacks
`trigger_sequence`, covers only a diagnostic subset, and is not safe as the
canonical N10 classification source.

## Task 3: Attribution vs Availability

For the 2,974 currently-unknown rows:

| Category | Count | Interpretation |
|---|---:|---|
| Raw counts available for all three fluke components and cleanly classifiable | 0 | No full direct raw-count set exists in committed artifacts for these rows. |
| Raw counts available but components fail to sum cleanly to dog score | 0 assessable | Cannot assess because no currently-unknown row has all needed direct raw counts available. |
| One or more raw counts genuinely missing in the held-out raw-count artifact | 1,010 | N03/N06 carry the same early-game null semantics; these are not solved by switching to held-out prediction artifacts. |
| No held-out raw-count artifact row because row is from 2015-2021 | 1,964 | Direct artifact unavailable for training-era rows. |
| `dog_score_at_trigger == 0` intentional `no_dog_points` case | 0 | The unknown problem is not no-score triggers. |

The missingness is overwhelmingly early-game. This matches the Phase 0/N07 null
policy: dog scoring decomposition features were often null when there was no
completed dog drive before the trigger. For N10, that null policy is too
conservative because the direct conditional question needs to classify the
observed score state itself, including early one-score fluky leads.

## Methodological Implication

The N10 fix cannot be a simple column-source swap to existing full-corpus raw
count artifacts, because such an artifact does not exist.

Any N10 implementation must choose a new attribution policy for early-game
dog points:

- Option A, lenient: reconstruct known fluke components and assume unaccounted
  points are sustained.
- Option B, strict: classify as fluky only when explicitly attributed fluke
  components meet the threshold without relying on unaccounted points.
- Option C, hybrid: classify clearly fluky and clearly sustained rows, and put
  unresolved attribution cases in an `attribution_unclear` bucket excluded from
  the headline hypothesis test.

The audit does not implement any of these options. It only shows that direct
raw-count artifacts are insufficient for the original N10 classification plan.

## Recommendation For Review

Option C is the safest methodology if the project wants to avoid overstating
fluky-lead evidence from early-game rows. Option A is more generous to the
fluky hypothesis but risks silently treating unknown point sources as sustained.
Option B is conservative but may shrink the fluky bucket enough to underpower
the direct hypothesis test.

Awaiting explicit selection before changing `_build_n10.py`.
