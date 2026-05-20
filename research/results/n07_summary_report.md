# N07 feature pool expansion test

**Primary finding:** N07 found a real but limited missing signal source:
possession-adjusted deficit pressure. 2 of 14 pre-registered features passed
all three inclusion gates: `deficit_per_remaining_possession` and
`clock_pressure_index`. Both are Category A possession-adjusted features.
However, the expanded 33-feature model still does **not** beat the strict
`fav_deficit x time_bucket` baseline_C on the comeback-erasure target.
Expanded Scheme U Brier improvement versus baseline_C on `deficit_erased` is
**-0.00263** with 95% CI **[-0.00631, +0.00110]**.

This is the natural endpoint for the project's historical-data methodology.
Phase 0 built and stability-tested the original feature pool. N03 produced a
calibrated final-win model with modest discrimination. N04 showed that the
model beats stale pre-game market probabilities, validating that current game
state matters. N05 and N06 then showed that the model does not beat a simple
deficit x time lookup table on either final-win or deficit-erased labels. N07
tested the most plausible missing categories and found that possession
pressure helps at the feature level, while fluke-score and efficiency-gap
hypotheses do not clear the strict baseline_C gate. Historical data has now
been pushed about as far as this framework can push it; the next validation
question requires live market comparison.

The 33-feature expanded model (30 original Phase 0 features + the 2 N07
passes + protected `fav_deficit`) is the recommended production candidate for
an N08 live-data scaffold. It is not edge-grade on historical baseline_C
validation, but it is the best historically tested model specification and it
includes the one new signal source N07 surfaced.

## Pre-registered feature verdicts

| Feature | Category | Indicators | Verdict | Best label | R6 folds | Mean dBrier vs alpha | baseline_C improvement | Corrected lower | Fail reason |
|---|---|---|---|---|---:|---:|---:|---:|---|
| `estimated_possessions_remaining` | A | none | FAIL | `favorite_final_win` | 3 | +0.01528 | +0.00781 | -0.00063 | baseline_C_bonferroni |
| `deficit_per_remaining_possession` | A | none | PASS | `favorite_final_win` | 3 | +0.02255 | +0.01511 | +0.00651 |  |
| `possessions_needed_to_tie` | A | none | FAIL | `favorite_final_win` | 3 | +0.01448 | +0.00709 | -0.00178 | baseline_C_bonferroni |
| `clock_pressure_index` | A | none | PASS | `favorite_final_win` | 3 | +0.02294 | +0.01551 | +0.00715 |  |
| `dog_points_from_turnovers_pct` | B | dog_points_from_turnovers_pct_is_null | FAIL | `favorite_final_win` | 1 | -0.00048 | -0.00790 | -0.01984 | baseline_C_bonferroni |
| `dog_points_from_returns_pct` | B | dog_points_from_returns_pct_is_null | FAIL | `favorite_final_win` | 1 | -0.00090 | -0.00833 | -0.02065 | r6_stability, magnitude, baseline_C_bonferroni |
| `dog_points_from_explosives_pct` | B | dog_points_from_explosives_pct_is_null | FAIL | `favorite_final_win` | 3 | +0.00739 | -0.00004 | -0.00984 | baseline_C_bonferroni |
| `dog_offensive_points_pct` | B | dog_offensive_points_pct_is_null | FAIL | `favorite_final_win` | 1 | -0.00206 | -0.00947 | -0.02176 | r6_stability, magnitude, baseline_C_bonferroni |
| `fav_yards_per_point_ratio` | B | fav_yards_per_point_ratio_is_null | FAIL | `favorite_final_win` | 3 | +0.00437 | -0.00308 | -0.01313 | baseline_C_bonferroni |
| `epa_per_play_gap` | C | epa_per_play_gap_is_null | FAIL | `favorite_final_win` | 3 | +0.00521 | -0.00219 | -0.01288 | baseline_C_bonferroni |
| `success_rate_gap` | C | success_rate_gap_is_null | FAIL | `favorite_final_win` | 3 | +0.00637 | -0.00105 | -0.01076 | baseline_C_bonferroni |
| `third_down_gap` | C | third_down_gap_is_null | FAIL | `favorite_final_win` | 3 | +0.00696 | -0.00049 | -0.01015 | baseline_C_bonferroni |
| `explosive_rate_gap` | C | explosive_rate_gap_is_null | FAIL | `favorite_final_win` | 3 | +0.00354 | -0.00387 | -0.01437 | baseline_C_bonferroni |
| `drive_yards_gap` | C | drive_yards_gap_is_null | FAIL | `favorite_final_win` | 3 | +0.00894 | +0.00152 | -0.00808 | baseline_C_bonferroni |

## Category summary

| Category | Candidates | Pass | Bonferroni alpha | Passing features |
|---|---:|---:|---:|---|
| A | 4 | 2 | 0.0125 | deficit_per_remaining_possession, clock_pressure_index |
| B | 5 | 0 | 0.0100 | none |
| C | 5 | 0 | 0.0100 | none |

## Expanded model

| Model | N | Model Brier | baseline_C Brier | Improvement | AUC model | AUC baseline_C |
|---|---:|---:|---:|---:|---:|---:|
| N07 expanded U | 3854 | 0.17771 | 0.17508 | -0.00263 | 0.7637 | 0.7659 |
| N06 reference | 3854 | 0.17861 | 0.17508 | -0.00352 | 0.7646 | 0.7659 |

## Data provenance

- Descriptive feature rows: 11,412.
- Empirical possessions/minute from cached drives: 0.420; locked value used for `estimated_possessions_remaining`: 0.450.
- Possession remaining uses `seconds_remaining_in_regulation`, equivalent to the corrected `(quarter - 1) * 900 + period_elapsed` clock calculation.
- `dog_offensive_points_pct` treats `dog_points_from_returns` as non-offensive points; turnover-created offensive scores remain offensive points and are separately represented by `dog_points_from_turnovers_pct`.

## Honest interpretation

N07 is a mixed but clarifying result. The possession-adjusted hypothesis is
supported: the model was missing structural pressure from deficit relative to
remaining possessions. The fluke-score hypothesis, which was one of the
project's original mechanistic ideas, is not supported under this strict test:
0 of 5 Category B features passed. Efficiency-gap differentials also failed
to beat baseline_C.

The expanded model is marginally better than N06 on Brier
(-0.00263 versus -0.00352 improvement against baseline_C), but the confidence
interval still crosses zero and the AUC remains slightly below baseline_C
(0.7637 versus 0.7659). This is not an edge-grade historical result. It is,
however, enough to justify carrying the 33-feature expanded model forward as
the live-data deployment candidate, where the relevant comparison becomes
actual live market prices rather than a historical deficit x time baseline.
