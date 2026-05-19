# N04 model vs pre-game market validation

**Primary finding:** Model produces statistically supported predictive edge over pre-game market consensus. All-fold Brier improvement is **+0.05847** with bootstrap 95% CI **[+0.04211, +0.07465]** under Scheme U; Scheme W2 is effectively identical at **+0.05847** with 95% CI **[+0.04210, +0.07429]**.

This validates the core research-phase claim: once the favorite reaches a trigger state, N03's calibrated trigger-state probability predicts the final favorite outcome more accurately than the pre-game market probability did for this subpopulation.

The win is calibration, not ranking. The pre-game market still ranks teams better overall (market AUC **0.6812** vs model AUC **0.6650**), but it is poorly calibrated for trigger-state rows because it does not know the favorite is now trailing. N03 wins by probability-level adjustment: model ECE **0.03484** vs market ECE **0.24840**.

## Primary metrics

| Scheme | Fold | Rows | Games | Model Brier | Market Brier | Brier improvement | 95% CI | Model ECE | Market ECE | Model AUC | Market AUC |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| U | 2022 | 1261 | 482 | 0.22899 | 0.29390 | 0.06491 | [0.03240, 0.09753] | 0.06942 | 0.24282 | 0.6740 | 0.6293 |
| U | 2023 | 1300 | 481 | 0.22056 | 0.27474 | 0.05418 | [0.02870, 0.07881] | 0.01695 | 0.24785 | 0.6804 | 0.7256 |
| U | 2024 | 1296 | 488 | 0.22749 | 0.28400 | 0.05651 | [0.02950, 0.08290] | 0.03383 | 0.25438 | 0.6508 | 0.6812 |
| W2 | 2022 | 1261 | 482 | 0.22899 | 0.29390 | 0.06491 | [0.03291, 0.09687] | 0.06942 | 0.24282 | 0.6740 | 0.6293 |
| W2 | 2023 | 1300 | 481 | 0.22056 | 0.27474 | 0.05418 | [0.02885, 0.07880] | 0.01695 | 0.24785 | 0.6804 | 0.7256 |
| W2 | 2024 | 1296 | 488 | 0.22749 | 0.28400 | 0.05651 | [0.02934, 0.08287] | 0.03383 | 0.25438 | 0.6508 | 0.6812 |
| U | all | 3857 | 1451 | 0.22565 | 0.28412 | 0.05847 | [0.04211, 0.07465] | 0.03484 | 0.24840 | 0.6650 | 0.6812 |
| W2 | all | 3857 | 1451 | 0.22565 | 0.28412 | 0.05847 | [0.04210, 0.07429] | 0.03484 | 0.24840 | 0.6650 | 0.6812 |

## Deficit pattern

The deficit pattern is the strongest mechanistic validation. Brier improvement increases monotonically from D=3 to D=21: **-0.00570**, **+0.03134**, **+0.09147**, **+0.16507**, **+0.34131**. That is the expected signature of a useful in-game-state model: the deeper the favorite's deficit, the more pre-game market probability overstates comeback probability.

| Scheme | Fold | Deficit | Rows | Games | Brier improvement | Model ECE | Market ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| U | all | 3 | 1451 | 1451 | -0.00570 | 0.03626 | 0.13456 |
| U | all | 7 | 1072 | 1072 | 0.03134 | 0.04218 | 0.21884 |
| U | all | 10 | 698 | 698 | 0.09147 | 0.08715 | 0.33890 |
| U | all | 14 | 458 | 458 | 0.16507 | 0.06094 | 0.41725 |
| U | all | 21 | 178 | 178 | 0.34131 | 0.01410 | 0.57679 |
| W2 | all | 3 | 1451 | 1451 | -0.00570 | 0.03626 | 0.13456 |
| W2 | all | 7 | 1072 | 1072 | 0.03134 | 0.04218 | 0.21884 |
| W2 | all | 10 | 698 | 698 | 0.09147 | 0.08715 | 0.33890 |
| W2 | all | 14 | 458 | 458 | 0.16507 | 0.06094 | 0.41725 |
| W2 | all | 21 | 178 | 178 | 0.34131 | 0.01410 | 0.57679 |

## Important caveats

This is predictive validation, not a betting-edge demonstration. It shows that trigger-state probabilities beat pre-game market probabilities for historical trigger events.

The tertiary favorite-side betting simulation lost money at the primary deployment-context setting: threshold **+0.08**, **25% Kelly**, no D=21 rows produced **89** bets, **35.96%** win rate, and **-33.27% ROI**. That result is consistent with the primary finding, not contradictory. A probability advantage over stale pre-game market probability does not imply a profitable edge over correctly priced live in-game markets.

Historical live in-game line data is unavailable for the 2022-2024 corpus, so live-line edge remains untested. Project conclusion: the methodology works for predictive probability adjustment; deployment-context profitability requires future live-line data collection.

## Market data provenance

- Test unique games: 1451
- Test trigger events per scheme: 3857
- Missing market probability rows: 0
- Spread-conversion fallback, no moneyline: 14 unique games
- Spread-conversion fallback, moneyline side conflict: 38 unique games
- CFBD `/lines` records include game `startDate`, but no provider-level line timestamp; cached fields are treated as latest available pre-game lines.

## Spread conversion

`logit(p_favorite_win) = -0.120507 * favorite_spread + -0.069094`. Training Brier 0.15396, ECE 0.01024, AUC 0.7478.

## Tertiary deployment-context snapshot

| Scheme | Threshold | Sizing | Bets | Win rate | Mean edge | ROI | Total PnL | Max drawdown |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| U | 0.08 | kelly_25 | 89 | 0.35955 | 0.12911 | -0.33269 | -29.60898 | 29.46307 |
| W2 | 0.08 | kelly_25 | 89 | 0.35955 | 0.12911 | -0.33269 | -29.60898 | 29.46307 |

Tertiary betting rows are deployment context only. They do not override the primary Brier-improvement validation gate.

## Interpretation

N04 validates predictive edge versus pre-game market consensus. The model wins because it corrects pre-game probability for observed trigger-state information, especially at deeper deficits. It does not prove live betting profitability; the live-line question remains open until going-forward live market data can be collected.
