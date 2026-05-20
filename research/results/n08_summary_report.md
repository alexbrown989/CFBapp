# N08 -- Stern-Winston baseline and conformal diagnostic

**Deployment recommendation:** `M3_N06_CONFORMAL` (N06 + conformal interval). Basis: `supported_brier_edge_then_weighted_ece_then_conformal_width`; weighted ECE on `deficit_erased` = **0.0483**.

**Primary finding:** deploy `M3_N06_CONFORMAL`: the N06 calibrated point predictions with split-conformal intervals added as a descriptive uncertainty layer. Locked Decision 4's ordering is decisive: all three variants clear the first criterion by beating at least one baseline, M1 and M3 tie on weighted ECE (**0.0483**) ahead of M2 (**0.0512**), and the conformal-interval criterion breaks the M1/M3 tie in favor of M3.

Important methodological context: no model beats baseline_C on `deficit_erased`. The supported Brier edge is against Stern-Winston baselines, which is large but mostly reflects that Stern-Winston is poorly calibrated for the path-dependent deficit-erasure label: B_SW_CFB ECE is **0.4111** versus model ECE **0.0396**. This is informative about Stern-Winston as an evaluation baseline, but it is not new evidence of comeback-detection edge beyond baseline_C.

For deployment, use N06 calibrated point predictions and display conformal intervals as descriptive uncertainty. The average conformal interval width is **0.9732**, so individual trigger predictions carry substantial uncertainty beyond the point probability. For N09 bet sizing, that width argues against narrow-confidence Kelly assumptions; eighth-Kelly or flat staking should be considered as primary simulation strategies.

N08 compares the locked N06 point model, the locked N07 expanded model, and the N06 conformal layer against baseline_C and two Stern-Winston analytical baselines. M1 and M2 are both trained on `deficit_erased`; `favorite_final_win` is reported only as a cross-label diagnostic.

## Stern-Winston variance

The PFR/NFL reference standard deviation is **13.45**. The empirical CFB 2015-2021 favorite final-margin standard deviation is **15.75** (variance **248.13**, n=2,860 games). CFB variance is approximately **37.2%** higher than the NFL/PFR reference. Headline Stern-Winston comparisons and the exported N09 price-conversion function use the empirical CFB standard deviation.

The `deficit_erased` Stern-Winston comparison is an approximation: the analytical model estimates final favorite win probability, not the path-dependent probability of tying or retaking the lead before game end.

## Comparison Matrix: deficit_erased
| Model | Baseline | Brier improvement | 95% CI | Model ECE | Baseline ECE | Model AUC | Baseline AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| M1_N06 | B_C | -0.0035 | [-0.0072, 0.0000] | 0.0396 | 0.0226 | 0.7646 | 0.7659 |
| M1_N06 | B_SW_PFR | 0.2022 | [0.1817, 0.2228] | 0.0396 | 0.4388 | 0.7646 | 0.7825 |
| M1_N06 | B_SW_CFB | 0.1774 | [0.1580, 0.1967] | 0.0396 | 0.4111 | 0.7646 | 0.7825 |
| M2_N07_EXP | B_C | -0.0026 | [-0.0063, 0.0011] | 0.0415 | 0.0226 | 0.7637 | 0.7659 |
| M2_N07_EXP | B_SW_PFR | 0.2030 | [0.1820, 0.2242] | 0.0415 | 0.4388 | 0.7637 | 0.7825 |
| M2_N07_EXP | B_SW_CFB | 0.1783 | [0.1585, 0.1982] | 0.0415 | 0.4111 | 0.7637 | 0.7825 |
| M3_N06_CONFORMAL | B_C | -0.0035 | [-0.0071, 0.0000] | 0.0396 | 0.0226 | 0.7646 | 0.7659 |
| M3_N06_CONFORMAL | B_SW_PFR | 0.2022 | [0.1815, 0.2228] | 0.0396 | 0.4388 | 0.7646 | 0.7825 |
| M3_N06_CONFORMAL | B_SW_CFB | 0.1774 | [0.1580, 0.1968] | 0.0396 | 0.4111 | 0.7646 | 0.7825 |

## Comparison Matrix: favorite_final_win
| Model | Baseline | Brier improvement | 95% CI | Model ECE | Baseline ECE | Model AUC | Baseline AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| M1_N06 | B_C | -0.0414 | [-0.0517, -0.0310] | 0.1817 | 0.0262 | 0.6712 | 0.6677 |
| M1_N06 | B_SW_PFR | 0.0118 | [-0.0104, 0.0352] | 0.1817 | 0.2313 | 0.6712 | 0.6831 |
| M1_N06 | B_SW_CFB | -0.0008 | [-0.0216, 0.0204] | 0.1817 | 0.2035 | 0.6712 | 0.6831 |
| M2_N07_EXP | B_C | -0.0424 | [-0.0529, -0.0318] | 0.1859 | 0.0262 | 0.6701 | 0.6677 |
| M2_N07_EXP | B_SW_PFR | 0.0107 | [-0.0117, 0.0335] | 0.1859 | 0.2313 | 0.6701 | 0.6831 |
| M2_N07_EXP | B_SW_CFB | -0.0019 | [-0.0227, 0.0197] | 0.1859 | 0.2035 | 0.6701 | 0.6831 |
| M3_N06_CONFORMAL | B_C | -0.0414 | [-0.0517, -0.0309] | 0.1817 | 0.0262 | 0.6712 | 0.6677 |
| M3_N06_CONFORMAL | B_SW_PFR | 0.0118 | [-0.0106, 0.0344] | 0.1817 | 0.2313 | 0.6712 | 0.6831 |
| M3_N06_CONFORMAL | B_SW_CFB | -0.0008 | [-0.0214, 0.0205] | 0.1817 | 0.2035 | 0.6712 | 0.6831 |

## Conformal intervals

N06 split-conformal intervals use validation-slice conformity scores with alpha=0.05. Overall held-out event coverage is **0.9549** and average interval width is **0.9732**.

| Fold | q_hat | Validation coverage | Test coverage | Avg width |
|---:|---:|---:|---:|---:|
| 2022 | 0.7850 | 0.9536 | 0.9699 | 0.9722 |
| 2023 | 0.8235 | 0.9530 | 0.9337 | 0.9739 |
| 2024 | 0.7701 | 0.9582 | 0.9614 | 0.9735 |

## Per-deficit trained-label pattern

| Model | Baseline | Deficit | n | Brier improvement | 95% CI |
|---|---|---:|---:|---:|---:|
| M1_N06 | B_C | 3 | 1450 | -0.0048 | [-0.0077, -0.0020] |
| M1_N06 | B_C | 7 | 1071 | -0.0030 | [-0.0077, 0.0020] |
| M1_N06 | B_C | 10 | 697 | -0.0048 | [-0.0133, 0.0036] |
| M1_N06 | B_C | 14 | 458 | -0.0006 | [-0.0106, 0.0089] |
| M1_N06 | B_C | 21 | 178 | 0.0009 | [-0.0092, 0.0097] |
| M1_N06 | B_SW_PFR | 3 | 1450 | 0.2719 | [0.2518, 0.2911] |
| M1_N06 | B_SW_PFR | 7 | 1071 | 0.2439 | [0.2201, 0.2672] |
| M1_N06 | B_SW_PFR | 10 | 697 | 0.1270 | [0.0984, 0.1554] |
| M1_N06 | B_SW_PFR | 14 | 458 | 0.0728 | [0.0485, 0.0969] |
| M1_N06 | B_SW_PFR | 21 | 178 | 0.0104 | [0.0007, 0.0217] |
| M1_N06 | B_SW_CFB | 3 | 1450 | 0.2456 | [0.2271, 0.2640] |
| M1_N06 | B_SW_CFB | 7 | 1071 | 0.2133 | [0.1913, 0.2354] |
| M1_N06 | B_SW_CFB | 10 | 697 | 0.1024 | [0.0754, 0.1285] |
| M1_N06 | B_SW_CFB | 14 | 458 | 0.0575 | [0.0358, 0.0792] |
| M1_N06 | B_SW_CFB | 21 | 178 | 0.0077 | [-0.0006, 0.0176] |
| M2_N07_EXP | B_C | 3 | 1450 | -0.0039 | [-0.0068, -0.0010] |
| M2_N07_EXP | B_C | 7 | 1071 | -0.0036 | [-0.0086, 0.0012] |
| M2_N07_EXP | B_C | 10 | 697 | -0.0018 | [-0.0108, 0.0071] |
| M2_N07_EXP | B_C | 14 | 458 | 0.0005 | [-0.0094, 0.0103] |
| M2_N07_EXP | B_C | 21 | 178 | 0.0022 | [-0.0072, 0.0113] |
| M2_N07_EXP | B_SW_PFR | 3 | 1450 | 0.2728 | [0.2525, 0.2919] |
| M2_N07_EXP | B_SW_PFR | 7 | 1071 | 0.2432 | [0.2188, 0.2670] |
| M2_N07_EXP | B_SW_PFR | 10 | 697 | 0.1300 | [0.1018, 0.1587] |
| M2_N07_EXP | B_SW_PFR | 14 | 458 | 0.0739 | [0.0493, 0.0985] |
| M2_N07_EXP | B_SW_PFR | 21 | 178 | 0.0116 | [-0.0041, 0.0294] |
| M2_N07_EXP | B_SW_CFB | 3 | 1450 | 0.2465 | [0.2283, 0.2651] |
| M2_N07_EXP | B_SW_CFB | 7 | 1071 | 0.2126 | [0.1896, 0.2350] |
| M2_N07_EXP | B_SW_CFB | 10 | 697 | 0.1054 | [0.0791, 0.1317] |
| M2_N07_EXP | B_SW_CFB | 14 | 458 | 0.0586 | [0.0369, 0.0809] |
| M2_N07_EXP | B_SW_CFB | 21 | 178 | 0.0090 | [-0.0054, 0.0253] |
| M3_N06_CONFORMAL | B_C | 3 | 1450 | -0.0048 | [-0.0077, -0.0019] |
| M3_N06_CONFORMAL | B_C | 7 | 1071 | -0.0030 | [-0.0079, 0.0019] |
| M3_N06_CONFORMAL | B_C | 10 | 697 | -0.0048 | [-0.0133, 0.0035] |
| M3_N06_CONFORMAL | B_C | 14 | 458 | -0.0006 | [-0.0103, 0.0089] |
| M3_N06_CONFORMAL | B_C | 21 | 178 | 0.0009 | [-0.0093, 0.0100] |
| M3_N06_CONFORMAL | B_SW_PFR | 3 | 1450 | 0.2719 | [0.2523, 0.2909] |
| M3_N06_CONFORMAL | B_SW_PFR | 7 | 1071 | 0.2439 | [0.2203, 0.2672] |
| M3_N06_CONFORMAL | B_SW_PFR | 10 | 697 | 0.1270 | [0.0977, 0.1562] |
| M3_N06_CONFORMAL | B_SW_PFR | 14 | 458 | 0.0728 | [0.0487, 0.0975] |
| M3_N06_CONFORMAL | B_SW_PFR | 21 | 178 | 0.0104 | [0.0009, 0.0216] |
| M3_N06_CONFORMAL | B_SW_CFB | 3 | 1450 | 0.2456 | [0.2268, 0.2636] |
| M3_N06_CONFORMAL | B_SW_CFB | 7 | 1071 | 0.2133 | [0.1912, 0.2351] |
| M3_N06_CONFORMAL | B_SW_CFB | 10 | 697 | 0.1024 | [0.0758, 0.1287] |
| M3_N06_CONFORMAL | B_SW_CFB | 14 | 458 | 0.0575 | [0.0357, 0.0791] |
| M3_N06_CONFORMAL | B_SW_CFB | 21 | 178 | 0.0077 | [-0.0007, 0.0177] |

## Verification

- Rebuilt N06 held-out probabilities matched committed artifact with max absolute difference `0`.
- Rebuilt N07 held-out probabilities matched committed artifact with max absolute difference `0`.
- Diagnostic prediction rows: 3,854.
- Price conversion spec: `stern_winston_favorite_win_probability_v1`, using empirical CFB std 15.7523 and pregame-spread coefficient 0.0 in v1.

## Honest interpretation

N08 sharpens the deployment choice without changing the research conclusion. M3 is defensible because it preserves the better-calibrated N06 point predictions and adds uncertainty intervals, not because it discovers new comeback-detection signal. Baseline_C remains unbeaten on the trained label. The Stern-Winston result says more about the limitations of final-margin analytical baselines for path-dependent `deficit_erased` than about historical edge. The conformal layer is useful precisely because the intervals are wide: it makes visible the uncertainty that a single calibrated probability can hide.
