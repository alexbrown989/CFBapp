# N06 deficit-erased model validation

**Primary finding:** Fitting directly on `deficit_erased` dramatically repaired the N03 label-calibration problem, but it did **not** produce edge over the deficit x time-bucket baseline_C. N03 under-predicted `deficit_erased` by roughly +15-30 percentage points across the middle probability deciles; N06 reduces that to a weighted mean absolute decile gap of **0.040** with max gap **0.106**. Even so, Scheme U Brier improvement (`baseline_C - model`) is **-0.00352** with 95% cluster-bootstrap CI **[-0.00724, +0.00013]**: calibrated, but flat against baseline.

**Mechanistic interpretation:** The model AUC (**0.7646**) is essentially tied with baseline_C AUC (**0.7659**). The 30 engineered features plus protected `fav_deficit` add no ranking improvement over a 20-cell deficit x time lookup table. Whatever signal the engineered features carry is being absorbed by their correlation with deficit and time; they do not carry independent comeback-erasure signal beyond what the structural variables encode.

**Cross-label confirmation:** N06 on `favorite_final_win` is materially worse than baseline_C: improvement **-0.04137** with CI **[-0.05153, -0.03086]**. That confirms the experiment was a clean A/B test: each model performs best on its trained label and badly on the other, and neither beats baseline_C on its own label.

**Per-deficit pattern:** N06 shows no supported positive per-deficit edge against baseline_C. D=3 is significantly worse, D=7/D=10/D=14 are near-zero with CIs crossing zero, and D=21 is a tiny positive estimate with a CI crossing zero. N04's monotonic per-deficit improvement against pre-game market does not replicate against baseline_C, confirming N05's interpretation that N04's pattern was about market staleness at deeper deficits, not model deep-deficit insight.

**Project conclusion:** The validated feature pool is exhausted relative to baseline_C for both labels. Future research requires either feature expansion (possession-adjusted deficit, trajectory features, fluke-score decomposition) or a different validation target, especially live market comparison once data is available.

**Methodology integrity:** N06 changes one variable from N03: the training label is `deficit_erased` instead of `favorite_final_win`. The 30 R6-validated features, protected `fav_deficit` structural variable, play-level deduplication, null handling, L1 model class, walk-forward windows, and isotonic calibration structure remain aligned with N03.

## Primary validation versus baseline_C

| Scheme | Label | N | Model Brier | Baseline C Brier | Improvement | 95% CI | Model ECE | Baseline ECE | Model AUC | Baseline AUC | Classification |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| U | `deficit_erased` | 3854 | 0.17861 | 0.17508 | -0.00352 | [-0.00724, +0.00013] | 0.03959 | 0.02256 | 0.7646 | 0.7659 | no |
| U | `favorite_final_win` | 3854 | 0.26412 | 0.22275 | -0.04137 | [-0.05153, -0.03086] | 0.18168 | 0.02623 | 0.6712 | 0.6677 | no |
| W2 | `deficit_erased` | 3854 | 0.17861 | 0.17508 | -0.00352 | [-0.00724, +0.00013] | 0.03959 | 0.02256 | 0.7646 | 0.7659 | no |
| W2 | `favorite_final_win` | 3854 | 0.26412 | 0.22275 | -0.04137 | [-0.05153, -0.03086] | 0.18168 | 0.02623 | 0.6712 | 0.6677 | no |

## Per-fold target-label metrics

| Scheme | Fold | N | Model Brier | Baseline C Brier | Improvement | 95% CI | Model ECE | Model AUC |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| U | 2022 | 1261 | 0.16963 | 0.16688 | -0.00275 | [-0.00999, +0.00447] | 0.06920 | 0.7878 |
| U | 2023 | 1297 | 0.18643 | 0.18361 | -0.00282 | [-0.00875, +0.00295] | 0.03870 | 0.7532 |
| U | 2024 | 1296 | 0.17951 | 0.17453 | -0.00498 | [-0.01077, +0.00080] | 0.04274 | 0.7603 |
| W2 | 2022 | 1261 | 0.16963 | 0.16688 | -0.00275 | [-0.00999, +0.00447] | 0.06920 | 0.7878 |
| W2 | 2023 | 1297 | 0.18643 | 0.18361 | -0.00282 | [-0.00875, +0.00295] | 0.03870 | 0.7532 |
| W2 | 2024 | 1296 | 0.17951 | 0.17453 | -0.00498 | [-0.01077, +0.00080] | 0.04274 | 0.7603 |

## Per-deficit target-label pattern

| Deficit | N | Actual rate | Mean model prob | Mean baseline C | Model Brier | Baseline C Brier | Improvement | 95% CI |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D=3 | 1450 | 83.4% | 80.6% | 81.2% | 0.14497 | 0.14018 | -0.00479 | [-0.00767, -0.00188] |
| D=7 | 1071 | 72.7% | 67.8% | 70.9% | 0.18790 | 0.18494 | -0.00296 | [-0.00781, +0.00182] |
| D=10 | 697 | 47.6% | 48.4% | 46.2% | 0.24262 | 0.23780 | -0.00482 | [-0.01325, +0.00375] |
| D=14 | 458 | 32.3% | 30.3% | 33.6% | 0.20581 | 0.20523 | -0.00058 | [-0.01029, +0.00909] |
| D=21 | 178 | 9.0% | 6.7% | 13.3% | 0.07601 | 0.07693 | +0.00092 | [-0.00910, +0.00970] |

## Threshold analysis

| X | N | Actual rate | Mean model prob | Mean baseline C | Actual - model | Actual - baseline C |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 1788 | 68.0% | 70.4% | 64.5% | -0.02352 | +0.03510 |
| 0.03 | 1124 | 62.4% | 67.8% | 59.2% | -0.05444 | +0.03155 |
| 0.05 | 756 | 60.1% | 65.5% | 54.8% | -0.05416 | +0.05256 |
| 0.08 | 414 | 56.3% | 61.5% | 47.5% | -0.05201 | +0.08815 |
| 0.10 | 314 | 53.5% | 60.1% | 44.5% | -0.06597 | +0.09032 |

## Quintiles and calibration deciles

Spearman(`calibrated_prob`, `deficit_erased`) = **0.4390**.

| Quintile | N | Prob range | Mean model prob | Mean baseline C | Actual rate | Actual - model | Actual - baseline C |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 853 | [0.00000, 0.39640] | 22.7% | 27.9% | 27.7% | +0.04938 | -0.00282 |
| 2 | 730 | [0.40000, 0.67568] | 52.9% | 55.1% | 58.5% | +0.05636 | +0.03414 |
| 3 | 996 | [0.67605, 0.77005] | 72.5% | 74.2% | 74.9% | +0.02418 | +0.00738 |
| 4 | 645 | [0.77011, 0.85057] | 81.2% | 81.7% | 85.1% | +0.03896 | +0.03407 |
| 5 | 630 | [0.85193, 1.00000] | 88.5% | 84.2% | 83.7% | -0.04867 | -0.00526 |

| Decile | N | Mean model prob | Mean baseline C | Actual rate | Actual - model | Actual - baseline C |
|---|---:|---:|---:|---:|---:|---:|
| 0 (0.0-0.1) | 181 | 3.9% | 10.2% | 6.6% | +0.02767 | -0.03576 |
| 1 (0.1-0.2) | 137 | 16.6% | 17.9% | 18.2% | +0.01681 | +0.00346 |
| 2 (0.2-0.3) | 312 | 25.0% | 33.4% | 35.6% | +0.10597 | +0.02166 |
| 3 (0.3-0.4) | 223 | 38.7% | 40.9% | 39.5% | +0.00783 | -0.01421 |
| 4 (0.4-0.5) | 135 | 40.2% | 45.9% | 42.2% | +0.01997 | -0.03726 |
| 5 (0.5-0.6) | 402 | 52.1% | 55.0% | 58.2% | +0.06151 | +0.03246 |
| 6 (0.6-0.7) | 676 | 67.0% | 66.6% | 69.2% | +0.02260 | +0.02585 |
| 7 (0.7-0.8) | 851 | 77.2% | 79.9% | 82.1% | +0.04980 | +0.02286 |
| 8 (0.8-0.9) | 751 | 86.0% | 83.7% | 84.7% | -0.01271 | +0.01034 |
| 9 (0.9-1.0) | 186 | 91.8% | 84.3% | 83.3% | -0.08515 | -0.00959 |

## Cross-label comparison

N06 is trained on `deficit_erased`. This table checks how the same probabilities behave against `favorite_final_win` and keeps the label distinction explicit.

| Model/use | Label | Brier improvement vs baseline_C | 95% CI | Model Brier | Baseline C Brier |
|---|---|---:|---|---:|---:|
| N06 Scheme U | `favorite_final_win` | -0.04137 | [-0.05153, -0.03086] | 0.26412 | 0.22275 |
| N03 reference | `favorite_final_win` | -0.00303 | [-0.00677, +0.00051] | 0.22565 | 0.22262 |
| N06 Scheme U | `deficit_erased` | -0.00352 | [-0.00724, +0.00013] | 0.17861 | 0.17508 |
| N03 reference | `deficit_erased` | -0.06123 | [-0.07244, -0.05029] | 0.23632 | 0.17508 |

## Feature selection and sensitivity

- U: selected 30 R6 features + 1 protected structural feature; indicators=19; dropped=none.
- W2: selected 30 R6 features + 1 protected structural feature; indicators=19; dropped=none.

| C | U weighted Brier | U weighted ECE | U weighted AUC | Nonzero feature union |
|---:|---:|---:|---:|---:|
| 0.1 | 0.17161 | 0.05608 | 0.7920 | 25 |
| 0.5 | 0.17191 | 0.05488 | 0.7902 | 31 |
| 1.0 | 0.17212 | 0.05151 | 0.7878 | 31 |
| 2.0 | 0.17237 | 0.05675 | 0.7886 | 31 |
| 10.0 | 0.17259 | 0.05797 | 0.7882 | 31 |

## Data and outputs

- N05 non-null `deficit_erased` event rows used: 11,412; excluded null rows: 4.
- Play-level model rows after deduplication: 7,852.
- Held-out prediction rows per scheme: 3,854; main parquet rows across U/W2: 7,708.
- Scheme E 2024 validation rows: 1,296.

## Honest interpretation

N06 appears to fix the label-calibration problem but still does not add Brier value beyond baseline_C. That would mean the right label matters for probability scale, but the current feature pool is mostly exhausted relative to the deficit/time baseline.
