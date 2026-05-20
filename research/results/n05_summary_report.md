# N05 favorite final-win and deficit-erased rate analysis

**Primary finding:** The model does **not** improve over a simple `fav_deficit x time_bucket` baseline on either label. For `favorite_final_win`, Brier improvement (`baseline_C - model`) is **-0.00303** with 95% cluster-bootstrap CI **[-0.00677, +0.00051]**, not distinguishable from zero. For `deficit_erased`, improvement is **-0.06123** with CI **[-0.07244, -0.05029]**, materially worse than baseline.

**Interpretation:** N03's calibrated probabilities largely encode deficit x time information rather than adding comeback-detection signal beyond what a naive lookup table provides. N04's positive Brier improvement against pre-game market probability is real, but mechanism-restricted: the model beats pre-game markets because pre-game markets do not condition on current game state, not because the model has discovered comeback patterns that a deficit/time baseline misses.

**Secondary finding:** Favorites erase deficits much more often than they win games after a trigger: `deficit_erased` rate **63.5%** versus `favorite_final_win` rate **43.3%**, a roughly **20.2 percentage-point** gap. The "favorite came back but lost" subpopulation is substantial and should be treated as its own future research object.

**Tertiary finding:** The model is systematically under-calibrated for the `deficit_erased` label. Across the middle probability deciles, actual deficit-erased rates exceed model probability by roughly 15-30 percentage points, consistent with the model being trained on `favorite_final_win` (43.3% base rate) rather than `deficit_erased` (63.5% base rate).

N05 distinguishes two labels throughout: `favorite_final_win` is the N03 target, while `deficit_erased` means the favorite tied or retook the lead after the trigger. The model was trained on `favorite_final_win`, not on `deficit_erased`.

## Q2 model versus deficit-by-time baseline

Baseline C is the training-years-only rate by `fav_deficit x time_bucket` using seasons 2015-2021. Positive Brier improvement means the model beat that baseline on held-out 2022-2024 Scheme U rows.

| Label | Rows | Model Brier | Baseline C Brier | Improvement | 95% bootstrap CI | Spearman(model, actual) | Classification |
|---|---:|---:|---:|---:|---|---:|---|
| `favorite_final_win` | 3857 | 0.22565 | 0.22262 | -0.00303 | [-0.00677, 0.00051] | 0.2839 | no |
| `deficit_erased` | 3854 | 0.23632 | 0.17508 | -0.06123 | [-0.07244, -0.05029] | 0.4212 | no |

### Threshold analysis

Label: `favorite_final_win`

| Threshold X | N | Actual rate | Mean model prob | Mean baseline C | Actual - model | Actual - baseline C | Wilson 95% CI | Bootstrap 95% CI |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.00 | 1666 | 0.43277 | 0.46266 | 0.40755 | -0.02989 | 0.02523 | [0.40916, 0.45669] | [0.39768, 0.47062] |
| 0.03 | 960 | 0.41458 | 0.46128 | 0.37716 | -0.04669 | 0.03742 | [0.38382, 0.44603] | [0.37018, 0.45994] |
| 0.05 | 687 | 0.39010 | 0.45426 | 0.35094 | -0.06416 | 0.03916 | [0.35434, 0.42709] | [0.34058, 0.44148] |
| 0.08 | 425 | 0.39059 | 0.46668 | 0.33798 | -0.07609 | 0.05261 | [0.34538, 0.43775] | [0.32791, 0.45455] |
| 0.10 | 265 | 0.40377 | 0.48360 | 0.33252 | -0.07982 | 0.07125 | [0.34648, 0.46382] | [0.32950, 0.48000] |

Label: `deficit_erased`

| Threshold X | N | Actual rate | Mean model prob | Mean baseline C | Actual - model | Actual - baseline C | Wilson 95% CI | Bootstrap 95% CI |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.00 | 190 | 0.27368 | 0.26275 | 0.21171 | 0.01094 | 0.06198 | [0.21524, 0.34109] | [0.19898, 0.35052] |
| 0.03 | 116 | 0.22414 | 0.26496 | 0.18743 | -0.04082 | 0.03670 | [0.15780, 0.30816] | [0.14167, 0.31579] |
| 0.05 | 70 | 0.32857 | 0.33370 | 0.23447 | -0.00513 | 0.09411 | [0.22999, 0.44499] | [0.21053, 0.45946] |
| 0.08 | 35 | 0.28571 | 0.31477 | 0.18261 | -0.02905 | 0.10311 | [0.16327, 0.45055] | [0.14286, 0.45455] |
| 0.10 | 18 | 0.27778 | 0.38392 | 0.21943 | -0.10614 | 0.05834 | [0.12500, 0.50873] | [0.09524, 0.50000] |

### Quintile analysis

Label: `favorite_final_win`; Spearman(model_prob, actual) = 0.2839

| Quintile | N | Prob range | Mean model prob | Mean baseline C | Actual rate | Actual - model | Actual - baseline C |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 829 | [0.00000, 0.23000] | 0.13865 | 0.18340 | 0.19662 | 0.05797 | 0.01323 |
| 2 | 833 | [0.23929, 0.42029] | 0.35065 | 0.38113 | 0.40936 | 0.05872 | 0.02823 |
| 3 | 906 | [0.42373, 0.52427] | 0.47468 | 0.49191 | 0.47020 | -0.00448 | -0.02172 |
| 4 | 761 | [0.52948, 0.58921] | 0.56831 | 0.57773 | 0.58081 | 0.01250 | 0.00308 |
| 5 | 528 | [0.59200, 1.00000] | 0.63918 | 0.59571 | 0.59280 | -0.04638 | -0.00291 |

Label: `deficit_erased`; Spearman(model_prob, actual) = 0.4212

| Quintile | N | Prob range | Mean model prob | Mean baseline C | Actual rate | Actual - model | Actual - baseline C |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 827 | [0.00000, 0.23000] | 0.13843 | 0.28252 | 0.28658 | 0.14814 | 0.00406 |
| 2 | 832 | [0.23929, 0.42029] | 0.35059 | 0.57730 | 0.59976 | 0.24917 | 0.02246 |
| 3 | 906 | [0.42373, 0.52427] | 0.47468 | 0.72437 | 0.73179 | 0.25711 | 0.00741 |
| 4 | 761 | [0.52948, 0.58921] | 0.56831 | 0.82096 | 0.85151 | 0.28320 | 0.03055 |
| 5 | 528 | [0.59200, 1.00000] | 0.63918 | 0.83614 | 0.82955 | 0.19036 | -0.00660 |

### Calibration deciles

Label: `favorite_final_win`

| Decile | N | Mean model prob | Actual rate | Calibration gap | Bootstrap 95% CI |
|---:|---:|---:|---:|---:|---|
| 0 (0.0-0.1) | 293 | 0.04953 | 0.08874 | 0.03921 | [0.05694, 0.12414] |
| 1 (0.1-0.2) | 200 | 0.13956 | 0.21000 | 0.07044 | [0.14925, 0.27228] |
| 2 (0.2-0.3) | 636 | 0.24587 | 0.32233 | 0.07646 | [0.28076, 0.36438] |
| 3 (0.3-0.4) | 201 | 0.36162 | 0.45274 | 0.09111 | [0.38308, 0.52261] |
| 4 (0.4-0.5) | 776 | 0.41933 | 0.43686 | 0.01753 | [0.39613, 0.47821] |
| 5 (0.5-0.6) | 1368 | 0.55449 | 0.54971 | -0.00478 | [0.51833, 0.58086] |
| 6 (0.6-0.7) | 341 | 0.64105 | 0.58944 | -0.05161 | [0.53529, 0.64223] |
| 7 (0.7-0.8) | 24 | 0.72467 | 0.66667 | -0.05800 | [0.45833, 0.83333] |
| 8 (0.8-0.9) | 15 | 0.84618 | 0.66667 | -0.17952 | [0.40000, 0.87500] |
| 9 (0.9-1.0) | 3 | 0.98830 | 1.00000 | 0.01170 | [1.00000, 1.00000] |

Label: `deficit_erased`

| Decile | N | Mean model prob | Actual rate | Calibration gap | Bootstrap 95% CI |
|---:|---:|---:|---:|---:|---|
| 0 (0.0-0.1) | 293 | 0.04953 | 0.13652 | 0.08699 | [0.09622, 0.17895] |
| 1 (0.1-0.2) | 200 | 0.13956 | 0.29000 | 0.15044 | [0.22280, 0.35885] |
| 2 (0.2-0.3) | 634 | 0.24592 | 0.47319 | 0.22727 | [0.42835, 0.51852] |
| 3 (0.3-0.4) | 201 | 0.36162 | 0.65672 | 0.29509 | [0.59000, 0.72277] |
| 4 (0.4-0.5) | 775 | 0.41935 | 0.65290 | 0.23355 | [0.61248, 0.69311] |
| 5 (0.5-0.6) | 1368 | 0.55449 | 0.82310 | 0.26861 | [0.79838, 0.84794] |
| 6 (0.6-0.7) | 341 | 0.64105 | 0.83871 | 0.19766 | [0.79706, 0.87834] |
| 7 (0.7-0.8) | 24 | 0.72467 | 0.87500 | 0.15033 | [0.74896, 1.00000] |
| 8 (0.8-0.9) | 15 | 0.84618 | 0.86667 | 0.02048 | [0.66667, 1.00000] |
| 9 (0.9-1.0) | 3 | 0.98830 | 1.00000 | 0.01170 | [1.00000, 1.00000] |

## Per-deficit model pattern

Label: `favorite_final_win`

| Deficit | N | Actual rate | Mean model prob | Mean baseline C | Model Brier | Baseline C Brier | Improvement | 95% CI |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D=3 | 1451 | 0.56857 | 0.55571 | 0.57301 | 0.24721 | 0.24560 | -0.00161 | [-0.00532, 0.00207] |
| D=7 | 1072 | 0.47201 | 0.44869 | 0.47458 | 0.24308 | 0.24169 | -0.00140 | [-0.00667, 0.00382] |
| D=10 | 698 | 0.33381 | 0.31248 | 0.30420 | 0.22639 | 0.21642 | -0.00997 | [-0.01626, -0.00384] |
| D=14 | 458 | 0.24017 | 0.20334 | 0.21836 | 0.18188 | 0.17785 | -0.00403 | [-0.01112, 0.00308] |
| D=21 | 178 | 0.06180 | 0.04769 | 0.09463 | 0.05459 | 0.05999 | 0.00540 | [-0.00265, 0.01269] |

Label: `deficit_erased`

| Deficit | N | Actual rate | Mean model prob | Mean baseline C | Model Brier | Baseline C Brier | Improvement | 95% CI |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D=3 | 1450 | 0.83448 | 0.55582 | 0.81172 | 0.22091 | 0.14018 | -0.08073 | [-0.09124, -0.07026] |
| D=7 | 1071 | 0.72736 | 0.44890 | 0.70933 | 0.26472 | 0.18494 | -0.07978 | [-0.09428, -0.06531] |
| D=10 | 697 | 0.47633 | 0.31260 | 0.46193 | 0.27299 | 0.23780 | -0.03519 | [-0.04952, -0.02112] |
| D=14 | 458 | 0.32314 | 0.20334 | 0.33639 | 0.22411 | 0.20523 | -0.01887 | [-0.03504, -0.00318] |
| D=21 | 178 | 0.08989 | 0.04769 | 0.13344 | 0.07871 | 0.07693 | -0.00178 | [-0.01442, 0.00946] |

## Q1 descriptive rates

Label: `favorite_final_win` overall rate = 0.43316 (4945/11416), Wilson 95% CI [0.42410, 0.44227], bootstrap 95% CI [0.41717, 0.44927].

By deficit:

| Deficit | Successes | N | Rate | Wilson 95% CI |
|---:|---:|---:|---:|---|
| D=3 | 2459 | 4311 | 0.57040 | [0.55557, 0.58511] |
| D=7 | 1510 | 3183 | 0.47440 | [0.45709, 0.49176] |
| D=10 | 633 | 2025 | 0.31259 | [0.29277, 0.33312] |
| D=14 | 301 | 1348 | 0.22329 | [0.20187, 0.24629] |
| D=21 | 42 | 549 | 0.07650 | [0.05709, 0.10180] |

By time bucket:

| Time bucket | Successes | N | Rate | Wilson 95% CI |
|---|---:|---:|---:|---|
| Q1 | 3205 | 5867 | 0.54628 | [0.53351, 0.55898] |
| Q2-first-half | 1226 | 3104 | 0.39497 | [0.37792, 0.41229] |
| Q3 | 370 | 1403 | 0.26372 | [0.24133, 0.28740] |
| Q4 | 144 | 1042 | 0.13820 | [0.11857, 0.16048] |

By deficit x time bucket:

| Deficit | Time bucket | Successes | N | Rate | Wilson 95% CI |
|---:|---|---:|---:|---:|---|
| D=3 | Q1 | 1805 | 3002 | 0.60127 | [0.58363, 0.61864] |
| D=3 | Q2-first-half | 455 | 843 | 0.53974 | [0.50599, 0.57313] |
| D=3 | Q3 | 125 | 272 | 0.45956 | [0.40131, 0.51894] |
| D=3 | Q4 | 74 | 194 | 0.38144 | [0.31602, 0.45147] |
| D=7 | Q1 | 1064 | 1948 | 0.54620 | [0.52402, 0.56820] |
| D=7 | Q2-first-half | 305 | 725 | 0.42069 | [0.38527, 0.45695] |
| D=7 | Q3 | 106 | 307 | 0.34528 | [0.29430, 0.40008] |
| D=7 | Q4 | 35 | 203 | 0.17241 | [0.12666, 0.23033] |
| D=10 | Q1 | 223 | 573 | 0.38918 | [0.35012, 0.42971] |
| D=10 | Q2-first-half | 301 | 838 | 0.35919 | [0.32742, 0.39225] |
| D=10 | Q3 | 87 | 367 | 0.23706 | [0.19641, 0.28315] |
| D=10 | Q4 | 22 | 247 | 0.08907 | [0.05956, 0.13117] |
| D=14 | Q1 | 105 | 313 | 0.33546 | [0.28543, 0.38949] |
| D=14 | Q2-first-half | 138 | 514 | 0.26848 | [0.23199, 0.30841] |
| D=14 | Q3 | 46 | 286 | 0.16084 | [0.12280, 0.20787] |
| D=14 | Q4 | 12 | 235 | 0.05106 | [0.02945, 0.08712] |
| D=21 | Q1 | 8 | 31 | 0.25806 | [0.13702, 0.43246] |
| D=21 | Q2-first-half | 27 | 184 | 0.14674 | [0.10285, 0.20508] |
| D=21 | Q3 | 6 | 171 | 0.03509 | [0.01618, 0.07443] |
| D=21 | Q4 | 1 | 163 | 0.00613 | [0.00108, 0.03393] |

By season:

| Season | Successes | N | Rate | Wilson 95% CI |
|---:|---:|---:|---:|---|
| 2015 | 493 | 998 | 0.49399 | [0.46305, 0.52497] |
| 2016 | 489 | 1129 | 0.43313 | [0.40450, 0.46221] |
| 2017 | 483 | 1116 | 0.43280 | [0.40401, 0.46205] |
| 2018 | 451 | 1104 | 0.40851 | [0.37988, 0.43778] |
| 2019 | 466 | 1101 | 0.42325 | [0.39438, 0.45265] |
| 2020 | 399 | 890 | 0.44831 | [0.41593, 0.48114] |
| 2021 | 479 | 1221 | 0.39230 | [0.36529, 0.41999] |
| 2022 | 573 | 1261 | 0.45440 | [0.42710, 0.48198] |
| 2023 | 561 | 1300 | 0.43154 | [0.40486, 0.45863] |
| 2024 | 551 | 1296 | 0.42515 | [0.39850, 0.45225] |

By feature subset:

Subset: `dog_scored_on_opening_drive`

| Bucket | Successes | N | Rate | Wilson 95% CI |
|---|---:|---:|---:|---|
| false | 3415 | 8340 | 0.40947 | [0.39896, 0.42007] |
| missing | 1137 | 1860 | 0.61129 | [0.58893, 0.63319] |
| true | 393 | 1216 | 0.32319 | [0.29750, 0.35000] |

Subset: `opening_drive_was_explosive_td`

| Bucket | Successes | N | Rate | Wilson 95% CI |
|---|---:|---:|---:|---|
| false | 3328 | 8026 | 0.41465 | [0.40392, 0.42547] |
| missing | 1137 | 1860 | 0.61129 | [0.58893, 0.63319] |
| true | 480 | 1530 | 0.31373 | [0.29097, 0.33742] |

Subset: `early_vs_late_season`

| Bucket | Successes | N | Rate | Wilson 95% CI |
|---|---:|---:|---:|---|
| early_week_le_8 | 2838 | 6462 | 0.43918 | [0.42712, 0.45132] |
| late_week_gt_8 | 2107 | 4954 | 0.42531 | [0.41161, 0.43913] |

Subset: `is_neutral_site`

| Bucket | Successes | N | Rate | Wilson 95% CI |
|---|---:|---:|---:|---|
| false | 4534 | 10347 | 0.43819 | [0.42866, 0.44778] |
| true | 411 | 1069 | 0.38447 | [0.35577, 0.41400] |

Label: `deficit_erased` overall rate = 0.63495 (7246/11412), Wilson 95% CI [0.62607, 0.64373], bootstrap 95% CI [0.61958, 0.65077].

By deficit:

| Deficit | Successes | N | Rate | Wilson 95% CI |
|---:|---:|---:|---:|---|
| D=3 | 3526 | 4309 | 0.81829 | [0.80649, 0.82952] |
| D=7 | 2277 | 3182 | 0.71559 | [0.69966, 0.73100] |
| D=10 | 938 | 2024 | 0.46344 | [0.44180, 0.48521] |
| D=14 | 444 | 1348 | 0.32938 | [0.30480, 0.35492] |
| D=21 | 61 | 549 | 0.11111 | [0.08748, 0.14015] |

By time bucket:

| Time bucket | Successes | N | Rate | Wilson 95% CI |
|---|---:|---:|---:|---|
| Q1 | 4624 | 5867 | 0.78814 | [0.77749, 0.79840] |
| Q2-first-half | 1810 | 3104 | 0.58312 | [0.56568, 0.60035] |
| Q3 | 575 | 1403 | 0.40984 | [0.38438, 0.43578] |
| Q4 | 237 | 1038 | 0.22832 | [0.20382, 0.25483] |

By deficit x time bucket:

| Deficit | Time bucket | Successes | N | Rate | Wilson 95% CI |
|---:|---|---:|---:|---:|---|
| D=3 | Q1 | 2541 | 3002 | 0.84644 | [0.83310, 0.85889] |
| D=3 | Q2-first-half | 681 | 843 | 0.80783 | [0.77986, 0.83301] |
| D=3 | Q3 | 195 | 272 | 0.71691 | [0.66064, 0.76714] |
| D=3 | Q4 | 109 | 192 | 0.56771 | [0.49699, 0.63577] |
| D=7 | Q1 | 1568 | 1948 | 0.80493 | [0.78674, 0.82192] |
| D=7 | Q2-first-half | 474 | 725 | 0.65379 | [0.61843, 0.68753] |
| D=7 | Q3 | 164 | 307 | 0.53420 | [0.47832, 0.58923] |
| D=7 | Q4 | 71 | 202 | 0.35149 | [0.28898, 0.41954] |
| D=10 | Q1 | 338 | 573 | 0.58988 | [0.54914, 0.62942] |
| D=10 | Q2-first-half | 432 | 838 | 0.51551 | [0.48168, 0.54920] |
| D=10 | Q3 | 131 | 367 | 0.35695 | [0.30965, 0.40721] |
| D=10 | Q4 | 37 | 246 | 0.15041 | [0.11113, 0.20043] |
| D=14 | Q1 | 165 | 313 | 0.52716 | [0.47185, 0.58180] |
| D=14 | Q2-first-half | 190 | 514 | 0.36965 | [0.32903, 0.41220] |
| D=14 | Q3 | 72 | 286 | 0.25175 | [0.20496, 0.30511] |
| D=14 | Q4 | 17 | 235 | 0.07234 | [0.04565, 0.11278] |
| D=21 | Q1 | 12 | 31 | 0.38710 | [0.23733, 0.56176] |
| D=21 | Q2-first-half | 33 | 184 | 0.17935 | [0.13065, 0.24116] |
| D=21 | Q3 | 13 | 171 | 0.07602 | [0.04496, 0.12571] |
| D=21 | Q4 | 3 | 163 | 0.01840 | [0.00628, 0.05271] |

By season:

| Season | Successes | N | Rate | Wilson 95% CI |
|---:|---:|---:|---:|---|
| 2015 | 674 | 998 | 0.67535 | [0.64568, 0.70368] |
| 2016 | 718 | 1129 | 0.63596 | [0.60748, 0.66352] |
| 2017 | 682 | 1116 | 0.61111 | [0.58218, 0.63928] |
| 2018 | 649 | 1104 | 0.58786 | [0.55857, 0.61654] |
| 2019 | 723 | 1101 | 0.65668 | [0.62813, 0.68413] |
| 2020 | 560 | 890 | 0.62921 | [0.59699, 0.66033] |
| 2021 | 755 | 1220 | 0.61885 | [0.59127, 0.64569] |
| 2022 | 845 | 1261 | 0.67010 | [0.64367, 0.69550] |
| 2023 | 806 | 1297 | 0.62143 | [0.59472, 0.64744] |
| 2024 | 834 | 1296 | 0.64352 | [0.61705, 0.66914] |

By feature subset:

Subset: `dog_scored_on_opening_drive`

| Bucket | Successes | N | Rate | Wilson 95% CI |
|---|---:|---:|---:|---|
| false | 5050 | 8339 | 0.60559 | [0.59505, 0.61603] |
| missing | 1604 | 1860 | 0.86237 | [0.84596, 0.87728] |
| true | 592 | 1213 | 0.48805 | [0.46000, 0.51617] |

Subset: `opening_drive_was_explosive_td`

| Bucket | Successes | N | Rate | Wilson 95% CI |
|---|---:|---:|---:|---|
| false | 4892 | 8025 | 0.60960 | [0.59887, 0.62021] |
| missing | 1604 | 1860 | 0.86237 | [0.84596, 0.87728] |
| true | 750 | 1527 | 0.49116 | [0.46614, 0.51622] |

Subset: `early_vs_late_season`

| Bucket | Successes | N | Rate | Wilson 95% CI |
|---|---:|---:|---:|---|
| early_week_le_8 | 4117 | 6458 | 0.63750 | [0.62570, 0.64914] |
| late_week_gt_8 | 3129 | 4954 | 0.63161 | [0.61808, 0.64494] |

Subset: `is_neutral_site`

| Bucket | Successes | N | Rate | Wilson 95% CI |
|---|---:|---:|---:|---|
| false | 6638 | 10343 | 0.64179 | [0.63250, 0.65097] |
| true | 608 | 1069 | 0.56876 | [0.53887, 0.59815] |

## Data quality

- Trigger events: 11,416
- Unique trigger plays: 7,854
- Missing play-by-play games: 0
- Missing trigger play IDs: 0
- Trigger score mismatches: 0
- No-post-trigger unique plays labeled null: 2
- Deficit-erased null event rows excluded label-wise: 4

## Interpretation

favorite_final_win is the N03 training label; deficit_erased is a newly computed literal comeback event. Differences between the two indicate whether the model is detecting final game recovery, temporary deficit erasure, or both.

If model-vs-baseline improvement is small or negative, N05 should be read as evidence that N03's apparent comeback detection mostly inherits deficit/time structure. If improvement is positive with a CI above zero, N05 supports incremental comeback-detection signal beyond the naive structural baseline.
