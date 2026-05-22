# N10 -- Direct conditional analysis of fluke-deficit comebacks

**Project-level finding.** N10 is the most direct test of the project's core mechanistic hypothesis. It resolves the question that drove the prior research arc: when a favorite falls behind because of fluky scoring (turnovers, returns, explosive plays), is genuinely the better pre-game team, and still has meaningful time remaining, do pre-game markets underprice that favorite's comeback probability?

The answer is unambiguous: **no**. In the exact headline condition, favorites win **43.9%** of the time against a mean pre-game no-vig market probability of **79.1%**. Markets overprice these favorites by about **35 percentage points**; they do not underprice them. Across the reliable three-way cells covering fluke composition, favorite strength, and time bucket, zero cells show actual final-win rate above market implied probability with statistical confidence.

The implication is project-level, not just notebook-level. Pre-game CFB markets price favorite comeback scenarios at least correctly, and possibly with extra favorite-side padding. The project's predictive-edge framework, which did not beat `baseline_C` across N06/N07/N08/N09, does not have a hidden mechanistic fluky-deficit angle that prior notebooks missed. The fluky-vs-sustained hypothesis is contradicted, not merely unsupported.

Live in-game market edge is a separate question that N10 cannot test. Future live-data work would need to test a fresh hypothesis: whether live in-game markets overreact to favorite deficits more than pre-game markets do. That live question should stand on its own rather than inheriting support from the historical pre-game analysis.

**Direct answer: NULL OR NEGATIVE.** The stricter clear-fluky compound condition does not show validated underpricing by pre-game markets. Favorite final-win rate is **43.9%** vs mean no-vig market probability **79.1%**; actual-minus-market is **-35.2%** with bootstrap CI **[-41.1%, -29.2%]**. Edge, if it exists, must come from live in-game prices that N10 cannot test.

N10 uses pre-game odds only. It never tests live market edge. Positive cells are candidate live-watch conditions for future collection, not betting recommendations.

## Fluke Classification Sanity Checks

The broad N07-attributed `fluky_lead` bucket did **not** separate aggregate yards-per-point cleanly from `sustained_lead`; this is why N10 uses Option C with a stricter headline guard. The broad bucket remains in descriptive/dashboard tables, while the direct Tier 7 hypothesis test uses `clear_fluky_lead`: `fluke_bucket=fluky_lead` plus dog completed-drive yards per point at or below the `sustained_lead` median. Dog early-down success rate is reported diagnostically rather than used as a gate.
`clear_fluky_lead` threshold: dog completed-drive yards per point <= **14.891**; rows flagged clear-fluky: **1,779**.
Option C attribution handling assigned **2,974** rows to `attribution_unclear`. These rows remain in descriptive tables but are excluded from the headline fluky-lead hypothesis test and candidate live-watch flags.
| fluke_bucket | n_events | mean_dog_yards_per_point | median_dog_yards_per_point | mean_dog_drive_yards_per_point_diagnostic | median_dog_drive_yards_per_point_diagnostic | mean_dog_success_rate | mean_epa_per_play_gap | mean_success_rate_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attribution_unclear | 2974 | 9.732 | 9.000 | nan | nan | 0.489 | -0.571 | nan |
| fluky_lead | 3318 | 14.858 | 13.774 | 15.016 | 13.571 | 0.452 | -0.238 | -0.037 |
| mixed_lead | 2907 | 13.921 | 13.059 | 16.262 | 14.000 | 0.469 | -0.290 | -0.077 |
| no_dog_points | 19 | 90.474 | 92.000 | 5.571 | 5.571 | 0.214 | 0.999 | 0.250 |
| sustained_lead | 2194 | 14.292 | 12.571 | 17.215 | 14.891 | 0.400 | -0.170 | -0.000 |

## Tier 4 Reliable Single-Dimension Context

### fluke_bucket

| bucket | n_events | n_games | thin_flag | final_win | deficit_erased | no_vig | final_minus_market |
| --- | --- | --- | --- | --- | --- | --- | --- |
| attribution_unclear | 2974 | 1816 | reliable | 58.4% | 84.5% | 71.0% | -12.6% |
| fluky_lead | 3318 | 1770 | reliable | 31.0% | 44.7% | 68.2% | -37.2% |
| mixed_lead | 2907 | 1592 | reliable | 34.8% | 52.8% | 67.6% | -32.9% |
| no_dog_points | 19 | 13 | unreliable | 84.2% | 100.0% | 80.4% | 3.8% |
| sustained_lead | 2194 | 1343 | reliable | 52.6% | 77.3% | 69.9% | -17.4% |

### spread_bucket

| bucket | n_events | n_games | thin_flag | final_win | deficit_erased | no_vig | final_minus_market |
| --- | --- | --- | --- | --- | --- | --- | --- |
| big_favorite | 3327 | 1285 | reliable | 47.0% | 67.0% | 74.5% | -27.5% |
| huge_favorite | 2141 | 990 | reliable | 69.8% | 82.0% | 88.6% | -18.8% |
| moderate_favorite | 3940 | 1379 | reliable | 35.1% | 57.9% | 61.8% | -26.7% |
| small_favorite | 2004 | 655 | reliable | 25.1% | 48.9% | 53.8% | -28.7% |

### time_bucket

| bucket | n_events | n_games | thin_flag | final_win | deficit_erased | no_vig | final_minus_market |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q1 | 5867 | 3002 | reliable | 54.6% | 78.8% | 70.4% | -15.8% |
| Q2 | 3104 | 1764 | reliable | 39.5% | 58.3% | 68.3% | -28.8% |
| Q3 | 1403 | 887 | reliable | 26.4% | 41.0% | 67.2% | -40.8% |
| Q4 | 1038 | 718 | reliable | 13.9% | 22.8% | 66.8% | -52.9% |

### deficit

| bucket | n_events | n_games | thin_flag | final_win | deficit_erased | no_vig | final_minus_market |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 4309 | 4309 | reliable | 57.1% | 81.8% | 70.7% | -13.7% |
| 7 | 3182 | 3182 | reliable | 47.5% | 71.6% | 69.6% | -22.2% |
| 10 | 2024 | 2024 | reliable | 31.3% | 46.3% | 67.7% | -36.5% |
| 14 | 1348 | 1348 | reliable | 22.3% | 32.9% | 66.7% | -44.3% |
| 21 | 549 | 549 | reliable | 7.7% | 11.1% | 64.9% | -57.3% |

## Tier 7 Direct Hypothesis Test

Definition: `clear_fluky_lead=True AND spread_bucket in {huge_favorite,big_favorite} AND time_bucket in {Q1,Q2}; clear_fluky_lead requires fluke_bucket=fluky_lead and dog completed-drive yards/point <= sustained_lead median (14.891)`.

- Events/games/seasons: **472 / 311 / 10** (`reliable`).
- `favorite_final_win`: **43.9%**, Wilson CI **[39.4%, 48.4%]**, actual-minus-no-vig **-35.2%**, bootstrap CI **[-41.1%, -29.2%]**.
- `deficit_erased`: **58.5%**, Wilson CI **[54.0%, 62.8%]**, actual-minus-no-vig **-20.6%**, bootstrap CI **[-26.8%, -14.3%]**.
- Held-out real-moneyline flat ROI: **-49.6%** on **135** bets, CI **[-62.9%, -36.1%]**.
- Held-out real + synthetic fallback flat ROI: **-49.6%** on **137** bets, CI **[-62.4%, -36.2%]**.

Subcells:
| cell | n_events | n_games | thin_flag | final_win | no_vig | diff |
| --- | --- | --- | --- | --- | --- | --- |
| big_favorite / Q1 | 92 | 62 | reliable | 40.2% | 73.9% | -33.7% |
| big_favorite / Q2 | 213 | 150 | reliable | 40.4% | 74.3% | -34.0% |
| huge_favorite / Q1 | 58 | 38 | reliable | 55.2% | 87.5% | -32.3% |
| huge_favorite / Q2 | 109 | 83 | reliable | 47.7% | 88.2% | -40.5% |

## Tier 8 Inverse Hypothesis Sanity Check

Definition: `fluke_bucket=sustained_lead AND spread_bucket=small_favorite AND time_bucket=Q4`.

- Events/games/seasons: **6 / 6 / 4** (`unreliable`).
- `favorite_final_win`: **33.3%**, actual-minus-no-vig **-20.3%**.
- Held-out real-moneyline flat ROI: **-14.8%** on **2** bets, CI **[-100.0%, 70.4%]**.

## Tier 2.5 Two-Way Findings

### fluke_x_spread

| cell | n_events | n_games | final_win | deficit_erased | no_vig | diff |
| --- | --- | --- | --- | --- | --- | --- |
| attribution_unclear / huge_favorite | 722 | 455 | 81.7% | 94.5% | 89.0% | -7.2% |
| sustained_lead / huge_favorite | 471 | 315 | 78.1% | 91.7% | 89.0% | -10.9% |
| attribution_unclear / big_favorite | 873 | 531 | 62.0% | 87.1% | 74.9% | -13.0% |
| attribution_unclear / moderate_favorite | 908 | 550 | 48.2% | 81.5% | 61.9% | -13.7% |
| sustained_lead / big_favorite | 608 | 374 | 56.9% | 83.2% | 74.3% | -17.4% |
| attribution_unclear / small_favorite | 471 | 280 | 35.5% | 70.5% | 53.5% | -18.0% |
| sustained_lead / moderate_favorite | 771 | 461 | 42.8% | 71.1% | 62.0% | -19.2% |
| sustained_lead / small_favorite | 344 | 193 | 31.7% | 60.8% | 53.7% | -22.0% |

### fluke_x_time

| cell | n_events | n_games | final_win | deficit_erased | no_vig | diff |
| --- | --- | --- | --- | --- | --- | --- |
| sustained_lead / Q1 | 1387 | 885 | 58.5% | 82.1% | 70.8% | -12.3% |
| attribution_unclear / Q1 | 2966 | 1813 | 58.4% | 84.6% | 71.0% | -12.6% |
| sustained_lead / Q2 | 664 | 435 | 47.0% | 72.7% | 68.5% | -21.6% |
| fluky_lead / Q1 | 710 | 438 | 48.2% | 66.1% | 70.4% | -22.2% |
| mixed_lead / Q1 | 796 | 431 | 39.2% | 62.6% | 67.6% | -28.4% |
| mixed_lead / Q2 | 1281 | 790 | 37.5% | 55.3% | 68.0% | -30.6% |
| fluky_lead / Q2 | 1146 | 786 | 37.2% | 53.1% | 68.5% | -31.3% |
| mixed_lead / Q3 | 531 | 358 | 30.9% | 45.8% | 67.3% | -36.4% |

### fluke_x_deficit

| cell | n_events | n_games | final_win | deficit_erased | no_vig | diff |
| --- | --- | --- | --- | --- | --- | --- |
| attribution_unclear / 3 | 1816 | 1816 | 60.1% | 85.6% | 71.3% | -11.2% |
| sustained_lead / 3 | 1222 | 1222 | 57.5% | 82.8% | 70.8% | -13.3% |
| attribution_unclear / 7 | 1138 | 1138 | 56.2% | 83.8% | 70.7% | -14.5% |
| fluky_lead / 3 | 699 | 699 | 53.2% | 75.3% | 70.2% | -17.0% |
| mixed_lead / 3 | 562 | 562 | 50.7% | 75.4% | 69.1% | -18.4% |
| sustained_lead / 7 | 701 | 701 | 50.4% | 77.5% | 70.0% | -19.6% |
| sustained_lead / 10 | 172 | 172 | 42.4% | 62.8% | 67.0% | -24.6% |
| mixed_lead / 7 | 684 | 684 | 39.0% | 60.4% | 68.0% | -28.9% |

### spread_x_time

| cell | n_events | n_games | final_win | deficit_erased | no_vig | diff |
| --- | --- | --- | --- | --- | --- | --- |
| huge_favorite / Q1 | 1336 | 748 | 79.9% | 91.9% | 89.0% | -9.1% |
| big_favorite / Q1 | 1690 | 855 | 58.3% | 81.5% | 74.8% | -16.5% |
| moderate_favorite / Q1 | 1893 | 943 | 44.5% | 74.2% | 61.9% | -17.4% |
| small_favorite / Q1 | 948 | 456 | 32.8% | 64.9% | 53.7% | -20.9% |
| huge_favorite / Q2 | 508 | 326 | 63.4% | 76.0% | 88.0% | -24.6% |
| big_favorite / Q2 | 899 | 518 | 45.6% | 65.7% | 74.5% | -28.9% |
| moderate_favorite / Q2 | 1135 | 616 | 32.0% | 52.2% | 61.9% | -29.9% |
| small_favorite / Q2 | 562 | 304 | 23.3% | 42.9% | 53.7% | -30.4% |

### deficit_x_time

| cell | n_events | n_games | final_win | deficit_erased | no_vig | diff |
| --- | --- | --- | --- | --- | --- | --- |
| 3 / Q1 | 3002 | 3002 | 60.1% | 84.6% | 71.2% | -11.1% |
| 7 / Q1 | 1948 | 1948 | 54.6% | 80.5% | 70.6% | -15.9% |
| 3 / Q2 | 843 | 843 | 54.0% | 80.8% | 70.2% | -16.2% |
| 3 / Q3 | 272 | 272 | 46.0% | 71.7% | 67.8% | -21.8% |
| 7 / Q2 | 725 | 725 | 42.1% | 65.4% | 68.5% | -26.4% |
| 10 / Q1 | 573 | 573 | 38.9% | 59.0% | 68.1% | -29.2% |
| 3 / Q4 | 192 | 192 | 38.5% | 56.8% | 69.8% | -31.2% |
| 10 / Q2 | 838 | 838 | 35.9% | 51.6% | 68.1% | -32.2% |

## Tier 1 Reliable Three-Way Cells

| cell | n_events | n_games | final_win | deficit_erased | no_vig | diff | candidate_live_watch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sustained_lead / 3 / Q1 | 836 | 836 | 61.4% | 84.7% | 71.3% | -9.9% | False |
| fluky_lead / 3 / Q3 | 104 | 104 | 55.8% | 72.1% | 66.7% | -10.9% | False |
| attribution_unclear / 3 / Q1 | 1813 | 1813 | 60.2% | 85.6% | 71.3% | -11.1% | False |
| fluky_lead / 3 / Q2 | 225 | 225 | 56.9% | 82.2% | 70.3% | -13.4% | False |
| fluky_lead / 3 / Q1 | 256 | 256 | 57.4% | 79.7% | 71.0% | -13.6% | False |
| mixed_lead / 3 / Q1 | 90 | 90 | 54.4% | 77.8% | 68.4% | -14.0% | False |
| attribution_unclear / 7 / Q1 | 1136 | 1136 | 56.2% | 83.9% | 70.7% | -14.5% | False |
| sustained_lead / 7 / Q1 | 471 | 471 | 55.2% | 80.9% | 70.6% | -15.4% | False |
| sustained_lead / 3 / Q2 | 331 | 331 | 53.5% | 82.2% | 70.0% | -16.5% | False |
| mixed_lead / 3 / Q2 | 282 | 282 | 52.5% | 78.0% | 70.4% | -17.9% | False |
| sustained_lead / 10 / Q1 | 57 | 57 | 50.9% | 68.4% | 69.0% | -18.1% | False |
| fluky_lead / 7 / Q1 | 172 | 172 | 52.9% | 70.3% | 72.0% | -19.0% | False |
| mixed_lead / 3 / Q3 | 122 | 122 | 46.7% | 75.4% | 68.0% | -21.2% | False |
| mixed_lead / 3 / Q4 | 68 | 68 | 45.6% | 61.8% | 67.0% | -21.4% | False |
| sustained_lead / 10 / Q2 | 90 | 90 | 43.3% | 64.4% | 66.1% | -22.8% | False |
| mixed_lead / 7 / Q1 | 168 | 168 | 44.0% | 66.7% | 67.9% | -23.9% | False |
| sustained_lead / 7 / Q2 | 190 | 190 | 44.2% | 72.1% | 68.5% | -24.3% | False |
| mixed_lead / 7 / Q3 | 127 | 127 | 41.7% | 58.3% | 67.3% | -25.6% | False |
| fluky_lead / 7 / Q2 | 212 | 212 | 42.5% | 62.3% | 68.6% | -26.1% | False |
| mixed_lead / 7 / Q2 | 320 | 320 | 40.3% | 63.4% | 68.4% | -28.1% | False |

## Candidate Live-Watch Conditions

N10 flags **0** candidate live-watch cells under the locked definition.

## Deliverables

- `n10_conditional_rates.parquet`: 11,412 trigger-event rows.
- `n10_conditional_analysis.json`: all tier matrices, direct/inverse tests, sanity diagnostics, and candidate live-watch cells.
- `n10_summary_report.md`: this human-readable report.

## Honest Interpretation

N10 is the project's direct conditional answer, but it still uses pre-game prices. If the direct fluky-deficit condition is positive, it means the historical pre-game market underpriced that subset before kickoff; it does not prove that live in-game prices after the favorite falls behind would remain exploitable. If the condition is null or underpowered, that is evidence that pre-game markets already encode much of the favorite-strength and game-context information in these trigger states. The next true market-edge test still requires live odds collection.
