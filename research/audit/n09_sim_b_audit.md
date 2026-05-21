# N09 Sim B Betting Result Audit

Date: 2026-05-20

Scope: focused audit of the N09 `B_model_edge` betting simulation at `edge_threshold >= 0.10`, before any N09 bucket revisions or commits. This audit does not modify N09 logic or outputs.

State check: repository HEAD and `origin/main` were both `5148ab190055bd5f308925a26c12b55b36839fbd`. Working tree contained only expected untracked executed notebooks, 02g diagnostics, and untracked N09 artifacts.

## Executive Conclusion

The reported flat-stake Sim B result is reproducible from the N09 parquet:

- `n_bets = 236`
- `favorite_final_win` win rate = `0.8136`
- ROI = `+0.4250`
- total flat-stake profit = `+100.3060` units
- `deficit_erased` rate = `0.9661`

However, this should **not** be treated as confirmed final-win betting edge in its current framing. The result is a real realized return for the implemented selection rule, but the selection rule is not a same-label expected-value test.

Critical finding: Sim B filters on `n06_prob`, which is trained on `deficit_erased`, then pays bets on `favorite_final_win`. The edge calculation compares a deficit-erasure probability to a final-win market break-even probability. That is a target mismatch, not direct outcome leakage, but it invalidates interpreting `edge_at_entry = n06_prob - market_implied_prob` as final-win betting edge.

Additional moderate finding: 32 of 236 high-edge bets use synthetic fallback odds derived from `baseline_C_favorite_final_win`, not real cached sportsbook moneylines. These fallback bets contribute `+31.376` units of the `+100.306` total profit. Excluding fallback bets leaves a positive ROI (`+0.3379` on 204 bets), so fallback pricing is not the whole result, but it materially inflates the headline.

Bottom line: Sim B is interesting as a descriptive selection pattern. It is not yet a verified betting-edge result.

## Task 1 - Five Hand-Checked Winning Bets

I selected five `B_model_edge`, threshold `0.10`, flat-stake bets that won and had usable direction-consistent sportsbook moneyline data. All five reconcile exactly to the N09 parquet. In all five cases, N09 uses raw sportsbook break-even probability `1 / decimal_odds`, not no-vig probability, for `market_implied_prob`.

No-vig formula used for audit math:

```text
decimal = 1 + odds / 100                         for positive American odds
decimal = 1 + 100 / abs(odds)                    for negative American odds
raw_prob = 1 / decimal
fav_no_vig = fav_raw_prob / (fav_raw_prob + dog_raw_prob)
```

### Bet 1: 401403876, Kansas State vs Missouri, 2022, D=3

Cached lines:

| Provider | Home ML | Away ML | Spread | Spread Open |
|---|---:|---:|---:|---:|
| Bovada | -300 | +250 | -7 | -8.5 |
| teamrankings | null | null | -7 | null |
| consensus | null | null | -7.5 | null |
| William Hill (New Jersey) | null | null | -7 | null |

N09 selected Kansas State home moneyline at Bovada.

Math:

| Item | Value |
|---|---:|
| Favorite American odds | -300 |
| Favorite decimal odds | `1.333333` |
| Dog decimal odds | `3.500000` |
| Favorite raw break-even probability | `0.750000` |
| Dog raw probability | `0.285714` |
| Overround | `1.035714` |
| Favorite no-vig probability | `0.724138` |
| N09 `market_implied_prob` | `0.750000` |
| `n06_prob` | `0.882353` |
| N09 edge | `0.882353 - 0.750000 = +0.132353` |
| Outcome | favorite won |
| Flat payout | `1.333333 - 1 = +0.333333` units |

Finding: payout and edge math reconcile. N09 uses raw break-even, not no-vig.

### Bet 2: 401403882, Penn State vs Auburn, 2022, D=3

Cached lines:

| Provider | Home ML | Away ML | Spread | Spread Open |
|---|---:|---:|---:|---:|
| Bovada | +120 | -140 | +2 | +3 |
| teamrankings | null | null | +2.5 | null |
| consensus | null | null | +2.5 | null |
| William Hill (New Jersey) | null | null | +2.5 | null |

N09 selected Penn State away moneyline at Bovada.

Math:

| Item | Value |
|---|---:|
| Favorite American odds | -140 |
| Favorite decimal odds | `1.714286` |
| Dog decimal odds | `2.200000` |
| Favorite raw break-even probability | `0.583333` |
| Dog raw probability | `0.454545` |
| Overround | `1.037879` |
| Favorite no-vig probability | `0.562044` |
| N09 `market_implied_prob` | `0.583333` |
| `n06_prob` | `0.850575` |
| N09 edge | `0.850575 - 0.583333 = +0.267241` |
| Outcome | favorite won |
| Flat payout | `1.714286 - 1 = +0.714286` units |

Finding: payout and edge math reconcile. N09 uses raw break-even, not no-vig.

### Bet 3: 401403927, Arkansas vs Auburn, 2022, D=3

Cached lines:

| Provider | Home ML | Away ML | Spread | Spread Open |
|---|---:|---:|---:|---:|
| Bovada | +160 | -185 | +4 | +4 |
| teamrankings | null | null | +4 | null |
| consensus | null | null | +4 | null |
| William Hill (New Jersey) | null | null | +4 | null |

N09 selected Arkansas away moneyline at Bovada.

Math:

| Item | Value |
|---|---:|
| Favorite American odds | -185 |
| Favorite decimal odds | `1.540541` |
| Dog decimal odds | `2.600000` |
| Favorite raw break-even probability | `0.649123` |
| Dog raw probability | `0.384615` |
| Overround | `1.033738` |
| Favorite no-vig probability | `0.627937` |
| N09 `market_implied_prob` | `0.649123` |
| `n06_prob` | `0.785047` |
| N09 edge | `0.785047 - 0.649123 = +0.135924` |
| Outcome | favorite won |
| Flat payout | `1.540541 - 1 = +0.540541` units |

Finding: payout and edge math reconcile. N09 uses raw break-even, not no-vig.

### Bet 4: 401403996, California vs Arizona, 2022, D=7

Cached lines:

| Provider | Home ML | Away ML | Spread | Spread Open |
|---|---:|---:|---:|---:|
| teamrankings | null | null | -3.5 | null |
| consensus | null | null | -3.5 | null |
| William Hill (New Jersey) | null | null | -3.5 | null |
| Bovada | -160 | +135 | -3.5 | -4 |

N09 selected California home moneyline at Bovada.

Math:

| Item | Value |
|---|---:|
| Favorite American odds | -160 |
| Favorite decimal odds | `1.625000` |
| Dog decimal odds | `2.350000` |
| Favorite raw break-even probability | `0.615385` |
| Dog raw probability | `0.425532` |
| Overround | `1.040916` |
| Favorite no-vig probability | `0.591195` |
| N09 `market_implied_prob` | `0.615385` |
| `n06_prob` | `0.785047` |
| N09 edge | `0.785047 - 0.615385 = +0.169662` |
| Outcome | favorite won |
| Flat payout | `1.625000 - 1 = +0.625000` units |

Finding: payout and edge math reconcile. N09 uses raw break-even, not no-vig.

### Bet 5: 401404018, Stanford vs Arizona State, 2022, D=7

Cached lines:

| Provider | Home ML | Away ML | Spread | Spread Open |
|---|---:|---:|---:|---:|
| Bovada | -150 | +130 | -3 | -3.5 |
| teamrankings | null | null | -3 | null |
| consensus | null | null | -3 | null |
| William Hill (New Jersey) | null | null | -3 | null |

N09 selected Stanford home moneyline at Bovada.

Math:

| Item | Value |
|---|---:|
| Favorite American odds | -150 |
| Favorite decimal odds | `1.666667` |
| Dog decimal odds | `2.300000` |
| Favorite raw break-even probability | `0.600000` |
| Dog raw probability | `0.434783` |
| Overround | `1.034783` |
| Favorite no-vig probability | `0.579832` |
| N09 `market_implied_prob` | `0.600000` |
| `n06_prob` | `0.769231` |
| N09 edge | `0.769231 - 0.600000 = +0.169231` |
| Outcome | favorite won |
| Flat payout | `1.666667 - 1 = +0.666667` units |

Finding: payout and edge math reconcile. N09 uses raw break-even, not no-vig.

Task severity: moderate. The five real-sportsbook examples reconcile, but they also confirm the code path uses raw break-even probabilities rather than no-vig probabilities. This does not by itself explain the high ROI.

Does this change the conclusion that Sim B is real? It confirms the real-sportsbook payout math is internally consistent. It does not validate the interpretation of Sim B as a final-win probability edge.

## Task 2 - No-Vig Conversion Methodology

Relevant code path:

- `american_raw_prob` and `american_decimal_odds`: `research/notebooks/_build_n09.py:316-333`
- `market_prices_for_favorite`: `research/notebooks/_build_n09.py:909-982`
- edge calculation: `research/notebooks/_build_n09.py:1000`
- payout calculation: `research/notebooks/_build_n09.py:1065-1069`

Actual N09 behavior:

1. Direction-consistent sportsbook providers are identified.
2. The best available decimal odds are selected by maximum decimal payout.
3. `best_market_prob = 1.0 / best_decimal_odds`.
4. `edge_at_entry = n06_prob - best_market_prob`.
5. Flat-stake payout uses the same selected decimal odds.

No provider-level no-vig probability is computed or used in Sim B.

Interpretation:

- For a realized betting simulation at listed book odds, raw break-even probability `1 / decimal_odds` is the relevant EV threshold. A bettor must beat the vig-inclusive break-even probability to have positive expected value at the offered price.
- For a probability comparison against market consensus, no-vig probability is the cleaner market belief estimate. N04 used that framing. N09 Sim B does not.
- Therefore the current `market_implied_prob` name is ambiguous. It is really `best_available_raw_break_even_prob`.

Task severity: moderate. This is a methodology/labeling mismatch, not the source of the inflated ROI. Using raw break-even is stricter than no-vig for edge filtering.

Does this change the conclusion that Sim B is real? It changes the framing. Sim B is a raw offered-price simulation, not a no-vig market-consensus probability comparison.

## Task 3 - Pre-Game Timing Audit

The cached line records used by N09 contain game-level `startDate`, but line entries do not contain line-level timestamps. The line-level keys are provider/price fields such as:

```text
provider, spread, spreadOpen, formattedSpread, overUnder, overUnderOpen,
homeMoneyline, awayMoneyline
```

Timestamp-like line keys found in the Sim B edge `>= 0.10` subset: none.

Because CFBD's cached `/lines` payload does not include line-level timestamps, this audit cannot verify whether a particular line snapshot was captured immediately at open, close, or any other pre-game time. I found no evidence in the cached schema that these are live in-game prices, but the exact timestamp distribution requested by the audit is unavailable from the data.

Task severity: data limitation. No evidence of in-game-line leakage was found, but the cache cannot prove line timing.

Does this change the conclusion that Sim B is real? It limits provenance confidence. It does not explain the high ROI.

## Task 4 - Per-Season Decomposition of Sim B

Flat-stake Sim B by edge threshold:

| Threshold | Season | n_bets | Win rate | ROI | Bootstrap ROI CI, 95% |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 2022 | 134 | 0.7388 | 0.5771 | [0.1384, 1.2804] |
| 0.00 | 2023 | 147 | 0.7959 | 0.1979 | [0.0942, 0.2990] |
| 0.00 | 2024 | 119 | 0.8571 | 0.3568 | [0.2474, 0.4583] |
| 0.05 | 2022 | 106 | 0.7736 | 0.4080 | [0.1657, 0.7148] |
| 0.05 | 2023 | 117 | 0.8034 | 0.2541 | [0.1334, 0.3699] |
| 0.05 | 2024 | 91 | 0.8681 | 0.4240 | [0.3015, 0.5398] |
| 0.10 | 2022 | 79 | 0.7595 | 0.4803 | [0.1658, 0.8773] |
| 0.10 | 2023 | 83 | 0.8313 | 0.3545 | [0.2090, 0.4883] |
| 0.10 | 2024 | 74 | 0.8514 | 0.4451 | [0.2951, 0.5778] |

Task severity: verified. The high-edge ROI is not driven by one anomalous year. It is positive in 2022, 2023, and 2024.

Does this change the conclusion that Sim B is real? It supports that the realized selection pattern is stable across seasons. It does not resolve the label-target mismatch.

## Task 5 - Price Distribution Check

For all 236 Sim B edge `>= 0.10` flat bets:

| Statistic | Decimal odds |
|---|---:|
| count | 236 |
| mean | 1.8114 |
| std | 1.0986 |
| min | 1.2632 |
| p05 | 1.3509 |
| p10 | 1.4082 |
| p25 | 1.5188 |
| median | 1.6897 |
| p75 | 1.8000 |
| p90 | 1.8696 |
| p95 | 2.0125 |
| p99 | 6.9388 |
| max | 12.2308 |

Histogram:

| Decimal odds bucket | Count |
|---|---:|
| [1.25, 1.50) | 41 |
| [1.50, 1.75) | 112 |
| [1.75, 2.00) | 70 |
| [2.00, 2.50) | 8 |
| [2.50, 3.00) | 0 |
| [3.00, 4.00) | 1 |
| [4.00, 5.00) | 1 |
| [5.00, 10.00) | 1 |
| [10.00, 100.00) | 2 |

The extreme payout outliers are not real sportsbook moneylines. They are synthetic fallback prices:

| Game | Favorite | Dog | Decimal odds | Source | Outcome | Profit |
|---:|---|---|---:|---|---:|---:|
| 401470891 | Oregon | North Carolina | 12.2308 | fallback synthetic | win | +11.2308 |
| 401551754 | UCF | Georgia Tech | 12.2308 | fallback synthetic | loss | -1.0000 |
| 401404066 | Oklahoma | Kent State | 8.2000 | fallback synthetic | win | +7.2000 |
| 401520238 | New Mexico | New Mexico State | 4.5965 | fallback synthetic | loss | -1.0000 |
| 401520273 | UTEP | UNLV | 3.7159 | fallback synthetic | loss | -1.0000 |

Fallback breakdown:

| Subset | n_bets | Win rate | ROI | Profit |
|---|---:|---:|---:|---:|
| All Sim B >= 0.10 | 236 | 0.8136 | 0.4250 | +100.3060 |
| Real moneyline only | 204 | 0.8186 | 0.3379 | +68.9304 |
| Synthetic fallback only | 32 | 0.7813 | 0.9805 | +31.3756 |

Real-moneyline-only decimal odds are much more reasonable:

| Statistic | Decimal odds, real moneyline only |
|---|---:|
| count | 204 |
| mean | 1.6457 |
| std | 0.1774 |
| min | 1.2632 |
| median | 1.6667 |
| p95 | 1.8929 |
| max | 2.2000 |

Task severity: moderate. Synthetic fallback odds materially inflate the headline ROI and create the extreme price outliers. They do not create the entire positive result, because real-moneyline-only bets remain strongly positive.

Does this change the conclusion that Sim B is real? Yes. The all-in `+42.5%` ROI should not be reported as a pure cached-sportsbook result. A real-moneyline-only version is still positive at `+33.8%`, but it remains subject to the critical label-target mismatch.

## Task 6 - Cross-Check Against N04

N04 was the correct same-label market comparison for `favorite_final_win`: model probability trained on favorite final win versus pre-game market probability.

For the 236 games selected by N09 Sim B edge `>= 0.10`, N04's `favorite_final_win` model shows the opposite signal:

| N04 check on Sim B games | Value |
|---|---:|
| N04 trigger-event rows | 359 |
| Unique games | 236 |
| Mean N04 Brier improvement vs market | -0.03155 |
| Mean N04 model edge vs market | -0.07133 |
| N04 model Brier | 0.22172 |
| Market Brier | 0.19017 |
| Actual final-win rate | 0.77159 |

For the deepest trigger row matched one-to-one to each Sim B game:

| N04 deepest-row check | Value |
|---|---:|
| Rows/games | 236 |
| Mean N04 Brier improvement vs market | -0.03561 |
| Mean N04 model edge vs market | -0.08240 |
| Mean N04 model probability | 0.53884 |
| Mean market probability | 0.62124 |
| Actual final-win rate | 0.81356 |

This explains the apparent discrepancy with the N04 Brier result. N09 Sim B is not selecting games where the final-win model finds positive final-win edge. It is selecting games where the deficit-erased model assigns high comeback-erasure probability relative to a final-win market break-even probability.

The realized final-win rate of the selected subset is high, but the same-label final-win model did not forecast a positive edge on those games. That means the Sim B result is not a straightforward extension of N04's probability-edge result.

Task severity: critical. This changes the interpretation of Sim B.

Does this change the conclusion that Sim B is real? Yes. The realized ROI is real, but it is not validated by the same-label market comparison. The clean explanation is target mismatch plus a high-rate selected subset, not confirmed final-win betting edge.

## Task 7 - Data Leakage Audit

Relevant code path:

- `heldout_u = n08.copy()`: `research/notebooks/_build_n09.py:984`
- deepest trigger deduplication by game: `research/notebooks/_build_n09.py:985-988`
- market rows constructed from game ID, favorite team, and fallback probability: `research/notebooks/_build_n09.py:992-998`
- edge filter: `research/notebooks/_build_n09.py:1000`
- Sim B selection: `research/notebooks/_build_n09.py:1017-1018`
- bet outcome and payout: `research/notebooks/_build_n09.py:1068-1069`

What enters the Sim B filter:

```text
edge_at_entry = n06_prob - best_market_prob
selected if edge_at_entry >= threshold
```

I found no direct use of `favorite_final_win` or realized profit in the filter selection. `favorite_final_win` is used after selection for payout and summaries, as expected.

The issue is not ordinary outcome leakage. The issue is target mismatch:

- `n06_prob` is trained and calibrated on `deficit_erased`.
- The bet pays on `favorite_final_win`.
- `best_market_prob` is a final-win price/break-even probability.
- Therefore `edge_at_entry` is not a valid final-win probability edge.

There is also a fallback-pricing issue:

- If no direction-consistent moneyline is found, N09 uses `baseline_C_favorite_final_win` as `fallback_prob`.
- It then creates synthetic decimal odds as `1 / fallback_prob`.
- These synthetic odds are included in realized ROI as if they were bettable prices.

Task severity: critical for interpretation, moderate for implementation. No direct outcome leakage was found, but the label-target mismatch invalidates treating Sim B as confirmed final-win betting edge. Synthetic fallback odds should be separated from real-moneyline ROI.

Does this change the conclusion that Sim B is real? Yes. The result is a real realized historical pattern from the implemented rule, but it is not an audited proof of final-win betting edge.

## Final Severity Summary

| Finding | Severity | Effect |
|---|---|---|
| Sim B uses `n06_prob` trained on `deficit_erased` to filter bets paid on `favorite_final_win` | Critical | Invalidates final-win betting-edge interpretation |
| N04 same-label final-win model shows negative edge on the same selected games | Critical | Confirms Sim B is not a same-label extension of N04 |
| 32 of 236 high-edge bets use synthetic fallback odds | Moderate | Inflates headline ROI; all-in ROI is not pure sportsbook result |
| N09 uses raw break-even probability, not no-vig | Moderate | Requires clearer naming/framing; not cause of inflated ROI |
| No line-level timestamps in CFBD cache | Data limitation | Cannot audit exact pre-game capture timing |
| Positive ROI appears in all three held-out seasons | Verified | Result is not one-year-only, but interpretation remains constrained |

## Recommendation Before N09 Commit

Do not frame Sim B edge `>= 0.10` as the project's strongest betting-edge result without revision.

At minimum, before committing N09:

1. Separate real-moneyline bets from synthetic fallback-price bets in all betting summaries.
2. Rename `market_implied_prob` in Section 3 or document it as raw offered-price break-even probability, not no-vig market probability.
3. Reframe Sim B as a deficit-erasure-model selection heuristic unless a same-label final-win probability is used for the edge filter.
4. If the goal is final-win betting edge, rerun Sim B with a same-label final-win probability source, such as the N04/N03 favorite-final-win model, and report N06 deficit-erasure filtering only as an exploratory descriptive overlay.

The current Sim B result is worth preserving as an audit finding, but not as confirmed betting edge.
