# N11 -- Top-25 favorite stratification and market efficiency analysis

## Project-Level Closing Finding

N11 is the final test of pre-game edge in the project's historical research arc. AP ranking is real and behaves as expected: top-5 teams are favored by more, win more often, and carry higher pre-game ratings. But ranking stratification does not reveal any pre-game market inefficiency. Every ranking bucket -- top_5, top_10, top_25, and unranked -- shows actual `favorite_final_win` rates below market-implied probability, with negative held-out ROI in every bucket.

Six notebooks have now tested pre-game edge from distinct angles:

- N04: model vs pre-game market showed positive Brier improvement, but the mechanism was restricted to trigger-state probability adjustment rather than betting edge.
- N05/N06: model vs deficit x time baseline showed no structural edge.
- N07/N08: methodology refinement and uncertainty diagnostics did not surface an edge under the stricter framework.
- N09: realized betting simulation showed the always-bet-favorite strategy loses, while the same-label positive filter was underpowered.
- N10: direct fluky-deficit hypothesis testing showed markets overprice favorites by 35 percentage points in the headline condition.
- N11: AP ranking stratification shows markets overprice every ranking bucket.

Conclusion: pre-game CFB markets correctly price, and often slightly overprice, favorite comeback scenarios. There is no hidden inefficiency in any stratification dimension the project has tested. The mispricing diminishes as ranking improves -- top_5 gap **-18.3%** versus unranked gap **-27.2%** -- but never closes to neutral or positive. This is consistent with markets rewarding team strength while still compounding favorite quality slightly past actual performance.

Live in-game market edge is a separate, untested hypothesis. The project's infrastructure -- N03-N08 models, N09-N11 stratifications, calibrated probabilities, and conformal intervals -- is ready for live-data testing in 2026. The pre-game edge story is closed.

**Hypothesis A: PARTIAL DESCRIPTIVE SUPPORT.** There are **1** matched cells where a ranked bucket's Wilson lower bound exceeds the unranked Wilson upper bound. The strongest cell is top_25 at D=3 / Q1 / small_favorite: 59.6% vs unranked 34.9%.

**Hypothesis B: RANKED FAVORITES ARE LESS OVERPRICED / MORE UNDERPRICED.** The average ranked-favorite actual-minus-market gap is -21.5% versus unranked -27.2%.

N11 flags **0** candidate live-watch cells under the locked N10-style criteria.

N11 is descriptive only: no model training, no feature selection, and no threshold tuning. It uses cached AP rankings and the committed N10 trigger-level market probabilities.

## Ranking Data Sanity Checks

Ranking data passed the locked sanity checks: top-5 favorites are more heavily favored on average, have higher pre-game Elo, and top-5 teams win more often across cached games than unranked teams. AP records with 26 rows are retained as top-25 cutoff ties.
| bucket | n_events | mean_spread | mean_fav_rating | all_game_win_pct | unique_fav_teams |
| --- | --- | --- | --- | --- | --- |
| top_5 | 649 | -13.5593 | 2018.2003 | 0.8175 | 29 |
| top_10 | 643 | -11.1664 | 1866.1198 | 0.7100 | 47 |
| top_25 | 1888 | -9.4928 | 1738.5016 | 0.6672 | 91 |
| unranked | 8232 | -7.4543 | 1506.3230 | 0.4500 | 132 |

## Ranking Bucket Summary

| bucket | n_events | n_games | final_win | deficit_erased | no_vig | final_minus_market | heldout_roi |
| --- | --- | --- | --- | --- | --- | --- | --- |
| top_5 | 649 | 281 | 58.9% | 74.0% | 77.2% | -18.3% | -24.7% |
| top_10 | 643 | 263 | 51.3% | 72.5% | 74.3% | -22.9% | -33.9% |
| top_25 | 1888 | 736 | 48.4% | 66.7% | 71.5% | -23.1% | -31.0% |
| unranked | 8232 | 3029 | 40.3% | 61.2% | 67.6% | -27.2% | -43.2% |

## Hypothesis A -- Matched Comeback Rates

**Hypothesis A: PARTIAL DESCRIPTIVE SUPPORT.** There are **1** matched cells where a ranked bucket's Wilson lower bound exceeds the unranked Wilson upper bound. The strongest cell is top_25 at D=3 / Q1 / small_favorite: 59.6% vs unranked 34.9%.
| cell | bucket | ranked_n | unranked_n | ranked_rate | unranked_rate |
| --- | --- | --- | --- | --- | --- |
| D=3 / Q1 / small_favorite | top_25 | 57 | 381 | 59.6% | 34.9% |

## Hypothesis B -- Market Efficiency By Ranking

**Hypothesis B: RANKED FAVORITES ARE LESS OVERPRICED / MORE UNDERPRICED.** The average ranked-favorite actual-minus-market gap is -21.5% versus unranked -27.2%.
| bucket | actual_minus_market | bootstrap_ci | heldout_roi | heldout_roi_ci |
| --- | --- | --- | --- | --- |
| top_5 | -18.3% | [-24.6%, -11.9%] | -24.7% | [-40.4%, -8.0%] |
| top_10 | -22.9% | [-29.4%, -16.3%] | -33.9% | [-49.0%, -17.6%] |
| top_25 | -23.1% | [-26.9%, -19.1%] | -31.0% | [-41.1%, -20.4%] |
| unranked | -27.2% | [-29.0%, -25.4%] | -43.2% | [-47.8%, -38.4%] |

## Candidate Live-Watch Cells

No ranking-stratified cell satisfies the locked candidate live-watch rule.

## Inverse Sanity Check

Definition: `ranking_bucket=unranked AND spread_bucket=small_favorite AND time_bucket=Q4`.
Events/games/seasons: **170 / 119 / 10** (`reliable`).
Favorite final-win rate: **8.8%**; actual-minus-market **-45.1%** with CI **[-50.2%, -39.3%]**.
Held-out real-moneyline ROI: **-73.6%** on **51** bets, CI **[-93.5%, -47.1%]**.
Methodology warning: **False**.

## Honest Interpretation

Ranking status behaves like real football strength, and N11 finds one narrow matched-cell descriptive separation, but it does not recover the project's pre-game edge hypothesis. Every ranking bucket remains below market-implied probability on aggregate, held-out ROI is negative in every ranking bucket, and zero cells satisfy the locked candidate live-watch rule. The central N10 conclusion remains intact: AP ranking stratification does not reveal a hidden pre-game market inefficiency for favorite comeback scenarios.

