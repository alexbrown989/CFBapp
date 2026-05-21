# N09 -- Trigger-state analysis, dashboard stratification, and counterfactual betting simulation

## Lead findings

**Primary finding:** pre-game CFB markets correctly price favorite comeback risk on average. The unfiltered always-bet-favorite strategy on trigger games produces ROI **-0.2241** with bootstrap 95% CI **[-0.2606, -0.1873]** across **1393** real-moneyline bets. This is a clean statistically significant loss, not a hidden broad betting edge against pre-game prices.

**Secondary finding, suggestive but underpowered:** the methodologically valid same-label edge filter using N03's `favorite_final_win` probability is positive at every tested edge threshold: ROI **0.4611** at edge >= 0.00 (**48** bets), **0.5202** at edge >= 0.05 (**23** bets), and **0.6499** at edge >= 0.10 (**7** bets). All CIs have positive lower bounds, but every real-moneyline sample is below the project's locked 50-event floor for rate-comparison claims. This is suggestive, not deployable. The underpower is itself informative: the same-label model rarely disagrees with pre-game markets by 10+ percentage points on trigger games, consistent with pre-game markets being well-calibrated for this subpopulation.

**Tertiary finding, real pattern with unverified mechanism:** the original N06-based Sim B track, which filters with the `deficit_erased` model while paying on `favorite_final_win`, produces ROI **0.3379** on **204** real-moneyline bets at edge >= 0.10. This is methodologically a cross-label heuristic, not a validated final-win betting edge. N04's same-label model showed negative edge on these same games, so the favorable realized outcomes are not explained by a probability framework we have validated. Preserve it as a research curiosity, not a deployable strategy.

**Project-level implication:** pre-game prices are approximately efficient on CFB trigger games. The same-label filter shows suggestive directional betting edge but needs much more data to confirm. The live-data path is the route to accumulating enough trigger events with market prices to test whether that same-label signal is real. Future N10+ live-data scaffold work should prioritize collecting trigger events and live market prices across multiple seasons.

## Section 1 -- baseline_C structural analysis

**Structural edge finding:** baseline_C remains the dominant structural signal. On held-out `deficit_erased`, the two-feature deficit + time logistic model has Brier **0.1763**, baseline_C has Brier **0.1751**, and N06 has Brier **0.1786**. N06 deviations from baseline_C are not reliably better: among the 100 highest-disagreement triggers, actual `deficit_erased` rate is **0.5700**, mean N06 probability is **0.4057**, and mean baseline_C probability is **0.4886**; closer aggregate rate = **baseline_C**.

The two-feature deficit + time-bucket logistic recovers most of N06's predictive performance on `deficit_erased` (Brier **0.1763** versus N06 **0.1786**). This continues the N05 through N08 structural-edge pattern: the model's deviations from baseline_C are not reliably better than the simple structural lookup. The 20-cell baseline_C table and season-stability diagnostics are written to `n09_baseline_analysis.json`.

## Section 2 -- dashboard stratifications

**Dashboard finding:** rich descriptive stratifications across **11** game-state dimensions are pre-computed for **11,412** trigger events and are ready for dashboard consumption. Dimensions include turnover composition, short-field proxy, explosive composition, EPA differential, pace, possessions remaining, team tempo/pass style, favorite and dog momentum, conference/neutral context, week bucket, and the single authorized deficit x favorite-momentum cross dimension.

Numeric stratification buckets are empirical quintiles of the full corpus. Composition buckets keep zero/no-points cases separate and use quintiles over the positive population. Remaining concentration is intentional only for intrinsic categorical buckets such as first-two-games and early-game momentum. Every bucket table reports both labels (`favorite_final_win` and `deficit_erased`) with `n_events`, `n_games`, `n_seasons`, `thin_flag`, and Wilson intervals.

## Section 3 -- counterfactual betting simulation against pre-game prices

**Audit-corrected betting-edge framing:** Sim B is reported two ways. The methodologically valid version uses N03's same-label `favorite_final_win` probability as `model_prob_final_win`. The N06 version remains as a `deficit_erased` selection heuristic and is not a final-win probability-edge claim. Headline ROI uses real-moneyline rows only; synthetic fallback prices are reported separately and are not treated as cached sportsbook results.

Unfiltered real-moneyline trigger strategy: **1393** bets, ROI **-0.2241**, bootstrap 95% CI **[-0.2606, -0.1873]**. This tests betting edge against pre-game prices only; it does not test live market edge.

Same-label Sim B (`B_final_win_model_edge`, threshold 0.00, real moneyline only): **48** bets, win rate **0.8333**, ROI **0.4611**, CI **[0.2584, 0.6407]**.
Same-label Sim B (`B_final_win_model_edge`, threshold 0.05, real moneyline only): **23** bets, win rate **0.8696**, ROI **0.5202**, CI **[0.2420, 0.7490]**.
Same-label Sim B (`B_final_win_model_edge`, threshold 0.10, real moneyline only): **7** bets, win rate **0.8571**, ROI **0.6499**, CI **[0.0686, 1.0112]**.
The same-label Sim B track is positive but underpowered: its real-moneyline sample is below the 50-bet reporting threshold at every tested edge threshold, so it is descriptive rather than bankable evidence.
Same-label Sim B all-price comparison at threshold 0.10: **12** bets, ROI **0.5867**, CI **[0.1252, 0.9519]**.
N06 deficit-erased heuristic Sim B at threshold 0.10, real moneyline only: **204** bets, win rate **0.8186**, ROI **0.3379**, CI **[0.2464, 0.4237]**. This is a selection heuristic, not a same-label betting-edge claim.
N06 heuristic synthetic-fallback subset at threshold 0.10: **32** rows, ROI **0.9805**. This subset is not part of headline betting-edge claims because the prices are baseline-derived synthetic odds.

Small-sample warning: 5 Sim B/D threshold(s) produced fewer than 50 selected games. They remain in the data but should be treated as descriptive curiosities only.

## Deliverables

- `n09_trigger_state_stratifications.parquet`: 11,412 rows.
- `n09_baseline_analysis.json`: Section 1 + Section 2 aggregate tables.
- `n09_betting_simulations.parquet`: 12,074 bet/staking rows.
- `n09_betting_summary.json`: Section 3 aggregate metrics, CIs, bankroll trajectories, and sample-size flags.

## Honest interpretation

N09 is application-facing, but it does not soften the research conclusion. Section 1 remains clean: baseline_C captures the dominant structural signal. Section 2 is descriptive dashboard data with empirical buckets and sample-size flags. Section 3 says pre-game markets are efficient on average, the same-label filter is promising but underpowered, and the N06 heuristic is a real pattern without a validated final-win mechanism. Live data collection is the next required step.
