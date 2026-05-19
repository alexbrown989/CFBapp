# Research Summary: CFB Live Edge Journal

This document summarizes the completed research phase of CFB Live Edge Journal. It is written for readers who want the story, findings, and limitations without reading the notebooks. The detailed audit trail lives in `research/corrections_log.md`; feature-level documentation lives in `research/results/feature_validation.schema.md`.

## Hypothesis and Motivation

The project began with a sports-market question: when a pre-game college football favorite falls behind in-game, can a model tell the difference between a favorite that is still likely to recover and a favorite that is genuinely being outplayed?

The motivating betting hypothesis was live-line oriented. If the market overreacts to early-game deficits, or underreacts to certain kinds of underdog control, then a trigger-state model might identify live moneylines with positive expected value. The model would need to condition on features that separate "early variance" from "real signal": explosive plays, sustained drives, turnovers, red-zone failures, down-distance efficiency, and game context.

The available data forced an important reframing. CFBD provides historical game, play, and pre-game line data, but not continuous historical live in-game moneyline snapshots for the 2022-2024 validation seasons. That means the completed research phase cannot prove live betting edge. Instead, N04 asks a cleaner predictive question:

> Given that the favorite has entered a trigger state, does the model's calibrated trigger-state probability predict the final game outcome more accurately than the pre-game market's implied probability did?

This is still a meaningful research test. If the answer is no, the model has not learned useful in-game information beyond the pre-game market. If the answer is yes, the model has validated predictive edge over stale pre-game consensus, and the next question becomes whether that edge survives comparison to actual live prices collected going forward.

## Methodology Overview

The research proceeded in three stages.

### Phase 0: Trigger and Feature Engineering

Notebook 01 defines the trigger-event corpus. A trigger event is a game state where the pre-game favorite trails by one of the configured deficit thresholds. The final corpus contains **11,416** trigger events over **7,854** unique trigger plays. Some plays satisfy multiple deficit thresholds: for example, a favorite trailing by 10 also satisfies the D=3 and D=7 trigger definitions. That multi-threshold structure became important later in N03.

Notebooks 02a through 02g test feature families one at a time. The feature families are:

- 02a: baseline efficiency and game-progress features
- 02b: opening-drive shock
- 02c: explosive versus sustained production
- 02d: turnovers and short fields
- 02e: red-zone failure
- 02f: down-distance efficiency
- 02g: week and home/neutral context

Across Phase 0, the project tested **40** candidate feature groups. **30** passed the R6 stability criterion and entered N03. R6 was deliberately permissive: a feature entered the candidate pool if it had at least 2 of 3 Brier-improving folds. No feature was pre-filtered out for redundancy or small magnitude. Instead, redundancy tags and magnitude skepticism were documented for N03 to handle through regularization and diagnostics.

### N03: Walk-Forward Probability Modeling

N03 fits the production probability model. The locked primary model is L1 logistic regression with `C=1.0`, standardized features, and isotonic calibration fit only on the validation slice for each walk-forward fold.

The walk-forward windows are:

| Fold | Train years | Validation year | Test year |
|---:|---|---:|---:|
| 2022 | 2015-2020 | 2021 | 2022 |
| 2023 | 2015-2021 | 2022 | 2023 |
| 2024 | 2015-2022 | 2023 | 2024 |

N03 uses **30** R6-validated Phase 0 features plus `fav_deficit` as a protected structural conditioning variable, for **31** core model features. It also adds missingness indicators for nullable features above the 5% null threshold. The final post-imputation model matrix has **50** columns: 31 core features plus **19** missingness indicators.

A key N03 design correction was deduplicating training to unique trigger plays. Training directly on all 11,416 trigger events would count the same play multiple times when it satisfied multiple deficit thresholds. N03 therefore trains on **7,854** unique trigger plays using the lowest qualifying deficit as the structural `fav_deficit`, then replicates held-out plays back to their qualifying deficit thresholds at scoring time for N04.

### N04: Model Versus Pre-Game Market

N04 compares N03 calibrated trigger-state probabilities to pre-game market probabilities from cached CFBD `/lines` data. Moneyline is preferred when available. If moneyline is unavailable or direction-conflicted, N04 falls back to an empirical spread-to-win-probability conversion fit on 2015-2021 games.

The primary metric is per-trigger Brier improvement:

```text
brier_improvement = brier_market - brier_model
```

Positive means the model's trigger-state probability beat the pre-game market probability. Confidence intervals use cluster bootstrap by `game_id` with **10,000** resamples.

Market coverage was sufficient for a clean test: N04 evaluated **3,857** test trigger events per Scheme U/W2 across **1,451** unique games with **zero** missing market-probability rows. Fallbacks were limited to **14** no-moneyline games and **38** moneyline side-conflict games.

## Key Methodological Findings

### The `chrono_key` Lookahead Leak Was Real

The most important Phase 0 correction was the lookahead leak post-mortem. Early feature code used CFBD `playNumber` as if it were a global chronological ordering key. It is not. `playNumber` can reset or behave inconsistently across drives, which means filtering "plays before trigger" by `playNumber < trigger_playNumber` can accidentally include future plays or omit real past plays.

The fix was the canonical `_chrono_key`, based on period, clock, and tie-breakers. It is documented in `research/notebooks/_lib_chrono.py` and `research/corrections_log.md`. The correction materially changed feature verdicts. Most notably, a previously celebrated 02b calibration standout, `fav_def_epa_first_drive`, collapsed from a leaky PASS to a corrected FAIL. Conversely, `plays_so_far` recovered as the strongest 02a signal after the chronological filter was fixed.

This was not cosmetic. For 02f down-distance rates, D10 disagreement between the canonical `_chrono_key` and leaky `playNumber` filters affected roughly **44-45%** of micro-quantized trigger rows per down-distance column, the widest Phase 0 footprint observed. The project treats the leak correction as load-bearing.

### 2024-Fold Weakness Was a Project-Wide Warning

Across Phase 0, many features performed worse on the newest 2024 fold than on 2022 or 2023. That pattern led to the explicit N03 policy of reporting multiple weighting schemes and treating 2024 as deployment-proximate evidence rather than just another fold.

N03 partially confirmed the concern. The model improved materially over the pre-game alpha baseline in 2022, was essentially flat in 2023, and did **not** beat alpha on 2024 Brier:

| Test fold | Alpha Brier | N03 Brier | Delta Brier |
|---:|---:|---:|---:|
| 2022 | 0.245734 | 0.221332 | +0.024402 |
| 2023 | 0.216000 | 0.215187 | +0.000812 |
| 2024 | 0.218155 | 0.220206 | -0.002051 |

That made N04 especially important. Encouragingly, N04's model-vs-market Brier improvement was consistent across all three test folds, including **2024 +0.05651**.

### No Feature Was a 3/3 ECE Savior

Calibration was the hardest part of the project. Several features improved Brier without consistently improving Expected Calibration Error. After the leak correction, no Phase 0 feature deserved to be treated as a reliable 3/3 Brier plus 3/3 ECE calibrator. That shaped the N03 decision to use isotonic calibration per fold and to report ECE as a primary diagnostic rather than assume feature engineering would solve calibration directly.

N03 weighted ECE was usable but fragile: Scheme U weighted ECE **0.041820** and weighted AUC **0.689016**. The all-fold number hides per-deficit weakness: D=10 ECE **0.0872** and D=14 ECE **0.0609**.

### Magnitude Skepticism Mattered

R6 was an entry rule, not proof that every passing feature carried practical signal. Several features passed mechanically with effects near the noise floor. For example, 02g context features produced an honest negative finding: `season_phase_bowl` and `fav_is_home` passed R6 mechanically, but their per-fold Brier magnitudes were too small to treat as load-bearing signal.

The project used a soft skepticism threshold of roughly **+0.005** Brier improvement. Features below that level were not discarded automatically, but their caveats were documented. N03 still allowed L1 regularization, permutation importance, and ablation to decide. The final production model retained all 31 core features under the strict three-signal pruning rule, which is informative in its own right: the pruning rule was conservative, and many engineered features were marginal relative to structural game-state variables.

## Results

### Phase 0 Feature Set

Phase 0 tested **40** candidate feature groups and passed **30** into N03. The strongest and most interpretable surviving signals include game progress (`plays_so_far`), favorite deficit structure, underdog drive efficiency, explosive-play production, short-field and turnover context, red-zone outcomes, and down-distance rates.

The feature set is documented in:

- `research/results/feature_validation.csv`
- `research/results/feature_validation.schema.md`
- `research/results/_02g_full_correlation_matrix.csv`

The schema sidecar is the best place to inspect individual feature caveats, redundancy tags, and per-fold magnitudes.

### N03 Model Performance

The production N03 model is a unified L1 logistic regression at `C=1.0` with isotonic calibration. Scheme U selected all **30** R6-validated features plus protected `fav_deficit`, and retained **19** missingness indicators.

Weighted N03 held-out performance:

| Scheme | Weighted Brier | Weighted ECE | Weighted AUC |
|---|---:|---:|---:|
| U | 0.218908 | 0.041820 | 0.689016 |
| W2 | 0.219233 | 0.042243 | 0.685529 |

N03 was not an edge-grade result by itself. Its discrimination was real but modest, and the 2024 alpha comparison was weak. Two structural variables dominated the fitted signal: `plays_so_far` with weighted signed standardized coefficient **-0.687**, and `fav_deficit` with **-0.629**. The engineered features contributed incrementally, but most were not load-bearing in isolation.

The N03 sensitivity checks supported the locked architecture without making it look magical. No C value in `{0.1, 0.5, 1.0, 2.0, 10.0}` strictly dominated `C=1.0`. Bin-specific models for D<=7 and D>=10 worsened calibration relative to the unified model.

### N04 Market Comparison

N04 produced the strongest positive result of the project:

| Scope | Brier improvement | 95% CI |
|---|---:|---:|
| U all folds | +0.05847 | [+0.04211, +0.07465] |
| W2 all folds | +0.05847 | [+0.04210, +0.07429] |

Per-fold improvements were consistent:

| Fold | Brier improvement |
|---:|---:|
| 2022 | +0.06491 |
| 2023 | +0.05418 |
| 2024 | +0.05651 |

The result is best understood as calibration improvement over stale pre-game probability. The market still ranked teams better overall: market AUC **0.6812** versus model AUC **0.6650**. But market calibration was poor for the trigger-state subpopulation: market ECE **0.24840** versus model ECE **0.03484**.

The per-deficit pattern is the strongest mechanistic evidence:

| Deficit | Brier improvement |
|---:|---:|
| D=3 | -0.00570 |
| D=7 | +0.03134 |
| D=10 | +0.09147 |
| D=14 | +0.16507 |
| D=21 | +0.34131 |

That monotonic increase is exactly what a useful trigger-state model should show. The deeper the favorite's deficit, the more stale the pre-game probability becomes, and the more the trigger-state model corrects it.

## Limitations

The biggest limitation is live-line data. N04 validates predictive edge versus pre-game market consensus, not live betting edge. Historical live in-game moneyline data is not available for the 2022-2024 corpus. Therefore, the project cannot yet answer whether the model would beat real live prices at the moment each trigger occurred.

The tertiary betting simulation is a useful warning. At the primary deployment-context setting - edge threshold **+0.08**, **25% Kelly**, D=21 excluded - the favorite-side policy produced **89** bets, **35.96%** win rate, and **-33.27% ROI**. That does not contradict the N04 result. It says a model can improve on stale pre-game probability without producing a profitable favorite-side strategy against prices that are not actual live prices.

Calibration remains fragile. N03 all-fold calibration looked acceptable in aggregate, but D=10 and D=14 were materially weaker. Any deployment system would need conservative stake sizing, per-deficit monitoring, and live calibration checks before risking capital.

The model's discrimination is also structurally dominated. `plays_so_far` and `fav_deficit` carry the largest standardized coefficients. That is not a bug - game progress and deficit are exactly the variables that should matter - but it means many engineered features are marginal contributors rather than standalone edges.

Finally, the research depended on careful corrections. The `chrono_key` leak, 2024-fold weakness, no-3/3-ECE-calibrator finding, and magnitude-skepticism policy all changed the interpretation. Readers should treat `research/corrections_log.md` as part of the result, not as incidental project history.

## What's Next

The research phase is complete. The next phase should not be another round of historical polishing unless new historical data becomes available. The important next steps are:

1. **Collect live in-game lines going forward.** This is the only way to test whether predictive edge survives against actual live prices.
2. **Validate deployment behavior by deficit bin.** D=10 and D=14 need special attention because N03 calibration was weakest there, even though N04 Brier improvement was strong.
3. **Build conservative deployment analytics before betting automation.** Any future staking should start with observation, calibration monitoring, and paper trading.
4. **Create frontend visualization.** A useful app would show trigger events, model probability, market probability when available, deficit-bin diagnostics, and calibration drift over time.

Project conclusion:

> Predictive edge versus pre-game market consensus is validated; live-line betting edge remains untested.

That is a successful research outcome. The methodology found real in-game information, but the betting question remains open until live market data exists.
