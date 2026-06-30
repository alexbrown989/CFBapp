# N12 -- Unified Probability Lookup Layer

**Purpose:** consolidate committed N03-N11 probability estimates, historical rates, model state, conformal intervals, and analytical price references into a long-format lookup layer for N13. N12 does not train, refit, fetch data, or create new research estimates.

## Critical Reproduction Gate

`_lib_lookup.score_live_trigger()` re-scored **7,708** committed N06 held-out rows from raw feature values.

- Raw probability max abs diff: `3.33066907388e-16`
- Calibrated probability max abs diff: `2.40918396344e-14`
- Tolerance: `1e-06`
- Passed: `True`

This is independent of the N06 re-export gate: it verifies the consolidated query helper, not just the N06 build script.

## Row Counts By Metric

| metric_name | rows |
| --- | ---: |
| `baseline_c_rate` | 40 |
| `conditional_rate_full` | 536 |
| `conformal_lower` | 3,854 |
| `conformal_upper` | 3,854 |
| `market_no_vig_historical` | 571 |
| `n06_calibrated_prob` | 7,708 |
| `ranking_rate` | 606 |
| `stern_winston_state_price` | 7,708 |

## Provenance Map

- `baseline_c_rate`: N09 baseline_C 20-cell table.
- `n06_calibrated_prob`: N06 committed held-out prediction parquet.
- `conformal_lower` / `conformal_upper`: N08 diagnostic prediction parquet.
- `stern_winston_state_price`: N08 CFB-specific Stern-Winston diagnostic probabilities.
- `conditional_rate_full` and N10 `market_no_vig_historical`: N10 tier-3 conditional matrix.
- `ranking_rate` and N11 `market_no_vig_historical`: N11 matched ranking matrix.
- Live scoring state: N06 full fitted-state provenance export, Scheme E as default deployment model.

## Conformal Uncertainty

The deployment conformal q-hat is **0.770**, producing wide prediction intervals consistent with N08's finding that per-trigger predictions carry large uncertainty. The live system (N13) should surface these intervals alongside point probabilities, not hide them. The width is the mathematical basis for conservative bet sizing: a point estimate of 40% with this interval width does not justify aggressive staking. This is honest uncertainty carried forward from committed research, not a defect in the lookup layer.

## No New Estimates

N12 reshapes existing committed artifacts and exports complete scoring state. The only calculations performed are deterministic key construction, schema normalization, and reproduction verification.
