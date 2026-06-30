# N12 Lookup Key Schema

N12 stores every estimate in long format. Live callers should build the same `lookup_key` strings below and filter by `metric_name` and `label`.

## Columns

| column | meaning |
| --- | --- |
| `lookup_key` | deterministic bucket or trigger-state identifier |
| `dimension_set` | pipe-delimited dimension names used by the key |
| `metric_name` | estimate type |
| `label` | `favorite_final_win` or `deficit_erased` |
| `value` | probability/rate estimate |
| `ci_lower`, `ci_upper` | confidence bounds when present |
| `n_events`, `n_games`, `n_seasons` | supporting sample size when applicable |
| `reliability_flag` | `reliable`, `thin`, `unreliable`, or `n_a` |
| `source_notebook`, `source_artifact` | provenance |

## Key Rules

- `baseline_c_rate`: `deficit={d}|time={t}` where `t` is one of `Q1`, `Q2-first-half`, `Q3`, `Q4`. Helper calls normalize `Q2` to `Q2-first-half` only for this metric.
- `conditional_rate_full`: `fluke={f}|deficit={d}|time={t}|spread={s}` using N10 buckets.
- `ranking_rate`: `rank={r}|deficit={d}|time={t}|spread={s}` using N11 AP ranking buckets.
- `market_no_vig_historical`: uses the same key as the N10 conditional or N11 ranking bucket that produced the historical market mean.
- `n06_calibrated_prob`, `conformal_lower`, `conformal_upper`, `stern_winston_state_price`: `scheme={scheme}|fold={fold}|game_id={game_id}|trigger_sequence={seq}|deficit={d}` for historical trigger-event provenance rows.

## Live Scoring

`research/notebooks/_lib_lookup.py::score_live_trigger(feature_dict)` uses the Scheme E N06 fitted state by default: train 2015-2023, validate 2024. For historical reproduction, pass `scheme` and `fold` explicitly.

Missingness indicators are inferred from raw core-feature nulls unless the caller supplies indicator columns directly. This is required because committed N06 prediction parquets preserve raw feature nulls but do not include indicator columns.
