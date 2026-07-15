# N13 Stage 2 Scoring Verification

Date: 2026-07-14

## Acceptance Result

PASS. Five cached 2024 triggers spanning all five deficit thresholds, Q1-Q4, and varied spread/ranking/fluke contexts were scored through `live.scoring.score_trigger()`.

- Tier 1 and Tier 2: exact equality with independently filtered rows from `n12_probability_lookup.parquet` for estimate value, sample sizes, and reliability flag.
- Tier 3: maximum absolute difference versus committed N06 calibrated predictions = `0` (required `< 1e-6`).
- Runtime parity guard: maximum absolute difference across identical cached feature snapshots = `0.0`.
- Synthetic drift check: changing only `plays_so_far` by 1 correctly sets `tier3_suspect=true` and identifies only that feature.
- Additive log compatibility: a 16-field Stage 1 record remains readable, with Stage 2 fields exposed as null.
- Tier 3 missing-feature guard: removing `dog_points_off_turnovers` suppresses N06 and leaves Tier 2 active; no imputation occurs.
- Network/API calls: 0.

| Game | Deficit | Time | Spread | Rank | Fluke | Tier | baseline_C final win | baseline_C deficit erased | N06 calibrated | abs diff |
|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 401628332 | D=3 | Q3 | moderate_favorite | top_25 | mixed_lead | 3 | 0.459559 | 0.716912 | 0.558823529412 | 0 |
| 401677090 | D=7 | Q4 | small_favorite | unranked | mixed_lead | 3 | 0.173267 | 0.351485 | 0.236363636364 | 0 |
| 401677086 | D=10 | Q1 | moderate_favorite | unranked | fluky_lead | 3 | 0.389180 | 0.589878 | 0.518518518519 | 0 |
| 401677082 | D=14 | Q2 | moderate_favorite | unranked | fluky_lead | 3 | 0.268482 | 0.369650 | 0.236363636364 | 0 |
| 401628374 | D=21 | Q1 | small_favorite | top_5 | fluky_lead | 3 | 0.258065 | 0.387097 | 0.142857142857 | 0 |

## Tier Behavior

Tier 1 is always present and remains the primary score+clock estimate. Tier 2 rows are labeled historical descriptive and retain N12 sample sizes, reliability, confidence bounds, and source provenance. Tier 3 appears only for explicitly certified cached historical feature dictionaries in this verification. Normal Stage 2 runtime suppresses Tier 3 with `unavailable - no live play feed`.

The parity guard comparison and append-only drift schema are ready, but the first weeks of the 2026 season remain the live-feed certification window. Until that window is clean, operational decisions should lean on Tier 1.
