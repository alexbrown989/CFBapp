# Code Correctness Audit v1

Date: 2026-05-20

Scope: Interpretation A only. This audit verifies whether the existing local research pipeline is doing what the project documentation says it does across CFBD cache ingestion, feature computation, and N03-N07 methodology implementation. It intentionally does not propose methodology changes, performance refactors, or preference-based rewrites.

State checked before audit:

- `HEAD`: `5c11bf5a8f6c19965e9333ad0afaf47a9ffafb7b`
- `origin/main`: `5c11bf5a8f6c19965e9333ad0afaf47a9ffafb7b`
- Working tree: clean except expected untracked executed notebooks and 02g diagnostic artifacts.

## Phase 1 Findings: CFBD Ingestion Correctness

Verdict: verified. No critical ingestion discrepancy found.

### Cache file chain

Observed cache chain:

| Data type | Raw cache location | Loader/parser | Loaded structure |
|---|---|---|---|
| Play-by-play | `research/data/cache/cfbd__plays__*.json` | `json.loads(...read_text(encoding="utf-8"))`; e.g. 02a-02f, N05 `load_all_cached("plays")` | `list[dict]`, then grouped by `game_id` into `plays_by_game` |
| Game metadata | `research/data/cache/cfbd__games__*.json` | `json.loads(...read_text(encoding="utf-8"))`; 02g and N05 metadata joins | `list[dict]`, keyed by `id` / `game_id` |
| Lines | `research/data/cache/cfbd__lines__*.json` | N04 `load_line_cache`; provider rows inside each game record | `line_by_game: dict[int, dict]` |
| Drives | `research/data/cache/cfbd__drives__*.json` | 02b-02f/N07 cache reloads | `list[dict]`, grouped by `game_id` |

Cache inventory from `research/data/cache/cfbd_call_log.csv`:

- `/plays`: 5,655 cache calls
- `/games`: 100 cache calls
- `/lines`: 100 cache calls
- `/drives`: 561 cache calls
- Other support endpoints: ratings, teams, conferences, archive.

### Spot-check parsing

Five random play-by-play games were inspected from raw cache rows through parsed dictionaries:

| game_id | Raw cache excerpt | Parsed structure check |
|---:|---|---|
| `401309890` | Oregon at Washington; first play P1 15:00 kickoff; last play P4 0:00 end of game | 174 parsed plays; `offense`, `defense`, `playType`, `playText`, `clock`, `driveNumber`, `playNumber`, and scores present |
| `401013446` | New Mexico State at BYU; final 10-45 | 213 parsed plays; team labels and score state present |
| `401309585` | Toledo at Bowling Green; final 49-17 | 182 parsed plays; clock and drive/play sequence parseable |
| `400868998` | Georgia at Missouri; final 28-27 | 214 parsed plays; descriptions and scoring fields present |
| `401112100` | Kansas at TCU; final 14-51 | 183 parsed plays; terminal play and scoring sequence parseable |

Five random game metadata rows were inspected:

- All sampled records had `id`, `season`, `week`, `seasonType`, `startDate`, `neutralSite`, `conferenceGame`, home/away teams, and final scores.
- `startDate` values are ISO-like UTC strings and parseable.
- Team labels in metadata matched labels in the corresponding play rows for sampled games.

Five random line records were inspected:

| game_id | Raw cache excerpt | Parsed structure check |
|---:|---|---|
| `400787341` | 2015 Central Michigan/Akron, provider `teamrankings`, spread `3`, formatted `Central Michigan -3` | Spread present; no moneyline |
| `401022526` | 2018 Boise State/Wyoming, provider `teamrankings`, spread `15.5` | Spread present; no moneyline |
| `401635589` | 2024 Miami/Louisville, provider `DraftKings`, away ML `-185`, home ML `154`, spread `4.5` | Two-sided moneyline and spread parse correctly |
| `401644736` | 2024 Buffalo/Northern Illinois, provider `Bovada`, away ML `390`, home ML `-550`, spread `-13` | Two-sided moneyline and spread parse correctly |
| `400938895` | 2017 Massachusetts/BYU, provider `teamrankings`, spread `-3.5` | Spread present; no moneyline |

### Chronological ordering

Canonical ordering is `_chrono_key(p) = (period, period_seconds_elapsed, driveNumber, playNumber)` in `research/notebooks/_lib_chrono.py:31-54`.

Ten random games were sorted by `_chrono_key`; all had zero inversions after sorting. Manual narrative spot-checks showed coherent game flow: kickoff, drive plays, scoring/PAT where present, next kickoff/drive, and terminal game events. Plays with identical clock values were ordered by `(driveNumber, playNumber)`, which matches the documented correction in `research/corrections_log.md:68-92`.

### Negative play-id encoding

The negative-ID observation is documented in `research/corrections_log.md:184-237`; `_chrono_key` explicitly does not use `play.id` (`research/notebooks/_lib_chrono.py:31-54`).

Audit sample:

- 115 games in the play cache contain negative `play.id` values.
- 19,828 cached plays use that negative integer-string encoding.
- Three sampled negative-ID games (`401310698`, `401310696`, `401309600`) sorted coherently under `_chrono_key`; negative IDs did not affect order because ordering uses period, elapsed seconds, drive number, and play number only.

### Lines direction-conflict handling

N04 direction-consistency logic is implemented per-provider in `research/notebooks/_build_n04.py:303-423`:

- `consistent_moneyline_rows` excludes a provider if its moneyline-implied favorite conflicts with the spread-implied favorite (`research/notebooks/_build_n04.py:345-363`).
- Direction-consistent `consensus` rows are preferred, then direction-consistent sportsbook rows (`research/notebooks/_build_n04.py:365-377`).
- If no usable moneyline row remains but a spread conversion is available, the record falls back to spread conversion and records `fallback_reason` (`research/notebooks/_build_n04.py:401-423`).

Artifact check from `research/results/n04_validation_results.parquet`:

- `market_status=moneyline`: 8,706 rows
- `market_status=spread_conversion`: 304 rows
- Spread-conversion fallback reasons: 243 rows from `moneyline_side_conflict`, 61 rows from `no_moneyline_available`.
- Unique test games using spread conversion: 38 direction-conflict games and 14 no-moneyline games.
- No `moneyline_side_conflict` fallback rows had zero recorded conflict providers; no `no_moneyline_available` fallback rows had positive conflict-provider counts.

## Phase 2 Findings: Feature Computation Correctness

Verdict: verified with moderate documentation/provenance findings. No critical feature-computation discrepancy found.

### Chrono-key use across feature builders

Highest-risk check: no canonical feature builder was found using raw `playNumber < trigger.playNumber` as the production pre-trigger gate.

| Script | Audit result |
|---|---|
| `_build_02a.py` | Canonical feature matrix slices `plays_before` by `_chrono_key(p) < trigger_chrono_key`; feature functions consume that slice. |
| `_build_02b.py` | Canonical feature matrix uses `_chrono_key` and `assert_no_lookahead`; drive-summary features separately use completed drive metadata. |
| `_build_02c.py` | Canonical explosive/sustained features use `_chrono_key` pre-trigger slicing. |
| `_build_02d.py` | Drive-level extractors are intentionally keyed by `driveNumber < trigger_drive_in_game`; `plays_before` is chrono-sliced only for the no-lookahead gate and diff-vs-leaky check (`research/notebooks/_build_02d.py:811-839`, `research/notebooks/_build_02d.py:1129-1148`). |
| `_build_02e.py` | Red-zone/fav-yards-per-point features use the corrected chrono pre-trigger gate in the canonical path. |
| `_build_02f.py` | Down-distance efficiency uses `_chrono_key` in canonical extraction and documents the leaky `playNumber` path as diagnostic-only (`research/notebooks/_build_02f.py:522-538`, `research/notebooks/_build_02f.py:1101-1114`). |
| `_build_02g.py` | Context features do not depend on play ordering; the chrono/leaky signatures are retained only to verify Category A byte identity (`research/notebooks/_build_02g.py:325-397`). |
| `_build_n05.py` | `deficit_erased` is computed by sorting plays with `_chrono_key` and scanning only post-trigger plays (`research/notebooks/_build_n05.py:259`, `research/notebooks/_build_n05.py:323-362`). |
| `_build_n07.py` | Newly derived play-dependent rates use `_chrono_key` for `plays_before` (`research/notebooks/_build_n07.py:571-582`). |

### Feature definitions and edge cases

The audited feature definitions match the schema sidecar at the level relevant for result correctness:

- 02a baseline efficiency features are computed from pre-trigger plays and evaluated per feature against the pre-game alpha baseline.
- 02b opening-drive shock features use drive/opening-drive metadata and document drive-1 null handling.
- 02c explosive/sustained features use pre-trigger score/play bucketing.
- 02d turnover/short-field features use completed drives and the documented in-progress dog-drive edge case.
- 02e red-zone/fav-yards-per-point features preserve the documented informative-missingness structure.
- 02f down-distance efficiency uses the documented effective-distance clamp and paired insufficient-sample indicators.
- 02g context features are game/trigger metadata only.
- N07's 14 pre-registered features match the authorized list exactly (`research/notebooks/_build_n07.py:161-247`) and are computed at `research/notebooks/_build_n07.py:604-659`.

Documented edge cases are handled in code:

- 02f negative distance: corrections log documents the clamp at `research/corrections_log.md:596-602`; the 02f audit/code path uses `max(0, min(distance, yardsToGoal))` and skips/counts negative raw fields.
- N07 `dog_points_from_explosives_pct > 1`: code asserts exactly 5 cases, sets the feature to `NaN`, and turns on the paired indicator (`research/notebooks/_build_n07.py:620-626`); corrections log records it at `research/corrections_log.md:956-960`.
- N07 negative `fav_yards_per_point_ratio`: code asserts exactly 5 cases, sets the feature to `NaN`, and turns on the paired indicator (`research/notebooks/_build_n07.py:645-651`); corrections log records it at `research/corrections_log.md:962-969`.

### Row-count and null-count checks

Artifact counts matched the documented trigger-event structure:

| Artifact | Rows | Unique trigger plays | Unique trigger events |
|---|---:|---:|---:|
| `n05_descriptive_rates.parquet` | 11,416 | 7,854 | 11,416 |
| `n07_descriptive_features.parquet` | 11,412 | 7,852 | 11,412 |
| `n03_calibrated_predictions.parquet` | 7,714 | 2,657 held-out plays | 3,857 held-out events |
| `n06_calibrated_predictions.parquet` | 7,708 | 2,656 held-out plays | 3,854 held-out events |
| `n07_expanded_model_predictions.parquet` | 7,708 | 2,656 held-out plays | 3,854 held-out events |

Feature-validation artifact counts:

- `research/results/feature_validation.csv`: 120 rows.
- Unique `passed_stability=True` features: 30.
- Passed rows: 90 (30 features x 3 test folds).
- Version row counts match notebook candidate counts:
  - 02a: 18 rows
  - 02b: 30 rows
  - 02c: 24 rows
  - 02d: 12 rows
  - 02e: 9 rows
  - 02f: 12 rows
  - 02g: 15 rows

N03/N06 missingness handling matches the locked policy:

- Indicator threshold is >5% null rate in the full corpus.
- Existing Phase 0 indicators are reused where present.
- Train-fold medians are fit in `fit_preprocessor(train_df, ...)` and then applied to train/validation/test (`research/notebooks/_build_n03.py:571-596`, `research/notebooks/_build_n06.py:622-647`).
- N03 and N06 specs both report 31 semantic core features, 19 indicator columns, and 50 post-imputation model columns (`research/results/n03_model_spec.json:5-7`, `research/results/n03_model_spec.json:824-825`; `research/results/n06_model_spec.json:8-10`, `research/results/n06_model_spec.json:828-829`).

### Redundancy-tag check

Pearson redundancy tags mostly match the canonical cross-feature correlation matrix:

| Tag | Matrix check |
|---|---:|
| `dog_off_epa_per_play -> fav_def_epa_per_play` | rho = +1.000 |
| `dog_third_down_success_rate -> dog_avg_drive_yards` | rho = +0.647 |
| `fav_red_zone_tds -> plays_so_far` | rho = +0.650 |
| `fav_red_zone_trips -> plays_so_far` | rho = +0.781 |

Finding MOD-01 below covers the one provenance mismatch: `dog_def_epa_per_play` has a redundancy tag in `feature_validation.csv`, but it is not represented in the 33x33 `_02g_full_correlation_matrix.csv` because it failed stability and is outside the validated feature matrix.

## Phase 3 Findings: Methodology Correctness

Verdict: verified. No critical methodology-implementation discrepancy found.

### Walk-forward windows

N03 and N06 implement the locked windows:

- Train 2015-2020, validate 2021, test 2022
- Train 2015-2021, validate 2022, test 2023
- Train 2015-2022, validate 2023, test 2024
- Scheme E: train 2015-2023, validate 2024

References: `research/notebooks/_build_n03.py:125-135`; `research/notebooks/_build_n06.py:133-143`.

N07 implements the same three walk-forward windows for the expansion test (`research/notebooks/_build_n07.py:110-113`). N07 did not specify a Scheme E deliverable, so no Scheme E absence is counted as a discrepancy.

### Calibration

Isotonic calibration is fit on validation slices only and then applied to held-out test slices:

- N03: `calibrator.fit(raw_val, y_val)` at `research/notebooks/_build_n03.py:674-675`; deployment/XGBoost diagnostic similarly uses validation raw probabilities at `research/notebooks/_build_n03.py:1228-1229`.
- N06: `calibrator.fit(raw_val, y_val)` at `research/notebooks/_build_n06.py:725-726`; XGBoost diagnostic at `research/notebooks/_build_n06.py:1279-1280`.
- N07: per-feature fit at `research/notebooks/_build_n07.py:761-762`; expanded model fit at `research/notebooks/_build_n07.py:925-926`.

No validation/test leakage found in the calibration code paths audited.

### Cluster bootstrap

Bootstrap implementation matches the documented cluster-by-game policy:

- N04: `N_BOOTSTRAPS = 10_000`, `BOOTSTRAP_SEED = 42`; bootstrap groups by `game_id` before resampling (`research/notebooks/_build_n04.py:87-88`, `research/notebooks/_build_n04.py:534-544`).
- N05: `BOOTSTRAP_RESAMPLES = 10_000`, `BOOTSTRAP_SEED = 42`; rate bootstrap aggregates by `game_id` (`research/notebooks/_build_n05.py:77-78`, `research/notebooks/_build_n05.py:145-150`).
- N06: `BOOTSTRAP_RESAMPLES = 10000`, `BOOTSTRAP_SEED = 42`; bootstrap groups by `game_id` (`research/notebooks/_build_n06.py:154-155`, `research/notebooks/_build_n06.py:1345-1349`).
- N07: `BOOTSTRAP_RESAMPLES = 10000`, `BOOTSTRAP_SEED = 42`; bootstrap groups by `game_id` (`research/notebooks/_build_n07.py:101-102`, `research/notebooks/_build_n07.py:407-411`).

### Baseline_C construction

Baseline_C is training-years-only and not contaminated by 2022-2024 held-out data:

- N05 uses `train_base = base[base["season"].isin(TRAIN_BASELINE_SEASONS)]`, then groups by `fav_deficit x time_bucket` (`research/notebooks/_build_n05.py:594-604`).
- N06 uses `n05_all["season"].between(2015, 2021)` before fitting the baseline tables (`research/notebooks/_build_n06.py:1363-1369`).
- N07 uses `wide_df["season"].between(2015, 2021)` before fitting baseline tables (`research/notebooks/_build_n07.py:719-737`).

### fav_deficit structural-variable exemption

N03 and N06 both mark `fav_deficit` as a structural conditioning variable and pruning-exempt:

- N03 model code/spec: `research/notebooks/_build_n03.py:929-934`, `research/notebooks/_build_n03.py:1368-1372`, `research/results/n03_model_spec.json:460-464`.
- N06 model code/spec: `research/notebooks/_build_n06.py:980-983`, `research/notebooks/_build_n06.py:1705-1709`, `research/results/n06_model_spec.json:463-467`.

N07's expanded candidate spec correctly treats the 33-feature model as 30 Phase 0 features + 2 N07 passes + protected `fav_deficit` structural feature (`research/results/n07_expanded_model_spec.json:2-17`).

### N06 label-change audit

The modeling framework elements are aligned between N03 and N06:

- Same 30 R6-validated features plus protected `fav_deficit`.
- Same L1 logistic regression model class and C=1.0 production setting.
- Same train-fold median imputation + missingness indicators.
- Same walk-forward windows and Scheme E extension.
- Same isotonic validation-only calibration structure.
- Same deduplicated one-row-per-trigger-play training structure.

N06 necessarily differs from N03 in data source/label plumbing, exclusion of the 4 N05 `deficit_erased=NaN` trigger events, baseline_C validation, and cross-label reporting. That is not counted as a methodology discrepancy because those differences are required by the N06 design.

### N07 pre-registration and Bonferroni accounting

N07 pre-registration matches the authorized 14-feature list exactly (`research/notebooks/_build_n07.py:161-247`):

- Category A: 4 possession-adjusted features.
- Category B: 5 fluke-score decomposition features.
- Category C: 5 efficiency-gap differential features.

Bonferroni alpha values match the locked design: `{"A": 0.05 / 4, "B": 0.05 / 5, "C": 0.05 / 5}` at `research/notebooks/_build_n07.py:250`, consumed in the baseline_C gate at `research/notebooks/_build_n07.py:814-851`.

## Phase 4 Findings: Cross-Cutting Checks

Verdict: mostly verified; documentation/provenance issues are listed below.

### Validated feature counts

`feature_validation.csv` has 30 unique R6-PASS features. This matches the N03 and N06 model specs:

- N03: `feature_pool_count=31`, `r6_validated_feature_count=30`, `structural_conditioning_feature_count=1`.
- N06: `feature_pool_count=31`, `r6_validated_feature_count=30`, `structural_conditioning_feature_count=1`.
- N07 expanded model: 33 semantic features = 30 original + 2 N07 passes + `fav_deficit`.

### Post-imputation model columns

N03 and N06 both document 19 indicator columns and 50 post-imputation model columns. The build scripts compute the same values from `len(model_core_features) + len(model_indicator_cols)` (`research/notebooks/_build_n03.py:1397-1398`; `research/notebooks/_build_n06.py:1735-1736`).

N07 is not directly comparable because it is an expansion test, not a full N03/N06-style pruning/spec pipeline. Its expanded production candidate is documented as semantic-feature count 33 in `research/results/n07_expanded_model_spec.json:2-17`.

### Executed notebook correspondence

Untracked executed notebooks are present per convention. Source-cell comparison found:

| Notebook | Executed/source code cells match? | Interpretation |
|---|---|---|
| 02g | No | Source/report prose changed after execution; diagnostic artifact only |
| 03 | Yes | Matches |
| 04 | No | Source/report prose changed after execution; diagnostic artifact only |
| 05 | No | Source/report prose changed after execution; diagnostic artifact only |
| 06 | No | Source/report prose changed after execution; diagnostic artifact only |
| 07 | Yes | Matches |

This is not result-critical because executed notebooks are intentionally untracked. It is a reproducibility/provenance issue if someone treats untracked executed notebooks as canonical.

## Summary

Grouped checks performed:

- Phase 1 ingestion/cache checks: 12
- Phase 2 feature-computation checks: 19
- Phase 3 methodology checks: 12
- Phase 4 cross-cutting checks: 8
- Total grouped checks: 51

Discrepancies found:

| Severity | Count | Summary |
|---|---:|---|
| CRITICAL | 0 | No methodology error found that requires rerunning analyses. |
| MODERATE | 3 | Documentation/provenance mismatches that could confuse future reruns or interpretation. |
| COSMETIC | 1 | Untracked executed-notebook mismatch. |

Recommendation: no analysis rerun is indicated by this audit. Before N08, fix or consciously accept the moderate documentation/provenance issues below, especially the N07 builder/spec reproducibility mismatch.

## Appendix: Discrepancies

### MOD-01: `redundant_with` documentation mixes structural-identity and Pearson-correlation semantics

What the code/data does:

- `feature_validation.csv` uses `redundant_with` both for the 02a structural EPA duplication tags and later Pearson `|rho| >= 0.6` redundancy tags.
- The schema sidecar's field definition says `redundant_with` is for per-row identical structural redundancy, "not statistical correlation" (`research/results/feature_validation.schema.md:71`).
- Later sidecar sections explicitly define Pearson-based tagging for `|rho| >= 0.6` (`research/results/feature_validation.schema.md:1104-1110`, `research/results/feature_validation.schema.md:1412-1428`, `research/results/feature_validation.schema.md:1522-1534`, `research/results/feature_validation.schema.md:1760-1766`).

What it should do:

- The sidecar should choose one canonical definition or explicitly state that the column is historical and contains both structural and Pearson-threshold tags, with the later N03 policy being "do not pre-prune on this tag; L1 handles redundancy."

Severity:

- MODERATE. Results are not affected because N03/N06 used all 30 R6-pass features and did not pre-prune based on `redundant_with`. The mismatch can mislead future feature-pool assembly.

### MOD-02: One `redundant_with` tag cannot be verified from the canonical 33x33 cross-feature matrix

What the code/data does:

- `dog_def_epa_per_play` has `redundant_with=fav_off_epa_per_play` in `feature_validation.csv` rows 2-4, but it failed R6 stability (`passed_stability=False`) and is absent from `research/results/_02g_full_correlation_matrix.csv`.
- The audit could verify the other Pearson tags from the matrix:
  - `dog_off_epa_per_play -> fav_def_epa_per_play`: rho = +1.000
  - `dog_third_down_success_rate -> dog_avg_drive_yards`: rho = +0.647
  - `fav_red_zone_tds -> plays_so_far`: rho = +0.650
  - `fav_red_zone_trips -> plays_so_far`: rho = +0.781

What it should do:

- Either document that failed features may still retain historical redundancy tags outside the canonical validated-set matrix, or include a separate provenance source for redundancy tags attached to failed features.

Severity:

- MODERATE. No production model impact because `dog_def_epa_per_play` is not in the 30 R6-pass feature pool.

### MOD-03: N07 committed spec/report include post-run documentation not regenerated by `_build_n07.py`

What the code/data does:

- The committed `research/results/n07_expanded_model_spec.json` starts with a `deployment_candidate` block marking the expanded 33-feature model as the N08 deployment candidate (`research/results/n07_expanded_model_spec.json:2-17`).
- `_build_n07.py` writes `expanded_model` without that `deployment_candidate` block (`research/notebooks/_build_n07.py:991-1025`).
- The committed `research/results/n07_summary_report.md` leads with the fuller project-arc endpoint framing (`research/results/n07_summary_report.md:3-28`).
- `_build_n07.py` still writes the shorter generated report framing at `research/notebooks/_build_n07.py:1087-1146`.

What it should do:

- Either update `_build_n07.py` so a rerun regenerates the committed spec/report exactly, or document that the final N07 report/spec were manually post-edited after execution and are canonical over the generator output.

Severity:

- MODERATE. Numeric results are not affected, but a future rerun would overwrite the canonical N07 documentation/spec framing.

### COS-01: Several untracked executed notebooks do not byte-match current source notebooks

What the code/data does:

- Source-cell comparison found mismatches for the untracked executed notebooks for 02g, 04, 05, and 06.
- 03 and 07 executed notebooks match current source notebook cell sources.

What it should do:

- Either leave as-is under the existing convention that executed notebooks are untracked diagnostic artifacts, or regenerate them after final documentation edits if they will be used as provenance.

Severity:

- COSMETIC. These artifacts are untracked by convention and do not affect committed results.
