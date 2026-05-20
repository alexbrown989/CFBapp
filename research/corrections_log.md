# Corrections log

This file records validated, post-merge corrections to Phase 0 notebooks
and their published deliverables. Each entry documents a specific bug
or data-format finding, the diagnostic path that surfaced it, the fix,
the verification scope, and the per-notebook impact on validated feature
sets.

Entries are append-only and chronological. When a correction supersedes
results from a prior commit, the prior commit hash and the
superseding-commit hash are both recorded so any reader can trace the
lineage. Per-feature verdict deltas (PASS ↔ FAIL) are summarised here
and detailed in the schema sidecar (`research/results/feature_validation.schema.md`)
under each notebook's section.

---

## 1. Lookahead leak in `plays_before` filter (2026-05-13)

### Diagnosis

While auditing 02c's P3 trigger logic for the post-explosive momentum
features, the `seconds_since_last_dog_explosive_play` extractor was
observed producing negative imputation values (a physical impossibility
— elapsed seconds since an event in the past cannot be negative). The
initial hypothesis was a sign-flip in the calculation, but tracing the
arithmetic showed the formula was correct; what was wrong was the input
play set.

The extractor iterates over `plays_before` — the set of plays the
trigger row's filter classified as having occurred before the trigger
play. That set was being constructed via `play.playNumber < trig.playNumber`,
on the assumption that `playNumber` was a globally-monotonic per-game
play index. It is not.

### Root cause

CFBD's `play.playNumber` field is a **per-drive** index, not a global
one. It resets to 1 at the start of each new drive. For a trigger at
drive 5, play 3 (`playNumber=3`), the leaky filter
`p.playNumber < 3` includes:

- plays 1 and 2 of drive 1 (correctly: in the past)
- plays 1 and 2 of drive 2 (correctly: in the past)
- plays 1 and 2 of drive 3 (correctly: in the past)
- plays 1 and 2 of drive 4 (correctly: in the past)
- **plays 1 and 2 of drive 5** (correctly: in the past, the
  trigger-drive prefix)
- **plays 1 and 2 of drives 6, 7, …** (incorrectly: these are
  *future* plays whose per-drive index happens to be < 3)

And it excludes:

- **plays 3, 4, 5, …** of drive 1 (incorrectly: these are real
  in-the-past plays whose per-drive index ≥ 3)
- same for drives 2, 3, 4

Two distinct R3 violations, both fixed by the same filter change:
**cross-drive forward leak** (future plays included) and
**within-drive truncation** (past late-drive plays excluded). The
truncation pattern affected drive-1-scoped features in 02b that
iterated `plays_before` filtered to drive 1 — the leaky filter saw a
truncated drive-1 prefix even when the trigger was many drives later
in the game.

### Fix

Replaced the `playNumber`-based comparison with a composite
chronological key:

```python
def _chrono_key(p: dict) -> tuple[int, int, int, int]:
    period = int(p.get("period") or 0)
    clock = p.get("clock") or {}
    m = clock.get("minutes")
    s = clock.get("seconds")
    elapsed = 900 - 60 * int(m) - int(s) if m is not None and s is not None else 0
    return (
        period,
        elapsed,
        int(p.get("driveNumber") or 0),
        int(p.get("playNumber") or 0),
    )
```

Primary chronological signal: the actual game clock (period +
elapsed-seconds-in-period). Secondary tiebreaker for plays sharing a
clock value (e.g., a TD recorded at 7:27 and the post-score kickoff at
the same 7:27): lex on `(driveNumber, playNumber)`. The per-drive-reset
problem is sidestepped because `driveNumber` increases monotonically
across the game, and within-drive `playNumber` is a valid tiebreaker
once `driveNumber` has already disambiguated.

The canonical definition lives in `research/notebooks/_lib_chrono.py`
as a single source of truth. Each `_build_0Xx.py` build script imports
`CHRONO_KEY_SOURCE` (the inlined Python source via
`inspect.getsource(...)`) and embeds it into the executed notebook's
cell body. Notebooks remain self-contained — no runtime import
dependency on the lib module — but the function definition is pinned
to one file so future amendments propagate to all three notebooks on
next regeneration.

The filter change in each `_build_0Xx.py`:

1. Sort `plays_by_game[gid]` by `_chrono_key` once at load time.
2. Compute `trig_chrono_key` from the trigger row's quarter and
   clock fields without looking up the trigger play in the cache.
3. Replace `p.playNumber < trig.playNumber` with
   `_chrono_key(p) < trig_chrono_key`.
4. Update `assert_no_lookahead` to verify the chrono ordering rather
   than the playNumber inequality.

### Verification

The `chrono_key` design was verified against the full Phase 0 corpus
(8,537 games, 1.54M plays, all seasons 2015-2024) via
`research/notebooks/_verify_chrono_key_composite.py`, with results
captured in `research/results/_verify_chrono_key_composite.stdout.txt`.

Procedure: for every adjacent play pair in chrono-key order, compare
the chrono-key ordering to the alternative orderings (lex on `play.id`,
lex on `(driveNumber, playNumber)`). Classify each disagreement by the
field driving the discrepancy.

Outcome: 22,775 adjacent disagreements identified across the full
corpus, all classified into named categories (a) through (d) by their
driving field. The residual category (c) — adjacent plays where the
chrono-key cannot disambiguate the within-clock-tick ordering and the
secondary `(driveNumber, playNumber)` tiebreaker resolves it — contains
749 disagreements. These are dominated by **Kickoff plays** where
CFBD's drive-attribution differs from the source-of-truth game-clock
ordering (e.g., a kickoff is logged as the last play of the receiving
drive in some seasons, the first play of the receiving drive in
others). The chrono-key resolves them via the
`(driveNumber, playNumber)` lex tiebreaker.

**Residual trigger-impact bound: 0.394%.** The 749 (c)-anomalies
upper-bound the fraction of trigger evaluations where the chrono-key
ordering might still differ from a hypothetical perfect oracle. Below
the noise floor of the per-fold Brier deltas in
`feature_validation.csv`.

### Audit-prediction validation

Before regenerating each notebook, an at-risk audit categorised
candidate features by extractor-structure exposure to the leak: drive-
metadata-only (no play iteration), drive-1-scoped (drive-level
aggregates), EPA aggregates over specific play subsets (drive-scoped
with play iteration), cumulative aggregates over all pre-trigger plays.
The post-execution diff between leaky and corrected verdicts validated
this categorisation:

- **Category-A (drive-metadata-only):** byte-identical pre/post — 5
  of 6 features in this category in 02b were unchanged across all
  three folds.
- **Category-B (cross-drive cumulative aggregates with recency
  dependence):** materially shifted — `seconds_since_last_dog_explosive_play`
  recovered from negative imputation values to physically plausible
  positives; `prior_drive_had_dog_explosive_play` collapsed from PASS
  to FAIL.
- **Category-C/D (drive-1-scoped EPA aggregates and derived flags):**
  shifted with verdicts mostly holding; one notable exception
  (`fav_def_epa_first_drive`) flipped to FAIL when truncation
  systematically biased the leaky EPA mean upward.

This pattern is mechanistically consistent with the diagnosis: the
features that survived the correction are the ones whose extractor
shape doesn't depend on which subset of past plays is included
(metadata-only, drive-level aggregates), and the features that
materially shifted are the ones whose extractor either (a) iterates
over future plays incorrectly classified as past (cross-drive forward
leak) or (b) truncates real past plays from drive-1 (within-drive
truncation).

### Related CFBD data observation

See section 2 for the negative `play.id` encoding observation that
surfaced during the same correction sweep. Not part of the leak fix;
documented in the same log because it was discovered during the same
diagnostic work.

---

## 2. CFBD `play.id` negative-encoding observation (2026-05-13)

CFBD's `play.id` field uses two encodings in the Phase 0 corpus. Both
are stored as **JSON strings**; the distinction is whether the string
represents a positive or negative integer:

- The dominant format: 18-digit positive-integer strings (e.g.,
  `'401628497104102201'`).
- A compact alternate format: small negative-integer strings (e.g.,
  `'-6654'`, `'-6655'`, `'-6656'`).

19,828 plays across 115 games use the negative encoding, spanning
seasons 2021, 2022, and 2024. Spot-verified examples (cache hit per
game; all plays in each carry the negative-string format and complete
`playText`):

- `game_id=401282236` (2021 regular season, week 8) — 181 plays,
  sample ids: `'-13070'`, `'-13083'`, `'-13072'`, `'-13071'`, `'-13073'`
- `game_id=401426342` (2022 regular season, week 6) — 192 plays,
  sample ids: `'-6659'`, `'-6654'`, `'-6655'`, `'-6656'`, `'-6657'`
- `game_id=401628492` (2024 regular season, week 4) — 168 plays,
  sample ids: `'-586'`, `'-582'`, `'-571'`, `'-572'`, `'-573'`

Both encodings carry complete play data: `playText`, drive metadata,
yardage, scoring flags, EPA, period/clock fields. There is no quality
difference in the underlying play records — only the `id` field's
numeric range differs. The build script's negative-id detection casts
via `int(p.get("id") or 0)` which handles both string representations
correctly.

### Why this is documented separately from a fix

During the lookahead correction work, an early diagnostic incorrectly
classified a 17-trigger sample of negative-id plays as "placeholder
rows" and proposed pre-filtering them. Verification of the broader
sample showed they are real plays, not placeholders, and the proposed
pre-filter would have dropped 19,828 real plays across 115 real games
— a 1.26% trigger-impact regression on a misread classification.

The composite `chrono_key` orders these plays correctly without
referencing `play.id`. They are retained in `plays_by_game` and
participate in all feature extraction. The negative-id observation is
recorded here as a **CFBD data-format finding**, not as a filtered-
around problem.

### When this matters

If any future code lex-sorts plays by `play.id` (which it shouldn't,
given `chrono_key` exists), the two encodings will not interleave
correctly: under numeric casting, negative integers sort before
positive integers, but a negative-id play and a positive-id play in
the same game can be in any chronological relationship. Under string
lex-sort, the ordering is even more arbitrary (`'-'` sorts before
digit characters in ASCII, so all negative-id strings sort before all
positive-id strings regardless of numeric value). Always use
`_chrono_key` for play ordering.

Origin of the dual encoding is unverified; possible explanation is
same-day vs. backfill source path differences in CFBD's ingestion
pipeline, but this is speculation. Not blocking; documented for future
reference.

---

## 3. Per-notebook impact summary

This section is filled in after each corrections commit lands, per the
plan documented in the corresponding commit body. Verdict deltas are
quoted relative to the verdicts that would have been published from the
last leaky-filter execution of each notebook.

### 02c — Explosive vs sustained drives + post-explosive momentum

**Superseded artifacts:** the implied verdicts from the leaky-filter
execution that produced the "8/8 PASS" verbal report. The 02c
deliverables CSV/sidecar from that execution were overwritten in the
working tree by a subsequent re-run before any commit landed, so the
leaky verdicts do not exist in any committed file; only the
correctness-of-this-correction is testable, not the byte-level diff.

**Last committed 02c lineage prior to corrections:**

- `4e19375` — scaffold (unexecuted notebook, no deliverables)
- `beacef4` — scoring-playType registry extension (unexecuted notebook
  with extended D12 attribution rules)

**Verdict delta under corrected chrono_key:** 8/8 PASS → **7/8 PASS**.

| Feature | Pre-correction | Post-correction | Notes |
|---|---|---|---|
| `dog_points_from_explosives` | PASS | PASS (3/3) | Strongest 02c signal (+0.020 mean Brier improvement) |
| `dog_points_from_sustained` | PASS | PASS (3/3) | Drive-volume aggregate, leak-robust |
| `dog_points_from_returns` | PASS | PASS (3/3) | Drive-volume aggregate, leak-robust |
| `dog_explosive_play_count` | PASS | PASS (3/3) | Cumulative count, leak-robust |
| `dog_avg_drive_yards` | PASS | PASS (3/3) | Drive-level mean, leak-robust |
| `dog_avg_drive_plays` | PASS | PASS (3/3) | Drive-level mean, leak-robust |
| `seconds_since_last_dog_explosive_play` | PASS (leaky) | PASS (2/3 Brier-improving folds under R6) | Imputation values now physically plausible (positive); brier improvements +0.00468 / +0.00780 / **−0.00269** — passes R6's floor (`sum(brier_improvement > 0) >= 2`) but doesn't strengthen across all test seasons. Marginal signal. Also resolves the negative-imputation symptom that initially surfaced the leak. |
| `prior_drive_had_dog_explosive_play` | PASS (leaky) | **FAIL (0/3)** | Binary momentum form collapsed to near-zero Brier deltas (−0.00009 / −0.00014 / −0.00010). Roebber 2022's NFL-WP transfer hypothesis does not hold for CFB comeback-trigger contexts under a clean filter. See `research/future_features.md` "Categorical-window momentum features" for the live alternative-shape hypothesis. |

**Net change to 02c validated set:** 7 features (was 8 under leaky
verdicts).

### 02a — Baseline efficiency features

**Superseded commits:**

- `9822cfc` — Phase 0 N02a scaffold (baseline efficiency features
  unexecuted)
- `2e17807` — docs(n02a): correct feature-count narrative; tag EPA
  redundancies in CSV (5-feature validated set under the leaky filter)

The 02a deliverables published at the time of `2e17807` claimed a
5-feature validated set (`fav_off_epa_per_play`, `fav_def_epa_per_play`,
`dog_off_epa_per_play`, `dog_def_epa_per_play`, `epa_divergence`), with
`plays_so_far` as the sole 1/3 FAIL. The CSV rows for those verdicts
are superseded by this commit; the prior CSV values for 02a remain in
the `2e17807` commit's tree for historical reference.

**Verdict delta under corrected chrono_key** (per
`feature_validation.csv v1_baseline_efficiency_only`, 18 rows):

| Feature | Pre-correction | Post-correction | Notes |
|---|---|---|---|
| `fav_def_epa_per_play` | PASS | PASS (3/3) | Mean Brier improvement +0.00316 across folds. Conditional identity with `dog_off_epa_per_play` (byte-identical 3 fold rows). |
| `dog_off_epa_per_play` | PASS | PASS (3/3) | Mean Brier improvement +0.00316. Tagged `redundant_with=fav_def_epa_per_play` from `2e17807` survives the correction. |
| `epa_divergence` | PASS (3/3 leaky) | PASS (3/3 corrected) | **10x magnitude collapse:** prior mean Brier improvement was +0.044 under the leak; corrected mean is +0.00094. Verdict holds but the prior "standout" framing was leak-inflated; N03 should treat this signal as marginal. |
| `plays_so_far` | **FAIL (1/3)** | **PASS (3/3)** | Mean Brier improvement +0.01472 (+0.022 / +0.013 / +0.009 across folds) -- **strongest single 02a signal** post-correction. The leak had actively destroyed a clean game-time-elapsed proxy by including future plays in the per-trigger play count for some triggers and excluding real past plays for others, distorting the cumulative-count signal. |
| `fav_off_epa_per_play` | PASS | **FAIL (0/3)** | Mean Brier improvement collapsed to -0.00057 across folds. The leak inflated this feature's signal; post-correction, the favorite's offensive performance over the pre-trigger window adds no signal beyond the baseline. Conditional identity with `dog_def_epa_per_play` (byte-identical 3 fold rows). |
| `dog_def_epa_per_play` | PASS | **FAIL (0/3)** | Mean Brier improvement -0.00057. Tagged `redundant_with=fav_off_epa_per_play` from `2e17807`; both members of the conditional-identity pair collapsed to FAIL together. |

**Net change to 02a validated set:** 4 features (was 5 under the leaky
filter; counting conditional-identity pairs once would give 3, but
preserving the `2e17807` convention of counting both members of a
conditional-identity pair gives 4).

The validated 02a set under the corrected filter:

- `fav_def_epa_per_play` (≡ `dog_off_epa_per_play` under conditional identity)
- `dog_off_epa_per_play`
- `epa_divergence` (weak; +0.00094 mean Brier improvement)
- `plays_so_far` (strongest 02a signal; +0.01472 mean Brier improvement)

Removed from the validated set:

- `fav_off_epa_per_play` (and its conditional-identity pair
  `dog_def_epa_per_play`)

The `plays_so_far` FAIL → PASS recovery is the single most significant
swing in this correction sweep -- the leak was actively destroying a
clean signal. The `epa_divergence` magnitude collapse from +0.044 to
+0.00094 (verdict holding) means N03 must revise its prior
interpretation: this feature is a survivor, not a standout.

### 02b — Opening-drive shock features

**Superseded commits:**

- `bfddc16` — Phase 0 N02b scaffold (opening-drive shock features
  unexecuted)
- `e1710a2` — feat(n02b): opening-drive shock validation results
  (7 PASS / 3 FAIL under the leaky filter)

The 02b deliverables published at the time of `e1710a2` claimed a
7-feature validated set and specifically flagged
`fav_def_epa_first_drive` in the commit body as "the only feature
across 02a + 02b to pass 3/3 on both Brier AND ECE — structurally
important for N03 because Kelly stake sizing reads directly off
calibrated probabilities." **That claim is empirically falsified
under the corrected `_chrono_key` filter** (see retraction in this
commit's body and in the 02b schema-sidecar section's "Post-correction
findings and retractions" subsection).

**Verdict delta under corrected chrono_key** (per
`feature_validation.csv v1_opening_drive_shock`, 30 rows):

| Feature | Pre-correction | Post-correction | Notes |
|---|---|---|---|
| `dog_received_opening_kickoff` | PASS | PASS (3/3) | Mean Brier −0.00071, mean ECE +0.00553. Category-A (drive-metadata-only, no play iteration) — verdict and fold-level signs identical pre/post by structure. |
| `opening_drive_was_td` | PASS | PASS (3/3 by R6; 2/3 Brier-improving folds) | Mean Brier +0.00154. Drive-1-scoped binary read from `drives_for_game`. |
| `opening_drive_was_explosive_td` | PASS | PASS (3/3 by R6; 2/3 Brier-improving folds) | Mean Brier +0.00195. **Core project hypothesis support.** 542 triggers (32% of evaluable) flipped from `leaky=0, chrono=1` in the same direction -- the leaky filter systematically *undercounted* explosive opening-drive TDs because the explosive play often had `playNumber >= trig_pn` and got truncated. Corrected feature is more accurate than the leaky one. Verdict shifted from "3/3 Brier-improving under leak" to "2/3 Brier-improving under correction" because one fold (2015-2021 -> 2023) happened to have the leaky distribution align with outcomes in a way the corrected distribution does not; the other two folds strengthened. |
| `dog_scored_on_opening_drive` | PASS | PASS (3/3 by R6; 2/3 Brier-improving folds) | Mean Brier +0.00002. |
| `fav_def_epa_after_first_drive` | PASS | PASS (3/3 Brier-improving folds) | Mean Brier +0.00313, mean ECE −0.01102. Verdict and signal preserved. |
| `defense_stabilized_flag` | PASS | PASS (3/3 Brier-improving folds) | Mean Brier +0.00479. **Mechanism note:** this derived feature `int(fav_def_epa_after_first_drive < fav_def_epa_first_drive)` survives despite one of its two inputs (`fav_def_epa_first_drive`) failing 0/3. The *direction of the inequality* between the two corrected EPA means carries signal even when the level of input A alone does not. N03 should treat this as a defense-trajectory feature, not as an EPA aggregate. |
| `fav_def_epa_first_drive` | PASS (leaky; 3/3 Brier AND 3/3 ECE) | **FAIL (0/3 Brier-improving)** | Mean Brier −0.00132, ECE 2/3 (one fold flipped). Per-fold Brier: −0.00147 / −0.00091 / −0.00157 — all three folds collapsed in sign. **Mechanism:** the within-drive truncation arm of the leak (see section 1) systematically removed late-drive-1 plays from `plays_before` for triggers in later drives with low `playNumber`. Late-drive-1 plays disproportionately include sacks, third-down conversions, and stops — removing them biased the leaky `fav_def_epa_first_drive` mean upward in a way that happened to correlate with comeback outcomes. **This is the most important finding of the entire correction sweep:** the prior commit's "calibration-standout" claim was leak artifact. Retracted in this commit's body. |
| `opening_drive_yards` | FAIL | FAIL (0/3) | Drive-shape feature; verdict preserved at FAIL under both filters. |
| `opening_drive_plays` | FAIL | FAIL (0/3) | Same. |
| `opening_drive_seconds` | FAIL | FAIL (0/3) | Same. |

**Net change to 02b validated set:** 6 features (was 7 under the leaky
filter). One verdict change: `fav_def_epa_first_drive` PASS → FAIL.

### Truncation-impact diagnostics (02b in-notebook cell)

The 02b notebook's Phase 02b-d2 cell computes three truncation-impact
diagnostics on the fly to mechanistically explain the corrections:

1. **Per-trigger drive-1 play count under leaky vs. chrono filters.**
   Reveals the magnitude of the within-drive truncation arm. Leaky
   filter saw a truncated drive-1 prefix for any trigger in a later
   drive with a small `playNumber`; the chrono filter sees the full
   real drive-1 prefix.
2. **`opening_drive_was_explosive_td` flip counts.** 542 triggers
   (32% of evaluable) flipped 0 → 1 under the chrono filter; zero
   triggers flipped 1 → 0. The flip is uniformly in the direction of
   "leaky undercounted explosive drive-1 TDs."
3. **`fav_def_epa_first_drive` EPA-mean shifts.** Distribution of
   `(chrono_epa - leaky_epa)` across triggers where both filters
   produced non-null values. 32% of triggers see EPA-mean shifts
   > 0.01 magnitude; 9% flip sign. Magnitude and direction of these
   shifts explain the PASS → FAIL collapse mechanistically.

See the executed notebook for the per-trigger distributions.

### Cross-notebook summary

After all three corrections commits, the validated set across
02a + 02b + 02c is **17 features** (was a "claimed 18" under the
prior leaky-filter narrative). The composition shifted meaningfully:
drive-volume and cumulative-aggregate features mostly survived;
recency-dependent EPA features mostly did not.

**Validated 02a set (4 features, post-correction):**

- `fav_def_epa_per_play` (≡ `dog_off_epa_per_play` under conditional identity)
- `dog_off_epa_per_play`
- `epa_divergence` (weak; +0.00094 mean Brier; was +0.04400 under leak — 10x collapse, verdict holds)
- `plays_so_far` (strongest 02a signal; +0.01472 mean Brier; FAIL → PASS recovery)

**Validated 02b set (6 features, post-correction):**

- `dog_received_opening_kickoff`
- `dog_scored_on_opening_drive`
- `opening_drive_was_td`
- `opening_drive_was_explosive_td` (core hypothesis support; +0.00195 mean Brier; 2/3 Brier-improving folds under R6 -- magnitudes smaller than the leak suggested but mechanism cleaner under correction since 542 triggers had explosive drive-1 plays the leak was undercounting)
- `fav_def_epa_after_first_drive`
- `defense_stabilized_flag` (3/3 Brier-improving despite one input failing 0/3 -- direction-of-inequality signal survives the input collapse)

**Validated 02c set (7 features, post-correction):**

- `dog_points_from_explosives` (strongest 02c signal, +0.020 mean Brier)
- `dog_points_from_sustained`
- `dog_points_from_returns`
- `dog_explosive_play_count`
- `dog_avg_drive_yards`
- `dog_avg_drive_plays`
- `seconds_since_last_dog_explosive_play` (weak; +0.00326 mean Brier; 2/3 Brier-improving folds under R6 -- one negative-Brier fold)

**Removed from the validated set by the corrections:**

- `fav_off_epa_per_play` (02a; PASS → FAIL; and its conditional-identity pair `dog_def_epa_per_play`)
- `fav_def_epa_first_drive` (02b; PASS → FAIL; was prior calibration-standout claim, now retracted)
- `prior_drive_had_dog_explosive_play` (02c; PASS → FAIL; binary momentum form fails under clean filter; Roebber 2022's NFL-WP-streaks transfer hypothesis does not hold in this CFB comeback-trigger context)

**Pattern observation.** The features that survived the correction
are predominantly:

- Drive-metadata-only features (no play iteration): byte-identical pre/post by structure.
- Cumulative aggregates over all pre-trigger plays where the play-set
  distortion was small in relative terms: `plays_so_far`, drive-volume
  features.
- Direction-of-inequality binaries that absorb noise in their inputs:
  `defense_stabilized_flag`.

The features that did not survive are predominantly:

- Recency-dependent EPA aggregates over specific play subsets:
  `fav_off_epa_per_play`, `dog_def_epa_per_play`, `fav_def_epa_first_drive`.
- Possession-level binary momentum form:
  `prior_drive_had_dog_explosive_play`.

This pattern is mechanistically consistent with the diagnosis in
section 1 -- the leak's two arms (cross-drive forward leak and
within-drive truncation) both have larger relative effects on
recency-dependent and small-N play subsets than on cumulative
aggregates over all pre-trigger plays.

**N03 implications.** Calibration support for Kelly stake sizing must
come from elsewhere in the validated set; no single feature in the
corrected 02a + 02b sets passes both 3/3 Brier AND 3/3 ECE. The N03
production-feature assembly should treat the 17 validated features as
the working set, with `redundant_with` filtering applied per the
existing 02a tags (which drops the two duplicate-identity 02a
features). See `research/future_features.md` for the live
categorical-window momentum hypothesis triggered by 02c's marginal
continuous-form result.

---

## 02d prediction-vs-result calibration

**Plan-time prediction:** 2/4 PASS (build-author prior, based on the
expected harshness of trigger-conditioning on small-magnitude
turnover/field-position features). **Observer hypothesis-watch:**
relative ordering -- `dog_points_off_turnovers` more likely to pass
than `short_field_tds_allowed`, again with an implicit ~2/4 base rate.

**Observed:** 4/4 PASS under R6 stability (>=2 of 3 walk-forward test
seasons with positive held-out Brier improvement).

Both the plan author's and the reviewer's independent priors were
systematically pessimistic about trigger-conditioning's harshness on
small-magnitude turnover/field-position features. Two independent
predictions 2x off the observed result is a calibration signal worth
recording.

**Cross-notebook correlation diagnostic (post-execution, pre-commit).**
Pearson correlations between the 4 new 02d features and the 16
validated features carried into 02d (17-feature set after dropping
`dog_off_epa_per_play` as byte-identical to `fav_def_epa_per_play`).
On the non-null intersection of each pair. Full matrix in
`research/results/_02d_correlations.csv` (untracked, diagnostic-only).

- **Zero pairs with `|rho| >= 0.6`.** No `redundant_with` tags applied.
- Four pairs in the meaningful (0.3-0.6) band: `fav_turnovers_so_far`
  with `plays_so_far` (rho=+0.501), `short_field_tds_allowed` with
  `dog_points_from_sustained` (+0.425), `short_field_tds_allowed`
  with `plays_so_far` (+0.349), and `fav_turnovers_so_far` with
  `dog_explosive_play_count` (+0.320). All flagged for N03 L1
  down-weighting awareness in the 02d schema sidecar.
- The conditional-identity flag from D11 (`dog_points_off_turnovers`
  vs `dog_points_from_returns`) confirmed at `rho=+0.042` -- the 2.67%
  co-occurrence is mirrored by the correlation; cleanly separable in
  both diagnostics.

**Verdict-vs-correlation cross-classification for the 4 02d features:**

| Feature | Brier mean | Max `|rho|` (validated) | Best-fit interpretation |
|---|---:|---:|---|
| `fav_turnovers_so_far` | +0.00649 | 0.501 (`plays_so_far`) | Real signal, but partly correlated with game-length proxy. |
| `dog_points_off_turnovers` | +0.00553 | 0.257 (`dog_points_from_sustained`) | Independent signal; cleanly separable from `dog_points_from_returns` (rho=+0.042). |
| `dog_avg_starting_field_pos` | +0.00173 | 0.244 (`dog_points_from_sustained`) | Independent of validated set but smallest Brier magnitudes; noise-fold-luck candidate. |
| `short_field_tds_allowed` | +0.00258 | 0.425 (`dog_points_from_sustained`) | Partial overlap with sustained-style scoring; L1 will likely down-weight. |

**No verdict changes.** R6 stability passed honestly for all four
features under the corrected `_chrono_key` filter. Correlation context
informs N03 feature-selection but does not invalidate the empirical
walk-forward Brier improvements.

**Open methodological question for 02e/02f/02g.** Are R6's stability
thresholds permissive enough that small-magnitude features pass when
they're closer to noise than signal? `dog_avg_starting_field_pos`
posted three Brier improvements all `< +0.005` per fold (the smallest
trio in the validated set to date) and the lowest cross-correlation
with the validated set, yet still cleared the 2-of-3 PASS bar. The
plan-time prior was directionally correct about this feature's
weakness (Cursor's plan-time confidence: 35-50%), but R6 doesn't
distinguish "passes with magnitude > X" from "passes with magnitude
near zero". Magnitudes `< +0.005` Brier improvement on a fold should
be treated with skepticism in N03 production feature-selection,
regardless of fold count.

This is a soft prior for 02e onwards: when a 4/4 PASS rate diverges
sharply from a calibrated 2/4 plan-time prior, run a correlation
diagnostic before committing. The cost is one cache-only diagnostic
script; the benefit is catching pseudo-independent signals that L1
will collapse anyway. The 02d run found zero outright redundancies
(`|rho| >= 0.6`) but four meaningful (0.3-0.6) overlaps that would
have been invisible without the diagnostic.

---

## 2024-fold weakness pattern (project-wide)

Documentation of systematic **walk-forward Δ Brier** weakening on **2024-test** folds surfaced during **Notebook 02e** empirical review ( **`research/notebooks/_diag_02e_fold_pattern.py`**, diagnostic-only script — presently **untracked**; rerun after updating **`feature_validation.csv`** ).

### Empirical aggregates

**Mean Δ Brier (unweighted) across the twenty-one PASS feature names that pre-date `v1_red_zone_failure`:**

| Test season | Mean Δ Brier |
|---|---:|
| 2022 | +0.00686 |
| 2023 | +0.00475 |
| 2024 | +0.00162 |

Roughly **4× erosion** comparing **2022** → **2024** (**+0.00686 → +0.00162**).

**Negative Δ Brier fold counts among all PASS cohort rows (distinct features × folds = **24 features** × **three** test seasons summarized as row counts):**

| Test season | # features with Δ Brier `< 0` | Denominator |
|---|---:|---:|
| 2022 | **1** | 24 PASS features |
| 2023 | **4** | 24 PASS features |
| 2024 | **6** | 24 PASS features |

The deterioration is visible **without** restricting to **`v1_red_zone_failure`** — i.e., it precedes interpreting any single new red-zone PASS.

### Interpretation (hypotheses, not adjudicated here)

Possibilities coexist:

1. **Temporal signal attenuation:** comeback‑equity / pre‑trigger structure may compress as league metadata, pacing, officiating trends, scoring environment, or calibration baselines drift (2015–train windows vs newest hold‑out seasons).
2. **Selection leakage into older regimes:** features earned **PASS** under **≥2-of-3** walk-forward uplift; uplift could concentrate on **early** eras even when aggregates still nominally PASS on **2024** — i.e., **overfit curvature** disguised behind R6 aggregates.
3. **Higher variance regime** for **2024**‑only — especially plausible if cohort **n_test** swings or parity noise rises; distinguishing true decay from sampling noise needs held‑out repeats **not** embodied in Phase‑0 scaffolding.

These are bookkeeping uncertainties for **N03**, not indictment of any lone feature.

### N03 implication

Treat **single-season** **held-out uplift** sceptically when newest season underperforms the portfolio mean. Calibration / stake‑sizing artefacts should emphasize **recent** eras — e.g., **over-weight 2024** when fitting isotonic calibration on validation slices, prefer **designating 2024 as primary calibration-validation cohort**, or withhold **recent season** folds from naive training mixtures so production behaviour tracks **nearest-to-live conditions**. The farthest chronological test season is materially **easier historically** yet **furthest culturally** from next-season deployment; blindly trusting uniform fold weighting biases deployment optimism. This finding also affects feature-selection design for **N03**, not just calibration. Some currently-**PASS** features may not retain their **R6** verdict under a **2024**-weighted re-test. Before training the production model, run a re-stability check that weights the **2024** fold **≥ 2×** the other folds and flag any feature whose verdict flips under that weighting. Such features should either be dropped from the production set or carry an explicit **`pre-2024-only`** caveat.

Cross-reference redundancy diagnostics from **Notebook 02e**: **`research/results/_02e_correlations.csv`** and hedged‑verdict prose in **`feature_validation.schema.md`** ( **`<!-- BEGIN: 02e red_zone_failure -->`** section).

Cross-reference **02f** redundancy: **`research/results/_02f_correlations.csv`** and the **02f** block in **`feature_validation.schema.md`** ( **`dog_third_down_success_rate` ↔ `dog_avg_drive_yards`**, **ρ ≈ +0.647**).


---

## 02f — D10 leak exposure (2026-05-14); cross-notebook DDL correlation

### Empirical D10 scope (executed 02f notebook)

On **Notebook 02f** owning-team down–distance rates, **D10** disagreement between canonical **`_chrono_key`** and leaky **`playNumber`** filters affected **~44–45%** of micro-quantized triggers per DDL column — the **widest Phase 0 footprint** observed so far (**02e** ~**26%**; **02b** ~**32%**).

**Interpretation:** **`_chrono_key` was load-bearing specifically for these DDL accumulators** — not a cosmetic ordering tweak. The **`MICRO_NAN_SENT`** quantization used in **D10** also shows the leak moving rows between **defined rate** and **NULL** states, not merely nudging numerators inside an otherwise stable support set.

### Negative `distance` / `yardsToGoal` audit (CFBD quirk)

Rare `/plays` rows carry **negative `distance`** (especially **Kickoff**-typed rows with **`down ∈ {1,2,3}`** in the provider feed). **`_effective_distance`** now clamps **`max(0, min(distance, yardsToGoal))`**, and strict field audit **counts-and-skips** negatives instead of aborting. A full cache scan (**not** capped at the staged **250k** “good-audit” quota) finds **116** such DN-keyed snaps; **`dog_third_down_success_rate`** cross-correlation tag is independent of this plumbing fix.

### Pearson redundancy vs cumulative PASS (**`|ρ| ≥ 0.6`**)

**`research/notebooks/_diag_02f_correlations.py`** emits **`research/results/_02f_correlations.csv`**: **four** 02f DDL rates × **twenty-four** other PASS numeric columns (**11,416** triggers).

One pair clears **0.6**: **`dog_third_down_success_rate`** vs **`dog_avg_drive_yards`** (**ρ ≈ +0.647**). **`feature_validation.csv`** row tag: **`redundant_with=dog_avg_drive_yards`** (**02d**/**02e** precedent: strongest partner wins). **`dog_early_down_success_rate`** peaked at **ρ ≈ +0.581** vs the same partner — advisory only (below gate).

---

## N03 null handling decision (2026-05-17)

Notebook 03 fits all Phase 0 R6-PASS features in a single walk-forward
model matrix. Phase 0 evaluated features one at a time, so each 02x
notebook could use a local null policy. N03 cannot complete-case drop
every row with any missing value: high-null features such as
`fav_yards_per_point`, `defense_stabilized_flag`, down-distance success
rates, drive-volume features, and turnover/short-field features would
remove a large share of the corpus.

The locked N03 policy is **train-fold-only median imputation plus paired
missingness indicators for every candidate feature with >5% full-corpus
null rate**. Medians are computed on training years only and then applied
to train, validation, and test slices. This preserves R16 no-leakage
discipline while keeping the full trigger corpus available for fitting.

Pure median imputation without indicators was rejected because some null
states are themselves informative. The clearest example is
`fav_yards_per_point`: the 02e/02f schema-sidecar work separated nulls
into bucket (a) "no completed favorite drives yet" and bucket (b)
"completed favorite drives exist but zero favorite offensive points."
Bucket (b) is a real game-state signal, not merely absent data. Median
imputing that state without an indicator would erase exactly the
information the feature was meant to preserve.

Existing Phase 0 indicator columns are reused rather than duplicated:

- `seconds_since_last_dog_explosive_play_is_null`
- `fav_yards_per_point_is_null`
- `fav_early_down_success_rate_insufficient_sample`
- `fav_third_down_success_rate_insufficient_sample`
- `dog_early_down_success_rate_insufficient_sample`
- `dog_third_down_success_rate_insufficient_sample`

For any other R6-PASS feature above the 5% null threshold, N03 creates a
fresh `{feature}_is_null` preprocessing column. These indicator columns
do not change the semantic 30-feature Phase 0 candidate pool; they
preserve information already present in the validated features' null
structure. N03 reports core-feature pruning decisions separately from
missingness-indicator diagnostics. A zeroed indicator does not, by
itself, drop the underlying core feature.

---

## N03 trigger-play deduplication and structural `fav_deficit` decision (2026-05-18)

During N03 verification, the first successful execution produced
prediction rows that were not unique on `game_id + trigger_play_id +
scheme + fold`. The verifier surfaced the issue before the artifacts
were accepted as canonical. The root cause is the trigger design itself:
one play can satisfy multiple deficit thresholds. For example, a play
where the favorite trails by 10 qualifies at the D=3, D=7, and D=10
thresholds. The natural trigger-event key is therefore
`game_id + fav_deficit + trigger_sequence`, not `game_id +
trigger_play_id`.

The diagnostic count was substantial: **11,416** trigger events but only
**7,854** unique trigger plays. The duplicate event rows were identical
on all 30 Phase 0 feature values, labels, scores, and model
probabilities; only the threshold identity differed. Training directly
on all 11,416 rows would count the same observed play multiple times,
inflate effective sample size, and over-weight multi-threshold game
states.

N03 therefore uses a mixed structure:

- **Training/evaluation matrix:** one row per unique trigger play
  (**7,854** rows), using the lowest qualifying deficit threshold as
  `fav_deficit`.
- **Model feature set:** **30** R6-validated Phase 0 features plus
  `fav_deficit` as a protected structural conditioning variable
  (**31** core model features before missingness indicators).
- **Prediction output:** held-out plays are replicated back to their
  qualifying deficit thresholds at scoring time. `fav_deficit` varies
  across those replicated rows, while the 30 Phase 0 feature values stay
  constant for the underlying play.

This was chosen over pure event-row training because pure event training
retains the duplicate-play weighting problem even if `fav_deficit`
enters as a feature. It was chosen over pure play-row scoring because
N04 needs deficit-threshold-specific probabilities: a market at D=3 is
not the same bet state as a market at D=7 for the same play. It was
chosen over dropping multi-threshold rows because those plays are real
game states and are needed for N04 threshold sensitivity.

`fav_deficit` is explicitly **not** a Phase 0 stability-tested feature.
It is a structural conditioning variable that defines the trigger event.
It is exempt from L1/permutation/ablation pruning; removing it would
collapse N04's threshold-specific scoring back to identical predictions
across deficit variants and reintroduce the duplicate-row ambiguity.
N03 reports the exemption in the pruning matrix and records
`structural_conditioning_variable: true` in `n03_model_spec.json`.

---

## N03 honest interpretation (2026-05-19)

N03 produced a usable probability model, but not an edge-grade result by
itself. The structural finding is: **the model has real but modest
discrimination, calibration remains fragile, and the newest fold does not
beat the Phase 0 pre-game alpha baseline on Brier**.

The locked production model is the unified L1 logistic regression at
`C=1.0`, with isotonic calibration per walk-forward validation slice. It uses
30 R6-validated Phase 0 features plus protected structural `fav_deficit`.
Weighted held-out performance:

| Scheme | Weighted Brier | Weighted ECE | Weighted AUC |
|---|---:|---:|---:|
| U | 0.218908 | 0.041820 | 0.689016 |
| W2 | 0.219233 | 0.042243 | 0.685530 |

### Pre-game alpha comparison

The key diagnostic is N03 versus the Phase 0 pre-game alpha baseline
(`pregame_spread`, `rating_gap`, `fav_pregame_rating`, `dog_pregame_rating`,
`spread_movement`, `spread_movement_is_null`). Positive Delta Brier means N03
beat alpha.

| Test fold | Alpha Brier | N03 Brier | Delta Brier | Alpha ECE | N03 ECE | Alpha AUC | N03 AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022 | 0.245734 | 0.221332 | +0.024402 | 0.076677 | 0.055814 | 0.6092 | 0.6933 |
| 2023 | 0.216000 | 0.215187 | +0.000812 | 0.094456 | 0.026130 | 0.7116 | 0.6986 |
| 2024 | 0.218155 | 0.220206 | -0.002051 | 0.056951 | 0.043515 | 0.6854 | 0.6751 |

This is the cold-water result. N03 improves calibration versus alpha on every
fold, but it does **not** improve 2024 Brier or 2024 AUC. The project should
not treat N03 as proof of deployable edge before N04 compares the model
against market probabilities directly.

### Calibration and structural dominance

Aggregate ECE around **0.042** hides material per-deficit calibration
weakness:

| Deficit | Event rows | ECE | Brier |
|---:|---:|---:|---:|
| D=3 | 1,451 | 0.0363 | 0.2472 |
| D=7 | 1,072 | 0.0422 | 0.2431 |
| D=10 | 698 | 0.0872 | 0.2264 |
| D=14 | 458 | 0.0609 | 0.1819 |
| D=21 | 178 | 0.0141 | 0.0546 |

Discrimination is also dominated by two structural game-state variables:
`plays_so_far` (weighted signed standardized coefficient **-0.687**) and
`fav_deficit` (**-0.629**). The engineered Phase 0 features contribute
incrementally, but most are marginal after time/deficit structure.

### Sensitivity diagnostics

Two informational diagnostics were run before commit. They are documented as
appendices, not production replacements.

**C sweep:** tested `C in {0.1, 0.5, 1.0, 2.0, 10.0}`. No C value strictly
dominated locked `C=1.0` by the pre-declared criterion of better weighted ECE
and better per-fold Brier-vs-alpha on all three folds. `C=2.0` slightly
improved weighted U ECE (**0.04112** vs **0.04182**) but did not improve
Brier-vs-alpha on all folds. `C=0.1` produced a non-empty three-signal pruning
set (`dog_explosive_play_count`, `dog_points_from_explosives`,
`opening_drive_was_explosive_td`, `fav_red_zone_tds`) but worsened the 2024
Brier-vs-alpha result.

**Bin-specific models:** D<=7 and D>=10 separate L1 models were fit at
`C=1.0`. They worsened event-level calibration relative to the unified model:

| Bin | Event rows | Bin-specific ECE | Unified ECE on same rows | Bin-specific Brier | Unified Brier |
|---|---:|---:|---:|---:|---:|
| D<=7 | 2,523 | 0.04286 | 0.03399 | 0.24751 | 0.24545 |
| D>=10 | 1,334 | 0.07195 | 0.06638 | 0.19468 | 0.18818 |

The unified architecture remains the right production choice. The bin-specific
negative result also suggests that D=10/D=14 mis-calibration is not solved by
splitting the sample; the split loses useful sample size and does not recover
calibration.

### N04 implications

N04 should be framed as a predictive-edge validation against pre-game market
probabilities, not as a live-betting CLV proof. The honest prior for finding
positive CLV/predictive edge is **10-20%**, not **40-50%**.

Operational implications for N04:

- Be skeptical of high-confidence Kelly sizing; report conservative sizing
  and threshold sensitivity.
- Report per-deficit CLV/probability-comparison results prominently,
  especially D=10 and D=14.
- Expect deployment-class behavior to look closer to the 2024 fold than the
  stronger 2022 fold.
- A clean negative N04 result should be treated as a meaningful project
  finding, not as a failed implementation.

---

## N04 honest interpretation (2026-05-19)

N04 produced a defensible positive result on the locked primary validation
gate: **N03's trigger-state probabilities beat pre-game market
probabilities for the historical trigger-state subpopulation**. Under
Scheme U, all-fold Brier improvement is **+0.05847** with cluster-bootstrap
95% CI **[+0.04211, +0.07465]**. Scheme W2 is effectively identical at
**+0.05847** with 95% CI **[+0.04210, +0.07429]**. Fold-level improvements
are consistent: **2022 +0.06491**, **2023 +0.05418**, and **2024
+0.05651**. The comparable 2024 result is reassuring given the Phase 0 and
N03 concerns about 2024-fold weakness.

The win is **calibration, not ranking**. The pre-game market still ranks
teams better overall (**market AUC 0.6812** vs **model AUC 0.6650**), but it
is poorly calibrated for trigger-state rows because it does not condition on
the favorite now trailing. N04's all-fold ECE comparison captures that
shift: **model ECE 0.03484** versus **market ECE 0.24840**. The model wins
by adjusting probabilities down to the observed in-game state, not by
re-ordering teams better than the market.

The deficit pattern is the strongest mechanistic validation. All-fold Brier
improvement increases monotonically with deficit: **D=3 -0.00570**,
**D=7 +0.03134**, **D=10 +0.09147**, **D=14 +0.16507**, and **D=21
+0.34131**. That is the expected signature of a useful in-game-state model:
as the favorite's deficit deepens, the stale pre-game probability becomes
increasingly over-optimistic and the trigger-state model corrects it.

This result does **not** prove live betting edge. Historical live in-game
line data is unavailable for the 2022-2024 corpus, so N04 can only compare
trigger-state probabilities to pre-game market probabilities. The tertiary
favorite-side betting simulation at the primary deployment-context setting
(edge threshold **+0.08**, **25% Kelly**, no D=21 rows) produced **89** bets,
**35.96%** win rate, and **-33.27% ROI**. That is honest evidence that this
specific favorite-side policy failed in hindsight. It is also consistent
with the primary result: a model can improve on stale pre-game probability
without beating correctly priced live in-game markets.

Project conclusion after N04: **predictive edge versus pre-game market
consensus is validated; live-line betting edge remains untested**.
Deployment-context profitability requires going-forward live market data
collection and a separate validation pass against actual live prices.

---

## N05 honest interpretation (2026-05-19)

N05 produced a clean negative result on the stricter model-vs-baseline
question. Against the training-years-only `fav_deficit x time_bucket`
baseline_C, the N03 Scheme U probabilities do **not** improve Brier score on
either label. For the N03 training label, `favorite_final_win`, Brier
improvement (`baseline_C - model`) is **-0.00303** with cluster-bootstrap 95%
CI **[-0.00677, +0.00051]**, which is not distinguishable from zero. For the
literal comeback label, `deficit_erased`, improvement is **-0.06123** with 95%
CI **[-0.07244, -0.05029]**, materially worse than the simple baseline.

This reframes N04's per-deficit pattern. N04's positive comparison against
pre-game market probability remains real, but the mechanism is narrower than
"the model discovered deep-deficit comeback signal." The more honest read is
that the pre-game market baseline becomes increasingly stale at deeper
in-game deficits, while a simple current-state baseline already captures much
of that structure. N03 beat pre-game market probabilities because pre-game
markets do not condition on current deficit and game time; N05 shows that N03
does not beat a naive deficit/time lookup table on held-out triggers.

The descriptive split is itself important: favorites erased the deficit after
trigger on **63.5%** of non-null trigger events, but won the game only
**43.3%** of trigger events. That roughly 20-percentage-point gap corresponds
to **2,326** trigger events where the favorite came back to tie or lead but
still lost the game. This "favorite came back but lost" subpopulation is large
enough to deserve direct follow-up rather than being treated as noise around
the final-win label.

The model is also systematically under-calibrated for `deficit_erased`.
Across the middle probability deciles, actual deficit-erased rates exceed N03
model probabilities by roughly 15-30 percentage points. That is consistent
with the model being trained on `favorite_final_win` rather than on the
literal deficit-erasure event, and it explains why final-win probabilities
are a poor direct proxy for comeback-erasure probabilities.

Future research should treat an N06-style model explicitly fit on
`deficit_erased` as the next candidate notebook. That model should be judged
against the same strict baseline_C construction and should keep
`favorite_final_win` separate from `deficit_erased` throughout.

Methodology note: this is exactly the kind of result that a looser analysis
could hide. The training-years-only baseline_C definition was load-bearing:
it prevented leakage from held-out seasons while still testing whether the
model adds signal beyond the two structural dimensions N03 leaned on most
heavily. Keeping that baseline strict surfaced the real negative finding
instead of allowing the positive N04 market comparison to overstate the
model's mechanism.

---

## N06 honest interpretation (2026-05-19)

N06 completed the natural label-change test: keep the N03 feature pool,
model class, null handling, play-level deduplication, walk-forward windows,
and isotonic calibration structure fixed, but train on `deficit_erased`
instead of `favorite_final_win`. This repaired the main N05 calibration
failure. N03 under-predicted `deficit_erased` by roughly 15-30 percentage
points across the middle probability deciles; N06 reduced the weighted mean
absolute decile gap to **0.040** with max gap **0.106**.

That calibration repair did **not** produce comeback-detection edge over the
strict baseline_C. Scheme U Brier improvement (`baseline_C - model`) on
`deficit_erased` is **-0.00352** with cluster-bootstrap 95% CI
**[-0.00724, +0.00013]**. Model Brier is **0.17861** versus baseline_C Brier
**0.17508**. The label change moved the model from "badly miscalibrated for
the wrong label" to "approximately calibrated but flat against the simple
baseline."

The cleanest single diagnostic is the AUC tie: N06 AUC on `deficit_erased` is
**0.7646**, while baseline_C AUC is **0.7659**. The 30 engineered features
plus protected `fav_deficit` do not rank comeback-erasure outcomes better
than the 20-cell `fav_deficit x time_bucket` lookup table. Whatever signal
the engineered features carry appears to be absorbed by their correlation
with deficit and time.

The cross-label result confirms experimental cleanliness. N06 performs badly
on `favorite_final_win`, with Brier improvement **-0.04137** and 95% CI
**[-0.05153, -0.03086]**. This is the expected mirror image of N03's poor
performance on `deficit_erased`: each model performs best on its trained
label and poorly on the other, and neither model beats baseline_C on its own
label.

The per-deficit pattern is also flat against baseline_C. N06 has no supported
positive edge at D=3, D=7, D=10, D=14, or D=21. D=3 is significantly worse;
D=7, D=10, and D=14 are near-zero with CIs crossing zero; D=21 is a tiny
positive estimate with CI crossing zero. This confirms N05's reframing of
N04: the monotonic per-deficit improvement over pre-game market probability
was about market staleness at deeper in-game deficits, not model
deep-deficit insight beyond current deficit and time.

Project implication: the validated Phase 0 feature pool is exhausted relative
to baseline_C for this question. Future research needs either feature
expansion or a different validation target. Candidate feature directions:
possession-adjusted deficit, trajectory features, and fluke-score
decomposition. Candidate validation direction: live market comparison once
historical or go-forward live-line data is available.
