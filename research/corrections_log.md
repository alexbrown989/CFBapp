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

*Filled in after Commit 3 lands.*

### Cross-notebook summary

*Filled in with the 02b commit (final commit in the corrections
sequence).*
