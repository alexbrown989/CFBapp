# Phase 0 tech debt

Items discovered during Phase 0 notebook execution that should be addressed
before N03 (or before paper-trading in Phase 4), but do not block the
current notebook from producing valid deliverables. Each item is logged
with location(s), symptom, severity, and a fix path so a single sweep can
clear them before downstream work.

Items 1 and 2 were identified during the 02a follow-up review of commit
`9822cfc`. Item 3 was identified during 02b notebook authoring (commit
`bfddc16`) when introducing sentinel-delimited splicing for the schema
sidecar. Add new items here as discovered.

---

## 1. Hardcoded monthly free-tier limit in budget cells

**STATUS: RESOLVED-FOR-02a at the 02a corrections commit in the
chrono_key sequence (alongside Commit 1 `28a1b4b`). 02a's budget cell
now uses 02c's dual-display format (stated 1,000 vs. actual 3,000).
Item remains OPEN for `research/notebooks/01_trigger_events.ipynb`
and any other notebook with the stale single-quota narrative.**

**Locations:**

- `research/notebooks/01_trigger_events.ipynb` — budget summary cell
  (still OPEN)
- `research/notebooks/02a_baseline_features.ipynb` — final budget
  summary cell (RESOLVED as of the 02a corrections commit; uses
  dual-display matching 02c)

**Symptom:** The 01 notebook still prints `"monthly free-tier limit:
1,000"` in the budget summary. The actual quota on the current CFBD
key is 3000/cycle, observed via `x-calllimit-remaining: 2999` after
one call in `_probe_cfbd_quota.py` on 2026-05-12 (see
`research/results/budget_reconciliation.md`).

**Severity:** Display-only. Budget enforcement does not depend on this
value — the in-notebook assertion only checks fresh CFBD calls for the
current run (must be 0 in 02a-onwards), not cumulative quota usage. The
displayed "remaining" number was wrong by 2,000 in 02a; fixed via the
dual-display pattern that surfaces both the stated free-tier ceiling
and the observed key-specific ceiling.

**Fix path (for the still-open 01 notebook):** Centralize as a single
constant `CFBD_MONTHLY_LIMIT = 3000` (or read from `backend/.env` /
a new `research/data/config.yaml`) so a future tier change updates
all notebooks in one place. Sweep before N03's own budget print.

---

## 2. sklearn `CalibratedClassifierCV(cv="prefit")` deprecation

**STATUS: RESOLVED at commit [hash] via sklearn pin to ==1.7.2 (held
below 1.8 to retain CalibratedClassifierCV(cv='prefit') and L1 logreg
semantics). Migration to FrozenEstimator deferred to Item 5 (production
cleanup sweep before N03).**

**Locations:**

- `research/notebooks/02a_baseline_features.ipynb`, cell `c02a000f`
  (`fit_calibrate_evaluate` helper)
- Will recur in every subsequent feature-group notebook (02b–g) that
  copies the helper, and in N03's production-model pipeline.

**Symptom:** sklearn 1.6 emits
`UserWarning: The 'cv="prefit"' option is deprecated in 1.6 and will be removed in 1.8. You can use CalibratedClassifierCV(FrozenEstimator(estimator)) instead.`
The 02a execution emitted this warning 84 times (42 fits × 2 warnings per
fit). Behavior unchanged on 1.6.x; hard break on sklearn 1.8 when the
deprecation drops.

**Fix path:** Once we pin sklearn ≥ 1.7 (which exposes
`sklearn.frozen.FrozenEstimator`), replace

```python
CalibratedClassifierCV(estimator=est, method="isotonic", cv="prefit")
```

with

```python
from sklearn.frozen import FrozenEstimator
CalibratedClassifierCV(FrozenEstimator(est), method="isotonic")
```

in the shared `fit_calibrate_evaluate` helper. Single point of edit per
notebook. Phase 0 finishes well before sklearn 1.8 is the system default,
but the fix is cheap and should land before 02b duplicates the helper.

---

## 3. 02a's `feature_validation.schema.md` writer clobbers 02b-g sections

**Location:**

- `research/notebooks/02a_baseline_features.ipynb`, cell `c02a0014` (the
  schema-write cell)

**Symptom:** 02a regenerates `feature_validation.schema.md` from scratch
on every run via `FEATURE_VALIDATION_SCHEMA.write_text(text, ...)`,
where `text` is a hard-coded f-string template containing only 02a's
content. Re-running 02a after 02b (or any 02c-g) has spliced its
section will overwrite the file and silently drop 02b-g's sentinel-
delimited sections. 02b's writer (added in commit `bfddc16`) uses
sentinel-delimited splicing
(`<!-- BEGIN: 02b opening_drive_shock --> ... <!-- END: ... -->`); 02a
does not.

**Severity:** Recoverable but silent. The CSV (`feature_validation.csv`)
is append-safe via the defensive `(feature, train_window, test_season)`
key drop, so data is fine. Only the schema-side documentation gets
clobbered; recovery is to re-run the affected 02b-g notebook(s). Risk
window: any 02a re-run between 02b's first execution and N03 silently
breaks the documentation invariant.

**Fix path:**

1. Convert 02a's writer to the same sentinel-delimited splicing pattern
   02b uses. Delimit 02a's content with
   `<!-- BEGIN: 02a baseline_efficiency --> ... <!-- END: 02a baseline_efficiency -->`.
   On write, read the existing sidecar, splice 02a's section in place
   (or append if markers absent), preserve everything else verbatim.
2. **Defensive assertion (do this first; cheap):** add a precondition
   to 02a's sidecar-write cell -- before overwriting, scan the existing
   sidecar for any `<!-- BEGIN: ... -->` sentinel whose tag is NOT
   `02a baseline_efficiency`; refuse to write and emit migration
   instructions (e.g., `found <!-- BEGIN: 02b opening_drive_shock -->;
   this 02a run would clobber 02b's section -- switch 02a to splicing
   mode per commit bfddc16 before re-running`). Prevents silent clobber
   of 02b-g sections if 02a re-runs during the 02b-g window before the
   full splicing migration lands.

Sweep before N03 in the same commit as items 1 and 2. The defensive
assertion is a ~5-line addition that buys forward safety even before
the full splicing migration is done; land it as soon as 02b's first
execution writes a `<!-- BEGIN: 02b ... -->` marker into the sidecar.

---

## 4. Scoring-playType `exclude` category strands 1,562 alt-encoding plays

**Locations:**

- `research/notebooks/02c_explosive_vs_sustained.ipynb`, cell `c02c0004`
  (registry definition: 14 playTypes mapped to `exclude` sentinel).
- `research/notebooks/02c_explosive_vs_sustained.ipynb`, points-bucket
  helper cell (the `if cat == "exclude": continue` short-circuit).

**Symptom:** 1,562 CFBD plays flagged `scoring == True` (~2% of all
recognized scoring across the 2015-2024 corpus) are routed to `exclude`
and contribute zero points to `dog_points_from_explosives`,
`dog_points_from_sustained`, or `dog_points_from_returns`. Excluded
playTypes are alt-encodings whose point value cannot be unambiguously
determined from `playType` alone (same-bucket-different-value or
cross-bucket ambiguity), or anomalous `scoring == True` false positives.

Concentration of exclusions by destination bucket (per the 02c
plan-approval investigation):

- Returns bucket alt-encodings (would have routed to
  `dog_points_from_returns` if disambiguated): `Punt` (314), `Kickoff`
  (130), `Blocked Punt` (174), `Sack` (148), `Interception` (27),
  `Blocked Field Goal` (21) -- ~814 plays, mostly worth 6+1 PAT each.
- PAT bucket alt-encodings (would have attributed to whichever bucket
  the preceding TD landed in via D12): `Uncategorized` (688) -- mostly
  1pt PATs, with at least 1 in 50 sampled being a 2pt conversion.
- Offensive bucket alt-encodings (would have routed to
  `dog_points_from_explosives` or `dog_points_from_sustained`):
  `Pass Reception` (30), `Rush` (21) -- ~51 plays, mostly OT or text-
  format anomalies that overlap safety-against-offense cases.
- Anomalous false-positives (no scoring text in playText): `Pass
  Incompletion` (3), `Penalty` (3), `End Period` (1), `Timeout` (1),
  `placeholder` (1) -- 9 plays.

**Severity:** Low if `dog_points_from_returns` passes stability under
the current exclusion. The conservative outcome is intentional: if the
feature passes stability with the cleaner (smaller) returns-bucket
values, that's real signal. If it fails stability and we'd previously
have mapped `exclude` -> dominant category, we couldn't distinguish
real signal from misclassification artifact.

**Fix path (deferred; conditional on Phase 0 stability outcome):**

If 02c stability shows `dog_points_from_returns` failing stability OR
N03 needs higher-volume returns signal, revisit the 1,562 excluded
plays via a text-branching registry extension. PlayText patterns are
documented per-playType with sampled evidence in:

- `research/results/_investigate_02c_unknown_scoring.csv` (328 sampled
  rows: 128 n=10 initial across all 18 playTypes + 200 n=50 verification
  for the four high-volume targets).
- `research/results/_investigate_02c_unknown_scoring.summary.json`
  (per-playType template counts and per-row exception details for the
  rows that didn't match the dominant template).
- `research/results/_investigate_02c_unknown_scoring_drive_attrib.csv`
  (the Fumble Recovery (Own) drive-attribution check that confirmed
  the existing points-bucket cell handles drive-level routing correctly
  via `drive_had_dog_explosive` lookup).

Implementation sketch: introduce a sub-registry `SCORING_PLAYTEXT_BRANCHES`
mapping each `exclude`d playType to a per-playText classifier function
(e.g., `Punt -> lambda txt: "safety_def" if "for a SAFETY" in txt else
"return_td"`). The points-bucket helper consults the sub-registry when
`cat == "exclude"` and applies the resolved category, including a
defensive raise for any text not matching the expected templates. The
n=50-verified template strings from the investigation CSV are the
authoritative input set.

**Identified during:** 02c notebook execution + post-investigation
mapping decision (this commit).

---

## 5. FrozenEstimator + l1_ratio=1.0 production migration

**Locations:** all 02a-g notebooks' `fit_calibrate_evaluate` helper
cells (the `CalibratedClassifierCV(estimator=..., cv="prefit")` and
`LogisticRegression(penalty="l1", ...)` constructions).

**Symptom:** sklearn 1.8 removes `cv="prefit"` (Item 2) and deprecates
`penalty="l1"` in favor of `l1_ratio=1.0` (Item 6). Current sklearn
pin (==1.7.2, held below 1.8) avoids both via Item 2's resolution, but
the modern API is required eventually.

**Fix path:** when N03 needs production-grade calibration API, do a
coordinated migration across all 02a-g and N03/N04 notebooks in a
single sweep commit:

- Replace `CalibratedClassifierCV(estimator=est, method="isotonic",
  cv="prefit")` with `CalibratedClassifierCV(FrozenEstimator(est),
  method="isotonic")`.
- Replace `LogisticRegression(penalty="l1", C=..., solver=...)`
  with `LogisticRegression(l1_ratio=1.0, C=..., solver=...)`.
- Bump sklearn pin past 1.8 in `backend/pyproject.toml`.
- Verify L1 behavior via coefficient sparsity check (per Item 6).
- Re-execute all 02a-g notebooks under the new pin and confirm
  `feature_validation.csv` rows are byte-identical or document
  any numerical drift.

Do NOT migrate piecemeal — single coordinated commit, not per-
notebook patches. Piecemeal migration creates a state where some
notebooks use the old API and others use the new, and the validated
feature set's reproducibility chain breaks.

**Identified during:** Commit 2 of the N02c re-execution prep
sequence (sklearn pin to `==1.7.2`).

---

## 6. LogisticRegression `penalty='l1'` -> silent L2 fallback in sklearn 1.8+

**Locations:** all 02a-g notebooks' `fit_calibrate_evaluate` helper
cells; any N03/N04 model code that uses `LogisticRegression`.

**Symptom:** sklearn 1.8 deprecated `penalty="l1"` in favor of
`l1_ratio=1.0`. Under 1.8+, passing `penalty="l1"` silently falls
back to L2 regularization via the new default `l1_ratio=0.0`,
emitting only a `FutureWarning`. This changes feature selection
behavior (L1 produces sparse coefficients; L2 does not), which
means stability-test results computed under 1.8+ would differ
from those computed under <1.8 without any visible error.

**Fix path:** when bumping sklearn pin past 1.8 (during Item 5
sweep), explicitly verify the swap is semantically equivalent:
fit `LogisticRegression(l1_ratio=1.0, C=1.0, solver="liblinear")`
on a small synthetic dataset and check `coef_` has the L1
sparsity pattern (some coefficients exactly zero). Document the
verification in the Item 5 sweep commit.

**Why this is its own item:** Item 2 (`cv="prefit"`) is a hard error
that halts execution. Item 6 is a silent behavioral change that
requires intentional testing to catch. They have different fix
horizons and different risk profiles -- bundling them under one
item would conflate "the calibrator API moved" with "the
regularization API moved AND can silently corrupt results."

**Identified during:** Commit 2 of the N02c re-execution prep
sequence (sklearn pin to `==1.7.2`).

---

## 7. Phase 0 notebooks run under global Python, not the backend venv

**Locations:**

- `backend/pyproject.toml` -- manifests `scikit-learn==1.7.2` and all
  other Phase 0 deps, but is not installed against any environment
  that Phase 0 notebook execution actually uses.
- All `research/notebooks/02*.ipynb` -- execute via `python -m jupyter
  nbconvert` against the user's global Python 3.12 install
  (`C:\Users\Alexander\AppData\Local\Programs\Python\Python312`,
  per the nbconvert traceback paths observed during 02c's halt).

**Symptom:** The manifest at `backend/pyproject.toml` and the runtime
where notebooks actually execute have diverged. The manifest serves
as intent-only documentation right now; the user's global site-
packages is the de-facto runtime. Sklearn version (and any future
pin change) must be applied as two independent operations: edit the
manifest, then `pip install --force-reinstall` against the global
environment. The two steps can silently drift -- the manifest is
the canonical record of which version SHOULD be installed, but
nothing currently enforces that the runtime matches.

**Severity:** Medium-high if reproducibility is needed before N03.
The Phase 0 `feature_validation.csv` rows are generated under
whatever sklearn the global Python has installed. If that
installation drifts away from `==1.7.2` (e.g., a future `pip install`
on an unrelated dep silently upgrades sklearn), the rows are no
longer reproducible from the manifest alone.

**Fix path:** Before N03 production code begins:

- Create a project venv (`.venv/` at the repo root or
  `backend/.venv/`).
- `pip install -e backend/` against that venv so the manifest's pins
  become the runtime.
- Switch notebook execution to that venv (either by reconfiguring
  Jupyter's kernelspec to point at `.venv/Scripts/python.exe` on
  Windows, or by running nbconvert via the venv's Python directly).
- Document the venv setup in `README.md` so it's reproducible.
- Verify by re-running 02a and 02b under the venv and confirming
  `feature_validation.csv` rows are byte-identical to the committed
  versions (or documenting any drift).

Until that lands, sklearn and any future pin changes must be applied
manually via `pip` against the global environment, with the
`pyproject.toml` edit serving as the canonical record of which
version SHOULD be installed.

**Identified during:** Commit 2 of the N02c re-execution prep
sequence -- the `scikit-learn==1.7.2` pin landed in
`backend/pyproject.toml` but had to be separately applied to the
global Python via `pip install --force-reinstall scikit-learn==1.7.2`
because notebooks don't run against the manifest-installed
environment.

---

## 8. 02e `fav_yards_per_point`: D7 bucket (b) far above plan tech-debt threshold

**Locations:**

- Plan / schema: `feature_validation.schema.md`, 02e **D7** subsection
  (bucket (b) count and `%` of triggers).
- Implementation: `research/notebooks/_build_02e.py`,
  `_classify_yards_per_point_null_bucket` plus D8 paired-indicator wiring.

**Symptom:** At execution (`v1_red_zone_failure`), bucket **(b)** –
drives exist but **0 fav offensive points** – hit **4,517** triggers
**(39.57%** of 11,416 in-scope triggers). The plan flagged **> 200**
as the threshold warranting **N03** review of a worst-case-imputation
alternative (`tech_debt` / “tech-debt entry for N03”). The empirical
ratio is roughly **22×** that gate. Despite this, walk-forward stability
passed **PASS** (**2/3** folds with positive Brier improvement — R6).

**Interpretation:** The paired `fav_yards_per_point_is_null` indicator
(absorbing structured missingness across both bucket (a) and (b))
may carry more of the signal than the median-imputed efficiency
scalar for triggers in bucket **(b)** — worth an explicit hypothesis
during N03 ablation (`NULL`/`Non-NULL`-only regressions vs full
paired design). Execution did **not** change the locked Mode B design;
this item captures follow-up analytics only.

**Severity:** Non-blocking for Phase 0 deliverables — same PASS bar as
committed rows in `feature_validation.csv`. Signals that **N03**
should revisit imputation-choice sensitivity when integrating this
feature.

**Fix path (N03 sweep):**

- Ablation table: baseline + continuous only; indicator only;
  paired-as-shipped — on overlapping train/test protocol.
- If continuous contribution is negligible in bucket **(b)** dominant
  regimes, consider tech-debt child item: directional imputation
  (bounded “bad efficiency” substitute for bucket **(b)** only) vs
  global median; must preserve R16 paired-indicator coherence.

**Identified during:** Notebook `02e_red_zone_failure` execution /
schema write (deliverables commit pending user authorization).

---

## Tracking

When an item is fixed, move it under a `## Resolved` section with the
resolving commit hash, instead of deleting, so the audit trail is
preserved alongside `feature_validation.schema.md`'s `Corrections` section
and `research/results/budget_reconciliation.md`.

---

## Resolved process incidents

### 2026-05-12: Co-authored-by trailer leak on four commits

Cursor agent shell auto-injected `Co-authored-by: Cursor` trailer on
commits 2e17807, bfddc16, 6d1fbc9, c7cd43f despite the standing rule
from 2e17807 review. Root cause: CURSOR_AGENT=1 environment variable
triggers Cursor's commit wrapper, which prepends the trailer to the
commit message after `-m`/`-F` is consumed but before the commit
object is written. The wrapper also re-runs on `git commit --amend`,
so post-commit stripping via `git interpret-trailers --trim-empty`
followed by amend does not work — the amend re-injects.

The naive `git commit --trailer "Co-authored-by="` does NOT suppress:
it appends a second empty trailer while leaving the auto-injected
`Co-authored-by: Cursor <cursoragent@cursor.com>` line intact.

Verified working invocation (tested 2026-05-12 on two no-op commits,
both reset; body contained zero `Co-authored-by` lines):

    git -c trailer.ifExists=replace \
        -c trailer.ifMissing=doNothing \
        commit --trailer "Co-authored-by=Removed" \
        -m "<message>"

History not rewritten — the four leaked commits remain in `origin/main`
with the trailer. Future commits use the verified incantation above
and verify with `git log -1 --format=%B` post-commit. If a
`Co-authored-by` line appears, halt and surface before pushing.
