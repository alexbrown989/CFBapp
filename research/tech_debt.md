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

**Locations:**

- `research/notebooks/01_trigger_events.ipynb` — budget summary cell
- `research/notebooks/02a_baseline_features.ipynb` — final budget summary cell

**Symptom:** Both notebooks print `"monthly free-tier limit: 1,000"` in
the budget summary. The actual quota on the current CFBD key is
3000/cycle, observed via `x-calllimit-remaining: 2999` after one call in
`_probe_cfbd_quota.py` on 2026-05-12 (see
`research/results/budget_reconciliation.md`).

**Severity:** Display-only. Budget enforcement does not depend on this
value — the in-notebook assertion only checks fresh CFBD calls for the
current run (must be 0 in 02a-onwards), not cumulative quota usage. The
displayed "remaining" number is wrong by 2,000.

**Fix path:** Centralize as a single constant `CFBD_MONTHLY_LIMIT = 3000`
(or read from `backend/.env` / a new `research/data/config.yaml`) so a
future tier change updates all notebooks in one place. Sweep before N03's
own budget print.

---

## 2. sklearn `CalibratedClassifierCV(cv="prefit")` deprecation

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
