# Phase 0 tech debt

Items discovered during Phase 0 notebook execution that should be addressed
before N03 (or before paper-trading in Phase 4), but do not block the
current notebook from producing valid deliverables. Each item is logged
with location(s), symptom, severity, and a fix path so a single sweep can
clear them before downstream work.

The two items below were identified during the 02a follow-up review of
commit `9822cfc`. Add new items here as discovered.

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

## Tracking

When an item is fixed, move it under a `## Resolved` section with the
resolving commit hash, instead of deleting, so the audit trail is
preserved alongside `feature_validation.schema.md`'s `Corrections` section
and `research/results/budget_reconciliation.md`.
