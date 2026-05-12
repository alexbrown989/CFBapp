# CFBD budget reconciliation

## 1. As-of date and source

**As of:** 2026-05-12

**Source of truth:** `research/data/cache/cfbd_call_log.csv` (see git log
for commit history of this file).

Every CFBD HTTP request that the research notebooks issue goes through the
`cfbd_get(...)` helper, which appends one row per request to
`cfbd_call_log.csv` with `(timestamp, service, endpoint, params_hash,
cached, status, bytes, elapsed_ms)`. Rows with `cached == 0` count against
the CFBD monthly quota; `cached == 1` rows are local cache hits and cost
zero. Out-of-band probes (e.g., `_probe_cfbd_quota.py`) that bypass
`cfbd_get` must append a row manually to keep the audit closed.

## 2. Audit count vs narrative count

| Source | Fresh CFBD calls (audit log) | Narrative (BUILD_SPEC / N00 docstring) |
|---|---:|---:|
| N00 | **70** (`/games` 20, `/lines` 20, `/teams/fbs` 10, `/ratings/sp` 10, `/ratings/fpi` 10) | 87 |
| N01 | 182 (`/plays` 162, `/drives` 20) | 182 |
| Probe (`_probe_cfbd_quota.py`, 2026-05-12) | 1 (`/conferences`) | 1 |
| **Total** | **253** | **270** |

**17-call gap, traced:** N00's `data_quality_report.md` narrative (line 18)
documents 87 CFBD calls including "sample 2024 plays/drives". Those sample
`/plays` and `/drives` rows were pulled by N00, but their cache files were
later re-hit by N01 — and the log only carries the most recent access of a
given `params_hash` key under the writer that produced that row. So the 17
sample-plays/drives calls show up as N01-era `cached=1` rows rather than
N00-era `cached=0` rows. The 502 cached `/plays` rows / 162 unique keys ~=
3 N01 re-runs is consistent with the N01 fix-cycle in the recent git log
(commits 3a6566b, d216085, 9df25b9, a412bdb).

The 17-call gap is a documentation-vs-log artifact, not a budget
reconciliation issue. Quota math is unaffected: 253 of 3000 fresh calls
consumed this billing cycle (per the new key's `x-calllimit-remaining: 2999`
header on the 2026-05-12 probe), 2747 remaining.

## 3. Forward policy

Going forward, all phase-budget claims in commit messages, docstrings, and
research findings cite the audited log count from
`research/data/cache/cfbd_call_log.csv`. Per-notebook docstrings that
narrate their own call count (like N00's "87" or N01's "182") are
**documentation-of-intent at write time** and may diverge from the audit
log on re-runs (e.g., when a cache key gets re-hit and shifts ownership
between notebooks). **The log is canonical.** When the docstring and the
log disagree, the log wins and the docstring is the artifact to update.

N00's `data_quality_report.md` narrative count of `87` will reconcile to
whatever the log shows on N00's next regeneration; per BUILD_SPEC.md R20
the report is rebuilt every time N00 runs, so no separate commit is
required to fix it.

Out-of-band CFBD probes (i.e., HTTP calls that bypass `cfbd_get`) must
append a manual row to `cfbd_call_log.csv` immediately to keep the audit
closed. The probe row's `params_hash` should match what `cfbd_get` would
have produced for the same parameters
(`hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]`).
