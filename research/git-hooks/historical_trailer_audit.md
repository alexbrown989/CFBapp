# Historical `Co-authored-by` trailer audit (reference)

**Generated:** 2026-05-14 (working-tree reference; not committed unless separately authorized.)

**Scope:** All commits reachable from `main` at audit time (`git rev-list --all --count` → **39**).  
**Pattern:** start-of-line `^Co-authored-by:` in full commit body (`git log -1 --format=%B` per commit).

## Summary

- **16 / 39** commits (~41%) contained at least one  
  `Co-authored-by: Cursor <cursoragent@cursor.com>`  
  line (Cursor shell / `CURSOR_AGENT=1` wrapper behavior).
- **Severity:** cosmetic for code integrity (trailers are metadata, not tree blobs); attribution / policy concern only.
- **Remediation policy (R24(f)):** no history rewrite for cosmetic reasons; prevention is **`commit-msg` hook** (see `0a1e1fe` area) + **B3 invocation** in `.cursorrules` + user-side Cursor Agent Attribution settings.

## Affected commits (short hash — first line of subject)

| Hash | Subject |
|------|---------|
| `e87af23` | chore: initial repo scaffolding for Phase 0 (Actions 1-4) |
| `dd9ff73` | docs: addendum A.7 — substitute pre-game Elo for SP+/FPI |
| `67e22b9` | feat(research): add Notebook 00 data audit (Phase 0, unexecuted) |
| `dedfb62` | feat(research): surface classification schema-drift in quality report |
| `3856908` | feat(research): enhance data quality report with classification drift detection |
| `a906c7f` | chore(research): untrack executed notebook copy from version control |
| `ed664ec` | Phase 0 N01 scaffold: trigger_events + trigger_outcomes (unexecuted) |
| `3a6566b` | fix(n01): assert resolves trigger play by id; pregame_df index without duplicate game_id |
| `d216085` | fix(N01): explicit per-subtype scoring registry + drive-aware sort + scorer rewrite |
| `9df25b9` | chore(n01): rename pre-fix outputs to *.broken.csv for diff clarity |
| `a412bdb` | chore(n01): commit fixed trigger_events deliverables |
| `9822cfc` | Phase 0 N02a scaffold: baseline efficiency features (unexecuted) |
| `2e17807` | docs(n02a): correct feature-count narrative; tag EPA redundancies in CSV |
| `bfddc16` | Phase 0 N02b scaffold: opening-drive shock features (unexecuted) |
| `6d1fbc9` | docs(tech_debt): add item 3 (02a sidecar writer clobbering) + defensive assertion suggestion |
| `c7cd43f` | docs: add research/future_features.md with momentum hypothesis for 02c |

## Cross-references

- `research/tech_debt.md` — **Resolved process incidents** (Co-authored-by trailer leak, B3 pattern).
- `.cursorrules` — R23 commit body standards; R24(f) history rewrites; **R24(g)** per-commit authorization.
