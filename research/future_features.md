# Future Features — Parking Lot

Per V5.1 lock-the-spec policy: any "what about adding X" idea that surfaces
during Phase 0 (or later) goes here instead of expanding `BUILD_SPEC.md`.
Items are revisited only after Phase 0 produces a verdict, not before.

The point is to stop iterating on the spec and actually run the research.
Each feature group already in `BUILD_SPEC.md` is another overfitting
opportunity; adding more before validating the existing set is worse, not
better.

## Format

Each entry: a one-line description, the football mechanism, and the data
source required. Don't speculate on whether it'll work — we'll find out by
testing.

## Open ideas

_(empty — add as they come up)_

## Items deferred from V5.1 by explicit rule

- **Overtime trigger modeling.** Per `.cursorrules` rule 15, OT is excluded
  from Phase 0. OT in college football has different dynamics (each team gets
  a possession from the 25, no clock, sudden-elimination rules). Modeling OT
  is a separate research question.

- **Re-collapse / second-deficit modeling.** Per `trigger_events` first-occurrence
  semantics, only the first time a deficit threshold is hit per game is
  recorded. A favorite that goes down 7, comes back, then goes down 7 again
  has only the first instance in the table. Modeling re-collapses is a
  separate research question.
