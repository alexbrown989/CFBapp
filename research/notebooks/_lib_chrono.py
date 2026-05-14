"""
Shared chronological-ordering helper for N02 build scripts.

Single source of truth for `_chrono_key(p)`. Each `_build_0Xx.py` script
imports `CHRONO_KEY_SOURCE` (the inlined Python source of the function)
and embeds it into the executed notebook's cell body. The notebook
remains self-contained -- no runtime import dependency on this module.

This pattern keeps the notebook deliverable portable (no notebook->lib
import coupling) while pinning the canonical function definition to one
file. If we ever amend the chrono key, all three build scripts pick up
the change on next regeneration, eliminating drift across notebooks.

Origin: lookahead-bias fix landed during corrections sweep
(see research/corrections_log.md). The earlier per-build-script
`playNumber < trig.playNumber` filter silently leaked future plays
because CFBD `playNumber` resets per drive.

Verification: full-corpus run over 8,537 games / 1.54M plays bounded
the chrono-key residual disagreement (vs `play.id` lex order) at
0.394% of triggers -- dominated by Kickoff plays where CFBD's
drive-attribution differs from the source-of-truth game clock. See
research/results/_verify_chrono_key_composite.stdout.txt.
"""

from __future__ import annotations

import inspect


def _chrono_key(p: dict) -> tuple[int, int, int, int]:
    """Composite chronological key for a CFBD play.

    Returns (period, period_seconds_elapsed, driveNumber, playNumber).

    Primary chronological signal: the actual game clock (period + elapsed-
    seconds-in-period). Secondary tiebreaker for plays sharing a clock
    value (e.g., a TD recorded at 7:27 and the post-score kickoff at the
    same 7:27): lex on (driveNumber, playNumber). See
    research/results/_verify_chrono_key_composite.stdout.txt for the
    full-corpus verification (8,537 games, 1.54M plays, 22,775 adjacent
    chrono-vs-id disagreements all classified, 749 residual (c) anomalies
    upper-bounding trigger impact at 0.394%).
    """
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


# Source-of-truth body for build-script embedding. Trim the leading
# indent so the function definition appears at column 0 inside the
# notebook cell. (`inspect.getsource` returns the def at whatever
# indent it has in this file -- here, column 0 -- so no dedent
# needed; we still strip a trailing newline so callers control it.)
CHRONO_KEY_SOURCE: str = inspect.getsource(_chrono_key).rstrip("\n")
