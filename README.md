# CFB Live Edge Journal

Personal sports-betting research and live-edge web app for college football.

> **Status: Phase 0 (research) — no app code yet.**
> The four hard gates in `BUILD_SPEC.md` must clear before any FastAPI route,
> React component, or Alembic migration is written. If the hypothesis is
> disproven by research, we ship nothing — and that is a successful outcome.

## Hypothesis

A pre-game FBS favorite trailing by D points at quarter Q has a true win
probability that exceeds the live moneyline's devigged implied probability by
≥X%, when conditioned on a feature set that distinguishes "real underdog
domination" from "early-game variance."

Primary success metrics:

- Positive Closing Line Value (CLV) across all three walk-forward test seasons
- Expected Calibration Error (ECE) < 0.05 on every test season

Profit is a secondary, noisy signal.

## Repo layout

```
CFBapp/
├── .cursorrules         Agent behavioral rules (V5 + V5.1 + owner additions)
├── BUILD_SPEC.md        Mission, schema DDL, Phase 0 gates, Phase 4 thresholds
├── README.md            This file
├── research/
│   ├── notebooks/       Phase 0 notebooks (00 audit → 04 CLV simulation)
│   ├── data/            Local data + audit artifacts (gitignored)
│   ├── results/         Validated filters, walk-forward metrics, findings
│   └── future_features.md
├── backend/             Phase 1+ — empty stubs only until Phase 0 verdict
│   ├── pyproject.toml
│   ├── .env.example
│   ├── app/
│   └── tests/
└── frontend/            Phase 3+ — empty
```

## Where we are right now

- [x] Repo scaffolded (Action 1)
- [x] `.cursorrules` written (Action 2)
- [x] `BUILD_SPEC.md` written (Action 3)
- [x] `backend/pyproject.toml` pinned to verified PyPI versions (Action 4)
- [ ] Dependencies installed
- [ ] Phase 0, Notebook 00 (data audit)

## Reading order for new contributors

1. `BUILD_SPEC.md` — full spec including V5.1 patch and owner addendum
2. `.cursorrules` — the 22 behavioral rules that govern any agent work in this repo
3. `research/notebooks/` — Phase 0 hypothesis tests (when they exist)

## License / personal use

This is a personal research project. No license granted; do not use for
real-money betting on someone else's behalf.
