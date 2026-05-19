# CFB Live Edge Journal

CFB Live Edge Journal is a research-first college football modeling project. It asks whether an in-game model can estimate a pre-game favorite's comeback probability after the favorite enters a deficit trigger state, and whether that trigger-state probability adds information beyond the pre-game betting market. The repository currently contains the completed research phase: trigger construction, feature validation, walk-forward modeling, and model-vs-market validation. Deployment work is intentionally pending until live in-game line data can be collected going forward.

## Research Hypothesis

The original hypothesis was that a pre-game FBS favorite trailing by a deficit threshold D at a game state Q has a true win probability that can be estimated better when conditioned on in-game features that distinguish "real underdog domination" from "early-game variance." Because historical live in-game moneyline data is not available for the 2022-2024 validation corpus, the final research question became:

> Given that the favorite has entered a trigger state, does the model's calibrated probability estimate predict the final game outcome more accurately than the pre-game market's implied probability did?

That is a predictive-edge question, not yet a live-betting-edge question.

## Headline Findings

The research phase produced a positive but carefully bounded result.

- **Predictive edge over pre-game market consensus is validated.** N04 reports all-fold Brier improvement of **+0.05847** for Scheme U with cluster-bootstrap 95% CI **[+0.04211, +0.07465]**. Scheme W2 is effectively identical at **+0.05847** with 95% CI **[+0.04210, +0.07429]**.
- **The win is calibration, not ranking.** The pre-game market still ranks teams better overall: market AUC **0.6812** versus model AUC **0.6650**. The model wins because it adjusts stale pre-game probability to the observed trigger state: model ECE **0.03484** versus market ECE **0.24840**.
- **The deficit pattern is mechanistically strong.** Brier improvement increases monotonically with deficit: D=3 **-0.00570**, D=7 **+0.03134**, D=10 **+0.09147**, D=14 **+0.16507**, D=21 **+0.34131**.
- **This is not a betting-edge proof.** The primary tertiary favorite-side betting simulation lost money: edge threshold +0.08, 25% Kelly, D=21 excluded produced **89** bets, **35.96%** win rate, and **-33.27% ROI**. Historical live-line data is unavailable, so live betting edge remains untested.
- **The modeling work surfaced real caution flags.** N03 had real but modest discrimination (weighted AUC **0.689016**) and fragile per-deficit calibration, especially D=10 ECE **0.0872** and D=14 ECE **0.0609**. N03 did not beat the Phase 0 pre-game alpha baseline on 2024 Brier (Delta Brier **-0.002051**).

The short version: the model validates trigger-state predictive information beyond pre-game market consensus, especially at deeper deficits. It does not yet demonstrate profitable live betting.

## Repository Navigation

- `BUILD_SPEC.md` - original project specification, phase gates, data model, and methodological constraints.
- `.cursorrules` - agent/process rules used during the research build, including commit discipline and no-history-rewrite rules.
- `README.md` - this top-level entry point.
- `RESEARCH_SUMMARY.md` - narrative writeup of the completed research phase.
- `research/`
  - `notebooks/` - canonical notebooks and deterministic builders/verifiers. Important sequence: `00_data_audit.ipynb`, `01_trigger_events.ipynb`, `02a` through `02g`, `03_walk_forward_validation.ipynb`, and `04_model_vs_market_validation.ipynb`.
  - `results/` - committed research outputs, including `feature_validation.csv`, `feature_validation.schema.md`, `n03_model_spec.json`, `n04_spec.json`, `n04_summary_report.md`, and prediction/validation parquet files.
  - `corrections_log.md` - chronological audit log of methodological corrections and honest interpretations.
  - `tech_debt.md` and `future_features.md` - deferred work, data plumbing issues, and future feature ideas.
  - `data/` - local data and cache area. Some raw/cache state is local by design.
- `backend/` - FastAPI/backend scaffold and pinned Python dependency manifest. App implementation is pending deployment-phase work.
- `frontend/` - frontend placeholder. Visualization work is pending.

## How To Reproduce

The committed result artifacts are the easiest way to inspect the research. Full reproduction requires the local CFBD cache under `research/data/cache` or equivalent CFBD access for the earlier data-building notebooks.

Recommended environment:

```powershell
cd backend
python -m pip install -e ".[research]"
cd ..
```

Notebook run order:

```text
research/notebooks/00_data_audit.ipynb
research/notebooks/01_trigger_events.ipynb
research/notebooks/02a_baseline_features.ipynb
research/notebooks/02b_opening_drive_shock.ipynb
research/notebooks/02c_explosive_vs_sustained.ipynb
research/notebooks/02d_turnover_and_short_field.ipynb
research/notebooks/02e_red_zone_failure.ipynb
research/notebooks/02f_down_distance_efficiency.ipynb
research/notebooks/02g_context_week_home_neutral.ipynb
research/notebooks/03_walk_forward_validation.ipynb
research/notebooks/04_model_vs_market_validation.ipynb
```

Most canonical notebooks have matching deterministic builders and verifiers, for example `_build_n04.py` and `_verify_n04.py`. The N03/N04 notebooks can be executed with:

```powershell
cd research/notebooks
python -m jupyter nbconvert --to notebook --execute 03_walk_forward_validation.ipynb --output 03_walk_forward_validation_executed.ipynb --ExecutePreprocessor.timeout=3600
python -m jupyter nbconvert --to notebook --execute 04_model_vs_market_validation.ipynb --output 04_model_vs_market_validation_executed.ipynb --ExecutePreprocessor.timeout=3600
```

Key files for review:

- `research/results/feature_validation.csv` - Phase 0 feature verdict rows.
- `research/results/feature_validation.schema.md` - feature-by-feature documentation and caveats.
- `research/results/n03_model_spec.json` - N03 model, calibration, pruning, sensitivity, and diagnostics.
- `research/results/n04_spec.json` - N04 market comparison configuration and structured summary.
- `research/results/n04_summary_report.md` - human-readable model-vs-market report.
- `research/corrections_log.md` - deeper audit trail, including lookahead leak post-mortem and N04 interpretation.

## Status

Research phase complete and committed. The current project conclusion is:

> Predictive edge versus pre-game market consensus is validated; live-line betting edge remains untested.

The next phase is not more backtest polishing. It is live-line data collection, deployment validation against actual in-game prices, and eventual frontend visualization if the going-forward validation supports it.

## License / Personal Use

This is a personal research project. No license is granted. Do not use it for real-money betting on someone else's behalf.
