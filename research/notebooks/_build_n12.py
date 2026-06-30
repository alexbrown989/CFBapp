"""Deterministic builder for research/notebooks/12_probability_lookup_layer.ipynb."""

from __future__ import annotations

import json
import pathlib
import textwrap

OUT = pathlib.Path(__file__).resolve().parent / "12_probability_lookup_layer.ipynb"

CELLS: list[tuple[str, str, str]] = []


def add(cell_type: str, cell_id: str, src: str) -> None:
    CELLS.append((cell_type, cell_id, textwrap.dedent(src).lstrip("\n")))


add("markdown", "m12_0000", """
# Notebook 12 -- Unified probability lookup layer

N12 consolidates committed N03-N11 probability/rate artifacts into a single
long-format lookup layer for N13 and dashboard use. It does not train models,
fetch data, or create new estimates.

The critical gate is independent live-scoring reproduction: the exported
`_lib_lookup.score_live_trigger()` helper must re-score committed N06 held-out
trigger rows from raw feature values and reproduce committed N06 probabilities
to within floating-point tolerance.
""")


add("code", "c12_0001", r"""
from __future__ import annotations

import importlib
import json
import math
import pathlib
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

NOTEBOOK_DIR = pathlib.Path(".").resolve()
if NOTEBOOK_DIR.name != "notebooks":
    NOTEBOOK_DIR = pathlib.Path("research/notebooks").resolve()
RESEARCH_DIR = NOTEBOOK_DIR.parent
REPO_ROOT = RESEARCH_DIR.parent
RESULTS_DIR = RESEARCH_DIR / "results"

sys.path.insert(0, str(NOTEBOOK_DIR))
import _lib_lookup

N05_RATES = RESULTS_DIR / "n05_descriptive_rates.parquet"
N06_PRED = RESULTS_DIR / "n06_calibrated_predictions.parquet"
N06_SPEC = RESULTS_DIR / "n06_model_spec.json"
N06_STATE = RESULTS_DIR / "n06_full_fitted_state.json"
N08_DIAG = RESULTS_DIR / "n08_diagnostic_predictions.parquet"
N08_PRICE_SPEC = RESULTS_DIR / "n08_price_conversion_spec.json"
N09_STRAT = RESULTS_DIR / "n09_trigger_state_stratifications.parquet"
N09_BASELINE = RESULTS_DIR / "n09_baseline_analysis.json"
N10_RATES = RESULTS_DIR / "n10_conditional_rates.parquet"
N10_ANALYSIS = RESULTS_DIR / "n10_conditional_analysis.json"
N11_RANKING = RESULTS_DIR / "n11_ranking_stratification.parquet"
N11_ANALYSIS = RESULTS_DIR / "n11_analysis_results.json"

N12_LOOKUP = RESULTS_DIR / "n12_probability_lookup.parquet"
N12_SCORING_SPEC = RESULTS_DIR / "n12_live_scoring_spec.json"
N12_KEY_SCHEMA = RESULTS_DIR / "n12_lookup_key_schema.md"
N12_SUMMARY = RESULTS_DIR / "n12_summary_report.md"
LIB_LOOKUP = NOTEBOOK_DIR / "_lib_lookup.py"

SOURCE_ARTIFACTS = [
    N05_RATES,
    N06_PRED,
    N06_SPEC,
    N06_STATE,
    N08_DIAG,
    N08_PRICE_SPEC,
    N09_STRAT,
    N09_BASELINE,
    N10_RATES,
    N10_ANALYSIS,
    N11_RANKING,
    N11_ANALYSIS,
    LIB_LOOKUP,
]
for path in SOURCE_ARTIFACTS:
    assert path.exists(), f"Missing N12 source artifact: {path}"

LOOKUP_COLUMNS = [
    "lookup_key",
    "dimension_set",
    "metric_name",
    "label",
    "value",
    "ci_lower",
    "ci_upper",
    "n_events",
    "n_games",
    "n_seasons",
    "reliability_flag",
    "source_notebook",
    "source_artifact",
]

print(f"[ok] N12 setup at {NOTEBOOK_DIR}")
print(f"[ok] source artifacts: {len(SOURCE_ARTIFACTS)}")
""")


add("code", "c12_0002", r"""
def rel(path: pathlib.Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def json_load(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    val = float(value)
    return None if math.isnan(val) or math.isinf(val) else val


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return int(value)


def add_lookup_row(
    rows: list[dict[str, Any]],
    *,
    lookup_key: str,
    dimension_set: str,
    metric_name: str,
    label: str,
    value: Any,
    source_notebook: str,
    source_artifact: pathlib.Path,
    ci_lower: Any = None,
    ci_upper: Any = None,
    n_events: Any = None,
    n_games: Any = None,
    n_seasons: Any = None,
    reliability_flag: str = "n_a",
) -> None:
    rows.append({
        "lookup_key": lookup_key,
        "dimension_set": dimension_set,
        "metric_name": metric_name,
        "label": label,
        "value": safe_float(value),
        "ci_lower": safe_float(ci_lower),
        "ci_upper": safe_float(ci_upper),
        "n_events": safe_int(n_events),
        "n_games": safe_int(n_games),
        "n_seasons": safe_int(n_seasons),
        "reliability_flag": reliability_flag or "n_a",
        "source_notebook": source_notebook,
        "source_artifact": rel(source_artifact),
    })


def ci_bounds(rec: dict[str, Any], *, prefer: str = "wilson") -> tuple[float | None, float | None]:
    ci = rec.get(f"{prefer}_ci") or rec.get("wilson_ci") or rec.get("bootstrap_ci") or {}
    return ci.get("lower"), ci.get("upper")


def add_rate_metric_rows(
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    metric_name: str,
    dimension_set: str,
    key_builder,
    source_notebook: str,
    source_artifact: pathlib.Path,
    ci_prefer: str = "wilson",
) -> None:
    for rec in records:
        key = key_builder(rec)
        for label in ["favorite_final_win", "deficit_erased"]:
            if label not in rec:
                continue
            label_rec = rec[label]
            lo, hi = ci_bounds(label_rec, prefer=ci_prefer)
            add_lookup_row(
                rows,
                lookup_key=key,
                dimension_set=dimension_set,
                metric_name=metric_name,
                label=label,
                value=label_rec.get("rate"),
                ci_lower=lo,
                ci_upper=hi,
                n_events=rec.get("n_events"),
                n_games=rec.get("n_games"),
                n_seasons=rec.get("n_seasons"),
                reliability_flag=rec.get("thin_flag", "n_a"),
                source_notebook=source_notebook,
                source_artifact=source_artifact,
            )


def add_market_rows(
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    dimension_set: str,
    key_builder,
    source_notebook: str,
    source_artifact: pathlib.Path,
) -> None:
    for rec in records:
        value = rec.get("mean_pregame_no_vig_implied_prob")
        if value is None:
            continue
        add_lookup_row(
            rows,
            lookup_key=key_builder(rec),
            dimension_set=dimension_set,
            metric_name="market_no_vig_historical",
            label="favorite_final_win",
            value=value,
            n_events=rec.get("n_events"),
            n_games=rec.get("n_games"),
            n_seasons=rec.get("n_seasons"),
            reliability_flag=rec.get("thin_flag", "n_a"),
            source_notebook=source_notebook,
            source_artifact=source_artifact,
        )


print("[ok] row helpers defined")
""")


add("code", "c12_0003", r"""
n05_rates = pd.read_parquet(N05_RATES)
n06_pred = pd.read_parquet(N06_PRED)
n08_diag = pd.read_parquet(N08_DIAG)
n09_strat = pd.read_parquet(N09_STRAT)
n10_rates = pd.read_parquet(N10_RATES)
n11_ranking = pd.read_parquet(N11_RANKING)
n09_baseline = json_load(N09_BASELINE)
n10_analysis = json_load(N10_ANALYSIS)
n11_analysis = json_load(N11_ANALYSIS)
n06_state = json_load(N06_STATE)
n08_price = json_load(N08_PRICE_SPEC)

assert len(n05_rates) == 11416
assert len(n09_strat) == 11412
assert len(n10_rates) == 11412
assert len(n11_ranking) == 11412
assert n06_state["reproduction_gate"]["passed"] is True

lookup_rows: list[dict[str, Any]] = []

# Metric: baseline_c_rate from N09 20-cell table, both labels.
baseline_records = n09_baseline["section1_baseline_C"]["twenty_cell_rate_table"]
for rec in baseline_records:
    key = _lib_lookup.build_lookup_key(
        "baseline_c_rate",
        deficit=rec["fav_deficit"],
        time_bucket=rec["time_bucket"],
    )
    lo = rec.get("bootstrap_ci", {}).get("lower")
    hi = rec.get("bootstrap_ci", {}).get("upper")
    add_lookup_row(
        lookup_rows,
        lookup_key=key,
        dimension_set="deficit|time_bucket",
        metric_name="baseline_c_rate",
        label=rec["label"],
        value=rec["rate"],
        ci_lower=lo,
        ci_upper=hi,
        n_events=rec.get("n_events"),
        n_games=rec.get("n_games"),
        n_seasons=rec.get("n_seasons"),
        reliability_flag=rec.get("thin_flag", "n_a"),
        source_notebook="N09",
        source_artifact=N09_BASELINE,
    )

# Metric: historical N06 calibrated probabilities from committed held-out predictions.
for rec in n06_pred.to_dict(orient="records"):
    key = _lib_lookup.build_lookup_key(
        "n06_calibrated_prob",
        scheme=rec["scheme"],
        fold=rec["fold"],
        game_id=rec["game_id"],
        trigger_sequence=rec["trigger_sequence"],
        deficit=rec["fav_deficit"],
    )
    add_lookup_row(
        lookup_rows,
        lookup_key=key,
        dimension_set="scheme|fold|game_id|trigger_sequence|deficit",
        metric_name="n06_calibrated_prob",
        label="deficit_erased",
        value=rec["calibrated_prob"],
        reliability_flag="n_a",
        source_notebook="N06",
        source_artifact=N06_PRED,
    )

# Metrics: conformal bounds and Stern-Winston state price from N08 diagnostics.
for rec in n08_diag.to_dict(orient="records"):
    key = _lib_lookup.build_lookup_key(
        "conformal_lower",
        scheme="U",
        fold=rec["fold"],
        game_id=rec["game_id"],
        trigger_sequence=rec["trigger_sequence"],
        deficit=rec["fav_deficit"],
    )
    for metric in ["conformal_lower", "conformal_upper"]:
        add_lookup_row(
            lookup_rows,
            lookup_key=key,
            dimension_set="fold|game_id|trigger_sequence|deficit",
            metric_name=metric,
            label="deficit_erased",
            value=rec[metric],
            reliability_flag="n_a",
            source_notebook="N08",
            source_artifact=N08_DIAG,
        )
    for label in ["favorite_final_win", "deficit_erased"]:
        add_lookup_row(
            lookup_rows,
            lookup_key=key,
            dimension_set="fold|game_id|trigger_sequence|deficit",
            metric_name="stern_winston_state_price",
            label=label,
            value=rec["baseline_sw_cfb_prob"],
            reliability_flag="n_a",
            source_notebook="N08",
            source_artifact=N08_DIAG,
        )

# Metric: conditional_rate_full and market reference from N10 dashboard tier.
conditional_records = n10_analysis["tier3_dashboard_only"]
conditional_key = lambda rec: _lib_lookup.build_lookup_key(
    "conditional_rate_full",
    fluke_bucket=rec["fluke_bucket"],
    deficit=rec["fav_deficit"],
    time_bucket=rec["time_bucket"],
    spread_bucket=rec["spread_bucket"],
)
add_rate_metric_rows(
    lookup_rows,
    conditional_records,
    metric_name="conditional_rate_full",
    dimension_set="fluke|deficit|time_bucket|spread",
    key_builder=conditional_key,
    source_notebook="N10",
    source_artifact=N10_ANALYSIS,
)
add_market_rows(
    lookup_rows,
    conditional_records,
    dimension_set="fluke|deficit|time_bucket|spread",
    key_builder=conditional_key,
    source_notebook="N10",
    source_artifact=N10_ANALYSIS,
)

# Metric: ranking_rate and market reference from N11 matched cells.
ranking_records = n11_analysis["matched_ranking_deficit_time_spread_cells"]
ranking_key = lambda rec: _lib_lookup.build_lookup_key(
    "ranking_rate",
    ranking_bucket=rec["ranking_bucket"],
    deficit=rec["fav_deficit"],
    time_bucket=rec["time_bucket"],
    spread_bucket=rec["spread_bucket"],
)
add_rate_metric_rows(
    lookup_rows,
    ranking_records,
    metric_name="ranking_rate",
    dimension_set="rank|deficit|time_bucket|spread",
    key_builder=ranking_key,
    source_notebook="N11",
    source_artifact=N11_ANALYSIS,
)
add_market_rows(
    lookup_rows,
    ranking_records,
    dimension_set="rank|deficit|time_bucket|spread",
    key_builder=ranking_key,
    source_notebook="N11",
    source_artifact=N11_ANALYSIS,
)

lookup_df = pd.DataFrame(lookup_rows, columns=LOOKUP_COLUMNS)
assert not lookup_df.empty
assert not lookup_df[["lookup_key", "dimension_set", "metric_name", "label", "source_notebook", "source_artifact"]].isna().any().any()
assert lookup_df["value"].notna().all()
lookup_df.to_parquet(N12_LOOKUP, index=False)

metric_counts = lookup_df.groupby("metric_name").size().sort_index().to_dict()
print(f"[ok] wrote {N12_LOOKUP.relative_to(REPO_ROOT)} rows={len(lookup_df):,}")
print(f"[ok] metric counts: {metric_counts}")
""")


add("code", "c12_0004", r"""
q_by_fold = {
    str(int(fold)): float(grp["conformal_q_hat"].iloc[0])
    for fold, grp in n08_diag.groupby("fold")
}
deployment_q = q_by_fold[str(int(n06_state["deployment_choice"]["fold"]))]

live_scoring_spec = {
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
    "artifact_purpose": "N12 live scoring and lookup spec assembled from committed artifacts; no model was fit and no new estimate was created.",
    "n06_fitted_state_source": rel(N06_STATE),
    "n06_fitted_state": {
        "indicator_meta": n06_state["indicator_meta"],
        "fits": n06_state["fits"],
    },
    "deployment_choice": n06_state["deployment_choice"],
    "conformal": {
        "source_artifact": rel(N08_DIAG),
        "method": "split_conformal_absolute_residual_width_from_N08",
        "q_hat_by_fold": q_by_fold,
        "deployment_q_hat": deployment_q,
        "deployment_fold": int(n06_state["deployment_choice"]["fold"]),
    },
    "stern_winston": {
        "source_artifact": rel(N08_PRICE_SPEC),
        "function_name": n08_price["function_name"],
        "formula": n08_price["formula"],
        "empirical_cfb_std": n08_price["parameters"]["empirical_cfb_std"],
        "empirical_cfb_variance": n08_price["parameters"]["empirical_cfb_variance"],
        "pfr_reference_std": n08_price["parameters"]["pfr_reference_std"],
        "training_years": n08_price["parameters"]["training_years"],
        "limitations": n08_price["limitations"],
    },
    "source_artifacts": [rel(path) for path in SOURCE_ARTIFACTS],
}
N12_SCORING_SPEC.write_text(json.dumps(live_scoring_spec, indent=2), encoding="utf-8")
print(f"[ok] wrote {N12_SCORING_SPEC.relative_to(REPO_ROOT)} size={N12_SCORING_SPEC.stat().st_size:,} bytes")
""")


add("code", "c12_0005", r"""
importlib.reload(_lib_lookup)
_lib_lookup.load_scoring_spec.cache_clear()
_lib_lookup.load_lookup.cache_clear()

feature_cache: dict[tuple[str, int], list[str]] = {}
for fit in live_scoring_spec["n06_fitted_state"]["fits"]:
    feature_cache[(fit["scheme"], int(fit["fold"]))] = fit["core_features"]

raw_diffs: list[float] = []
cal_diffs: list[float] = []
for rec in n06_pred.to_dict(orient="records"):
    scheme = rec["scheme"]
    fold = int(rec["fold"])
    features = {feat: rec.get(feat) for feat in feature_cache[(scheme, fold)]}
    scored = _lib_lookup.score_live_trigger(
        features,
        scheme=scheme,
        fold=fold,
        spec_path=N12_SCORING_SPEC,
    )
    raw_diffs.append(abs(float(scored["raw_model_prob"]) - float(rec["raw_model_prob"])))
    cal_diffs.append(abs(float(scored["calibrated_prob"]) - float(rec["calibrated_prob"])))

reproduction_check = {
    "function": "research/notebooks/_lib_lookup.py::score_live_trigger",
    "source_predictions": rel(N06_PRED),
    "rows_checked": int(len(n06_pred)),
    "raw_model_prob_max_abs_diff": float(max(raw_diffs)),
    "calibrated_prob_max_abs_diff": float(max(cal_diffs)),
    "calibrated_prob_mean_abs_diff": float(np.mean(cal_diffs)),
    "tolerance": 1e-6,
    "passed": bool(max(cal_diffs) < 1e-6),
}
assert reproduction_check["passed"], reproduction_check

live_scoring_spec["n12_reproduction_check"] = reproduction_check
N12_SCORING_SPEC.write_text(json.dumps(live_scoring_spec, indent=2), encoding="utf-8")

# Verify baseline_C rows are direct copies of N09 values.
lookup_df = pd.read_parquet(N12_LOOKUP)
for rec in baseline_records:
    key = _lib_lookup.build_lookup_key("baseline_c_rate", deficit=rec["fav_deficit"], time_bucket=rec["time_bucket"])
    row = lookup_df[
        (lookup_df["lookup_key"] == key)
        & (lookup_df["metric_name"] == "baseline_c_rate")
        & (lookup_df["label"] == rec["label"])
    ]
    assert len(row) == 1
    assert abs(float(row["value"].iloc[0]) - float(rec["rate"])) < 1e-15

print(f"[ok] score_live_trigger reproduction rows={reproduction_check['rows_checked']:,}")
print(f"[ok] raw max abs diff={reproduction_check['raw_model_prob_max_abs_diff']:.12g}")
print(f"[ok] calibrated max abs diff={reproduction_check['calibrated_prob_max_abs_diff']:.12g}")
""")


add("code", "c12_0006", r"""
schema_lines = [
    "# N12 Lookup Key Schema",
    "",
    "N12 stores every estimate in long format. Live callers should build the same `lookup_key` strings below and filter by `metric_name` and `label`.",
    "",
    "## Columns",
    "",
    "| column | meaning |",
    "| --- | --- |",
    "| `lookup_key` | deterministic bucket or trigger-state identifier |",
    "| `dimension_set` | pipe-delimited dimension names used by the key |",
    "| `metric_name` | estimate type |",
    "| `label` | `favorite_final_win` or `deficit_erased` |",
    "| `value` | probability/rate estimate |",
    "| `ci_lower`, `ci_upper` | confidence bounds when present |",
    "| `n_events`, `n_games`, `n_seasons` | supporting sample size when applicable |",
    "| `reliability_flag` | `reliable`, `thin`, `unreliable`, or `n_a` |",
    "| `source_notebook`, `source_artifact` | provenance |",
    "",
    "## Key Rules",
    "",
    "- `baseline_c_rate`: `deficit={d}|time={t}` where `t` is one of `Q1`, `Q2-first-half`, `Q3`, `Q4`. Helper calls normalize `Q2` to `Q2-first-half` only for this metric.",
    "- `conditional_rate_full`: `fluke={f}|deficit={d}|time={t}|spread={s}` using N10 buckets.",
    "- `ranking_rate`: `rank={r}|deficit={d}|time={t}|spread={s}` using N11 AP ranking buckets.",
    "- `market_no_vig_historical`: uses the same key as the N10 conditional or N11 ranking bucket that produced the historical market mean.",
    "- `n06_calibrated_prob`, `conformal_lower`, `conformal_upper`, `stern_winston_state_price`: `scheme={scheme}|fold={fold}|game_id={game_id}|trigger_sequence={seq}|deficit={d}` for historical trigger-event provenance rows.",
    "",
    "## Live Scoring",
    "",
    "`research/notebooks/_lib_lookup.py::score_live_trigger(feature_dict)` uses the Scheme E N06 fitted state by default: train 2015-2023, validate 2024. For historical reproduction, pass `scheme` and `fold` explicitly.",
    "",
    "Missingness indicators are inferred from raw core-feature nulls unless the caller supplies indicator columns directly. This is required because committed N06 prediction parquets preserve raw feature nulls but do not include indicator columns.",
]
N12_KEY_SCHEMA.write_text("\n".join(schema_lines) + "\n", encoding="utf-8")

summary_rows = [
    {"metric_name": k, "rows": int(v)}
    for k, v in metric_counts.items()
]
summary_lines = [
    "# N12 -- Unified Probability Lookup Layer",
    "",
    "**Purpose:** consolidate committed N03-N11 probability estimates, historical rates, model state, conformal intervals, and analytical price references into a long-format lookup layer for N13. N12 does not train, refit, fetch data, or create new research estimates.",
    "",
    "## Critical Reproduction Gate",
    "",
    f"`_lib_lookup.score_live_trigger()` re-scored **{reproduction_check['rows_checked']:,}** committed N06 held-out rows from raw feature values.",
    "",
    f"- Raw probability max abs diff: `{reproduction_check['raw_model_prob_max_abs_diff']:.12g}`",
    f"- Calibrated probability max abs diff: `{reproduction_check['calibrated_prob_max_abs_diff']:.12g}`",
    f"- Tolerance: `{reproduction_check['tolerance']}`",
    f"- Passed: `{reproduction_check['passed']}`",
    "",
    "This is independent of the N06 re-export gate: it verifies the consolidated query helper, not just the N06 build script.",
    "",
    "## Row Counts By Metric",
    "",
    "| metric_name | rows |",
    "| --- | ---: |",
]
for row in summary_rows:
    summary_lines.append(f"| `{row['metric_name']}` | {row['rows']:,} |")

summary_lines.extend([
    "",
    "## Provenance Map",
    "",
    "- `baseline_c_rate`: N09 baseline_C 20-cell table.",
    "- `n06_calibrated_prob`: N06 committed held-out prediction parquet.",
    "- `conformal_lower` / `conformal_upper`: N08 diagnostic prediction parquet.",
    "- `stern_winston_state_price`: N08 CFB-specific Stern-Winston diagnostic probabilities.",
    "- `conditional_rate_full` and N10 `market_no_vig_historical`: N10 tier-3 conditional matrix.",
    "- `ranking_rate` and N11 `market_no_vig_historical`: N11 matched ranking matrix.",
    "- Live scoring state: N06 full fitted-state provenance export, Scheme E as default deployment model.",
    "",
    "## Conformal Uncertainty",
    "",
    f"The deployment conformal q-hat is **{deployment_q:.3f}**, producing wide prediction intervals consistent with N08's finding that per-trigger predictions carry large uncertainty. The live system (N13) should surface these intervals alongside point probabilities, not hide them. The width is the mathematical basis for conservative bet sizing: a point estimate of 40% with this interval width does not justify aggressive staking. This is honest uncertainty carried forward from committed research, not a defect in the lookup layer.",
    "",
    "## No New Estimates",
    "",
    "N12 reshapes existing committed artifacts and exports complete scoring state. The only calculations performed are deterministic key construction, schema normalization, and reproduction verification.",
])
N12_SUMMARY.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

print(f"[ok] wrote {N12_KEY_SCHEMA.relative_to(REPO_ROOT)}")
print(f"[ok] wrote {N12_SUMMARY.relative_to(REPO_ROOT)}")
print("[ok] N12 deliverables complete")
""")


def build() -> None:
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for cell_type, cell_id, src in CELLS:
        nb["cells"].append({
            "cell_type": cell_type,
            "id": cell_id,
            "metadata": {},
            "source": src.splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
        })
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"[ok] wrote {OUT} ({OUT.stat().st_size:,} bytes, {len(CELLS)} cells)")


if __name__ == "__main__":
    build()
