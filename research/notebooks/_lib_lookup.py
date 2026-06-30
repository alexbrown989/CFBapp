"""Pure lookup helpers for N12/N13 probability estimates.

This module has no network access and performs no fitting. It reads the N12
long-format lookup table and live-scoring spec, then returns existing estimates
or applies the committed N06 fitted state to caller-supplied feature values.
"""

from __future__ import annotations

import json
import math
import pathlib
from functools import lru_cache
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
RESULTS_DIR = REPO_ROOT / "research" / "results"
DEFAULT_LOOKUP_PATH = RESULTS_DIR / "n12_probability_lookup.parquet"
DEFAULT_SPEC_PATH = RESULTS_DIR / "n12_live_scoring_spec.json"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _as_float(value: Any) -> float:
    if _is_missing(value):
        return math.nan
    return float(value)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _interp_isotonic(raw_prob: float, x_thresholds: list[float], y_thresholds: list[float]) -> float:
    if not x_thresholds:
        raise ValueError("isotonic calibration has no thresholds")
    return float(np.interp(raw_prob, x_thresholds, y_thresholds, left=y_thresholds[0], right=y_thresholds[-1]))


def normalize_baseline_time_bucket(time_bucket: str) -> str:
    """N09/N05 baseline_C uses Q2-first-half; N10/N11 use quarter labels."""
    return "Q2-first-half" if str(time_bucket) == "Q2" else str(time_bucket)


def build_lookup_key(metric_name: str, **kwargs: Any) -> str:
    """Build the deterministic N12 lookup key for a supported metric."""
    if metric_name == "baseline_c_rate":
        return f"deficit={int(kwargs['deficit'])}|time={normalize_baseline_time_bucket(kwargs['time_bucket'])}"
    if metric_name == "conditional_rate_full":
        return (
            f"fluke={kwargs['fluke_bucket']}|deficit={int(kwargs['deficit'])}|"
            f"time={kwargs['time_bucket']}|spread={kwargs['spread_bucket']}"
        )
    if metric_name == "ranking_rate":
        return (
            f"rank={kwargs['ranking_bucket']}|deficit={int(kwargs['deficit'])}|"
            f"time={kwargs['time_bucket']}|spread={kwargs['spread_bucket']}"
        )
    if metric_name in {"n06_calibrated_prob", "conformal_lower", "conformal_upper", "stern_winston_state_price"}:
        return (
            f"scheme={kwargs.get('scheme', 'n_a')}|fold={int(kwargs['fold'])}|"
            f"game_id={kwargs['game_id']}|trigger_sequence={int(kwargs['trigger_sequence'])}|"
            f"deficit={int(kwargs['deficit'])}"
        )
    if metric_name == "market_no_vig_historical":
        if "ranking_bucket" in kwargs and kwargs.get("ranking_bucket") is not None:
            return build_lookup_key("ranking_rate", **kwargs)
        return build_lookup_key("conditional_rate_full", **kwargs)
    raise ValueError(f"Unsupported metric_name for lookup key: {metric_name}")


@lru_cache(maxsize=4)
def load_lookup(path: str | pathlib.Path = DEFAULT_LOOKUP_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


@lru_cache(maxsize=4)
def load_scoring_spec(path: str | pathlib.Path = DEFAULT_SPEC_PATH) -> dict[str, Any]:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def get_estimates(
    deficit: int,
    time_bucket: str,
    fluke_bucket: str | None = None,
    spread_bucket: str | None = None,
    ranking_bucket: str | None = None,
    *,
    lookup_path: str | pathlib.Path = DEFAULT_LOOKUP_PATH,
) -> pd.DataFrame:
    """Return all N12 rows matching the supplied live game-state buckets."""
    lookup = load_lookup(lookup_path)
    keys = [build_lookup_key("baseline_c_rate", deficit=deficit, time_bucket=time_bucket)]
    if fluke_bucket is not None and spread_bucket is not None:
        keys.append(
            build_lookup_key(
                "conditional_rate_full",
                fluke_bucket=fluke_bucket,
                deficit=deficit,
                time_bucket=time_bucket,
                spread_bucket=spread_bucket,
            )
        )
    if ranking_bucket is not None and spread_bucket is not None:
        keys.append(
            build_lookup_key(
                "ranking_rate",
                ranking_bucket=ranking_bucket,
                deficit=deficit,
                time_bucket=time_bucket,
                spread_bucket=spread_bucket,
            )
        )
    return lookup[lookup["lookup_key"].isin(keys)].copy()


def get_baseline_c(
    deficit: int,
    time_bucket: str,
    *,
    label: str = "deficit_erased",
    lookup_path: str | pathlib.Path = DEFAULT_LOOKUP_PATH,
) -> pd.DataFrame:
    key = build_lookup_key("baseline_c_rate", deficit=deficit, time_bucket=time_bucket)
    lookup = load_lookup(lookup_path)
    return lookup[
        (lookup["lookup_key"] == key)
        & (lookup["metric_name"] == "baseline_c_rate")
        & (lookup["label"] == label)
    ].copy()


def _select_fit(spec: dict[str, Any], scheme: str | None, fold: int | None) -> dict[str, Any]:
    if scheme is None and fold is None:
        dep = spec["deployment_choice"]
        scheme = dep["scheme"]
        fold = int(dep["fold"])
    for fit in spec["n06_fitted_state"]["fits"]:
        if fit["scheme"] == scheme and int(fit["fold"]) == int(fold):
            return fit
    raise KeyError(f"No N06 fitted state for scheme={scheme!r}, fold={fold!r}")


def score_live_trigger(
    feature_dict: dict[str, Any],
    *,
    scheme: str | None = None,
    fold: int | None = None,
    spec_path: str | pathlib.Path = DEFAULT_SPEC_PATH,
) -> dict[str, float | str | int]:
    """Score one trigger with the committed N06 fitted state.

    If `scheme` and `fold` are omitted, the Scheme E deployment model is used.
    Historical verification passes the row's committed scheme/fold explicitly.
    Missingness indicators are inferred from raw core-feature nulls unless the
    caller supplies the indicator column directly.
    """
    spec = load_scoring_spec(spec_path)
    fit = _select_fit(spec, scheme, fold)
    indicator_meta = spec["n06_fitted_state"]["indicator_meta"]

    model_values: dict[str, float] = {}
    for feature in fit["core_features"]:
        raw_val = feature_dict.get(feature)
        if _is_missing(raw_val):
            model_values[feature] = float(fit["imputation_medians"][feature])
        else:
            model_values[feature] = float(raw_val)

    for indicator in fit["indicator_columns"]:
        if indicator in feature_dict and not _is_missing(feature_dict[indicator]):
            model_values[indicator] = float(feature_dict[indicator])
        else:
            core = indicator_meta[indicator]["core_feature"]
            model_values[indicator] = 1.0 if _is_missing(feature_dict.get(core)) else 0.0

    z = float(fit["logistic_regression"]["intercept"])
    means = fit["standard_scaler"]["mean"]
    scales = fit["standard_scaler"]["scale"]
    coefs = fit["logistic_regression"]["coefficients"]
    for col in fit["model_columns"]:
        scaled = (model_values[col] - float(means[col])) / float(scales[col])
        z += scaled * float(coefs[col])

    raw = _sigmoid(z)
    iso = fit["isotonic_calibration"]
    calibrated = _interp_isotonic(raw, iso["x_thresholds"], iso["y_thresholds"])
    q_by_fold = spec.get("conformal", {}).get("q_hat_by_fold", {})
    q = q_by_fold.get(str(fold if fold is not None else fit["fold"]))
    if q is None:
        q = spec.get("conformal", {}).get("deployment_q_hat")
    q = float(q) if q is not None else math.nan
    lower = max(0.0, calibrated - q) if not math.isnan(q) else math.nan
    upper = min(1.0, calibrated + q) if not math.isnan(q) else math.nan
    return {
        "scheme": fit["scheme"],
        "fold": int(fit["fold"]),
        "raw_model_prob": float(raw),
        "calibrated_prob": float(calibrated),
        "conformal_lower": float(lower),
        "conformal_upper": float(upper),
        "conformal_q_hat": float(q),
    }


def stern_winston_price(
    score_diff: float,
    seconds_remaining: float,
    *,
    spec_path: str | pathlib.Path = DEFAULT_SPEC_PATH,
) -> float:
    """Compute the N08 Stern-Winston CFB state-price approximation."""
    spec = load_scoring_spec(spec_path)
    params = spec["stern_winston"]
    seconds = min(max(float(seconds_remaining), 0.0), 3600.0)
    if seconds <= 0:
        return 1.0 if float(score_diff) >= 0.5 else 0.0
    std = float(params["empirical_cfb_std"]) * math.sqrt(seconds / 3600.0)
    z = (0.5 - float(score_diff)) / std
    return float(1.0 - NormalDist().cdf(z))
