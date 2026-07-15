"""Tiered N13 scoring backed exclusively by committed N12 artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from .config import ensure_n12_lookup_import_path
from .trigger_detect import TriggerEvent

ensure_n12_lookup_import_path()
import _lib_lookup  # type: ignore  # noqa: E402


LABELS = ("favorite_final_win", "deficit_erased")
TIER3_CERTIFIED_SOURCES = {"cached_historical", "live_parity_certified"}


class LookupIntegrityError(RuntimeError):
    """Committed lookup state is missing or ambiguous for a required key."""


@dataclass(frozen=True)
class HistoricalEstimate:
    metric_name: str
    label: str
    lookup_key: str
    value: float
    ci_lower: float | None
    ci_upper: float | None
    n_events: int | None
    n_games: int | None
    n_seasons: int | None
    reliability_flag: str
    source_notebook: str
    source_artifact: str
    estimate_type: str = "historical_descriptive"

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "HistoricalEstimate":
        return cls(
            metric_name=str(row["metric_name"]),
            label=str(row["label"]),
            lookup_key=str(row["lookup_key"]),
            value=float(row["value"]),
            ci_lower=_optional_float(row.get("ci_lower")),
            ci_upper=_optional_float(row.get("ci_upper")),
            n_events=_optional_int(row.get("n_events")),
            n_games=_optional_int(row.get("n_games")),
            n_seasons=_optional_int(row.get("n_seasons")),
            reliability_flag=str(row.get("reliability_flag") or "n_a"),
            source_notebook=str(row.get("source_notebook") or ""),
            source_artifact=str(row.get("source_artifact") or ""),
        )

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class Tier3Estimate:
    calibrated_prob: float
    raw_model_prob: float
    conformal_lower: float
    conformal_upper: float
    conformal_q_hat: float
    scheme: str
    fold: int
    feature_source: str
    parity_status: str

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class ScoringContext:
    spread_bucket: str | None = None
    ranking_bucket: str | None = None
    fluke_bucket: str | None = None
    time_bucket: str | None = None
    tier3_features: Mapping[str, Any] | None = None
    tier3_certified: bool = False
    tier3_feature_source: str | None = None
    tier3_scheme: str | None = None
    tier3_fold: int | None = None
    tier3_unavailable_reason: str | None = "unavailable - no live play feed"


@dataclass(frozen=True)
class ScoringResult:
    tier_used: int
    tier_1: dict[str, HistoricalEstimate]
    tier_2: dict[str, list[HistoricalEstimate]]
    tier_3: Tier3Estimate | None
    tier_3_unavailable_reason: str | None
    tier_reasons: dict[str, str]
    spread_bucket: str | None
    ranking_bucket: str | None
    fluke_bucket: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "tier_used": self.tier_used,
            "tier_1": {label: estimate.as_dict() for label, estimate in self.tier_1.items()},
            "tier_2": {
                metric: [estimate.as_dict() for estimate in estimates]
                for metric, estimates in self.tier_2.items()
            },
            "tier_3": None if self.tier_3 is None else self.tier_3.as_dict(),
            "tier_3_unavailable_reason": self.tier_3_unavailable_reason,
            "tier_reasons": dict(self.tier_reasons),
            "spread_bucket": self.spread_bucket,
            "ranking_bucket": self.ranking_bucket,
            "fluke_bucket": self.fluke_bucket,
        }

    def as_log_fields(self) -> dict[str, object]:
        baseline_final = self.tier_1["favorite_final_win"]
        baseline_erased = self.tier_1["deficit_erased"]
        conditional = _find_estimate(self.tier_2, "conditional_rate_full", "favorite_final_win")
        ranking = _find_estimate(self.tier_2, "ranking_rate", "favorite_final_win")
        market = _preferred_market_estimate(self.tier_2, prefer_conditional=self.fluke_bucket is not None)
        return {
            "tier_used": self.tier_used,
            "tier_reasons": dict(self.tier_reasons),
            "baseline_c_final_win": baseline_final.value,
            "baseline_c_deficit_erased": baseline_erased.value,
            "baseline_c_n_events": baseline_final.n_events,
            "baseline_c_reliability": baseline_final.reliability_flag,
            "conditional_rate_final_win": None if conditional is None else conditional.value,
            "conditional_rate_n_events": None if conditional is None else conditional.n_events,
            "ranking_rate_final_win": None if ranking is None else ranking.value,
            "ranking_rate_n_events": None if ranking is None else ranking.n_events,
            "market_no_vig_historical": None if market is None else market.value,
            "n06_calibrated_prob": None if self.tier_3 is None else self.tier_3.calibrated_prob,
            "conformal_lower": None if self.tier_3 is None else self.tier_3.conformal_lower,
            "conformal_upper": None if self.tier_3 is None else self.tier_3.conformal_upper,
            "n06_unavailable_reason": self.tier_3_unavailable_reason,
            "spread_bucket": self.spread_bucket,
            "ranking_bucket": self.ranking_bucket,
            "fluke_bucket": self.fluke_bucket,
        }


class ScoringEngine:
    """Select the highest certified tier while retaining every lower tier."""

    def score_trigger(self, trigger_state: TriggerEvent, context: ScoringContext) -> ScoringResult:
        deficit = int(trigger_state.threshold_crossed)
        quarter_bucket = context.time_bucket or _period_time_bucket(trigger_state.period)
        normalized_baseline_bucket = _lib_lookup.normalize_baseline_time_bucket(quarter_bucket)

        tier_1 = {
            label: self._required_baseline(deficit, normalized_baseline_bucket, label)
            for label in LABELS
        }
        tier_reasons = {
            "tier_1": (
                f"available - committed baseline_C lookup for deficit={deficit}, "
                f"time={normalized_baseline_bucket}"
            )
        }

        estimates = _lib_lookup.get_estimates(
            deficit=deficit,
            time_bucket=quarter_bucket,
            fluke_bucket=context.fluke_bucket,
            spread_bucket=context.spread_bucket,
            ranking_bucket=context.ranking_bucket,
        )
        tier_2: dict[str, list[HistoricalEstimate]] = {}
        for row in estimates.to_dict(orient="records"):
            metric = str(row["metric_name"])
            if metric == "baseline_c_rate":
                continue
            tier_2.setdefault(metric, []).append(HistoricalEstimate.from_row(row))

        if tier_2:
            available = ", ".join(sorted(tier_2))
            tier_reasons["tier_2"] = f"available historical descriptive enrichment: {available}"
            tier_used = 2
        else:
            tier_reasons["tier_2"] = "suppressed - no complete N12 enrichment key could be built"
            tier_used = 1

        tier_3, unavailable = self._score_tier_3(context)
        if tier_3 is None:
            tier_reasons["tier_3"] = f"suppressed - {unavailable}"
        else:
            tier_reasons["tier_3"] = (
                f"available - all 31 N06 core features supplied from certified source "
                f"{tier_3.feature_source}"
            )
            tier_used = 3

        return ScoringResult(
            tier_used=tier_used,
            tier_1=tier_1,
            tier_2=tier_2,
            tier_3=tier_3,
            tier_3_unavailable_reason=unavailable,
            tier_reasons=tier_reasons,
            spread_bucket=context.spread_bucket,
            ranking_bucket=context.ranking_bucket,
            fluke_bucket=context.fluke_bucket,
        )

    @staticmethod
    def _required_baseline(deficit: int, time_bucket: str, label: str) -> HistoricalEstimate:
        rows = _lib_lookup.get_baseline_c(deficit=deficit, time_bucket=time_bucket, label=label)
        if len(rows) != 1:
            key = _lib_lookup.build_lookup_key(
                "baseline_c_rate", deficit=deficit, time_bucket=time_bucket
            )
            raise LookupIntegrityError(
                f"expected one baseline_C row for key={key!r}, label={label!r}; got {len(rows)}"
            )
        return HistoricalEstimate.from_row(rows.iloc[0].to_dict())

    @staticmethod
    def _score_tier_3(context: ScoringContext) -> tuple[Tier3Estimate | None, str | None]:
        if not context.tier3_certified:
            return None, context.tier3_unavailable_reason or "feature source is not parity-certified"
        if context.tier3_feature_source not in TIER3_CERTIFIED_SOURCES:
            return None, "feature source is not an allowed certified source"
        if context.tier3_features is None:
            return None, "certified Tier 3 feature dictionary was not supplied"

        required = _required_core_features(context.tier3_scheme, context.tier3_fold)
        missing = sorted(required - set(context.tier3_features))
        if missing:
            return None, f"missing required N06 core features: {', '.join(missing)}"

        scored = _lib_lookup.score_live_trigger(
            dict(context.tier3_features),
            scheme=context.tier3_scheme,
            fold=context.tier3_fold,
        )
        return (
            Tier3Estimate(
                calibrated_prob=float(scored["calibrated_prob"]),
                raw_model_prob=float(scored["raw_model_prob"]),
                conformal_lower=float(scored["conformal_lower"]),
                conformal_upper=float(scored["conformal_upper"]),
                conformal_q_hat=float(scored["conformal_q_hat"]),
                scheme=str(scored["scheme"]),
                fold=int(scored["fold"]),
                feature_source=str(context.tier3_feature_source),
                parity_status="certified_cached" if context.tier3_feature_source == "cached_historical" else "live_certified",
            ),
            None,
        )


def score_trigger(trigger_state: TriggerEvent, context: ScoringContext) -> ScoringResult:
    """Primary Stage 2 interface."""
    return ScoringEngine().score_trigger(trigger_state, context)


def _required_core_features(scheme: str | None, fold: int | None) -> set[str]:
    spec = _lib_lookup.load_scoring_spec()
    if scheme is None and fold is None:
        deployment = spec["deployment_choice"]
        scheme = str(deployment["scheme"])
        fold = int(deployment["fold"])
    matches = [
        fit
        for fit in spec["n06_fitted_state"]["fits"]
        if fit["scheme"] == scheme and int(fit["fold"]) == int(fold)
    ]
    if len(matches) != 1:
        raise LookupIntegrityError(f"expected one N06 fit for scheme={scheme}, fold={fold}; got {len(matches)}")
    return set(matches[0]["core_features"])


def _period_time_bucket(period: int) -> str:
    if period in (1, 2, 3):
        return f"Q{period}"
    if period >= 4:
        return "Q4"
    raise ValueError(f"period must be positive, got {period}")


def _find_estimate(
    tier_2: Mapping[str, list[HistoricalEstimate]], metric: str, label: str
) -> HistoricalEstimate | None:
    matches = [estimate for estimate in tier_2.get(metric, []) if estimate.label == label]
    return matches[0] if matches else None


def _preferred_market_estimate(
    tier_2: Mapping[str, list[HistoricalEstimate]], *, prefer_conditional: bool
) -> HistoricalEstimate | None:
    markets = tier_2.get("market_no_vig_historical", [])
    if not markets:
        return None
    prefix = "fluke=" if prefer_conditional else "rank="
    preferred = [estimate for estimate in markets if estimate.lookup_key.startswith(prefix)]
    return preferred[0] if preferred else markets[0]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _optional_int(value: Any) -> int | None:
    number = _optional_float(value)
    return None if number is None else int(number)
