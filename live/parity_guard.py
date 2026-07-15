"""Post-game live-vs-cached feature parity audit for Tier 3."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PARITY_LOG_FIELDS = (
    "record_type",
    "compared_at",
    "trigger_id",
    "game_id",
    "feature",
    "live_value",
    "cached_value",
    "abs_diff",
    "tolerance",
    "drifted",
    "tier3_suspect",
)


@dataclass(frozen=True)
class FeatureDrift:
    record_type: str
    compared_at: str
    trigger_id: str
    game_id: str
    feature: str
    live_value: object
    cached_value: object
    abs_diff: float | None
    tolerance: float
    drifted: bool
    tier3_suspect: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ParityComparison:
    trigger_id: str
    game_id: str
    records: tuple[FeatureDrift, ...]

    @property
    def tier3_suspect(self) -> bool:
        return any(record.drifted for record in self.records)

    @property
    def max_abs_diff(self) -> float:
        values = [record.abs_diff for record in self.records if record.abs_diff is not None]
        return max(values, default=0.0)


class RuntimeParityGuard:
    """Compare features after the completed-game cache becomes available."""

    def __init__(self, default_tolerance: float = 1e-9) -> None:
        if default_tolerance < 0:
            raise ValueError("default_tolerance must be non-negative")
        self.default_tolerance = float(default_tolerance)

    def compare(
        self,
        *,
        trigger_id: str,
        game_id: str,
        live_features: Mapping[str, Any],
        cached_features: Mapping[str, Any],
        tolerances: Mapping[str, float] | None = None,
    ) -> ParityComparison:
        compared_at = datetime.now(timezone.utc).isoformat()
        records: list[FeatureDrift] = []
        for feature in sorted(set(live_features) | set(cached_features)):
            tolerance = float((tolerances or {}).get(feature, self.default_tolerance))
            live = live_features.get(feature)
            cached = cached_features.get(feature)
            diff, drifted = _compare_values(live, cached, tolerance)
            records.append(
                FeatureDrift(
                    record_type="tier3_feature_parity",
                    compared_at=compared_at,
                    trigger_id=str(trigger_id),
                    game_id=str(game_id),
                    feature=feature,
                    live_value=_json_value(live),
                    cached_value=_json_value(cached),
                    abs_diff=diff,
                    tolerance=tolerance,
                    drifted=drifted,
                    tier3_suspect=drifted,
                )
            )
        return ParityComparison(str(trigger_id), str(game_id), tuple(records))

    @staticmethod
    def append_jsonl(comparison: ParityComparison, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8", newline="\n") as handle:
            for record in comparison.records:
                payload = record.as_dict()
                missing = [field for field in PARITY_LOG_FIELDS if field not in payload]
                if missing:
                    raise ValueError(f"parity record missing fields: {missing}")
                handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _compare_values(live: Any, cached: Any, tolerance: float) -> tuple[float | None, bool]:
    if _is_missing(live) and _is_missing(cached):
        return 0.0, False
    if _is_missing(live) or _is_missing(cached):
        return None, True
    try:
        diff = abs(float(live) - float(cached))
    except (TypeError, ValueError):
        return (0.0, False) if live == cached else (None, True)
    return diff, diff > tolerance


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _json_value(value: Any) -> object:
    if _is_missing(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
