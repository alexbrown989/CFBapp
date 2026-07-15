from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

from .trigger_detect import TriggerEvent


REQUIRED_TRIGGER_FIELDS = (
    "timestamp",
    "game_id",
    "season",
    "week",
    "favorite",
    "dog",
    "pregame_spread",
    "fav_score",
    "dog_score",
    "period",
    "clock",
    "deficit",
    "threshold_crossed",
    "possession",
    "data_source",
    "poll_number",
)

SCORING_FIELDS = (
    "tier_used",
    "tier_reasons",
    "baseline_c_final_win",
    "baseline_c_deficit_erased",
    "baseline_c_n_events",
    "baseline_c_reliability",
    "conditional_rate_final_win",
    "conditional_rate_n_events",
    "ranking_rate_final_win",
    "ranking_rate_n_events",
    "market_no_vig_historical",
    "n06_calibrated_prob",
    "conformal_lower",
    "conformal_upper",
    "n06_unavailable_reason",
    "spread_bucket",
    "ranking_bucket",
    "fluke_bucket",
)


class JSONLTriggerLogger:
    """Append-only local trigger log. No remote writes or secret values."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: TriggerEvent, scoring_fields: Mapping[str, object] | None = None) -> None:
        record = event.as_dict()
        missing = [field for field in REQUIRED_TRIGGER_FIELDS if field not in record]
        if missing:
            raise ValueError(f"trigger record missing required fields: {missing}")
        if scoring_fields is not None:
            unknown = sorted(set(scoring_fields) - set(SCORING_FIELDS))
            if unknown:
                raise ValueError(f"unknown scoring log fields: {unknown}")
            record.update({field: scoring_fields.get(field) for field in SCORING_FIELDS})
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    def append_many(self, events: Iterable[TriggerEvent]) -> None:
        for event in events:
            self.append(event)


def read_trigger_records(path: str | Path) -> list[dict[str, object]]:
    """Read Stage 1 or Stage 2 records through the additive schema."""
    records: list[dict[str, object]] = []
    source = Path(path)
    if not source.exists():
        return records
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = [field for field in REQUIRED_TRIGGER_FIELDS if field not in record]
        if missing:
            raise ValueError(f"trigger log line {line_number} missing Stage 1 fields: {missing}")
        for field in SCORING_FIELDS:
            record.setdefault(field, None)
        records.append(record)
    return records
