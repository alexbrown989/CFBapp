from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

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


class JSONLTriggerLogger:
    """Append-only local trigger log. No remote writes or secret values."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: TriggerEvent) -> None:
        record = event.as_dict()
        missing = [field for field in REQUIRED_TRIGGER_FIELDS if field not in record]
        if missing:
            raise ValueError(f"trigger record missing required fields: {missing}")
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    def append_many(self, events: Iterable[TriggerEvent]) -> None:
        for event in events:
            self.append(event)
