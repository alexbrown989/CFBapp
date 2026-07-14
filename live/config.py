from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_THRESHOLDS = (3, 7, 10, 14, 21)


@dataclass(frozen=True)
class Settings:
    season: int = 2026
    poll_interval_seconds: float = 25.0
    deficit_thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS
    top25_only: bool = True
    data_source: str = "stub"
    cfbd_base_url: str = "https://api.collegefootballdata.com"

    @classmethod
    def from_env(cls) -> "Settings":
        raw_thresholds = os.getenv("N13_DEFICIT_THRESHOLDS", "3,7,10,14,21")
        thresholds = tuple(sorted({int(value.strip()) for value in raw_thresholds.split(",") if value.strip()}))
        if not thresholds or any(value <= 0 for value in thresholds):
            raise ValueError("N13_DEFICIT_THRESHOLDS must contain positive integers")
        return cls(
            season=int(os.getenv("N13_SEASON", "2026")),
            poll_interval_seconds=float(os.getenv("N13_POLL_INTERVAL_SECONDS", "25")),
            deficit_thresholds=thresholds,
            top25_only=os.getenv("N13_TOP25_ONLY", "1") == "1",
            data_source=os.getenv("N13_DATA_SOURCE", "stub").lower(),
            cfbd_base_url=os.getenv("CFBD_BASE_URL", "https://api.collegefootballdata.com").rstrip("/"),
        )
