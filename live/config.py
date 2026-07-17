from __future__ import annotations

import os
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Mapping


DEFAULT_THRESHOLDS = (3, 7, 10, 14, 21)
REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_NOTEBOOKS_DIR = REPO_ROOT / "research" / "notebooks"
DEFAULT_RUNTIME_CONFIG_PATH = REPO_ROOT / "live" / "config.local.json"


def ensure_n12_lookup_import_path() -> Path:
    """Expose the committed N12 helper module without copying its logic."""
    path = str(RESEARCH_NOTEBOOKS_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    return RESEARCH_NOTEBOOKS_DIR


@dataclass(frozen=True)
class Settings:
    season: int = 2026
    poll_interval_seconds: float = 25.0
    deficit_thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS
    top25_only: bool = True
    data_source: str = "stub"
    cfbd_base_url: str = "https://api.collegefootballdata.com"
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8000
    runtime_config_path: Path = DEFAULT_RUNTIME_CONFIG_PATH

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
            dashboard_host="127.0.0.1",
            dashboard_port=int(os.getenv("N13_DASHBOARD_PORT", "8000")),
            runtime_config_path=Path(
                os.getenv("N13_RUNTIME_CONFIG_PATH", str(DEFAULT_RUNTIME_CONFIG_PATH))
            ),
        )


@dataclass(frozen=True)
class RiskConfig:
    """Personal local policy inputs; these are not research estimates."""

    bankroll: float = 1000.0
    kelly_fraction: float = 0.25
    max_bankroll_fraction_per_bet: float = 0.05
    ruin_comfort_threshold: float = 0.05
    drawdown_floor: float = 0.50
    season_bets: int = 50

    def __post_init__(self) -> None:
        if self.bankroll <= 0:
            raise ValueError("bankroll must be positive")
        for name in (
            "kelly_fraction",
            "max_bankroll_fraction_per_bet",
            "ruin_comfort_threshold",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if not 0.0 <= self.drawdown_floor < 1.0:
            raise ValueError("drawdown_floor must be in [0,1)")
        if int(self.season_bets) <= 0:
            raise ValueError("season_bets must be positive")

    def as_dict(self) -> dict[str, object]:
        return {
            "bankroll": float(self.bankroll),
            "kelly_fraction": float(self.kelly_fraction),
            "max_bankroll_fraction_per_bet": float(self.max_bankroll_fraction_per_bet),
            "ruin_comfort_threshold": float(self.ruin_comfort_threshold),
            "drawdown_floor": float(self.drawdown_floor),
            "season_bets": int(self.season_bets),
        }


class RuntimeConfigStore:
    """Thread-safe local JSON persistence for personal dashboard settings."""

    ALLOWED_FIELDS = frozenset(RiskConfig.__dataclass_fields__)

    def __init__(self, path: str | Path, defaults: RiskConfig | None = None) -> None:
        self.path = Path(path)
        self.defaults = defaults or RiskConfig()
        self._lock = RLock()
        self._current = self._load()

    def get(self) -> RiskConfig:
        with self._lock:
            return self._current

    def update(self, values: Mapping[str, object]) -> RiskConfig:
        unknown = sorted(set(values) - self.ALLOWED_FIELDS)
        if unknown:
            raise ValueError(f"unknown risk config fields: {unknown}")
        with self._lock:
            merged = self._current.as_dict()
            merged.update(values)
            candidate = RiskConfig(
                bankroll=float(merged["bankroll"]),
                kelly_fraction=float(merged["kelly_fraction"]),
                max_bankroll_fraction_per_bet=float(merged["max_bankroll_fraction_per_bet"]),
                ruin_comfort_threshold=float(merged["ruin_comfort_threshold"]),
                drawdown_floor=float(merged["drawdown_floor"]),
                season_bets=int(merged["season_bets"]),
            )
            self._persist(candidate)
            self._current = candidate
            return candidate

    def _load(self) -> RiskConfig:
        if not self.path.exists():
            return self.defaults
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("runtime config root must be an object")
            merged = self.defaults.as_dict()
            merged.update(payload)
            return RiskConfig(
                bankroll=float(merged["bankroll"]),
                kelly_fraction=float(merged["kelly_fraction"]),
                max_bankroll_fraction_per_bet=float(merged["max_bankroll_fraction_per_bet"]),
                ruin_comfort_threshold=float(merged["ruin_comfort_threshold"]),
                drawdown_floor=float(merged["drawdown_floor"]),
                season_bets=int(merged["season_bets"]),
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid runtime config {self.path}: {exc}") from exc

    def _persist(self, config: RiskConfig) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(config.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(self.path)
