from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Iterable, Protocol


TIER_ERROR = "CFBD Tier 2 subscription required; see research/PROJECT_STATE.md"


class LiveAccessError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScoreboardGameState:
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    period: int
    clock: str
    status: str
    possession: str | None = None
    data_source: str = "stub"
    poll_number: int = 0
    observed_at: str = ""

    def __post_init__(self) -> None:
        if not self.observed_at:
            object.__setattr__(self, "observed_at", datetime.now(timezone.utc).isoformat())


class ScoreboardSource(Protocol):
    def poll(self) -> list[ScoreboardGameState]: ...


class ScoreboardStub:
    """Replay pre-normalized scoreboard batches through the production interface."""

    def __init__(self, batches: Iterable[Iterable[ScoreboardGameState]]) -> None:
        self._batches = [list(batch) for batch in batches]
        self._cursor = 0
        self._poll_number = 0

    def poll(self) -> list[ScoreboardGameState]:
        self._poll_number += 1
        if self._cursor >= len(self._batches):
            return []
        batch = self._batches[self._cursor]
        self._cursor += 1
        return [replace(state, data_source="stub", poll_number=self._poll_number) for state in batch]


class ScoreboardLive:
    """Paid CFBD scoreboard adapter. Disabled until August certification."""

    def __init__(self, api_key: str | None = None, base_url: str = "https://api.collegefootballdata.com") -> None:
        self.api_key = api_key or os.getenv("CFBD_API_KEY")
        self.base_url = base_url.rstrip("/")
        self._poll_number = 0

    def _assert_enabled(self) -> None:
        enabled = os.getenv("N13_ENABLE_LIVE_SCOREBOARD", "0") == "1"
        tier_confirmed = os.getenv("CFBD_TIER2_CONFIRMED", "0") == "1"
        if not enabled or not tier_confirmed or not self.api_key:
            raise LiveAccessError(TIER_ERROR)

    def poll(self) -> list[ScoreboardGameState]:
        self._assert_enabled()
        self._poll_number += 1
        query = urllib.parse.urlencode({"classification": "fbs"})
        request = urllib.request.Request(
            f"{self.base_url}/scoreboard?{query}",
            headers={"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, list):
            raise LiveAccessError("CFBD /scoreboard returned a non-list payload")
        return [self._normalize_game(item) for item in payload]

    def _normalize_game(self, item: dict[str, object]) -> ScoreboardGameState:
        home = item.get("homeTeam")
        away = item.get("awayTeam")
        if not isinstance(home, dict) or not isinstance(away, dict):
            raise LiveAccessError("CFBD /scoreboard response is missing nested homeTeam/awayTeam objects")

        # AUGUST VERIFICATION REQUIRED: public OpenAPI confirms the nested
        # homeTeam/awayTeam objects but not their live score/name keys. Verify
        # these candidates against a paid real response before activation.
        home_name = _first_value(home, "name", "school", "team")
        away_name = _first_value(away, "name", "school", "team")
        home_score = _first_value(home, "points", "score")
        away_score = _first_value(away, "points", "score")
        if home_name is None or away_name is None or home_score is None or away_score is None:
            raise LiveAccessError(
                "CFBD /scoreboard nested team schema did not match the pre-activation mapping; "
                "verify AUGUST VERIFICATION REQUIRED fields"
            )

        clock = item.get("clock")
        if isinstance(clock, dict):
            clock = f"{int(clock.get('minutes', 0))}:{int(clock.get('seconds', 0)):02d}"
        status = item.get("status")
        if isinstance(status, dict):
            status = _first_value(status, "type", "name", "state", "detail")

        return ScoreboardGameState(
            game_id=str(_first_value(item, "id", "gameId")),
            season=int(_first_value(item, "season", "year") or 0),
            week=int(item.get("week") or 0),
            home_team=str(home_name),
            away_team=str(away_name),
            home_score=int(home_score),
            away_score=int(away_score),
            period=int(item.get("period") or 0),
            clock=str(clock or "0:00"),
            status=str(status or "unknown"),
            possession=_optional_string(_first_value(item, "possession", "possessionTeam")),
            data_source="live",
            poll_number=self._poll_number,
            observed_at=datetime.now(timezone.utc).isoformat(),
        )


def _first_value(mapping: dict[str, object], *keys: str) -> object | None:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _optional_string(value: object | None) -> str | None:
    return None if value is None else str(value)
