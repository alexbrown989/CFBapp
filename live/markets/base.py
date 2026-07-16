"""Shared read-only market contracts and resilient public HTTP transport."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol


class MarketDataError(RuntimeError):
    """A public market-data request or response was unusable."""


class MarketAuthenticationError(MarketDataError):
    """A read endpoint unexpectedly required authentication."""


class MarketMappingError(MarketDataError):
    """A game could not be mapped without outcome-inversion risk."""


@dataclass(frozen=True)
class GameMarket:
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    favorite: str
    dog: str
    pregame_spread: float
    kickoff: str | None = None

    def __post_init__(self) -> None:
        participants = {normalize_team(self.home_team), normalize_team(self.away_team)}
        if normalize_team(self.favorite) not in participants or normalize_team(self.dog) not in participants:
            raise ValueError("favorite and dog must match the game's home/away teams")
        if normalize_team(self.favorite) == normalize_team(self.dog):
            raise ValueError("favorite and dog must be different teams")


@dataclass(frozen=True)
class MarketRef:
    venue: str
    market_id: str
    favorite_outcome_id: str
    favorite_side: str
    dog_outcome_id: str | None
    favorite_team: str
    dog_team: str
    mapping_confidence: str
    mapping_reason: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.favorite_side not in {"yes", "no", "token"}:
            raise ValueError(f"unsupported favorite_side={self.favorite_side!r}")
        if self.mapping_confidence != "exact":
            raise MarketMappingError("only exact team/date market mappings may be scored")


@dataclass(frozen=True)
class Quote:
    venue: str
    market_id: str
    favorite_outcome_id: str
    best_bid: float
    best_ask: float
    mid: float
    spread: float
    implied_prob_raw: float
    implied_prob_no_vig: float
    depth_top_levels: tuple[Mapping[str, float], ...]
    volume_since_last_poll: float | None
    timestamp: str
    is_stale: bool
    dog_best_ask: float
    source_timestamp: str | None = None
    no_vig_method: str = "two_sided_best_ask_normalization"

    def __post_init__(self) -> None:
        for name in ("best_bid", "best_ask", "mid", "implied_prob_raw", "implied_prob_no_vig", "dog_best_ask"):
            value = float(getattr(self, name))
            if not 0.0 < value < 1.0:
                raise MarketDataError(f"{self.venue} {name} must be in (0,1), got {value}")
        if self.best_bid > self.best_ask:
            raise MarketDataError(f"{self.venue} crossed favorite book: bid={self.best_bid}, ask={self.best_ask}")
        if self.spread < 0:
            raise MarketDataError(f"{self.venue} spread must be non-negative")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MappingDecision:
    timestamp: str
    game_id: str
    venue: str
    status: str
    market_id: str | None
    favorite: str
    dog: str
    favorite_outcome_id: str | None
    favorite_side: str | None
    mapping_reason: str
    inversion_guard_passed: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class MarketClient(Protocol):
    venue: str

    def find_market(self, game: GameMarket) -> MarketRef | None: ...

    def get_quote(self, market_ref: MarketRef) -> Quote | None: ...


Transport = Callable[[str, float], tuple[int, Mapping[str, str], bytes]]


class PublicHTTPClient:
    """GET-only JSON client. It has no credential or custom-header surface."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 15.0,
        max_attempts: int = 3,
        backoff_seconds: float = 0.25,
        transport: Transport | None = None,
    ) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.max_attempts = int(max_attempts)
        self.backoff_seconds = float(backoff_seconds)
        self.transport = transport or _urllib_transport

    def get_json(self, base_url: str, path: str, params: Mapping[str, object] | None = None) -> Any:
        query = urllib.parse.urlencode({key: value for key, value in (params or {}).items() if value is not None})
        url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
        if query:
            url = f"{url}?{query}"
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                status, headers, body = self.transport(url, self.timeout_seconds)
                if status in (401, 403):
                    raise MarketAuthenticationError(
                        f"public market-data endpoint unexpectedly required authentication: {url} ({status})"
                    )
                if status == 429 or status >= 500:
                    retry_after = _float_or_none(headers.get("Retry-After"))
                    if attempt + 1 < self.max_attempts:
                        time.sleep(retry_after if retry_after is not None else self.backoff_seconds * (2**attempt))
                        continue
                if status < 200 or status >= 300:
                    raise MarketDataError(f"market-data GET failed: {url} ({status})")
                return json.loads(body.decode("utf-8"))
            except MarketAuthenticationError:
                raise
            except (MarketDataError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 < self.max_attempts:
                    time.sleep(self.backoff_seconds * (2**attempt))
                    continue
        raise MarketDataError(f"market-data GET failed after {self.max_attempts} attempts: {url}: {last_error}")


def normalize_team(value: str) -> str:
    aliases = {
        "ole miss": "mississippi",
        "miami fl": "miami",
        "miami florida": "miami",
        "uconn": "connecticut",
    }
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return aliases.get(normalized, normalized)


def team_in_text(team: str, text: str) -> bool:
    needle = normalize_team(team)
    haystack = normalize_team(text)
    return bool(needle) and re.search(rf"(?:^| )({re.escape(needle)})(?: |$)", haystack) is not None


def parse_json_array(value: object, field_name: str) -> list[str]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list) or not all(isinstance(item, (str, int)) for item in parsed):
        raise MarketDataError(f"{field_name} must be a JSON array")
    return [str(item) for item in parsed]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_timestamp(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value)
    try:
        if text.isdigit():
            number = int(text)
            seconds = number / 1000 if number > 10_000_000_000 else number
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (ValueError, OSError):
        return None


def is_timestamp_stale(value: object, stale_after_seconds: float, now: datetime | None = None) -> bool:
    parsed = parse_timestamp(value)
    if parsed is None:
        return False
    current = now or datetime.now(timezone.utc)
    return (current - parsed).total_seconds() > stale_after_seconds


def _urllib_transport(url: str, timeout_seconds: float) -> tuple[int, Mapping[str, str], bytes]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "N13-read-only-market-client/1.0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return int(response.status), dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), dict(exc.headers.items()), exc.read()


def _float_or_none(value: object) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
