from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Mapping, Sequence

from .data_source import ScoreboardGameState


REAL_SPORTSBOOKS = {
    "Bovada",
    "Caesars",
    "DraftKings",
    "ESPN Bet",
    "William Hill (New Jersey)",
}


@dataclass(frozen=True)
class WatchGame:
    game_id: str
    season: int
    week: int
    favorite: str
    dog: str
    pregame_spread: float
    home_team: str
    away_team: str
    spread_provider_used: str

    def favorite_scores(self, state: ScoreboardGameState) -> tuple[int, int]:
        if state.game_id != self.game_id:
            raise ValueError(f"state game {state.game_id} does not match watch game {self.game_id}")
        if self.favorite == self.home_team:
            return state.home_score, state.away_score
        if self.favorite == self.away_team:
            return state.away_score, state.home_score
        raise ValueError(f"favorite {self.favorite!r} is neither home nor away team")


def build_watchlist(
    games: Iterable[Mapping[str, object]],
    line_records: Iterable[Mapping[str, object]],
    *,
    ranked_teams: Iterable[str] | None = None,
    manual_teams: Iterable[str] | None = None,
    top25_only: bool = True,
) -> dict[str, WatchGame]:
    scope = {str(team) for team in (ranked_teams or [])} | {str(team) for team in (manual_teams or [])}
    if top25_only and not scope:
        raise ValueError("top-25 watch-list construction requires rankings input or a manual team list")

    lines_by_game = {str(record.get("id") or record.get("gameId")): record for record in line_records}
    result: dict[str, WatchGame] = {}
    for game in games:
        game_id = str(game.get("id") or game.get("gameId"))
        home = str(game.get("homeTeam") or game.get("home") or "")
        away = str(game.get("awayTeam") or game.get("away") or "")
        if not game_id or not home or not away:
            continue
        if top25_only and not ({home, away} & scope):
            continue
        selected = select_home_spread(lines_by_game.get(game_id, {}))
        if selected is None:
            continue
        home_spread, provider = selected
        if home_spread == 0:
            continue
        favorite, dog = (home, away) if home_spread < 0 else (away, home)
        favorite_spread = home_spread if home_spread < 0 else -home_spread
        result[game_id] = WatchGame(
            game_id=game_id,
            season=int(game.get("season") or game.get("year") or 0),
            week=int(game.get("week") or 0),
            favorite=favorite,
            dog=dog,
            pregame_spread=float(favorite_spread),
            home_team=home,
            away_team=away,
            spread_provider_used=provider,
        )
    return result


def select_home_spread(record: Mapping[str, object]) -> tuple[float, str] | None:
    lines = record.get("lines") or []
    if not isinstance(lines, Sequence):
        return None
    consensus = [line for line in lines if isinstance(line, Mapping) and line.get("provider") == "consensus" and _spread(line) is not None]
    if consensus:
        return float(_spread(consensus[0])), "consensus"
    sportsbook = [line for line in lines if isinstance(line, Mapping) and line.get("provider") in REAL_SPORTSBOOKS and _spread(line) is not None]
    if not sportsbook:
        return None

    # Exclude provider-side direction conflicts before averaging. A single
    # malformed spread must not reverse the favorite selected by the majority.
    positive = [line for line in sportsbook if float(_spread(line)) > 0]
    negative = [line for line in sportsbook if float(_spread(line)) < 0]
    if len(positive) == len(negative):
        return None
    direction_consistent = positive if len(positive) > len(negative) else negative
    values = [float(_spread(line)) for line in direction_consistent]
    providers = [str(line.get("provider")) for line in direction_consistent]
    label = f"single_provider_{providers[0]}" if len(values) == 1 else "multi_sportsbook_avg_direction_consistent"
    return mean(values), label


def _spread(line: Mapping[str, object]) -> float | None:
    value = line.get("spread")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
