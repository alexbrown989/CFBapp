from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .data_source import ScoreboardGameState
from .watchlist import WatchGame


@dataclass(frozen=True)
class TriggerEvent:
    timestamp: str
    game_id: str
    season: int
    week: int
    favorite: str
    dog: str
    pregame_spread: float
    fav_score: int
    dog_score: int
    period: int
    clock: str
    deficit: int
    threshold_crossed: int
    possession: str | None
    data_source: str
    poll_number: int

    def as_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "game_id": self.game_id,
            "season": self.season,
            "week": self.week,
            "favorite": self.favorite,
            "dog": self.dog,
            "pregame_spread": self.pregame_spread,
            "fav_score": self.fav_score,
            "dog_score": self.dog_score,
            "period": self.period,
            "clock": self.clock,
            "deficit": self.deficit,
            "threshold_crossed": self.threshold_crossed,
            "possession": self.possession,
            "data_source": self.data_source,
            "poll_number": self.poll_number,
        }


class TriggerDetector:
    """Stateful threshold detector shared by stub and live scoreboards."""

    def __init__(self, thresholds: Iterable[int] = (3, 7, 10, 14, 21)) -> None:
        self.thresholds = tuple(sorted({int(value) for value in thresholds}))
        if not self.thresholds or any(value <= 0 for value in self.thresholds):
            raise ValueError("thresholds must contain positive integers")
        self._armed: dict[str, dict[int, bool]] = {}

    def process(self, state: ScoreboardGameState, game: WatchGame) -> list[TriggerEvent]:
        fav_score, dog_score = game.favorite_scores(state)
        deficit = fav_score - dog_score
        trailing_margin = max(0, -deficit)
        armed = self._armed.setdefault(game.game_id, {threshold: True for threshold in self.thresholds})
        fired: list[TriggerEvent] = []

        for threshold in self.thresholds:
            if trailing_margin < threshold:
                armed[threshold] = True
                continue
            if not armed[threshold]:
                continue
            armed[threshold] = False
            fired.append(
                TriggerEvent(
                    timestamp=state.observed_at,
                    game_id=game.game_id,
                    season=game.season,
                    week=game.week,
                    favorite=game.favorite,
                    dog=game.dog,
                    pregame_spread=game.pregame_spread,
                    fav_score=fav_score,
                    dog_score=dog_score,
                    period=state.period,
                    clock=state.clock,
                    deficit=deficit,
                    threshold_crossed=threshold,
                    possession=state.possession,
                    data_source=state.data_source,
                    poll_number=state.poll_number,
                )
            )
        return fired
