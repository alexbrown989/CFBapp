from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path

from .config import Settings
from .data_source import ScoreboardLive, ScoreboardSource, ScoreboardStub
from .logger import JSONLTriggerLogger
from .trigger_detect import TriggerDetector
from .watchlist import WatchGame


class LiveMonitor:
    """Polling coordinator; depends only on the normalized source interface."""

    def __init__(
        self,
        source: ScoreboardSource,
        watchlist: dict[str, WatchGame],
        detector: TriggerDetector,
        logger: JSONLTriggerLogger,
        poll_interval_seconds: float,
    ) -> None:
        self.source = source
        self.watchlist = watchlist
        self.detector = detector
        self.logger = logger
        self.poll_interval_seconds = poll_interval_seconds

    def poll_once(self) -> int:
        event_count = 0
        for state in self.source.poll():
            game = self.watchlist.get(state.game_id)
            if game is None:
                continue
            events = self.detector.process(state, game)
            self.logger.append_many(events)
            for event in events:
                print(
                    f"TRIGGER {event.game_id} {event.favorite} D={event.threshold_crossed} "
                    f"score={event.fav_score}-{event.dog_score} Q{event.period} {event.clock}"
                )
            event_count += len(events)
        return event_count

    async def poll_forever(self) -> None:
        while True:
            self.poll_once()
            await asyncio.sleep(self.poll_interval_seconds)


def create_source(settings: Settings) -> ScoreboardSource:
    if settings.data_source == "stub":
        return ScoreboardStub([])
    if settings.data_source == "live":
        return ScoreboardLive(base_url=settings.cfbd_base_url)
    raise ValueError(f"unsupported N13_DATA_SOURCE={settings.data_source!r}")


def create_monitor(settings: Settings | None = None) -> LiveMonitor:
    settings = settings or Settings.from_env()
    return LiveMonitor(
        source=create_source(settings),
        watchlist={},
        detector=TriggerDetector(settings.deficit_thresholds),
        logger=JSONLTriggerLogger(Path(__file__).parent / "logs" / "triggers.jsonl"),
        poll_interval_seconds=settings.poll_interval_seconds,
    )


try:
    from fastapi import FastAPI

    _monitor = create_monitor()
    _poll_task: asyncio.Task[None] | None = None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        global _poll_task
        if os.getenv("N13_AUTO_POLL", "0") == "1":
            _poll_task = asyncio.create_task(_monitor.poll_forever())
        try:
            yield
        finally:
            if _poll_task is not None:
                _poll_task.cancel()
                try:
                    await _poll_task
                except asyncio.CancelledError:
                    pass

    app = FastAPI(title="N13 Stage 1 Live Monitor", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "data_source": type(_monitor.source).__name__,
            "watchlist_games": len(_monitor.watchlist),
            "read_only": True,
        }

    @app.post("/poll-once")
    def poll_once() -> dict[str, int]:
        return {"triggers_detected": _monitor.poll_once()}
except ImportError:
    app = None


def main() -> int:
    parser = argparse.ArgumentParser(description="N13 Stage 1 scoreboard monitor")
    parser.add_argument("--once", action="store_true", help="run one poll in the configured mode and exit")
    args = parser.parse_args()
    monitor = create_monitor()
    if args.once:
        print(f"poll complete: {monitor.poll_once()} trigger(s)")
        return 0
    asyncio.run(monitor.poll_forever())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
