from __future__ import annotations

from pathlib import Path

from .config import Settings
from .data_source import ScoreboardLive, ScoreboardSource, ScoreboardStub
from .logger import JSONLMarketSeriesLogger, JSONLRecordLogger, JSONLTriggerLogger
from .markets import KalshiClient, PolymarketClient
from .markets.service import MarketService
from .monitor import LiveMonitor, default_scoring_context
from .scoring import score_trigger
from .trigger_detect import TriggerDetector


def create_source(settings: Settings) -> ScoreboardSource:
    if settings.data_source == "stub":
        return ScoreboardStub([])
    if settings.data_source == "live":
        return ScoreboardLive(base_url=settings.cfbd_base_url)
    raise ValueError(f"unsupported N13_DATA_SOURCE={settings.data_source!r}")


def create_monitor(settings: Settings | None = None) -> LiveMonitor:
    settings = settings or Settings.from_env()
    log_dir = Path(__file__).parent / "logs"
    market_service = MarketService(
        [KalshiClient(), PolymarketClient()],
        market_series_logger=JSONLMarketSeriesLogger(log_dir / "market_series.jsonl"),
        mapping_logger=JSONLRecordLogger(
            log_dir / "market_mappings.jsonl",
            (
                "timestamp", "game_id", "venue", "status", "market_id", "favorite", "dog",
                "favorite_outcome_id", "favorite_side", "mapping_reason", "inversion_guard_passed",
            ),
        ),
        error_logger=JSONLRecordLogger(
            log_dir / "market_errors.jsonl",
            ("timestamp", "game_id", "venue", "operation", "error_type", "message"),
        ),
    )
    return LiveMonitor(
        source=create_source(settings),
        watchlist={},
        detector=TriggerDetector(settings.deficit_thresholds),
        logger=JSONLTriggerLogger(log_dir / "triggers.jsonl"),
        poll_interval_seconds=settings.poll_interval_seconds,
        scorer=score_trigger,
        context_provider=default_scoring_context,
        market_service=market_service,
    )
