from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Mapping

from .bootstrap import create_monitor
from .config import RuntimeConfigStore, Settings
from .logger import read_trigger_records
from .monitor import LiveMonitor


DASHBOARD_PATH = Path(__file__).parent / "static" / "dashboard.html"
DASHBOARD_CSS_PATH = Path(__file__).parent / "static" / "dashboard.css"
DASHBOARD_JS_PATH = Path(__file__).parent / "static" / "dashboard.js"
DASHBOARD_PANELS_JS_PATH = Path(__file__).parent / "static" / "dashboard_panels.js"
LOCALHOST = "127.0.0.1"


def create_app(
    monitor: LiveMonitor | None = None,
    config_store: RuntimeConfigStore | None = None,
    settings: Settings | None = None,
):
    """Create the localhost-only FastAPI presentation layer."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("FastAPI is required to serve the Stage 4 dashboard") from exc

    resolved_settings = settings or Settings.from_env()
    if resolved_settings.dashboard_host != LOCALHOST:
        raise ValueError("Stage 4 dashboard must bind to 127.0.0.1")
    resolved_monitor = monitor or create_monitor(resolved_settings)
    resolved_store = config_store or RuntimeConfigStore(resolved_settings.runtime_config_path)
    poll_task: asyncio.Task[None] | None = None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal poll_task
        if os.getenv("N13_AUTO_POLL", "0") == "1":
            poll_task = asyncio.create_task(resolved_monitor.poll_forever())
        try:
            yield
        finally:
            if poll_task is not None:
                poll_task.cancel()
                try:
                    await poll_task
                except asyncio.CancelledError:
                    pass

    dashboard = FastAPI(title="N13 Live Monitor", version="0.4.0", lifespan=lifespan)
    dashboard.state.monitor = resolved_monitor
    dashboard.state.config_store = resolved_store
    dashboard.state.bind_host = LOCALHOST

    @dashboard.get("/")
    def dashboard_page():
        if not DASHBOARD_PATH.exists():
            raise HTTPException(status_code=500, detail="dashboard asset is missing")
        return FileResponse(DASHBOARD_PATH, media_type="text/html")

    @dashboard.get("/static/dashboard.css")
    def dashboard_css():
        if not DASHBOARD_CSS_PATH.exists():
            raise HTTPException(status_code=500, detail="dashboard asset is missing")
        return FileResponse(DASHBOARD_CSS_PATH, media_type="text/css")

    @dashboard.get("/static/dashboard_panels.js")
    def dashboard_panels_js():
        if not DASHBOARD_PANELS_JS_PATH.exists():
            raise HTTPException(status_code=500, detail="dashboard asset is missing")
        return FileResponse(DASHBOARD_PANELS_JS_PATH, media_type="text/javascript")

    @dashboard.get("/static/dashboard.js")
    def dashboard_js():
        if not DASHBOARD_JS_PATH.exists():
            raise HTTPException(status_code=500, detail="dashboard asset is missing")
        return FileResponse(DASHBOARD_JS_PATH, media_type="text/javascript")

    @dashboard.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "data_source": type(resolved_monitor.source).__name__,
            "mode": resolved_monitor.mode,
            "watchlist_games": len(resolved_monitor.watchlist),
            "read_only": True,
            "bind_host": LOCALHOST,
        }

    @dashboard.get("/api/state")
    def api_state() -> dict[str, object]:
        return resolved_monitor.dashboard_snapshot(resolved_store.get())

    @dashboard.get("/api/triggers")
    def api_triggers(limit: int = 100) -> dict[str, object]:
        bounded = min(500, max(1, int(limit)))
        records = read_trigger_records(resolved_monitor.logger.path)
        return {
            "triggers": list(reversed(records[-bounded:])),
            "snapshots": resolved_monitor.recent_trigger_snapshots(resolved_store.get(), bounded),
            "count": min(len(records), bounded),
        }

    @dashboard.get("/api/game/{game_id}")
    def api_game(game_id: str) -> dict[str, object]:
        snapshot = resolved_monitor.dashboard_snapshot(resolved_store.get())
        for game in snapshot["games"]:
            if game["game_id"] == str(game_id):
                return game
        raise HTTPException(status_code=404, detail="game is not in the current watch list")

    @dashboard.get("/api/config")
    def api_config() -> dict[str, object]:
        return {
            **resolved_store.get().as_dict(),
            "bind_host": LOCALHOST,
            "mode": resolved_monitor.mode,
        }

    @dashboard.post("/api/config")
    def update_config(values: Mapping[str, object]) -> dict[str, object]:
        try:
            updated = resolved_store.update(values)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return updated.as_dict()

    @dashboard.post("/poll-once")
    def poll_once() -> dict[str, int]:
        return {"triggers_detected": resolved_monitor.poll_once()}

    return dashboard
