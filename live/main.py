from __future__ import annotations

import argparse
import asyncio

from .api import LOCALHOST, create_app
from .bootstrap import create_monitor, create_source
from .config import Settings
from .monitor import LiveMonitor


try:
    app = create_app()
except RuntimeError:
    app = None


def main() -> int:
    parser = argparse.ArgumentParser(description="N13 localhost live monitor")
    parser.add_argument("--once", action="store_true", help="run one poll and exit")
    parser.add_argument("--serve", action="store_true", help="serve the dashboard on localhost")
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.serve:
        import uvicorn

        uvicorn.run(
            "live.main:app",
            host=LOCALHOST,
            port=settings.dashboard_port,
            reload=False,
        )
        return 0
    monitor = create_monitor(settings)
    if args.once:
        print(f"poll complete: {monitor.poll_once()} trigger(s)")
        return 0
    asyncio.run(monitor.poll_forever())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
