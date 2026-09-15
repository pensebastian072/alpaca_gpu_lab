"""Daily refresh job (research pipeline, idempotent, --once).

Steps, each fail-soft (a failed step logs and continues so one flaky network
call doesn't kill the refresh):
  1. incremental bar backfill, current + previous month (all timeframes; the
     resume logic makes re-runs cheap)
  2. incremental news pull + FinBERT scoring (only unscored ids)
  3. coverage report
  4. re-score latest bars with the best REGISTERED batteries' models is a
     later enhancement — for now the flag republishes the latest scorecards
     so staleness reflects reality.
  5. publish flag file

CLI: .venv\\Scripts\\python.exe -m src.daily --once
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, publish  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("daily")


def _step(name, fn):
    try:
        fn()
        log.info("step ok: %s", name)
    except Exception:  # noqa: BLE001 — fail-soft by design
        log.error("step FAILED: %s\n%s", name, traceback.format_exc())


def _incremental_bars() -> None:
    from src.data.alpaca_backfill import backfill

    now = datetime.now(timezone.utc)
    start = (now.replace(day=1) - timedelta(days=1)).replace(day=1)  # prev month
    eq = [config.UNIVERSE_22[k]["ticker"] for k in config.EQUITY_22]
    for tf in ("1Min", "1Hour", "1Day"):
        backfill(eq, tf, start, now, config.DEFAULT_FEED, overwrite=True)
        backfill(["BTC/USD"], tf, start, now, config.DEFAULT_FEED,
                 overwrite=True, asset_class="crypto")


def _incremental_news() -> None:
    from src.data.news_backfill import backfill_news
    from src.news.score import run_bulk

    now = datetime.now(timezone.utc)
    start = (now.replace(day=1) - timedelta(days=1)).replace(day=1)
    backfill_news(start, now, overwrite=True)
    run_bulk()


def _coverage() -> None:
    from src.data.coverage import main as coverage_main

    coverage_main(["1Min", "1Hour", "1Day"])


def _publish_flag() -> None:
    scorecards: dict = {}
    for p in sorted(config.SCORECARDS.glob("*.json")):
        try:
            res = json.loads(p.read_text(encoding="utf-8"))
            bid = res.get("battery_id")
            if bid:
                scorecards[bid] = res  # latest file per battery wins (sorted)
        except json.JSONDecodeError:
            continue
    state = publish.build_state(scorecards, signals={})
    out = publish.publish(state)
    log.info("flag -> %s (status=%s)", out, state["status"])


def main() -> None:
    _step("incremental_bars", _incremental_bars)
    _step("incremental_news", _incremental_news)
    _step("coverage", _coverage)
    _step("publish_flag", _publish_flag)
    log.info("daily refresh complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="single pass (the only mode)")
    ap.parse_args()
    main()
