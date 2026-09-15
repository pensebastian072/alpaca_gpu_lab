"""Alpaca news -> partitioned Parquet (headlines back to ~2015).

Same resumable monthly-chunk pattern as alpaca_backfill.py:
  market_data/raw/news/year=Y/month=M/news.parquet
One request stream covers ALL universe symbols (news items carry a symbol
list); dedup on the Alpaca news id. Data client only (CLAUDE.md rule 1).

CLI:
  python -m src.data.news_backfill --start 2020-01-01
  python -m src.data.news_backfill --start 2015-01-01 --end 2020-01-01
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.data.alpaca_backfill import (  # noqa: E402
    _load_keys, _month_windows, _trust_windows_certs)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("news_backfill")

NEWS_ROOT_NAME = "news"


def _symbols() -> list[str]:
    # equity tickers only — Alpaca news is keyed to equities (+ crypto majors)
    return [config.UNIVERSE_22[k]["ticker"] for k in config.EQUITY_22] + ["BTCUSD"]


def backfill_news(start: datetime, end: datetime, overwrite: bool = False) -> None:
    _trust_windows_certs()
    key, sec = _load_keys()
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest

    client = NewsClient(key, sec)
    root = config.RAW / NEWS_ROOT_NAME
    total = 0

    for win_start, win_end in _month_windows(start, end):
        y, m = win_start.year, win_start.month
        part = root / f"year={y}" / f"month={m:02d}"
        out = part / "news.parquet"
        if out.exists() and not overwrite:
            log.info("skip %04d-%02d (present)", y, m)
            continue
        rows = []
        try:
            # NO limit param: alpaca-py treats `limit` as a TOTAL item cap and
            # paginates internally until exhausted when it is absent.
            req = NewsRequest(
                symbols=",".join(_symbols()),
                start=win_start, end=win_end,
                include_content=False,
            )
            news = client.get_news(req)
            for item in news.data.get("news", []):
                d = item.model_dump() if hasattr(item, "model_dump") else dict(item)
                rows.append({
                    "id": str(d.get("id")),
                    "created_at": d.get("created_at"),
                    "headline": d.get("headline"),
                    "summary": d.get("summary"),
                    "source": d.get("source"),
                    "symbols": ",".join(d.get("symbols") or []),
                })
        except Exception as e:  # noqa: BLE001 — one bad month shouldn't abort
            log.warning("%04d-%02d fetch failed: %s", y, m, e)
            continue
        if not rows:
            log.info("%04d-%02d no news", y, m)
            continue
        df = pd.DataFrame(rows).drop_duplicates(subset="id")
        part.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        total += len(df)
        log.info("%04d-%02d: %d articles", y, m, len(df))

    log.info("done. %d articles under %s", total, root)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    backfill_news(
        datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc),
        datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc),
        a.overwrite,
    )
