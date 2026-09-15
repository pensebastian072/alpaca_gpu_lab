"""Alpaca historical bars -> partitioned Parquet (the data spine).

Research / DATA ONLY (CLAUDE.md rule 1): market-data client only, never the
trading client. No orders, ever.

Design (per the collection-layer guidance):
  - request multiple symbols together, one month at a time;
  - alpaca-py handles next_page_token pagination internally;
  - write each completed (symbol, month) chunk to Parquet immediately, so years
    of data never sit in RAM and an interrupted run resumes cleanly;
  - partition layout: raw/bars_<tf>/symbol=X/year=Y/month=M/bars.parquet
  - IEX feed by default (free); note the feed — IEX != SIP.

TLS-interception box: export the Windows CA bundle so alpaca's http client
verifies against the injected root CA.

Crypto (BTC) goes through CryptoHistoricalDataClient (no feed arg, 24/7 bars);
API symbol "BTC/USD" is stored under partition symbol=BTCUSD.

Usage:
  python -m src.data.alpaca_backfill --symbols SPY QQQ --timeframe 1Min \
      --start 2022-01-01 --end 2026-07-01
  python -m src.data.alpaca_backfill            # UNIVERSE, DEFAULT_* from config
  python -m src.data.alpaca_backfill --universe 22 --timeframe 1Hour --start 2016-01-01
  python -m src.data.alpaca_backfill --symbols BTC/USD --asset-class crypto --timeframe 1Hour
"""
from __future__ import annotations

import argparse
import logging
import os
import ssl
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("alpaca_backfill")


def _trust_windows_certs() -> None:
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
    import certifi

    bundle = config.DATA / "_windows_ca_bundle.pem"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    parts = [Path(certifi.where()).read_text(encoding="utf-8")]
    for store in ("ROOT", "CA"):
        for der, enc_type, _trust in ssl.enum_certificates(store):
            if enc_type == "x509_asn":
                parts.append(ssl.DER_cert_to_PEM_cert(der))
    bundle.write_text("\n".join(parts), encoding="utf-8")
    for var in ("CURL_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        os.environ[var] = str(bundle)


def _load_keys() -> tuple[str, str]:
    envf = config.REPO / ".env"
    if envf.exists():
        for line in envf.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    key, sec = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
    if not key or not sec:
        sys.exit(f"Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY in {envf}")
    return key, sec


def _timeframe(tf: str):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    table = {
        "1Min": TimeFrame.Minute,
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "1Hour": TimeFrame.Hour,
        "4Hour": TimeFrame(4, TimeFrameUnit.Hour),
        "1Day": TimeFrame.Day,
    }
    if tf not in table:
        sys.exit(f"--timeframe must be one of {list(table)}")
    return table[tf]


def _month_windows(start: datetime, end: datetime):
    """Yield (month_start, next_month_start) UTC pairs covering [start, end)."""
    cur = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    while cur < end:
        nxt = datetime(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1, tzinfo=timezone.utc)
        yield cur, min(nxt, end)
        cur = nxt


def _part_symbol(sym: str) -> str:
    """Partition-safe symbol name: 'BTC/USD' -> 'BTCUSD'."""
    return sym.replace("/", "")


def backfill(symbols, timeframe, start, end, feed, overwrite=False,
             asset_class="equity") -> None:
    _trust_windows_certs()
    key, sec = _load_keys()
    if asset_class == "crypto":
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest

        client = CryptoHistoricalDataClient(key, sec)

        def _fetch(pending, tf, win_start, win_end):
            req = CryptoBarsRequest(symbol_or_symbols=pending, timeframe=tf,
                                    start=win_start, end=win_end)
            return client.get_crypto_bars(req).df
    else:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest

        client = StockHistoricalDataClient(key, sec)

        def _fetch(pending, tf, win_start, win_end):
            req = StockBarsRequest(symbol_or_symbols=pending, timeframe=tf,
                                   start=win_start, end=win_end, feed=feed)
            return client.get_stock_bars(req).df

    tf = _timeframe(timeframe)
    root = config.RAW / f"bars_{timeframe}"
    total_rows = 0

    for win_start, win_end in _month_windows(start, end):
        y, m = win_start.year, win_start.month
        # resume: skip a month only if every symbol partition already exists
        pending = [s for s in symbols if overwrite or not
                   (root / f"symbol={_part_symbol(s)}" / f"year={y}" / f"month={m:02d}" / "bars.parquet").exists()]
        if not pending:
            log.info("skip %04d-%02d (all %d symbols present)", y, m, len(symbols))
            continue

        try:
            df = _fetch(pending, tf, win_start, win_end)
        except Exception as e:  # noqa: BLE001 — one bad month shouldn't abort the run
            log.warning("%04d-%02d fetch failed: %s", y, m, e)
            continue
        if df is None or df.empty:
            log.info("%04d-%02d no data", y, m)
            continue

        df = df.reset_index()  # columns: symbol, timestamp, open, high, low, close, volume, ...
        for sym, g in df.groupby("symbol"):
            part = root / f"symbol={_part_symbol(sym)}" / f"year={y}" / f"month={m:02d}"
            part.mkdir(parents=True, exist_ok=True)
            g = g.drop(columns=["symbol"]).reset_index(drop=True)
            g.to_parquet(part / "bars.parquet", index=False)
            total_rows += len(g)
            log.info("%s %04d-%02d: %d bars -> %s", sym, y, m, len(g), part.name)

    log.info("done. %d bars written under %s (feed=%s, asset_class=%s)",
             total_rows, root, feed, asset_class)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--universe", choices=["core7", "22"], default=None,
                    help="shortcut: core7 (original 1Min set) or the 22-asset equity leg")
    ap.add_argument("--asset-class", choices=["equity", "crypto"], default="equity")
    ap.add_argument("--timeframe", default=config.DEFAULT_TIMEFRAME)
    ap.add_argument("--start", default=config.DEFAULT_START)
    ap.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--feed", default=config.DEFAULT_FEED, choices=["iex", "sip"])
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.symbols:
        symbols = a.symbols
    elif a.universe == "22":
        symbols = [config.UNIVERSE_22[k]["ticker"] for k in config.EQUITY_22]
    elif a.universe == "core7":
        symbols = config.CORE7
    else:
        symbols = config.UNIVERSE
    backfill(
        symbols,
        a.timeframe,
        datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc),
        datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc),
        a.feed,
        a.overwrite,
        a.asset_class,
    )
