"""Measure real bid-ask half-spreads from sampled Alpaca quotes.

Every backtest on this box charges a flat 5 bps per side to every asset and nobody
has ever measured the real number. That assumption decides whether qlib's +9.0 bps
gross 5-day spread is an edge or a loss, and it decides the international-ETF
reversal outright.

The free route was tried first and failed: `research_ledger/cost/spread_ohlc.py`
proves that the Corwin-Schultz and Abdi-Ranaldo high-low estimators return exactly
0.0 for any spread below ~40 bps at ETF volatility, which is one to two orders of
magnitude above what these ETFs actually quote. Real quotes are the only source.

Sampling, not backfilling
-------------------------
Full quote history is billions of rows and would answer nothing extra. This pulls a
SAMPLED calendar -- by default one session per month -- and computes per-symbol
half-spread percentiles over the regular session only. That gives a per-symbol cost
level; it does not give a daily time series, and callers must not pretend otherwise.

Honest limits, printed with every report:
  * FREE PLAN = IEX FEED, a single venue with a few percent of consolidated volume.
    Quoted spreads read WIDER than the true NBBO, so every cost here is an
    OVERSTATEMENT. A strategy that survives these numbers survives reality --
    which is the direction of bias you want in a cost test.
  * Regular session only (14:30-21:00 UTC). Opening and closing auctions are
    excluded; the first minutes are the widest part of the day.
  * Quoted spread only. Market impact is not measured and is not included.
  * A sampled session is not a stress day. Spreads blow out exactly when signals
    claim to work, so the p90 column matters more than the median.

Data-only, per this repo's rule 1: StockHistoricalDataClient with the market-data
keys. No trading client is imported here and no order is ever placed. Every response
is cached to `market_data/raw/quotes/` and replayed from disk on re-run, the same
discipline the box applies to metered LiveVol pulls even though this feed is free.

    python -m src.data.quotes_sample --sessions 24
    python -m src.data.quotes_sample --report        # from cache, no API calls
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.data.alpaca_backfill import _load_keys, _trust_windows_certs  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("quotes_sample")

QUOTES_ROOT = config.RAW / "quotes"
SESSION_OPEN = time(14, 30)      # 09:30 ET in UTC (standard time)
SESSION_CLOSE = time(21, 0)      # 16:00 ET in UTC
FLAT_ASSUMPTION = 0.0005         # the per-side cost every repo currently charges
MIN_QUOTES = 200                 # below this a symbol-session is not summarised


def _sample_sessions(n_sessions: int, end: date | None = None) -> list[date]:
    """One session per month walking back, on the 15th (or the prior weekday).

    A fixed day-of-month avoids accidentally sampling only option-expiry Fridays or
    only month-ends, both of which have atypical liquidity.
    """
    end = end or (datetime.now(timezone.utc).date() - timedelta(days=1))
    out: list[date] = []
    y, m = end.year, end.month
    while len(out) < n_sessions:
        d = date(y, m, 15)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        if d < end:
            out.append(d)
        m -= 1
        if m == 0:
            y, m = y - 1, 12
        if y < 2015:
            break
    return sorted(out)


# Stratified intraday windows (UTC start, minutes). A single midday window would
# understate the spread badly -- the open is several times wider than lunchtime -- and
# pulling the whole session is ~1.8M quotes for SPY alone. Three short windows spanning
# open / midday / close capture the intraday shape at a fraction of the rows.
DEFAULT_WINDOWS = ((time(14, 30), 10),      # first 10 min: the widest part of the day
                   (time(17, 0), 10),       # midday: the tightest
                   (time(20, 45), 10))      # into the close


def _cache_path(symbol: str, day: date, tag: str = "windows") -> Path:
    return (QUOTES_ROOT / f"symbol={symbol.replace('/', '')}"
            / f"date={day.isoformat()}" / f"{tag}.parquet")


def fetch_session(client, symbol: str, day: date, feed: str,
                  overwrite: bool = False, windows=DEFAULT_WINDOWS
                  ) -> pd.DataFrame | None:
    """One symbol-session of quotes, cached.

    `windows=None` pulls the full regular session (accurate but very large);
    otherwise the stratified windows above, which is the default.
    """
    tag = "full" if windows is None else "windows"
    path = _cache_path(symbol, day, tag)
    if path.exists() and not overwrite:
        try:
            return pd.read_parquet(path)
        except Exception:
            log.warning("cache unreadable, refetching: %s", path)

    from alpaca.data.requests import StockQuotesRequest

    spans = ([(datetime.combine(day, SESSION_OPEN, tzinfo=timezone.utc),
               datetime.combine(day, SESSION_CLOSE, tzinfo=timezone.utc))]
             if windows is None else
             [(datetime.combine(day, w, tzinfo=timezone.utc),
               datetime.combine(day, w, tzinfo=timezone.utc)
               + timedelta(minutes=mins)) for w, mins in windows])

    frames = []
    for start, end in spans:
        try:
            req = StockQuotesRequest(symbol_or_symbols=symbol, start=start, end=end,
                                     feed=feed)
            part = client.get_stock_quotes(req).df
        except Exception as e:  # noqa: BLE001 -- one window must not abort the run
            log.warning("%s %s %s fetch failed: %s", symbol, day, start.time(), e)
            continue
        if part is not None and not part.empty:
            frames.append(part)
    if not frames:
        return None
    df = pd.concat(frames)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)
    keep = [c for c in ("bid_price", "ask_price", "bid_size", "ask_size")
            if c in df.columns]
    df = df[keep]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def summarise(df: pd.DataFrame) -> dict | None:
    """Per-session half-spread percentiles. Crossed/locked quotes are dropped."""
    if df is None or df.empty:
        return None
    bid = pd.to_numeric(df.get("bid_price"), errors="coerce")
    ask = pd.to_numeric(df.get("ask_price"), errors="coerce")
    ok = bid.notna() & ask.notna() & (bid > 0) & (ask > bid)
    bid, ask = bid[ok], ask[ok]
    if len(bid) < MIN_QUOTES:
        return None
    mid = (bid + ask) / 2.0
    half = (ask - bid) / 2.0 / mid          # proportional half-spread
    half = half[np.isfinite(half)]
    if half.empty:
        return None
    return {"n_quotes": int(len(half)),
            "half_median": float(half.median()),
            "half_p25": float(half.quantile(0.25)),
            "half_p90": float(half.quantile(0.90)),
            "half_mean": float(half.mean())}


def collect(symbols: list[str], sessions: list[date], feed: str,
            overwrite: bool = False, windows=DEFAULT_WINDOWS) -> pd.DataFrame:
    _trust_windows_certs()
    key, sec = _load_keys()
    from alpaca.data.historical import StockHistoricalDataClient

    client = StockHistoricalDataClient(key, sec)
    rows = []
    for day in sessions:
        for sym in symbols:
            df = fetch_session(client, sym, day, feed, overwrite=overwrite,
                               windows=windows)
            s = summarise(df)
            if s:
                rows.append({"symbol": sym, "date": day.isoformat(), **s})
        log.info("%s: %d symbol-sessions summarised so far", day, len(rows))
    return pd.DataFrame(rows)


def load_cached(symbols: list[str] | None = None) -> pd.DataFrame:
    """Rebuild the summary purely from cache -- no API calls, free and repeatable."""
    rows = []
    if not QUOTES_ROOT.exists():
        return pd.DataFrame()
    for sym_dir in sorted(QUOTES_ROOT.glob("symbol=*")):
        sym = sym_dir.name.split("=", 1)[1]
        if symbols and sym not in symbols:
            continue
        for day_dir in sorted(sym_dir.glob("date=*")):
            # prefer a full-session pull when both exist -- it is the better estimate
            f = next((c for c in (day_dir / "full.parquet",
                                  day_dir / "windows.parquet",
                                  day_dir / "quotes.parquet") if c.exists()), None)
            if f is None:
                continue
            try:
                s = summarise(pd.read_parquet(f))
            except Exception:
                continue
            if s:
                rows.append({"symbol": sym,
                             "date": day_dir.name.split("=", 1)[1], **s})
    return pd.DataFrame(rows)


def per_symbol(sessions_df: pd.DataFrame) -> pd.DataFrame:
    if sessions_df.empty:
        return pd.DataFrame()
    g = sessions_df.groupby("symbol")
    out = pd.DataFrame({
        "n_sessions": g.size(),
        "n_quotes": g["n_quotes"].sum(),
        "half_median": g["half_median"].median(),
        "half_p90": g["half_p90"].median(),
        "half_worst_session": g["half_median"].max(),
    })
    out["bps"] = out["half_median"] * 1e4
    out["vs_flat_5bps"] = out["half_median"] / FLAT_ASSUMPTION
    return out.sort_values("half_median")


def write_report(sessions_df: pd.DataFrame, feed: str) -> Path:
    sym = per_symbol(sessions_df)
    rep = {
        "as_of": date.today().isoformat(),
        "feed": feed,
        "bias": ("IEX is a single venue with a small share of consolidated volume, so "
                 "these quoted spreads are WIDER than the true NBBO. Every cost here "
                 "is an overstatement, which makes it a conservative cost test."),
        "flat_assumption_per_side": FLAT_ASSUMPTION,
        "session_window_utc": [SESSION_OPEN.isoformat(), SESSION_CLOSE.isoformat()],
        "n_symbols": int(len(sym)),
        "n_symbol_sessions": int(len(sessions_df)),
        "per_symbol": sym.reset_index().to_dict(orient="records"),
        "per_session": sessions_df.to_dict(orient="records"),
        "caveats": [
            "quoted spread only -- market impact is NOT measured or included.",
            "regular session only; the open and close auctions, which are the widest "
            "part of the day, are excluded.",
            "one sampled session per month -- this is a per-symbol LEVEL, not a daily "
            "time series, and it does not capture stress days. Use half_p90 for a "
            "stress-aware figure.",
            "free plan = IEX feed; spreads read wider than consolidated NBBO.",
        ],
    }
    out = config.JOURNAL / f"quote_spreads_{rep['as_of']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=24,
                    help="how many monthly sessions to sample")
    ap.add_argument("--symbols", type=str, default="",
                    help="comma list; default = the 22-asset equity universe")
    ap.add_argument("--feed", type=str, default="iex", choices=["iex", "sip"])
    ap.add_argument("--report", action="store_true",
                    help="rebuild from cache only, no API calls")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--full-session", action="store_true",
                    help="pull the whole regular session instead of the three "
                         "stratified windows (accurate, very large, slow)")
    args = ap.parse_args()
    windows = None if args.full_session else DEFAULT_WINDOWS

    symbols = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
               or list(config.EQUITY_22))

    if args.report:
        sessions_df = load_cached(symbols)
        if sessions_df.empty:
            print("no cached quotes yet -- run without --report first")
            return 1
    else:
        days = _sample_sessions(args.sessions)
        log.info("sampling %d sessions x %d symbols (%s feed)",
                 len(days), len(symbols), args.feed)
        sessions_df = collect(symbols, days, args.feed, overwrite=args.overwrite,
                              windows=windows)
        if sessions_df.empty:
            print("no quotes returned -- check keys, feed entitlement and date range")
            return 1

    out = write_report(sessions_df, args.feed)
    sym = per_symbol(sessions_df)

    print("=" * 84)
    print(f"Measured half-spreads from real quotes ({args.feed} feed)")
    print("=" * 84)
    print(f"{len(sym)} symbols, {len(sessions_df)} symbol-sessions, "
          f"{int(sym['n_quotes'].sum()):,} quotes")
    print(f"\n  {'symbol':<8s} {'sess':>5s} {'quotes':>10s} {'half_med':>10s} "
          f"{'bps':>7s} {'p90_bps':>8s} {'x flat 5bps':>11s}")
    for s, r in sym.iterrows():
        print(f"  {s:<8s} {int(r['n_sessions']):>5d} {int(r['n_quotes']):>10,d} "
              f"{r['half_median']:>10.7f} {r['bps']:>7.2f} "
              f"{r['half_p90']*1e4:>8.2f} {r['vs_flat_5bps']:>11.2f}")

    med = float(sym["half_median"].median())
    print(f"\n  median across symbols: {med*1e4:.2f} bps per side "
          f"({med/FLAT_ASSUMPTION:.2f}x the flat 5 bps assumption)")
    cheaper = int((sym["half_median"] < FLAT_ASSUMPTION).sum())
    print(f"  {cheaper} of {len(sym)} symbols are CHEAPER than the flat assumption")
    print("\n  IEX bias: these are OVERSTATEMENTS of true NBBO spreads, so an edge "
          "that survives them survives reality.")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
