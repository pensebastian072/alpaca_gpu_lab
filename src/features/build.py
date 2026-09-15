"""Polars feature engineering + triple-barrier labels (CPU).

Research only. Features feed a trade-quality FILTER (not a signal generator).
CPU Polars by design — GPU earns its keep at model-train/param-search, not
feature-eng at these row counts (CLAUDE.md GPU discipline).

Per-symbol features (plan section 2, minus microstructure which needs quote
data we don't backfill yet):
  returns   : log returns over 1/3/5/15/30 bars
  volatility: rolling return-std (15/30/60), ATR, realized vol, high-low range
  volume    : relative volume, volume z-score, volume acceleration
  trend     : EMA distance, VWAP distance, rolling slope, breakout distance
  session   : minutes-since-open, session bucket, day-of-week

Label (triple-barrier, de Prado): from each RTH entry bar, look forward
`horizon` bars; upper barrier = entry + tp_mult*ATR, lower = entry -
sl_mult*ATR. label = 1 if upper hit before lower, else 0 (vertical-barrier
timeout counts as 0 — no tradable up-move). Default tp/sl = 1.0 / 0.75.

Cross-market features (SPY-vs-QQQ, vs-yield via IEF/TLT, vs-VIXY) are a second
pass that joins per-symbol frames on timestamp — see build_cross_market().

CLI:
  python -m src.features.build --symbol SPY --timeframe 1Min
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402

# Regular trading hours in exchange-local time (handles DST via tz-convert).
RTH_OPEN = (9, 30)
RTH_CLOSE = (16, 0)
NY_TZ = "America/New_York"


def load_bars(symbol: str, timeframe: str) -> pl.DataFrame:
    """Scan all monthly Parquet partitions for one symbol, sorted by time."""
    root = config.RAW / f"bars_{timeframe}" / f"symbol={symbol}"
    files = sorted(root.glob("year=*/month=*/bars.parquet"))
    if not files:
        sys.exit(f"No parquet for {symbol} {timeframe} under {root}")
    df = pl.concat([pl.read_parquet(f) for f in files])
    return df.unique(subset="timestamp").sort("timestamp")


def _session_filter(df: pl.DataFrame, all_hours: bool = False) -> pl.DataFrame:
    """Keep only regular-hours bars; add local-time helper columns.

    `all_hours=True` keeps every bar and is meant for 24/7 assets (crypto), where the RTH
    filter is not a data-quality screen but a distortion: it threw away 82% of BTCUSD's
    hourly bars (48,812 -> 8,703) and graded a round-the-clock market only during US equity
    hours. Equity/ETF behaviour is unchanged.
    """
    local = pl.col("timestamp").dt.convert_time_zone(NY_TZ)
    df = df.with_columns([
        local.alias("ts_local"),
        # cast to Int32: hour()*60 overflows the default i8 (>127) and wraps
        (local.dt.hour().cast(pl.Int32) * 60 + local.dt.minute().cast(pl.Int32)).alias("min_of_day"),
        local.dt.weekday().alias("dow"),  # 1=Mon .. 7=Sun
    ])
    if all_hours:
        return df
    open_m = RTH_OPEN[0] * 60 + RTH_OPEN[1]
    close_m = RTH_CLOSE[0] * 60 + RTH_CLOSE[1]
    return df.filter(
        (pl.col("min_of_day") >= open_m) & (pl.col("min_of_day") < close_m)
        & (pl.col("dow") <= 5)
    )


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    c = pl.col("close")
    logret = (c / c.shift(1)).log()
    df = df.with_columns(logret.alias("ret_1"))

    feats = []
    # returns over multiple horizons
    for k in (1, 3, 5, 15, 30):
        feats.append((c / c.shift(k)).log().alias(f"ret_{k}"))
    # volatility
    for w in (15, 30, 60):
        feats.append(pl.col("ret_1").rolling_std(w).alias(f"vol_{w}"))
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - c.shift(1)).abs(),
        (pl.col("low") - c.shift(1)).abs(),
    )
    feats.append(tr.alias("true_range"))
    feats.append(((pl.col("high") - pl.col("low")) / c).alias("hl_range_rel"))
    # volume
    vol = pl.col("volume")
    rv_mean = vol.rolling_mean(30)
    feats.append((vol / rv_mean).alias("rel_volume"))
    feats.append(((vol - rv_mean) / vol.rolling_std(30)).alias("vol_zscore"))
    feats.append((vol - vol.shift(1)).alias("vol_accel"))
    # trend
    ema20 = c.ewm_mean(span=20)
    feats.append((c / ema20 - 1).alias("ema20_dist"))
    feats.append((c / pl.col("vwap") - 1).alias("vwap_dist"))
    feats.append((c - c.shift(10)).alias("slope_10"))
    feats.append((c / c.rolling_max(30) - 1).alias("breakout_up"))
    feats.append((c / c.rolling_min(30) - 1).alias("breakout_dn"))
    # session
    feats.append((pl.col("min_of_day") - (RTH_OPEN[0] * 60 + RTH_OPEN[1])).alias("min_since_open"))

    df = df.with_columns(feats)
    # ATR after true_range exists
    df = df.with_columns(pl.col("true_range").rolling_mean(30).alias("atr"))
    return df


def add_labels(
    df: pl.DataFrame,
    symbol: str,
    horizons: list[int] | None = None,
    tp_mult: float = 1.0,
    sl_mult: float = 0.75,
    truncate_at_session_close: bool = True,
) -> pl.DataFrame:
    """Multi-horizon labels: triple-barrier + forward return + MFE/MAE + net edge.

    Per horizon h (bars), from each entry bar i:
      tb_{h}      : 1 if +tp_mult*ATR hit before -sl_mult*ATR within h bars,
                    0 if SL hit first or timeout (no tradable up-move),
                    null if the window would cross the session close (or runs
                    off the end of data) — truncated windows systematically
                    mislabel near the close, so they are EXCLUDED, not shortened.
      fwd_ret_{h} : log close-to-close forward return, null if truncated.
      mfe_{h}     : max favorable excursion (highest high - entry) / ATR.
      mae_{h}     : max adverse excursion (entry - lowest low) / ATR.
      net_edge_{h}: fwd_ret_{h} - 2 * cost_per_side(symbol)  (round trip,
                    fractional cost — never absolute units).
      tb_ret_{h}  : realized barrier-exit return NET of round-trip cost:
                    +tp_mult*ATR/close if TP hit first, -sl_mult*ATR/close if
                    SL hit first, fwd_ret on timeout — the honest PnL of the
                    long trade the tb label describes.

    Deliberately NO "next candle up" target (noisy, untradable).
    Vectorized numpy (O(n*h) elementwise passes) — the old per-row python loop
    was O(n*h) interpreted and unusable at millions of rows.
    """
    import numpy as np

    if horizons is None:
        horizons = config.HORIZONS_BARS
    cost = config.cost_per_side(symbol)

    close = df["close"].to_numpy().astype(np.float64)
    high = df["high"].to_numpy().astype(np.float64)
    low = df["low"].to_numpy().astype(np.float64)
    atr = np.asarray(df["atr"].to_numpy(), dtype=np.float64)
    # session id: bars are RTH-filtered, so the local calendar date IS the session
    day = df["ts_local"].dt.date().to_numpy()
    n = len(df)

    out_cols: list[pl.Series] = []
    for h in horizons:
        ok_atr = np.isfinite(atr) & (atr > 0)
        up = close + tp_mult * atr
        dn = close - sl_mult * atr

        # window i+1..i+h must stay inside data. For INTRADAY timeframes it must
        # also stay inside the same session (no overnight hold — truncated
        # windows near the close mislabel). For 1Hour/4Hour/1Day the strategy
        # legitimately holds across sessions, so only the data bound applies.
        in_data = np.arange(n) + h < n
        if truncate_at_session_close:
            same_day = np.zeros(n, dtype=bool)
            same_day[in_data] = day[np.arange(n)[in_data] + h] == day[in_data]
            valid = in_data & same_day & ok_atr
        else:
            valid = in_data & ok_atr

        first_up = np.full(n, h + 1, dtype=np.int32)   # bar offset of first TP hit
        first_dn = np.full(n, h + 1, dtype=np.int32)   # bar offset of first SL hit
        for j in range(1, h + 1):
            hi_j = np.empty(n)
            lo_j = np.empty(n)
            hi_j[: n - j] = high[j:]
            lo_j[: n - j] = low[j:]
            hi_j[n - j:] = -np.inf
            lo_j[n - j:] = np.inf
            hit_up = (hi_j >= up) & (first_up == h + 1)
            hit_dn = (lo_j <= dn) & (first_dn == h + 1)
            first_up[hit_up] = j
            first_dn[hit_dn] = j

        tb = np.where(first_up < first_dn, 1, 0).astype(np.int8)
        # None where the window is truncated/invalid
        tb_col = [int(tb[i]) if valid[i] else None for i in range(n)]

        fwd = np.full(n, np.nan)
        fwd[: n - h] = np.log(close[h:] / close[: n - h])
        fwd[~valid] = np.nan

        # realized barrier-exit return of the long trade, net of costs
        tp_hit = first_up < first_dn
        sl_hit = first_dn < first_up
        tb_ret = np.where(tp_hit, tp_mult * atr / close,
                          np.where(sl_hit, -sl_mult * atr / close, fwd))
        tb_ret = tb_ret - 2.0 * cost
        tb_ret[~valid] = np.nan

        # future-window extremes over i+1..i+h via sliding windows
        mfe = np.full(n, np.nan)
        mae = np.full(n, np.nan)
        if n > h:
            from numpy.lib.stride_tricks import sliding_window_view
            hw = sliding_window_view(high[1:], h).max(axis=1)   # rows 0..n-h-1
            lw = sliding_window_view(low[1:], h).min(axis=1)
            m = n - h
            mfe[:m] = (hw[:m] - close[:m]) / atr[:m]
            mae[:m] = (close[:m] - lw[:m]) / atr[:m]
        mfe[~valid] = np.nan
        mae[~valid] = np.nan

        out_cols.extend([
            pl.Series(f"tb_{h}", tb_col, dtype=pl.Int8),
            pl.Series(f"fwd_ret_{h}", fwd),
            pl.Series(f"tb_ret_{h}", tb_ret),
            pl.Series(f"mfe_{h}", mfe),
            pl.Series(f"mae_{h}", mae),
            pl.Series(f"net_edge_{h}", fwd - 2.0 * cost),
        ])

    return df.with_columns(out_cols)


# Reference symbols whose returns become cross-market context for every target.
REF_SYMBOLS = ["SPY", "QQQ", "IEF", "TLT", "VIXY"]


def build_market_context(timeframe: str) -> pl.DataFrame:
    """timestamp + 1/15-bar returns for each reference symbol, RTH-aligned.

    Joined onto any target symbol to build relative-strength / regime features.
    """
    ctx = None
    for sym in REF_SYMBOLS:
        d = _session_filter(load_bars(sym, timeframe)).select([
            "timestamp",
            (pl.col("close") / pl.col("close").shift(1)).log().alias(f"{sym}_ret1"),
            (pl.col("close") / pl.col("close").shift(15)).log().alias(f"{sym}_ret15"),
        ])
        ctx = d if ctx is None else ctx.join(d, on="timestamp", how="full", coalesce=True)
    # ctx is dense in timestamp (union of all refs) but sparse per column;
    # forward-fill so every row carries each ref's last-known return.
    ctx = ctx.sort("timestamp")
    return ctx.with_columns(pl.all().exclude("timestamp").forward_fill())


def add_cross_market(df: pl.DataFrame, ctx: pl.DataFrame, symbol: str) -> pl.DataFrame:
    """Join market context and derive relative-strength + regime features."""
    # as-of (backward) join: illiquid refs (IEF/VIXY) have sparse IEX minute
    # bars; exact-timestamp join leaves most minutes null. Carry the last-known
    # ref bar instead — correct "current regime state" semantics, null-free.
    df = df.sort("timestamp").join_asof(ctx.sort("timestamp"), on="timestamp", strategy="backward")
    s15 = pl.col("ret_15")
    df = df.with_columns([
        pl.col("SPY_ret15").alias("mkt_ret15"),
        (s15 - pl.col("SPY_ret15")).alias("rs_vs_spy"),   # relative strength vs market
        (s15 - pl.col("QQQ_ret15")).alias("rs_vs_qqq"),
        pl.col("IEF_ret15").alias("yield7y_ret15"),        # bonds up ~ yields down (risk-off)
        pl.col("TLT_ret15").alias("yield20y_ret15"),
        pl.col("VIXY_ret15").alias("vix_ret15"),           # vol regime
    ])
    raw_ctx = [c for c in ctx.columns if c != "timestamp"]  # {sym}_ret1/_ret15
    return df.drop(raw_ctx)


def is_24_7(symbol: str) -> bool:
    """Crypto trades continuously — no RTH window, and no session close to truncate at."""
    return config.UNIVERSE_22.get(symbol, {}).get("asset_class") == "crypto"


def build_symbol(symbol: str, timeframe: str, ctx: pl.DataFrame | None = None,
                 enrich: bool = False, rg_ctx: pl.DataFrame | None = None,
                 sentiment: bool = False, all_hours: bool | None = None) -> Path:
    all_hours = is_24_7(symbol) if all_hours is None else all_hours
    df = load_bars(symbol, timeframe)
    df = _session_filter(df, all_hours=all_hours)
    df = add_features(df)
    if ctx is not None:
        df = add_cross_market(df, ctx, symbol)
    if enrich:
        # chart-derived context ONLY: xa_* (cross-asset) + rg_* (regime).
        # News (ns_*) is decoupled behind the separate `sentiment` flag so
        # chart batteries stay pure price/volume/cross-asset/regime.
        from src.features.cross_asset import add_cross_asset, build_context_for
        from src.features.regime import add_regime, build_regime_context
        df = add_cross_asset(df, build_context_for(symbol))
        df = add_regime(df, rg_ctx if rg_ctx is not None else build_regime_context())
    if sentiment:
        from src.features.sentiment import add_sentiment
        df = add_sentiment(df, symbol)  # neutral columns if no scores yet
    # intraday TFs don't hold overnight; hourly+ do. A 24/7 asset has no session close,
    # so truncating there would cut labels at an hour that means nothing to it.
    intraday = timeframe in ("1Min", "5Min", "15Min")
    df = add_labels(df, symbol, truncate_at_session_close=intraday and not all_hours)
    out = config.FEATURES / f"{symbol}_{timeframe}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    xm = "with cross-market" if ctx is not None else "per-symbol only"
    rates = []
    for h in config.HORIZONS_BARS:
        lab = df.filter(pl.col(f"tb_{h}").is_not_null())
        pos = lab.filter(pl.col(f"tb_{h}") == 1).height
        rates.append(f"tb_{h}: n={lab.height} pos={pos / max(lab.height, 1):.3f}")
    session = "24/7 bars" if all_hours else "RTH bars"
    print(f"{symbol}: {df.height} {session} ({xm}) -> {out}\n  " + "; ".join(rates))
    return out


def build_all(timeframe: str, enrich: bool = False, sentiment: bool = False) -> None:
    print("building market context...")
    ctx = build_market_context(timeframe)
    rg_ctx = None
    if enrich:
        from src.features.regime import build_regime_context
        print("building regime context...")
        rg_ctx = build_regime_context()
    for sym in config.UNIVERSE:
        build_symbol(sym, timeframe, ctx=ctx, enrich=enrich, rg_ctx=rg_ctx,
                     sentiment=sentiment)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--timeframe", default=config.DEFAULT_TIMEFRAME)
    ap.add_argument("--all", action="store_true", help="build every UNIVERSE symbol with cross-market features")
    ap.add_argument("--cross", action="store_true", help="add cross-market features (single symbol)")
    ap.add_argument("--enrich", action="store_true",
                    help="add hourly cross-asset (xa_*) + daily regime (rg_*) context")
    ap.add_argument("--sentiment", action="store_true",
                    help="also add news sentiment (ns_*) context — decoupled, off by default")
    a = ap.parse_args()
    if a.all:
        build_all(a.timeframe, enrich=a.enrich, sentiment=a.sentiment)
    elif a.cross:
        build_symbol(a.symbol, a.timeframe, ctx=build_market_context(a.timeframe),
                     enrich=a.enrich, sentiment=a.sentiment)
    else:
        build_symbol(a.symbol, a.timeframe, enrich=a.enrich, sentiment=a.sentiment)
