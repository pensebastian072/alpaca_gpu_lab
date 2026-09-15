"""Intraday cross-asset relationship features over the 22-asset hourly spine.

Math adapted from hq-trading-system/analytics/macro_engine.py (corr windows,
instability flag, beta, divergence z, absorption ratio, lead-lag margin) and
macro_gpu_lab/macro_gpu_lab/features.py (slow/fast corr + drift conventions) —
here computed at 1Hour granularity as MODEL FEATURES for intraday horizons,
not as a standalone strategy (daily strategies on these relationships are
exhausted — see CLAUDE.md rule 7).

Two layers:
  market_context(rets)  — shared across targets: Kritzman absorption ratio +
      15/60-bar shift, average pairwise correlation, credit stress (HYG/LQD z),
      divergence z-scores for the 9 canonical spreads (hq SPREADS remapped to
      the ETF-proxy tickers).
  target_context(rets, sym) — per target: rolling corr to anchors
      (SPY/UUP/GLD/TLT/HYG) slow-60 + fast-20 + sign-instability flag, beta vs
      SPY/UUP + 20-bar beta drift, lead-lag best lag vs SPY (±6 bars, only
      meaningful if it beats contemporaneous |corr| by 0.10 — hq LL_MARGIN).

All columns are backward-looking (rolling windows end at the current bar).
Everything is as-of joined backward onto minute rows by build.build_symbol —
last-known hourly state, the same semantics as add_cross_market.

Hourly data is small (~12k stamps x 22 assets) — pandas/numpy is fine here;
the GPU earns its keep at training, not at this matrix size.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402

ANCHORS = ["SPY", "UUP", "GLD", "TLT", "HYG"]
SLOW, FAST = 60, 20          # hq CORR_WINDOW / CORR_WINDOW_FAST
Z_WINDOW = 120               # hq Z_WINDOW
BETA_WINDOW = 60             # hq BETA_WINDOW
LL_MAX_LAG, LL_MARGIN = 6, 0.10
ABSORPTION_WINDOW = 60
UNSTABLE_MIN_ABS = 0.2

# hq macro_config.SPREADS remapped to the Alpaca ETF proxies
SPREADS = {
    "copper_gold": ("CPER", "GLD"),
    "qqq_tlt": ("QQQ", "TLT"),
    "gold_dxy": ("GLD", "UUP"),
    "spy_hyg": ("SPY", "HYG"),
    "btc_qqq": ("BTCUSD", "QQQ"),
    "tip_tlt": ("TIP", "TLT"),
    "lqd_hyg": ("LQD", "HYG"),
    "eem_spy": ("EEM", "SPY"),
    "xlf_spy": ("XLF", "SPY"),
}


def load_hourly_panel() -> pd.DataFrame:
    """Wide close-price frame (UTC timestamp x symbol) from the 1Hour spine."""
    root = config.RAW / "bars_1Hour"
    cols = {}
    for sym_dir in sorted(root.glob("symbol=*")):
        sym = sym_dir.name.split("=")[1]
        files = sorted(sym_dir.glob("year=*/month=*/bars.parquet"))
        if not files:
            continue
        df = pl.concat([pl.read_parquet(f, columns=["timestamp", "close"]) for f in files])
        df = df.unique(subset="timestamp").sort("timestamp")
        s = df.to_pandas().set_index("timestamp")["close"]
        cols[sym] = s
    wide = pd.DataFrame(cols).sort_index()
    # Grid = stamps where SPY actually traded (RTH hours). BTC trades 24/7 so
    # it has prices at those stamps too; keeping BTC's overnight stamps instead
    # would flood every rolling window with all-NaN equity rows and kill the
    # absorption/corr features. ffill bridges sparse thin-ticker gaps first.
    spy_traded = wide["SPY"].notna() if "SPY" in wide else wide.notna().any(axis=1)
    # filter FIRST, then ffill: otherwise BTC's overnight stamps consume the
    # ffill budget before the next RTH stamp and thin tickers stay NaN
    return wide[spy_traded].ffill(limit=8)


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))


def market_context(rets: pd.DataFrame) -> pd.DataFrame:
    """Shared market-state columns, indexed like rets."""
    out = pd.DataFrame(index=rets.index)

    # Kritzman absorption ratio: share of variance in top-k eigenvectors
    basket = [c for c in rets.columns if c in config.UNIVERSE_22]
    R = rets[basket]
    k = max(1, int(np.ceil(len(basket) / 5)))  # hq ABSORPTION_K_DIVISOR=5
    absorb = np.full(len(R), np.nan)
    vals = R.to_numpy()
    for i in range(ABSORPTION_WINDOW, len(R)):
        win = vals[i - ABSORPTION_WINDOW:i]
        ok = ~np.isnan(win).any(axis=0)
        if ok.sum() < 6:
            continue
        c = np.corrcoef(win[:, ok], rowvar=False)
        if np.isnan(c).any():
            continue
        ev = np.linalg.eigvalsh(c)
        absorb[i] = float(ev[-k:].sum() / ev.sum())
    out["xa_absorption"] = absorb
    out["xa_absorption_chg15"] = out["xa_absorption"] - out["xa_absorption"].shift(15)
    out["xa_absorption_chg60"] = out["xa_absorption"] - out["xa_absorption"].shift(60)

    # average pairwise correlation (mean off-diagonal of the same window)
    avg_corr = np.full(len(R), np.nan)
    for i in range(ABSORPTION_WINDOW, len(R)):
        win = vals[i - ABSORPTION_WINDOW:i]
        ok = ~np.isnan(win).any(axis=0)
        if ok.sum() < 6:
            continue
        c = np.corrcoef(win[:, ok], rowvar=False)
        m = ok.sum()
        avg_corr[i] = float((c.sum() - m) / (m * (m - 1)))
    out["xa_avg_corr"] = avg_corr

    # credit stress: z of log(HYG/LQD)
    if "HYG" in rets and "LQD" in rets:
        spread = np.log(rets["HYG"].add(1).cumprod() / rets["LQD"].add(1).cumprod())
        out["xa_credit_stress"] = (
            (spread - spread.rolling(Z_WINDOW).mean()) / spread.rolling(Z_WINDOW).std())

    # divergence z for the canonical spreads
    cum = rets.add(1).cumprod()
    for name, (a, b) in SPREADS.items():
        if a in cum and b in cum:
            s = np.log(cum[a] / cum[b])
            out[f"xa_z_{name}"] = (s - s.rolling(Z_WINDOW).mean()) / s.rolling(Z_WINDOW).std()
    return out


def target_context(rets: pd.DataFrame, sym: str) -> pd.DataFrame:
    """Per-target relationship columns vs the anchor assets."""
    out = pd.DataFrame(index=rets.index)
    if sym not in rets:
        return out
    r = rets[sym]

    for a in ANCHORS:
        if a not in rets or a == sym:
            continue
        slow = r.rolling(SLOW, min_periods=SLOW // 2).corr(rets[a])
        fast = r.rolling(FAST, min_periods=FAST // 2).corr(rets[a])
        out[f"xa_corr_{a}"] = slow
        out[f"xa_corr_{a}_fast"] = fast
        # hq unstable_flag: fast/slow sign disagreement, magnitude-aware
        out[f"xa_corr_{a}_unstable"] = (
            (np.sign(slow) != np.sign(fast))
            & (slow.abs().combine(fast.abs(), max) >= UNSTABLE_MIN_ABS)
        ).astype(float)

    for base in ("SPY", "UUP"):
        if base not in rets or base == sym:
            continue
        cov = r.rolling(BETA_WINDOW, min_periods=BETA_WINDOW // 2).cov(rets[base])
        var = rets[base].rolling(BETA_WINDOW, min_periods=BETA_WINDOW // 2).var()
        beta = cov / var
        out[f"xa_beta_{base}"] = beta
        out[f"xa_beta_{base}_chg"] = beta - beta.shift(20)

    # lead-lag vs SPY: best lag in ±LL_MAX_LAG by |rolling corr|, reported only
    # when it beats contemporaneous by LL_MARGIN (hq lead_lag_scan convention)
    if "SPY" in rets and sym != "SPY":
        base = rets["SPY"]
        corrs = {}
        for k in range(-LL_MAX_LAG, LL_MAX_LAG + 1):
            corrs[k] = r.rolling(Z_WINDOW, min_periods=Z_WINDOW // 2).corr(base.shift(k))
        mat = pd.DataFrame(corrs)
        vals = mat.to_numpy()  # columns ordered -LL_MAX_LAG..+LL_MAX_LAG
        absvals = np.abs(vals)
        has_any = ~np.isnan(absvals).all(axis=1)
        best_col = np.zeros(len(mat), dtype=int)
        best_col[has_any] = np.nanargmax(absvals[has_any], axis=1)
        best_val = np.where(has_any, absvals[np.arange(len(mat)), best_col], np.nan)
        contemp = absvals[:, LL_MAX_LAG]  # lag 0 column
        significant = (best_val - contemp >= LL_MARGIN) & has_any
        best_lag = best_col - LL_MAX_LAG
        out["xa_leadlag_spy_lag"] = np.where(significant, best_lag, 0).astype(float)
        out["xa_leadlag_spy_corr"] = np.where(
            significant, vals[np.arange(len(mat)), best_col], 0.0)
    return out


def build_context_for(sym: str) -> pl.DataFrame:
    """Hourly (timestamp + xa_*) polars frame for one target symbol.

    LEAK GUARD: Alpaca hourly bars are stamped at the bar START but their
    close is only known at bar END. Context stamps are shifted +1 hour so a
    backward as-of join from any minute row can never see a close that hadn't
    printed yet.
    """
    prices = load_hourly_panel()
    rets = log_returns(prices)
    ctx = pd.concat([market_context(rets), target_context(rets, sym)], axis=1)
    ctx.index = ctx.index + pd.Timedelta(hours=1)
    ctx.index.name = "timestamp"
    out = pl.from_pandas(ctx.reset_index())
    return out.with_columns(pl.col("timestamp").dt.cast_time_unit("us")).sort("timestamp")


def add_cross_asset(df: pl.DataFrame, xa_ctx: pl.DataFrame) -> pl.DataFrame:
    """As-of join (backward) hourly relationship state onto minute rows."""
    return df.sort("timestamp").join_asof(
        xa_ctx.sort("timestamp"), on="timestamp", strategy="backward")
