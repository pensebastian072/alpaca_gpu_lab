"""Daily regime-context columns from the near-miss daily signals.

VRP, vol-managed sizing, VIX term structure and net-liquidity trend all
showed PF > 1.5 in qlib_lab's daily batteries but failed the gate under
n_trials deflation — they re-enter here ONLY as context features for intraday
models (CLAUDE.md rule 7), never as standalone strategies.

Sources, all fail-safe (missing source -> NaN column, never raises):
  - own 1Day spine (SPY realized vol, VIXY momentum);
  - macro_gpu_lab's daily panel parquet READ-ONLY for true index levels
    (VIX level — VIXY is a roll-drag ETP, usable as returns but not levels);
  - qlib_lab's fred_liquidity parquet READ-ONLY for net liquidity.

Output: daily frame (timestamp + rg_* columns), as-of joined backward onto
minute rows like every other context layer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402

QLIB_LAB_DIR = Path(r"C:\Users\<your-user>\qlib_lab")
VOL_TARGET_ANN = 0.15  # vol-managed scalar target (annualized)


def _load_daily_close(sym: str) -> pd.Series | None:
    root = config.RAW / "bars_1Day" / f"symbol={sym}"
    files = sorted(root.glob("year=*/month=*/bars.parquet"))
    if not files:
        return None
    df = pl.concat([pl.read_parquet(f, columns=["timestamp", "close"]) for f in files])
    s = df.unique(subset="timestamp").sort("timestamp").to_pandas()
    return s.set_index("timestamp")["close"]


def _macro_panel_vix() -> pd.Series | None:
    """True VIX level from macro_gpu_lab's daily panel (read-only, fail-safe)."""
    try:
        panels = sorted((config.MACRO_GPU_LAB_DIR / "data").glob("panel_*.parquet"))
        if not panels:
            return None
        df = pl.read_parquet(panels[-1]).to_pandas()
        # wide layout: 22 asset close columns + the saved pandas index
        # ("__index_level_0__", naive ms datetimes)
        date_col = next((c for c in df.columns
                         if "index_level" in c or c.lower() == "date"), None)
        if date_col is None or "VIX" not in df.columns:
            return None
        s = df.set_index(date_col)["VIX"].dropna()
        s.index = pd.to_datetime(s.index, utc=True)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        return s if s.index.is_monotonic_increasing and len(s) else None
    except Exception:  # noqa: BLE001 — context is optional, never fatal
        return None
    return None


def _net_liquidity() -> pd.Series | None:
    try:
        p = QLIB_LAB_DIR / "data" / "fred_liquidity.parquet"
        if not p.exists():
            return None
        df = pl.read_parquet(p).to_pandas()
        cols = {c.lower(): c for c in df.columns}
        date_col = cols.get("date") or df.columns[0]
        val_col = cols.get("net_liquidity") or cols.get("netliq") or df.columns[-1]
        s = df.set_index(date_col)[val_col]
        s.index = pd.to_datetime(s.index, utc=True)
        return s.sort_index()
    except Exception:  # noqa: BLE001
        return None


def build_regime_context() -> pl.DataFrame:
    """Daily (timestamp + rg_*) frame. Missing sources leave NaN columns."""
    spy = _load_daily_close("SPY")
    if spy is None:
        raise FileNotFoundError("1Day spine missing SPY — run the daily backfill")
    idx = spy.index
    out = pd.DataFrame(index=idx)

    ret = np.log(spy / spy.shift(1))
    rv20 = ret.rolling(20).std() * np.sqrt(252)
    out["rg_spy_rv20"] = rv20
    out["rg_spy_mom20"] = np.log(spy / spy.shift(20))
    # vol-managed scalar: target/realized, capped at 2 (near-miss W03 as context)
    out["rg_vol_managed"] = (VOL_TARGET_ANN / rv20).clip(upper=2.0)

    vix = _macro_panel_vix()
    if vix is not None:
        vix = vix[~vix.index.duplicated(keep="last")]
        v = vix.reindex(idx, method="ffill")
        out["rg_vix"] = v
        out["rg_vix_chg20"] = v - v.shift(20)
        # VRP: implied^2 - realized^2 (near-miss H07 as context)
        out["rg_vrp"] = (v / 100.0) ** 2 - rv20**2
    else:
        out["rg_vix"] = np.nan
        out["rg_vix_chg20"] = np.nan
        out["rg_vrp"] = np.nan

    vixy = _load_daily_close("VIXY")
    if vixy is not None:
        vr = np.log(vixy / vixy.shift(1)).reindex(idx)
        # VIXY drifts down in contango, up in backwardation — 20d drift as a
        # tradable term-structure proxy (near-miss V1 as context)
        out["rg_vixy_drift20"] = vr.rolling(20).sum()

    nl = _net_liquidity()
    if nl is not None:
        nl = nl[~nl.index.duplicated(keep="last")]
        n = nl.reindex(idx, method="ffill")
        out["rg_netliq_chg13w_z"] = (
            (n - n.shift(65)) - (n - n.shift(65)).rolling(252).mean()
        ) / (n - n.shift(65)).rolling(252).std()
    else:
        out["rg_netliq_chg13w_z"] = np.nan

    # LEAK GUARD: daily bars are stamped at day START but the close (and every
    # column above) is only known after the session ends. Shift +1 day so any
    # intraday row only ever sees strictly-prior-day regime state.
    out.index = out.index + pd.Timedelta(days=1)
    out.index.name = "timestamp"
    res = pl.from_pandas(out.reset_index())
    return res.with_columns(pl.col("timestamp").dt.cast_time_unit("us")).sort("timestamp")


def add_regime(df: pl.DataFrame, rg_ctx: pl.DataFrame) -> pl.DataFrame:
    """As-of join (backward): each minute row carries yesterday's regime state.

    Daily bars are stamped at session times; joining backward means an intraday
    row only ever sees regime values computed from data at or before its own
    timestamp — same no-peek semantics as the other context layers.
    """
    return df.sort("timestamp").join_asof(
        rg_ctx.sort("timestamp"), on="timestamp", strategy="backward")
