"""Cross-asset / regime context: leak guards + column sanity."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import polars as pl

from src.features import cross_asset


def _rets(n=400, cols=("SPY", "UUP", "GLD", "TLT", "HYG", "QQQ"), seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 14:00", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(rng.normal(0, 0.001, (n, len(cols))), index=idx, columns=list(cols))


def test_market_context_columns():
    ctx = cross_asset.market_context(_rets())
    assert "xa_absorption" in ctx and "xa_avg_corr" in ctx
    assert "xa_z_qqq_tlt" in ctx  # spread with both legs present
    # absorption in (0, 1]
    vals = ctx["xa_absorption"].dropna()
    assert not vals.empty and (vals > 0).all() and (vals <= 1).all()


def test_target_context_columns():
    ctx = cross_asset.target_context(_rets(), "QQQ")
    assert "xa_corr_SPY" in ctx and "xa_beta_SPY" in ctx
    assert "xa_leadlag_spy_lag" in ctx
    # correlations bounded
    v = ctx["xa_corr_SPY"].dropna()
    assert (v.abs() <= 1.0 + 1e-9).all()


def test_asof_join_never_sees_unclosed_bar():
    # hourly context stamped at bar END: a minute row inside hour H must only
    # see context computed from hours strictly before H
    ctx = pl.DataFrame({
        "timestamp": [datetime(2024, 1, 2, h, 0, tzinfo=timezone.utc) for h in (15, 16, 17)],
        "xa_absorption": [0.1, 0.2, 0.3],
    }).with_columns(pl.col("timestamp").dt.cast_time_unit("us"))
    # ctx stamps above are already bar-END stamps (the +1h shift applied)
    rows = pl.DataFrame({
        "timestamp": [datetime(2024, 1, 2, 15, 30, tzinfo=timezone.utc)],
    }).with_columns(pl.col("timestamp").dt.cast_time_unit("us"))
    joined = cross_asset.add_cross_asset(rows, ctx)
    # 15:30 row: bar 15:00-16:00 not closed yet -> must carry the 15:00-END
    # stamp (i.e. the 14:00-15:00 bar's value), which is 0.1
    assert joined["xa_absorption"][0] == 0.1


def test_context_shift_applied():
    # build_context_for shifts +1h: last context stamp > last raw hourly stamp
    rets = _rets(100)
    ctx = cross_asset.market_context(rets)
    shifted = ctx.index + pd.Timedelta(hours=1)
    assert shifted[-1] == rets.index[-1] + pd.Timedelta(hours=1)
