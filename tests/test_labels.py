"""Label correctness on synthetic bars: barrier hits, truncation, costs."""
from datetime import datetime, timedelta, timezone

import polars as pl

from src import config
from src.features.build import add_labels


def _bars(closes, highs=None, lows=None, start=None, bar_minutes=1):
    """Minimal RTH bar frame with a constant ATR-friendly shape."""
    n = len(closes)
    start = start or datetime(2024, 3, 4, 14, 30, tzinfo=timezone.utc)  # 09:30 NY
    ts = [start + timedelta(minutes=bar_minutes * i) for i in range(n)]
    highs = highs or [c + 0.5 for c in closes]
    lows = lows or [c - 0.5 for c in closes]
    df = pl.DataFrame({
        "timestamp": ts,
        "open": closes, "high": highs, "low": lows, "close": closes,
        "volume": [1000.0] * n, "vwap": closes,
    }).with_columns(pl.col("timestamp").dt.cast_time_unit("us"))
    df = df.with_columns(pl.col("timestamp").dt.convert_time_zone("America/New_York").alias("ts_local"))
    # constant ATR = 1.0 so barriers are entry +1.0 / -0.75
    return df.with_columns(pl.lit(1.0).alias("atr"))


def test_tp_hit_before_sl():
    # flat, then bar 2 spikes high through +1*ATR without touching -0.75*ATR
    closes = [100.0] * 40
    highs = [100.5] * 40
    lows = [99.8] * 40
    highs[2] = 101.5  # hits up barrier (101.0)
    df = add_labels(_bars(closes, highs, lows), "SPY", horizons=[5])
    assert df["tb_5"][0] == 1


def test_sl_hit_before_tp():
    closes = [100.0] * 40
    highs = [100.4] * 40
    lows = [99.8] * 40
    lows[1] = 99.0  # hits down barrier (99.25) at bar 1
    highs[3] = 101.5  # TP later — too late
    df = add_labels(_bars(closes, highs, lows), "SPY", horizons=[5])
    assert df["tb_5"][0] == 0


def test_timeout_is_zero():
    closes = [100.0] * 40  # never moves enough either way
    df = add_labels(_bars(closes), "SPY", horizons=[5])
    assert df["tb_5"][0] == 0


def test_truncation_at_session_close():
    # 20 bars in one session: entries whose 5-bar window would run past the
    # last bar of the day must be null, not short-horizon mislabels
    closes = [100.0] * 20
    df = add_labels(_bars(closes), "SPY", horizons=[5])
    # last 5 entries have no same-session i+5 bar
    assert df["tb_5"][19] is None
    assert df["tb_5"][15] is None
    assert df["tb_5"][14] is not None
    import math
    assert math.isnan(df["fwd_ret_5"][15])  # float cols carry NaN, not null


def test_two_sessions_do_not_bleed():
    # session 1: 10 bars, session 2: 10 bars next day — window can't cross
    day1 = _bars([100.0] * 10)
    day2 = _bars([100.0] * 10, start=datetime(2024, 3, 5, 14, 30, tzinfo=timezone.utc))
    df = pl.concat([day1, day2])
    out = add_labels(df, "SPY", horizons=[5])
    # entry 7 in day 1: bars 8..12 cross into day 2 -> null
    assert out["tb_5"][7] is None
    # entry 4 in day 1: bars 5..9 stay in day 1 -> labeled
    assert out["tb_5"][4] is not None


def test_net_edge_subtracts_round_trip_cost():
    closes = [100.0 + 0.1 * i for i in range(40)]
    df = add_labels(_bars(closes), "SPY", horizons=[5])
    c = config.cost_per_side("SPY")
    row = df.filter(pl.col("fwd_ret_5").is_not_nan()).row(0, named=True)
    assert abs(row["net_edge_5"] - (row["fwd_ret_5"] - 2 * c)) < 1e-9


def test_mfe_mae_in_atr_units():
    closes = [100.0] * 40
    highs = [100.2] * 40
    lows = [99.9] * 40
    highs[3] = 100.8  # best excursion +0.8 ATR (ATR=1) — below TP so no early exit
    lows[2] = 99.5    # worst -0.5 ATR — above SL (99.25)
    df = add_labels(_bars(closes, highs, lows), "SPY", horizons=[5])
    assert abs(df["mfe_5"][0] - 0.8) < 1e-9
    assert abs(df["mae_5"][0] - 0.5) < 1e-9
