"""Leakage + holdout-lock guarantees.

1. Features are backward-looking: rebuilding on a time-truncated frame
   reproduces identical feature rows (macro_gpu_lab test pattern).
2. The default dataset loader returns zero holdout (>=2026) rows, and
   unlocking requires the env flag.
3. Quarterly walk-forward keeps a horizon gap between train and test.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from src import config
from src.features.build import add_features
from src.models import dataset


def _synth_bars(n=600, seed=0):
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    start = datetime(2024, 3, 4, 14, 30, tzinfo=timezone.utc)
    ts = [start + timedelta(minutes=i) for i in range(n)]
    df = pl.DataFrame({
        "timestamp": ts,
        "open": closes, "high": closes * 1.001, "low": closes * 0.999,
        "close": closes, "volume": rng.uniform(500, 1500, n), "vwap": closes,
    }).with_columns(pl.col("timestamp").dt.cast_time_unit("us"))
    return df.with_columns(
        pl.col("timestamp").dt.convert_time_zone("America/New_York").alias("ts_local"),
        (pl.col("timestamp").dt.hour() * 60 + pl.col("timestamp").dt.minute())
        .cast(pl.Int32).alias("min_of_day"),
    )


def test_features_backward_looking_only():
    df = _synth_bars()
    full = add_features(df)
    cut = 400
    trunc = add_features(df.head(cut))
    fcols = [c for c in trunc.columns if c not in
             ("timestamp", "ts_local", "min_of_day")]
    a = full.head(cut).select(fcols).to_numpy()
    b = trunc.select(fcols).to_numpy()
    # identical where both defined (NaN == NaN counts as equal)
    mask = ~(np.isnan(a) & np.isnan(b))
    assert np.allclose(a[mask], b[mask], equal_nan=True)


def _panel_fixture(tmp_path, monkeypatch, years=(2025, 2026)):
    """Write a tiny fake feature parquet spanning the holdout boundary."""
    rows = []
    for y in years:
        for m in (1, 4, 7, 10):
            for d in range(1, 4):
                rows.append({
                    "timestamp": datetime(y, m, d, 15, 0, tzinfo=timezone.utc),
                    "close": 100.0, "ret_1": 0.0, "tb_5": 1,
                })
    df = pl.DataFrame(rows).with_columns(pl.col("timestamp").dt.cast_time_unit("us"))
    monkeypatch.setattr(config, "FEATURES", tmp_path)
    df.write_parquet(tmp_path / "SPY_1Min.parquet")


def test_holdout_locked_by_default(tmp_path, monkeypatch):
    _panel_fixture(tmp_path, monkeypatch)
    panel = dataset.load_panel(["SPY"])
    assert panel.filter(pl.col("timestamp") >= datetime(2026, 1, 1, tzinfo=timezone.utc)).is_empty()
    assert not panel.is_empty()  # 2025 rows survive


def test_holdout_unlock_needs_env(tmp_path, monkeypatch):
    _panel_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "HOLDOUT_UNLOCK", False)
    with pytest.raises(PermissionError):
        dataset.load_panel(["SPY"], unlock_holdout=True)
    monkeypatch.setattr(config, "HOLDOUT_UNLOCK", True)
    panel = dataset.load_panel(["SPY"], unlock_holdout=True)
    assert not panel.filter(pl.col("timestamp") >= datetime(2026, 1, 1, tzinfo=timezone.utc)).is_empty()


def test_quarterly_walk_forward_gap():
    n = 2000
    start = datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc)
    ts = [start + timedelta(hours=6 * i) for i in range(n)]  # spans ~16 months
    panel = pl.DataFrame({"timestamp": ts, "x": list(range(n))}).with_columns(
        pl.col("timestamp").dt.cast_time_unit("us"))
    horizon = 30
    seen = 0
    for train, test, label in dataset.quarterly_walk_forward(panel, horizon):
        seen += 1
        assert train["timestamp"].max() < test["timestamp"].min()
        # gap: at least `horizon` panel stamps removed before test start
        stamps = panel["timestamp"].unique().sort()
        between = stamps.filter(
            (stamps >= train["timestamp"].max()) & (stamps < test["timestamp"].min()))
        assert len(between) >= horizon
    assert seen >= 2
