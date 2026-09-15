"""Phase-2 harness: rank-IC, rank acceptance, magnitude, cross-sectional."""
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from src.models.evaluate import walk_forward, walk_forward_xsect


def _panel(n_days=700, symbols=("AAA", "BBB", "CCC", "DDD"), seed=0, signal=0.0):
    """Daily panel spanning ~2yr (>=8 quarters). tb_ret/fwd_ret carry
    `signal`*f1 + noise so a fit_predict returning f1 has controllable rank-IC."""
    rng = np.random.default_rng(seed)
    rows = []
    start = datetime(2023, 1, 2, 15, 0, tzinfo=timezone.utc)
    for i in range(n_days):
        ts = start + timedelta(days=i)
        for s in symbols:
            f1 = rng.normal()
            ret = signal * f1 + rng.normal(0, 0.002)
            rows.append({"timestamp": ts, "symbol": s, "f1": f1, "f2": rng.normal(),
                         "tb_5": int(ret > 0), "tb_ret_5": float(ret),
                         "fwd_ret_5": float(ret), "mfe_5": abs(ret) + 0.5 * abs(f1)})
    return pl.DataFrame(rows).with_columns(pl.col("timestamp").dt.cast_time_unit("us"))


def _predict_f1(Xtr, ytr, Xte):
    return Xte[:, 0]  # score = f1 (first feature column)


def test_ic_detects_signal():
    strong = walk_forward(_panel(signal=0.01), 5, _predict_f1, accept="rank")
    none = walk_forward(_panel(signal=0.0), 5, _predict_f1, accept="rank")
    assert strong["ic"]["ic_mean"] > 0.2      # planted signal shows up
    assert abs(none["ic"]["ic_mean"] or 0) < 0.1  # no signal -> ~0 IC


def test_rank_acceptance_trades_top_q():
    r = walk_forward(_panel(signal=0.005), 5, _predict_f1, accept="rank", q=0.10)
    # rank mode always trades something (unlike the vacuous absolute threshold)
    assert len(r["pnls"]) > 0
    assert r["n_stamps"] > 0


def test_absolute_vs_rank_acceptance_differ():
    p = _panel(signal=0.005)
    # a predictor whose scores never reach 0.6 -> absolute trades nothing
    ab = walk_forward(p, 5, lambda a, b, X: np.full(len(X), 0.3),
                      accept="absolute", threshold=0.6)
    rk = walk_forward(p, 5, _predict_f1, accept="rank", q=0.10)
    assert len(ab["pnls"]) == 0
    assert len(rk["pnls"]) > 0


def test_magnitude_target_regression():
    # target mfe_5 (continuous); predictor returns f1 which correlates with mfe
    r = walk_forward(_panel(signal=0.005), 5, _predict_f1, accept="rank",
                     target_col="mfe_5", ic_col="mfe_5", ret_col="tb_ret_5")
    assert r["ic"]["ic_mean"] is not None


def test_xsect_longshort_produces_pnl():
    r = walk_forward_xsect(_panel(signal=0.004), 5, _predict_f1, q=0.34)
    assert r["n_stamps"] > 0
    assert len(r["pnls"]) > 0
    # net long-short return is finite
    assert all(np.isfinite(r["pnls"]))


def test_xsect_ic_zero_when_no_signal():
    r = walk_forward_xsect(_panel(signal=0.0, seed=3), 5, _predict_f1, q=0.34)
    assert abs(r["ic"]["ic_mean"] or 0) < 0.1
