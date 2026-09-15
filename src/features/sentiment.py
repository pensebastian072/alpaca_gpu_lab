"""News-sentiment features from precomputed scores (fail-safe to neutral).

Reads market_data/scores/news_scores.parquet (written offline by
src/news/score.py — never an LLM on any live path) and produces per-symbol
minute-joinable columns:
  ns_score_60m   : exponentially decay-weighted mean sentiment, trailing 60 min
  ns_count_60m   : article count, trailing 60 min
  ns_accel       : count(last 60m) - count(prior 60m)
  ns_high_impact : any high-impact article in trailing 60 min
  ns_mkt_score_60m / ns_mkt_count_60m : market-wide (all symbols) aggregates

LEAK GUARD: an article enters the trailing window only from its created_at
timestamp forward (as-of backward join on event time).
Missing scores file, or a symbol with no news -> all-neutral columns
(score 0, count 0) — the fail-safe default (CLAUDE.md rule 6).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402

SCORES_PATH = config.SCORES / "news_scores.parquet"
WINDOW = "60min"
DECAY_HALFLIFE_MIN = 30.0

NEUTRAL = {"ns_score_60m": 0.0, "ns_count_60m": 0.0, "ns_accel": 0.0,
           "ns_high_impact": 0.0, "ns_mkt_score_60m": 0.0, "ns_mkt_count_60m": 0.0}


def _load_scores() -> pd.DataFrame | None:
    if not SCORES_PATH.exists():
        return None
    df = pl.read_parquet(SCORES_PATH).to_pandas()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, format="mixed")
    return df.sort_values("created_at")


def _rolling_features(events: pd.DataFrame, stamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Trailing-window aggregates of event rows evaluated at each stamp."""
    out = pd.DataFrame(index=stamps, columns=list(NEUTRAL), dtype=float)
    out[:] = 0.0
    if events.empty:
        return out
    ev_ts = events["created_at"].to_numpy()
    ev_score = events["score"].to_numpy(dtype=float)
    ev_impact = events["high_impact"].to_numpy(dtype=bool)
    win = pd.Timedelta(WINDOW)
    lo = np.searchsorted(ev_ts, (stamps - win).to_numpy(), side="left")
    lo2 = np.searchsorted(ev_ts, (stamps - 2 * win).to_numpy(), side="left")
    hi = np.searchsorted(ev_ts, stamps.to_numpy(), side="right")
    for i, (a, a2, b) in enumerate(zip(lo, lo2, hi)):
        cnt = b - a
        out.iat[i, 1] = cnt                       # ns_count_60m
        out.iat[i, 2] = cnt - (a - a2)            # ns_accel
        if cnt:
            ages_min = (stamps[i] - pd.DatetimeIndex(ev_ts[a:b])).total_seconds() / 60.0
            w = np.exp(-np.log(2) * ages_min / DECAY_HALFLIFE_MIN)
            out.iat[i, 0] = float(np.average(ev_score[a:b], weights=w))  # ns_score_60m
            out.iat[i, 3] = float(ev_impact[a:b].any())                  # ns_high_impact
    return out


def build_sentiment_context(symbol: str, stamps: pl.Series) -> pl.DataFrame:
    """(timestamp + ns_*) frame aligned to the given minute stamps."""
    idx = pd.DatetimeIndex(stamps.to_pandas()).tz_convert("UTC")
    scores = _load_scores()
    if scores is None:
        ctx = pd.DataFrame(NEUTRAL, index=idx)
    else:
        mine = scores[scores["symbols"].fillna("").str.contains(symbol, regex=False)]
        own = _rolling_features(mine, idx)
        mkt = _rolling_features(scores, idx)
        own["ns_mkt_score_60m"] = mkt["ns_score_60m"]
        own["ns_mkt_count_60m"] = mkt["ns_count_60m"]
        ctx = own
    ctx.index.name = "timestamp"
    res = pl.from_pandas(ctx.reset_index())
    return res.with_columns(pl.col("timestamp").dt.cast_time_unit("us")).sort("timestamp")


def add_sentiment(df: pl.DataFrame, symbol: str) -> pl.DataFrame:
    """Attach ns_* columns to a minute feature frame (exact stamp alignment)."""
    ctx = build_sentiment_context(symbol, df["timestamp"])
    return df.join(ctx, on="timestamp", how="left").with_columns(
        [pl.col(c).fill_null(0.0) for c in NEUTRAL])
