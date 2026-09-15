"""Pooled model dataset over feature parquets + the 2026 holdout lock.

- load_panel(): stacks per-symbol feature parquets into one (symbol, timestamp)
  frame, float32 feature columns (GPU discipline), CLIPPED at HOLDOUT_START by
  default. Reading holdout rows requires BOTH unlock_holdout=True AND env
  ALPACA_GPU_HOLDOUT_UNLOCK=yes — flipped exactly once at the end of the
  research program (CLAUDE.md rule 5).
- quarterly_walk_forward(): expanding train / one-quarter test splits over the
  panel's unique timestamps, with a `gap` of label-horizon bars between train
  end and test start so overlapping labels cannot leak (same ethos as the
  imported walk_forward_splits; quarters here map to the user's pre-registered
  validation scheme).
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402

# Columns that are identifiers/labels/raw prices — never model features.
NON_FEATURES = {"timestamp", "ts_local", "symbol", "open", "high", "low",
                "close", "volume", "trade_count", "vwap", "day"}
LABEL_PREFIXES = ("tb_", "fwd_ret_", "mfe_", "mae_", "net_edge_", "label")


def feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in NON_FEATURES and not c.startswith(LABEL_PREFIXES)]


def load_panel(
    symbols: list[str],
    timeframe: str = "1Min",
    unlock_holdout: bool = False,
) -> pl.DataFrame:
    """Stack per-symbol feature parquets; clip the 2026 holdout by default."""
    frames = []
    for sym in symbols:
        p = config.FEATURES / f"{sym}_{timeframe}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"missing feature parquet: {p} — run src.features.build")
        frames.append(pl.read_parquet(p).with_columns(pl.lit(sym).alias("symbol")))
    panel = pl.concat(frames, how="diagonal").sort(["timestamp", "symbol"])

    if unlock_holdout:
        if not config.HOLDOUT_UNLOCK:
            raise PermissionError(
                "holdout is locked: set ALPACA_GPU_HOLDOUT_UNLOCK=yes to read "
                f">= {config.HOLDOUT_START}. This happens exactly once, at the end."
            )
    else:
        panel = panel.filter(
            pl.col("timestamp") < pl.lit(config.HOLDOUT_START).str.to_datetime(time_zone="UTC")
        )

    # float32 features for the GPU path
    casts = [pl.col(c).cast(pl.Float32) for c in feature_columns(panel)
             if panel[c].dtype in (pl.Float64,)]
    return panel.with_columns(casts)


def quarterly_walk_forward(
    panel: pl.DataFrame,
    horizon_bars: int,
    min_train_quarters: int = 2,
):
    """Yield (train_df, test_df, quarter_label) expanding splits.

    Train = everything before the test quarter minus a `horizon_bars` gap
    (measured in panel timestamps) so labels computed at train end cannot
    overlap the test window.
    """
    stamps = panel["timestamp"].unique().sort()
    quarters = (
        panel.select(
            pl.col("timestamp").dt.year().alias("y"),
            ((pl.col("timestamp").dt.month() - 1) // 3 + 1).alias("q"),
        )
        .unique()
        .sort(["y", "q"])
        .rows()
    )
    for y, q in quarters[min_train_quarters:]:
        q_start_month = 3 * (q - 1) + 1
        q_start = pl.datetime(y, q_start_month, 1, time_zone="UTC")
        nxt_y, nxt_m = (y + 1, 1) if q == 4 else (y, q_start_month + 3)
        q_end = pl.datetime(nxt_y, nxt_m, 1, time_zone="UTC")

        test = panel.filter((pl.col("timestamp") >= q_start) & (pl.col("timestamp") < q_end))
        if test.is_empty():
            continue
        test_start = test["timestamp"].min()
        # gap: drop the last `horizon_bars` panel timestamps before the test start
        pre = stamps.filter(stamps < test_start)
        if len(pre) <= horizon_bars:
            continue
        train_cutoff = pre[len(pre) - horizon_bars]
        train = panel.filter(pl.col("timestamp") < train_cutoff)
        if train.is_empty():
            continue
        yield train, test, f"{y}Q{q}"
