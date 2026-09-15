"""Data-spine health report: what's on disk, how complete, where the gaps are.

Scans market_data/raw/bars_<tf>/symbol=*/ partitions and writes
journal/data/coverage.json with, per (timeframe, symbol):
  rows, first/last timestamp, months present, months missing inside the span
  (a month with zero bars between first and last = a hole worth investigating;
  thin tickers legitimately have sparse minutes but not absent months).

Read-only over the parquet store. The daily job reuses this as its
"is the spine healthy" check.

CLI: python -m src.data.coverage [--timeframes 1Min 1Hour 1Day]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402


def _months_between(first: str, last: str) -> list[str]:
    """All YYYY-MM strings from first to last inclusive."""
    fy, fm = int(first[:4]), int(first[5:7])
    ly, lm = int(last[:4]), int(last[5:7])
    out = []
    y, m = fy, fm
    while (y, m) <= (ly, lm):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + (m == 12), (m % 12) + 1)
    return out


def symbol_coverage(tf_root: Path, sym_dir: Path) -> dict:
    files = sorted(sym_dir.glob("year=*/month=*/bars.parquet"))
    if not files:
        return {"rows": 0}
    present = set()
    rows = 0
    first_ts = last_ts = None
    for f in files:
        y = f.parents[1].name.split("=")[1]
        m = f.parents[0].name.split("=")[1]
        present.add(f"{y}-{m}")
        lf = pl.scan_parquet(f).select(
            pl.len().alias("n"),
            pl.col("timestamp").min().alias("lo"),
            pl.col("timestamp").max().alias("hi"),
        ).collect()
        rows += int(lf["n"][0])
        lo, hi = lf["lo"][0], lf["hi"][0]
        if lo is not None:
            first_ts = lo if first_ts is None else min(first_ts, lo)
            last_ts = hi if last_ts is None else max(last_ts, hi)
    if first_ts is None:
        return {"rows": 0}
    span = _months_between(str(first_ts)[:7], str(last_ts)[:7])
    missing = sorted(set(span) - present)
    return {
        "rows": rows,
        "first": str(first_ts),
        "last": str(last_ts),
        "months_present": len(present),
        "months_missing_in_span": missing,
    }


def build_report(timeframes: list[str]) -> dict:
    report: dict = {
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feed": config.DEFAULT_FEED,
        "timeframes": {},
    }
    expected = set(config.UNIVERSE_22)  # partition names (BTCUSD, not BTC/USD)
    for tf in timeframes:
        root = config.RAW / f"bars_{tf}"
        per_symbol = {}
        if root.exists():
            for sym_dir in sorted(root.glob("symbol=*")):
                sym = sym_dir.name.split("=")[1]
                per_symbol[sym] = symbol_coverage(root, sym_dir)
        have = {s for s, c in per_symbol.items() if c.get("rows", 0) > 0}
        report["timeframes"][tf] = {
            "symbols": per_symbol,
            "n_symbols": len(have),
            "universe22_missing": sorted(expected - have),
        }
    return report


def main(timeframes: list[str]) -> dict:
    report = build_report(timeframes)
    config.DATA_REPORTS.mkdir(parents=True, exist_ok=True)
    out = config.DATA_REPORTS / "coverage.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for tf, block in report["timeframes"].items():
        missing = block["universe22_missing"]
        print(f"{tf}: {block['n_symbols']} symbols"
              + (f", universe22 missing: {missing}" if missing else " (universe22 complete)"))
    print(f"wrote {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframes", nargs="+", default=["1Min", "1Hour", "1Day"])
    main(ap.parse_args().timeframes)
