"""B09 (direction) + B10 (magnitude) — grade Kronos forecasts through the gate.

Kronos runs in its own venv and writes market_data/kronos/forecast_<tf>.parquet
(see Kronos/research/alpaca_gpu_forecast.py). Here we join that onto the feature
panel, derive per-horizon signals, and grade them with the SAME rank-IC +
acceptance + PBO/DSR gate as every other battery — no re-porting, Kronos is
just a precomputed score source (score_col path in evaluate.py).

Signals per horizon h:
  kron_dir_{h} = pred_close_{h}/base_close - 1      (B09: direction/conviction)
  kron_mag_{h} = (pred_high_{h}-pred_low_{h})/base  (B10: predicted range —
                 applies the Phase-2 lesson that MAGNITUDE is the forecastable
                 quantity; IC graded vs realized mfe_{h})

B09 runs both single-name direction and, at hourly TFs, cross-sectional (where
Phase-2 B08 found PF>1). B10 is the magnitude play.

CLI:
  .venv\\Scripts\\python.exe -m src.models.kronos_signals --battery B09 --timeframe 4Hour
  .venv\\Scripts\\python.exe -m src.models.kronos_signals --battery B10 --timeframe 1Hour
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.experiments import registry  # noqa: E402
from src.models import dataset  # noqa: E402
from src.models.evaluate import run_battery  # noqa: E402

TARGETS = ["SPY", "QQQ", "IWM", "BND"]
CORE7 = ["SPY", "QQQ", "IWM", "BND", "IEF", "TLT", "VIXY"]
HORIZONS = config.HORIZONS_BARS


def _noop(Xtr, ytr, Xte):  # never called in score_col mode
    raise RuntimeError("score_col mode should not train")


def load_kronos_panel(symbols: list[str], timeframe: str, suffix: str = "") -> pl.DataFrame:
    """Feature panel joined with Kronos forecast-derived signal columns.

    suffix="_ft" reads the fine-tuned-model forecast (B11) instead of base.
    """
    fc = config.DATA / "kronos" / f"forecast_{timeframe}{suffix}.parquet"
    if not fc.exists():
        raise FileNotFoundError(
            f"no Kronos forecast at {fc} — run Kronos/research/alpaca_gpu_forecast.py "
            f"--timeframe {timeframe} first")
    kron = pl.read_parquet(fc)
    # normalize timestamp to UTC-aware microsecond to match the feature panel
    ts = pl.col("timestamp").cast(pl.Datetime("us"))
    if kron["timestamp"].dtype == pl.Datetime(time_zone=None):
        ts = ts.dt.replace_time_zone("UTC")
    else:
        ts = ts.dt.convert_time_zone("UTC")
    kron = kron.with_columns(ts.alias("timestamp"))
    exprs = []
    for h in HORIZONS:
        exprs.append((pl.col(f"pred_close_{h}") / pl.col("base_close") - 1.0).alias(f"kron_dir_{h}"))
        exprs.append(((pl.col(f"pred_high_{h}") - pl.col(f"pred_low_{h}")) / pl.col("base_close"))
                     .alias(f"kron_mag_{h}"))
    kron = kron.with_columns(exprs).select(
        ["timestamp", "symbol"] + [f"kron_dir_{h}" for h in HORIZONS]
        + [f"kron_mag_{h}" for h in HORIZONS])

    panel = dataset.load_panel(symbols, timeframe)
    return panel.join(kron, on=["timestamp", "symbol"], how="inner")


def register(battery: str, family: str, hypothesis: str, model: str) -> None:
    try:
        registry.load_registration(battery)
    except registry.UnregisteredBattery:
        registry.register(battery, hypothesis=hypothesis, dataset_family=family,
                          horizons=HORIZONS, model=model,
                          param_grid={"source": "kronos-small", "zero_shot": True},
                          n_trials=len(HORIZONS))


def _tag(symbols: list[str], suffix: str) -> str:
    """Battery-id tag for a non-default universe, so each cohort gets its own
    registration and scorecard instead of silently overwriting CORE7's."""
    if suffix:
        return suffix
    if len(symbols) == 1:
        return f"_{symbols[0].lower()}"
    return f"_{len(symbols)}sym"


def run_b09(timeframe: str, symbols: list[str] | None = None, suffix: str = "") -> dict:
    """Kronos direction — single-name (rank accept) + hourly cross-sectional."""
    syms = symbols or CORE7
    tag = "" if (syms == CORE7 and not suffix) else _tag(syms, suffix)
    battery = f"B09_kronos_dir_{timeframe}{tag}"
    family = f"kronos_{'core7' if syms == CORE7 else tag.strip('_')}_{timeframe.lower()}"
    register(battery, family,
             f"Kronos-small zero-shot forecast direction (pred_close vs base) on "
             f"{timeframe} bars over {','.join(syms)} ranks triple-barrier outcomes "
             f"with positive rank-IC and clears the gate — unlike the prior copper "
             f"zero-shot FAIL.",
             "kronos-direction")
    panel = load_kronos_panel(syms, timeframe, suffix)
    grid = [({"model": "kronos"}, _noop)]
    # cross-sectional needs a cross-section: hourly TFs with >1 name (B08 found PF>1)
    mode = "xsect" if (timeframe in ("1Hour", "4Hour") and len(syms) > 1) else "direction"
    res = run_battery(battery, grid, syms, timeframe=timeframe, mode=mode,
                      accept="rank", q=0.34 if mode == "xsect" else 0.10,
                      score_col="kron_dir_{h}", panel=panel)
    return res


def run_b10(timeframe: str, symbols: list[str] | None = None, suffix: str = "") -> dict:
    """Kronos magnitude — predicted range vs realized excursion (Phase-2 lesson)."""
    syms = symbols or TARGETS
    tag = "" if (syms == TARGETS and not suffix) else _tag(syms, suffix)
    battery = f"B10_kronos_mag_{timeframe}{tag}"
    family = f"kronos_{'core7' if syms == TARGETS else tag.strip('_')}_{timeframe.lower()}"
    register(battery, family,
             f"Kronos-small predicted high-low range forecasts realized excursion "
             f"(mfe) on {timeframe} bars over {','.join(syms)} with positive rank-IC — "
             f"applying the Phase-2 finding that magnitude is the forecastable quantity.",
             "kronos-magnitude")
    panel = load_kronos_panel(syms, timeframe, suffix)
    grid = [({"model": "kronos-mag"}, _noop)]
    res = run_battery(battery, grid, syms, timeframe=timeframe, mode="magnitude",
                      accept="rank", q=0.10, score_col="kron_mag_{h}", panel=panel)
    return res


def run_b11(timeframe: str = "1Hour") -> dict:
    """B11 — FINE-TUNED Kronos (SPY 60m/730d) vs base B10. Magnitude + direction
    on SPY at the fine-tune's native timeframe. Reads the _ft forecast."""
    battery = f"B11_kronos_finetuned_{timeframe}"
    family = f"kronos_ft_spy_{timeframe.lower()}"
    register(battery, family,
             f"A Kronos model fine-tuned on SPY {timeframe} bars forecasts SPY "
             f"excursion/direction with higher rank-IC than the zero-shot base "
             f"(B10) — testing whether fine-tuning lets a foundation model read "
             f"these charts where zero-shot could not.",
             "kronos-finetuned")
    panel = load_kronos_panel(["SPY"], timeframe, suffix="_ft")
    grid = [({"model": "kronos-ft"}, _noop)]
    res = run_battery(battery, grid, ["SPY"], timeframe=timeframe, mode="magnitude",
                      accept="rank", q=0.10, score_col="kron_mag_{h}", panel=panel)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", required=True, choices=["B09", "B10", "B11"])
    ap.add_argument("--timeframe", default="4Hour", choices=["1Hour", "4Hour", "15Min", "5Min", "1Min"])
    ap.add_argument("--symbols", nargs="+", default=None,
                    help="universe to grade (default CORE7/TARGETS); a non-default set "
                         "gets its own battery id so cohorts never overwrite each other")
    ap.add_argument("--suffix", default="", help="forecast-file suffix, e.g. _btc / _all23")
    a = ap.parse_args()
    if a.battery == "B11":
        res = run_b11(a.timeframe)
    else:
        res = ({"B09": run_b09, "B10": run_b10}[a.battery])(a.timeframe, a.symbols, a.suffix)
    ics = {}
    for c in res["combos"]:
        for h, v in c["horizons"].items():
            ic = v.get("ic") or {}
            ics[h] = {"ic": ic.get("ic_mean"), "t": ic.get("ic_t"),
                      "n": v.get("n_trades"), "PF": round(v.get("profit_factor") or 0, 2),
                      "pass": v.get("passes")}
    print(json.dumps({"battery": res["battery_id"], "status": res["status"],
                      "by_horizon": ics, "scorecard": res["scorecard_path"]}, indent=2))


if __name__ == "__main__":
    main()
