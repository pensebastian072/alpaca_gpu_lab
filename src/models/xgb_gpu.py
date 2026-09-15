"""B02 — XGBoost-CUDA parameter search on the core-7 1Min panel.

First real GPU workload (no XGBoost prior art in any sibling repo).
6 GB discipline: float32 features, QuantileDMatrix per fold (histogram bins,
not raw rows, live in VRAM), never the whole panel resident. The 3050 is
shared — check nvidia-smi before launching.

Pre-registered grid: depth{4,6,8} x eta{0.05,0.1} x min_child_weight{5,50}
= 12 combos x 3 horizons = 36 trials, charged to the ledger before results.
Model selection happens INSIDE train folds only (last 20% of train rows as
early-stopping eval); the OOS harness in evaluate.py is untouched.

CLI: .venv\\Scripts\\python.exe -m src.models.xgb_gpu [--register-only] [--cpu]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.experiments import registry  # noqa: E402
from src.gpu import set_seed  # noqa: E402
from src.models.evaluate import run_battery  # noqa: E402

FAMILY = "intraday_core7_1min"
SYMBOLS = ["SPY", "QQQ", "IWM", "BND"]

# One module, three pre-registered batteries: the grid is identical, the
# feature set present in the parquets at run time differs (B03 needs an
# --enrich rebuild first, B04 additionally needs scored news + sentiment).
BATTERIES = {
    "B02_xgb_core7": (
        "GPU gradient boosting extracts more of the intraday triple-barrier "
        "signal than the linear/RF baselines (B01) on the same features, "
        "enough to clear the gate after cumulative n_trials deflation."),
    "B03_xgb_cross_asset": (
        "Hourly cross-asset relationship state (correlation/beta/absorption/"
        "divergence/lead-lag) plus daily regime context adds intraday "
        "triple-barrier signal beyond the per-symbol features of B02."),
    "B04_xgb_sentiment": (
        "Trailing-60m news sentiment (FinBERT-scored Alpaca news) adds "
        "intraday triple-barrier signal beyond price+cross-asset features."),
    # B04 as executed answered nothing: every row reported n_trades = 0 because
    # absolute acceptance at p >= 0.60 never fired, so it graded a FILTER that never
    # triggered rather than the signal. This is the same hypothesis under the rank
    # acceptance B06-B08 already use (top decile per fold), which cannot return zero
    # trades. New id rather than a re-run of B04, so that record stays intact.
    "B04b_xgb_sentiment_rank": (
        "Trailing-60m news sentiment (FinBERT-scored Alpaca news) adds intraday "
        "triple-barrier signal beyond price+cross-asset features -- evaluated with "
        "RANK acceptance (top decile per fold) because absolute acceptance at 0.60 "
        "produced zero trades in B04 and therefore tested nothing."),
}
BATTERY = "B02_xgb_core7"  # default; overridden by --battery

GRID = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1],
    "min_child_weight": [5, 50],
}
N_ROUNDS = 500
EARLY_STOP = 30
TRAIN_TIME_LOG: list[dict] = []


def _combos() -> list[dict]:
    keys = list(GRID)
    return [dict(zip(keys, vals)) for vals in itertools.product(*GRID.values())]


def make_fit_predict(params: dict, device: str = "cuda"):
    def fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
        import xgboost as xgb

        # early-stopping eval = chronological last 20% of train (never test data)
        cut = int(len(Xtr) * 0.8)
        full = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "device": device,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": config.SEED,
            **params,
        }
        dtrain = xgb.QuantileDMatrix(Xtr[:cut], label=ytr[:cut])
        deval = xgb.QuantileDMatrix(Xtr[cut:], label=ytr[cut:], ref=dtrain)
        t0 = time.perf_counter()
        booster = xgb.train(
            full, dtrain, num_boost_round=N_ROUNDS,
            evals=[(deval, "eval")], early_stopping_rounds=EARLY_STOP,
            verbose_eval=False,
        )
        TRAIN_TIME_LOG.append({"device": device, "params": params,
                               "rows": int(len(Xtr)),
                               "seconds": round(time.perf_counter() - t0, 2),
                               "best_iter": int(booster.best_iteration)})
        dte = xgb.DMatrix(Xte)
        return booster.predict(dte, iteration_range=(0, booster.best_iteration + 1))

    return fit_predict


def ensure_registered(battery: str) -> None:
    try:
        registry.load_registration(battery)
    except registry.UnregisteredBattery:
        registry.register(
            battery,
            hypothesis=BATTERIES[battery],
            dataset_family=FAMILY,
            horizons=config.HORIZONS_BARS,
            model="xgboost-cuda",
            param_grid=GRID,
            n_trials=len(_combos()) * len(config.HORIZONS_BARS),
        )


def main(register_only: bool = False, device: str = "cuda",
         battery: str = BATTERY, accept: str = "absolute") -> None:
    set_seed(config.SEED)
    ensure_registered(battery)
    if register_only:
        print(f"{battery} registered ({len(_combos()) * len(config.HORIZONS_BARS)} trials).")
        return
    grid = [({"model": "xgb", "device": device, **p}, make_fit_predict(p, device))
            for p in _combos()]
    res = run_battery(battery, grid, SYMBOLS, accept=accept)
    # GPU-vs-CPU evidence + per-combo timings into the journal
    config.LOGS.mkdir(parents=True, exist_ok=True)
    (config.LOGS / f"xgb_train_times_{battery}.json").write_text(
        json.dumps(TRAIN_TIME_LOG, indent=2), encoding="utf-8")
    n_pass = sum(1 for c in res["combos"] if c["all_pass"])
    print(json.dumps({"status": res["status"],
                      "combos_passing_all_horizons": n_pass,
                      "scorecard": res["scorecard_path"]}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--register-only", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="timing comparison only")
    ap.add_argument("--battery", choices=list(BATTERIES), default=BATTERY)
    # Absolute acceptance at p >= 0.60 can legitimately produce ZERO trades, which
    # grades the filter and not the signal (that is what happened to B04). Rank
    # acceptance trades the top decile of each fold and cannot return nothing.
    ap.add_argument("--accept", choices=["absolute", "rank"], default="absolute")
    a = ap.parse_args()
    main(a.register_only, "cpu" if a.cpu else "cuda", a.battery, a.accept)
