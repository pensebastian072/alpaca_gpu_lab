"""B07 — magnitude: predict how FAR price runs, not which way.

XGBoost regressor on chart-only features predicting mfe_{h} (max favorable
excursion in ATR units). Volatility/excursion is the forecastable quantity
where sign mostly isn't — this is the higher-prior target.

Graded two ways:
  - rank-IC (predicted vs realized excursion) — the honest forecasting metric,
    the headline result;
  - a long-harvest PnL on the top-decile predicted-excursion bars, run through
    the gate — tests whether knowing WHERE big moves happen helps a harvester.

CLI: .venv\\Scripts\\python.exe -m src.models.magnitude [--register-only]
"""
from __future__ import annotations

import argparse
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

SYMBOLS = ["SPY", "QQQ", "IWM", "BND"]
GRID = [{"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 50},
        {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 50}]
N_ROUNDS = 400
TRAIN_TIMES: list[dict] = []


def make_regressor(params: dict, device: str = "cuda"):
    def fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
        import xgboost as xgb

        # drop rows with NaN/inf target (mfe undefined on truncated windows)
        ok = np.isfinite(ytr)
        Xtr, ytr = Xtr[ok], ytr[ok].astype(np.float32)
        full = {"objective": "reg:squarederror", "eval_metric": "rmse",
                "tree_method": "hist", "device": device,
                "subsample": 0.8, "colsample_bytree": 0.8, "seed": config.SEED,
                **params}
        cut = int(len(Xtr) * 0.8)
        dtrain = xgb.QuantileDMatrix(Xtr[:cut], label=ytr[:cut])
        deval = xgb.QuantileDMatrix(Xtr[cut:], label=ytr[cut:], ref=dtrain)
        t0 = time.perf_counter()
        booster = xgb.train(full, dtrain, num_boost_round=N_ROUNDS,
                            evals=[(deval, "eval")], early_stopping_rounds=30,
                            verbose_eval=False)
        TRAIN_TIMES.append({"device": device, "params": params,
                            "seconds": round(time.perf_counter() - t0, 2)})
        return booster.predict(xgb.DMatrix(Xte),
                               iteration_range=(0, booster.best_iteration + 1))
    return fit_predict


def _ids(tf: str) -> tuple[str, str]:
    if tf == "1Min":
        return "B07_magnitude", "intraday_core7_1min"
    return f"B07_magnitude_{tf}", f"intraday_core7_{tf.lower()}"


def ensure_registered(tf: str = "1Min") -> None:
    battery, family = _ids(tf)
    try:
        registry.load_registration(battery)
    except registry.UnregisteredBattery:
        registry.register(
            battery,
            hypothesis=(
                f"On {tf} bars, chart features predict the SIZE of the coming "
                "move (mfe in ATR units) with positive rank-IC, even where "
                "direction is unpredictable; concentrating a long harvester on "
                "the top-decile predicted-excursion bars improves after-cost "
                "expectancy."
            ),
            dataset_family=family,
            horizons=config.HORIZONS_BARS,
            model="xgboost-regressor",
            param_grid={"grid": GRID, "target": "mfe", "accept": "rank"},
            n_trials=len(GRID) * len(config.HORIZONS_BARS),
        )


def main(register_only: bool = False, device: str = "cuda", timeframe: str = "1Min") -> None:
    set_seed(config.SEED)
    battery, _ = _ids(timeframe)
    ensure_registered(timeframe)
    if register_only:
        print(f"{battery} registered.")
        return
    grid = [({"model": "xgb-reg", **p}, make_regressor(p, device)) for p in GRID]
    res = run_battery(battery, grid, SYMBOLS, timeframe=timeframe,
                      mode="magnitude", accept="rank", q=0.10)
    ics = [{"params": c["params"],
            "ic": {h: v["ic"]["ic_mean"] for h, v in c["horizons"].items()}}
           for c in res["combos"]]
    print(json.dumps({"status": res["status"], "timeframe": timeframe,
                      "magnitude_ic": ics, "scorecard": res["scorecard_path"]}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--register-only", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--timeframe", default="1Min",
                    choices=["1Min", "5Min", "15Min", "1Hour", "4Hour"])
    a = ap.parse_args()
    main(a.register_only, "cpu" if a.cpu else "cuda", a.timeframe)
