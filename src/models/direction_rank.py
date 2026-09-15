"""B06 — direction with rank acceptance + rank-IC diagnostic (the fork battery).

Same chart-only features + model families as B01/B02, but two changes that
answer the question the absolute-threshold batteries could not:
  1. accept="rank" — trade the top-decile of each fold's scores, so the verdict
     is about ranking power, not whether the model ever hits p>=0.60.
  2. rank-IC logged per model/horizon — the cheap "is there ANY signal" number.

Run FIRST. If IC ~ 0 across logreg/RF/xgb and all horizons, direction is not
predictable from these chart features and B07/B08 on the same panel are long
shots — report and stop shallow. If IC > 0, there is something to monetize.

CLI: .venv\\Scripts\\python.exe -m src.models.direction_rank [--register-only]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.experiments import registry  # noqa: E402
from src.gpu import set_seed  # noqa: E402
from src.models.baselines import _logreg  # noqa: E402
from src.models.evaluate import run_battery  # noqa: E402
from src.models.xgb_gpu import make_fit_predict  # noqa: E402

SYMBOLS = ["SPY", "QQQ", "IWM", "BND"]
XGB_PARAMS = {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 50}


def _ids(tf: str) -> tuple[str, str]:
    """(battery_id, dataset_family) per timeframe. 1Min keeps the original
    unsuffixed id so it stays comparable with the first B06 run."""
    if tf == "1Min":
        return "B06_direction_rank", "intraday_core7_1min"
    return f"B06_direction_rank_{tf}", f"intraday_core7_{tf.lower()}"


def ensure_registered(tf: str = "1Min") -> None:
    battery, family = _ids(tf)
    try:
        registry.load_registration(battery)
    except registry.UnregisteredBattery:
        registry.register(
            battery,
            hypothesis=(
                f"Chart-only features RANK triple-barrier direction on {tf} bars "
                "even if they never reach 60% confidence: trading the top decile "
                "of each fold's predicted p_up yields positive rank-IC and "
                "net-positive after-cost expectancy. Tests whether the FAILs "
                "were a vacuous-threshold artifact or a real absence of signal."
            ),
            dataset_family=family,
            horizons=config.HORIZONS_BARS,
            model="logreg+xgb",
            param_grid={"accept": "rank", "q": 0.10, "xgb": XGB_PARAMS},
            n_trials=2 * len(config.HORIZONS_BARS),  # RF dropped (no GPU, redundant IC)
        )


def main(register_only: bool = False, device: str = "cuda", timeframe: str = "1Min") -> None:
    set_seed(config.SEED)
    battery, _ = _ids(timeframe)
    ensure_registered(timeframe)
    if register_only:
        print(f"{battery} registered.")
        return
    grid = [
        ({"model": "logreg"}, _logreg),
        ({"model": "xgb", **XGB_PARAMS}, make_fit_predict(XGB_PARAMS, device)),
    ]
    res = run_battery(battery, grid, SYMBOLS, timeframe=timeframe,
                      mode="direction", accept="rank", q=0.10)
    ics = {c["params"]["model"]: {h: v["ic"]["ic_mean"] for h, v in c["horizons"].items()}
           for c in res["combos"]}
    print(json.dumps({"status": res["status"], "timeframe": timeframe,
                      "ic_mean_by_model": ics, "scorecard": res["scorecard_path"]}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--register-only", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--timeframe", default="1Min",
                    choices=["1Min", "5Min", "15Min", "1Hour", "4Hour"])
    a = ap.parse_args()
    main(a.register_only, "cpu" if a.cpu else "cuda", a.timeframe)
