"""B08 — cross-sectional long-short across the core-7 names.

Per non-overlapping entry stamp, rank the tradable names by predicted up-prob,
long the top third / short the bottom third, one net PnL per stamp (gross
fwd_ret per leg, per-leg round-trip cost). This is the relative bet where the
xa_* cross-asset relationship features are supposed to pay off — "which name
moves most vs the others," not "does SPY go up."

NOTE: core-7 is 7 tradable names (SPY/QQQ/IWM/BND + IEF/TLT/VIXY) — thin for a
cross-section. This proves the mechanic; its natural home is the full-22 1Hour
panel (deferred, P10-scale). Rank-IC across the name panel is the headline.

CLI: .venv\\Scripts\\python.exe -m src.models.xsect [--register-only]
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
from src.models.evaluate import run_battery  # noqa: E402
from src.models.xgb_gpu import make_fit_predict  # noqa: E402

# all core-7 tradable as a cross-section (targets + proxies)
SYMBOLS = ["SPY", "QQQ", "IWM", "BND", "IEF", "TLT", "VIXY"]
XGB_PARAMS = {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 50}


def _ids(tf: str) -> tuple[str, str]:
    if tf == "1Min":
        return "B08_xsect", "intraday_core7_1min"
    return f"B08_xsect_{tf}", f"intraday_core7_{tf.lower()}"


def ensure_registered(tf: str = "1Min") -> None:
    battery, family = _ids(tf)
    try:
        registry.load_registration(battery)
    except registry.UnregisteredBattery:
        registry.register(
            battery,
            hypothesis=(
                f"On {tf} bars, chart + cross-asset (xa_*) features rank the "
                "core-7 names cross-sectionally: a per-stamp long-top/short-bottom "
                "book has positive rank-IC and net-positive after-cost "
                "expectancy. Relative bets where the relationship features earn "
                "their keep."
            ),
            dataset_family=family,
            horizons=config.HORIZONS_BARS,
            model="xgboost-xsect",
            param_grid={"xgb": XGB_PARAMS, "long_short_q": 0.34},
            n_trials=len(config.HORIZONS_BARS),
        )


def main(register_only: bool = False, device: str = "cuda", timeframe: str = "1Min") -> None:
    set_seed(config.SEED)
    battery, _ = _ids(timeframe)
    ensure_registered(timeframe)
    if register_only:
        print(f"{battery} registered.")
        return
    grid = [({"model": "xgb", **XGB_PARAMS}, make_fit_predict(XGB_PARAMS, device))]
    res = run_battery(battery, grid, SYMBOLS, timeframe=timeframe, mode="xsect", q=0.34)
    ics = {h: v["ic"]["ic_mean"] for h, v in res["combos"][0]["horizons"].items()}
    print(json.dumps({"status": res["status"], "timeframe": timeframe,
                      "xsect_ic_by_horizon": ics, "scorecard": res["scorecard_path"]}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--register-only", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--timeframe", default="1Min",
                    choices=["1Min", "5Min", "15Min", "1Hour", "4Hour"])
    a = ap.parse_args()
    main(a.register_only, "cpu" if a.cpu else "cuda", a.timeframe)
