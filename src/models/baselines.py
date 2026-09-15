"""B01 — logistic regression + RandomForest baselines on the core-7 1Min panel.

The reference bar every GPU model must beat. RF kwargs start from
macro_gpu_lab config.RF_KWARGS (n_estimators=300, max_depth=6, balanced).
Expected first outcome: honest FAIL -> SHADOW. Report it.

CLI: .venv\\Scripts\\python.exe -m src.models.baselines [--register-only]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.experiments import registry  # noqa: E402
from src.gpu import set_seed  # noqa: E402
from src.models.evaluate import run_battery  # noqa: E402

BATTERY = "B01_baselines_core7"
FAMILY = "intraday_core7_1min"
SYMBOLS = ["SPY", "QQQ", "IWM", "BND"]  # labeled targets (refs are context)

RF_KWARGS = dict(n_estimators=300, max_depth=6, class_weight="balanced",
                 random_state=config.SEED, n_jobs=-1)
LOGREG_KWARGS = dict(C=1.0, max_iter=2000, class_weight="balanced")


def _logreg(Xtr, ytr, Xte) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(**LOGREG_KWARGS).fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1]


def _rf(Xtr, ytr, Xte) -> np.ndarray:
    from sklearn.ensemble import RandomForestClassifier

    m = RandomForestClassifier(**RF_KWARGS).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def ensure_registered() -> None:
    try:
        registry.load_registration(BATTERY)
    except registry.UnregisteredBattery:
        registry.register(
            BATTERY,
            hypothesis=(
                "Intraday (5/15/30-bar) triple-barrier outcomes on liquid ETFs "
                "are predictable from backward-looking price/volume/session/"
                "cross-market features well enough that accepting only "
                "p>=0.60 setups yields positive net expectancy after costs. "
                "Daily horizons on this universe are exhausted (macro_gpu_lab/"
                "qlib_lab all FAIL); intraday is genuinely new ground."
            ),
            dataset_family=FAMILY,
            horizons=config.HORIZONS_BARS,
            model="logreg+rf",
            param_grid={"logreg": LOGREG_KWARGS, "rf": RF_KWARGS},
            n_trials=2 * len(config.HORIZONS_BARS),  # 2 models x 3 horizons
        )


def main(register_only: bool = False) -> None:
    set_seed(config.SEED)
    ensure_registered()
    if register_only:
        print(f"{BATTERY} registered.")
        return
    grid = [({"model": "logreg", **LOGREG_KWARGS}, _logreg),
            ({"model": "rf", **RF_KWARGS}, _rf)]
    res = run_battery(BATTERY, grid, SYMBOLS)
    print(json.dumps(
        {"status": res["status"],
         "scorecard": res["scorecard_path"],
         "gates": {c["params"]["model"]: {h: v["passes"] for h, v in c["horizons"].items()}
                   for c in res["combos"]}},
        indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--register-only", action="store_true")
    main(ap.parse_args().register_only)
