"""P2b -- GPU XGBoost-CUDA meta-model on the v5 ledger (deep_v5_meta_v1).

Predicts which v5 entries win (meta-label) from point-in-time features, expanding
walk-forward, then trades only the top-decile predicted-win entries and runs that filtered
book through the canonical gate. Reads the feature parquet exported by options_desk.

Honest: labels are BS-MODELED v5 PnL (no OPRA yet). Small n (1044, 81% base win), so expect
the gate to bite; GPU is used as specified but is not the bottleneck at this scale. New
hypothesis, not a v5 retune.

    .venv\\Scripts\\python.exe -m src.research.meta_model
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config

FEATURES = Path(r"C:\Users\<your-user>\options_desk\journal\scorecards\deep_v5_meta_features.parquet")
OUT = config.REPO / "journal" / "scorecards" / "deep_v5_meta_v1.json"
N_FOLDS = 5
TOP_DECILE = 0.10
FEATURE_COLS = ["pop", "entry_iv", "credit_usd", "max_risk_usd", "return_on_risk", "iv_rv",
                "trend_dist_50", "trend_dist_200", "rv20", "magnitude_pct", "vrp_pct",
                "term_ratio", "p_eruption_5d", "p_eruption_21d", "entry_month", "entry_dow",
                "is_short_put", "is_short_call", "is_iron_condor"]


def _gate():
    root = str(config.MACRO_GPU_LAB_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    from macro_gpu_lab.validate import evaluate_gate, walk_forward_splits
    return evaluate_gate, walk_forward_splits


def _cluster_by_date(dates: list[str], pnls: list[float]) -> list[float]:
    by: dict[str, list[float]] = {}
    for d, p in zip(dates, pnls):
        by.setdefault(d, []).append(p)
    return [float(np.mean(v)) for _, v in sorted(by.items())]


def _rank_ic(prob: np.ndarray, y: np.ndarray) -> float:
    if len(y) < 5 or np.unique(y).size < 2:
        return float("nan")
    pr = pd.Series(prob).rank()
    yr = pd.Series(y).rank()
    return float(pr.corr(yr))


def run() -> dict:
    evaluate_gate, walk_forward_splits = _gate()
    import xgboost as xgb

    df = pd.read_parquet(FEATURES).sort_values("entry_date").reset_index(drop=True)
    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["win"].to_numpy(dtype=int)
    pnl = df["pnl_usd"].to_numpy(dtype=float)
    dates = df["entry_date"].astype(str).to_numpy()
    n = len(df)

    params = {"objective": "binary:logistic", "eval_metric": "logloss",
              "tree_method": "hist", "device": "cuda", "max_depth": 4,
              "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
              "min_child_weight": 20, "seed": 42}

    oos_prob = np.full(n, np.nan)
    fold_ics, folds = [], []
    for tr, te in walk_forward_splits(n, N_FOLDS, 5):
        if np.unique(y[tr]).size < 2:
            continue
        dtr = xgb.QuantileDMatrix(X[tr], label=y[tr].astype(np.float32))
        booster = xgb.train(params, dtr, num_boost_round=200,
                            evals=[(dtr, "train")], verbose_eval=False)
        p = booster.predict(xgb.DMatrix(X[te]))
        oos_prob[te] = p
        fold_ics.append(_rank_ic(p, y[te]))
        folds.append({"train": int(tr.size), "test": int(te.size),
                      "test_start": dates[te[0]], "test_end": dates[te[-1]]})

    scored = ~np.isnan(oos_prob)
    ns = int(scored.sum())
    # top-decile predicted-win filter, per the pooled OOS scores
    thr = np.nanquantile(oos_prob[scored], 1 - TOP_DECILE)
    keep = scored & (oos_prob >= thr)

    def _stats(mask):
        p = pnl[mask]
        w = int((p > 0).sum())
        return {"n": int(mask.sum()), "win_rate": round(w / max(mask.sum(), 1), 4),
                "mean_pnl_usd": round(float(p.mean()), 2) if mask.sum() else None,
                "total_pnl_usd": round(float(p.sum()), 2)}

    base = _stats(scored)
    filt = _stats(keep)
    base_cluster = _cluster_by_date(list(dates[scored]), list(pnl[scored]))
    filt_cluster = _cluster_by_date(list(dates[keep]), list(pnl[keep]))
    gate_base = evaluate_gate(np.asarray(base_cluster), n_trials=1)
    gate_filt = evaluate_gate(np.asarray(filt_cluster), n_trials=1)

    mean_ic = float(np.nanmean(fold_ics)) if fold_ics else float("nan")
    return {
        "experiment_id": "deep_v5_meta_v1", "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "SHADOW; BS-modeled labels (upper bound); GPU XGBoost-CUDA",
        "n_scored": ns, "walk_forward_folds": folds,
        "oos_rank_ic_mean": round(mean_ic, 4), "oos_rank_ic_per_fold": [round(x, 4) for x in fold_ics],
        "top_decile_threshold": round(float(thr), 4),
        "unfiltered_v5": {**base, "cluster_gate": gate_base},
        "filtered_topdecile": {**filt, "cluster_gate": gate_filt},
        "predictions": {
            "topdecile_win_gt_base": filt["win_rate"] > base["win_rate"],
            "filtered_mean_gt_unfiltered": (filt["mean_pnl_usd"] or 0) > (base["mean_pnl_usd"] or 0),
            "rank_ic_positive": mean_ic > 0,
        },
        "note": ("Meta-filter on v5; not a retune. Labels modeled (no OPRA). Confirm on real "
                 "options once OPRA is signed. Gate on cluster-adjusted PnL, honest."),
    }


def main() -> None:
    import subprocess
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        print("GPU:", smi.stdout.strip() or "n/a")
    except Exception:  # noqa: BLE001
        pass
    report = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    b, f = report["unfiltered_v5"], report["filtered_topdecile"]
    print(f"deep_v5_meta_v1 -> {OUT}")
    print(f"OOS rank-IC (win predictability): {report['oos_rank_ic_mean']} per-fold {report['oos_rank_ic_per_fold']}")
    print(f"unfiltered v5:  n={b['n']} win={b['win_rate']:.0%} mean=${b['mean_pnl_usd']} "
          f"DSR={ (b['cluster_gate'].get('deflated_sharpe') or {}).get('ratio') }")
    print(f"top-decile:     n={f['n']} win={f['win_rate']:.0%} mean=${f['mean_pnl_usd']} "
          f"DSR={ (f['cluster_gate'].get('deflated_sharpe') or {}).get('ratio') }")
    print(f"predictions: {report['predictions']}")


if __name__ == "__main__":
    main()
