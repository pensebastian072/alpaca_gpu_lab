"""The evaluation harness every model funnels through — registry-gated.

Honest-PnL construction (adapted from macro_gpu_lab/models/baseline_rf.py::
oos_portfolio_pnls and qlib_lab/pipeline.py):
  - quarterly expanding walk-forward with a horizon gap (src/models/dataset.py);
  - within each test quarter, entries only at every `horizon`-th unique panel
    timestamp -> forward windows never overlap;
  - at each entry stamp, score all symbols; ACCEPT where p >= threshold
    (pre-registered, default 0.60);
  - pool accepted symbols into ONE PnL observation (mean of realized
    tb_ret_{h}, already net of round-trip fractional costs) — no per-asset
    pseudo-replication (the honest-DSR trick);
  - gate = evaluate_gate(pnls, n_trials=registry.n_trials_for(family)) per
    horizon; ALL horizons must pass; GPU bootstrap CI attached.

Every executed (model, params, horizon) combo is charged to the ledger
BEFORE its result is computed — you pay the deflation for looking, not for
liking what you saw.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config, gate  # noqa: E402
from src.backtest_ci import confidence  # noqa: E402
from src.experiments import registry  # noqa: E402
from src.models import dataset  # noqa: E402

ACCEPT_THRESHOLD = 0.60  # absolute-mode acceptance probability
ACCEPT_QUANTILE = 0.10   # rank-mode: trade the top decile of each fold's scores


def _ic_summary(ic_folds: list[float]) -> dict:
    """Aggregate per-fold rank-IC into a signal-strength verdict.

    ic_t = mean/std * sqrt(n_folds) — a rough t-stat on 'is IC>0 across folds'.
    This is the cheap upstream 'is there ANY signal' number: |ic_t| ~< 1 and
    hit_rate ~ 0.5 means the features do not rank the outcome, and no
    acceptance rule downstream can manufacture edge.
    """
    xs = [x for x in ic_folds if x is not None and np.isfinite(x)]
    if len(xs) < 2:
        return {"ic_mean": None, "ic_std": None, "ic_t": None,
                "ic_hit_rate": None, "n_folds": len(xs)}
    a = np.asarray(xs)
    std = float(a.std(ddof=1))
    return {
        "ic_mean": round(float(a.mean()), 5),
        "ic_std": round(std, 5),
        "ic_t": round(float(a.mean() / std * np.sqrt(len(a))), 3) if std > 0 else None,
        "ic_hit_rate": round(float((a > 0).mean()), 3),
        "n_folds": len(a),
    }


def _impute(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill non-finite feature cells with per-column train medians (0 if a
    column is entirely NaN, e.g. rg_netliq when the FRED source is absent).

    Dropping rows on such columns would wipe the whole panel; tree models
    tolerate NaN but logreg/RF do not, so impute uniformly.
    """
    med = np.nanmedian(np.where(np.isfinite(Xtr), Xtr, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isfinite(Xtr), Xtr, med).astype(np.float32)
    Xte = np.where(np.isfinite(Xte), Xte, med).astype(np.float32)
    return Xtr, Xte


def _spearman(pred: np.ndarray, outcome: np.ndarray) -> float | None:
    """Rank correlation, fail-safe to None (constant input / too few points)."""
    m = np.isfinite(pred) & np.isfinite(outcome)
    if m.sum() < 10:
        return None
    if np.all(pred[m] == pred[m][0]) or np.all(outcome[m] == outcome[m][0]):
        return None  # constant input -> correlation undefined
    try:
        from scipy.stats import spearmanr
        r = spearmanr(pred[m], outcome[m]).correlation
        return float(r) if np.isfinite(r) else None
    except Exception:  # noqa: BLE001 — diagnostic must never crash a run
        return None


def _metrics(pnls: list[float], accepted_tb: list[int], n_stamps: int) -> dict:
    """Tracked metrics: expectancy, PF, maxDD, Sharpe/Sortino, precision, reject%."""
    if not pnls:
        return {"n_trades": 0}
    arr = np.asarray(pnls)
    gains = arr[arr > 0].sum()
    losses = -arr[arr <= 0].sum()
    equity = np.cumsum(arr)
    dd = float((np.maximum.accumulate(equity) - equity).max())
    downside = arr[arr < 0]
    sortino = float(arr.mean() / downside.std(ddof=1)) if len(downside) > 1 and downside.std(ddof=1) > 0 else None
    return {
        "n_trades": len(pnls),
        "expectancy": float(arr.mean()),
        "profit_factor_after_cost": float(gains / losses) if losses > 0 else None,
        "max_drawdown": dd,
        "sortino": sortino,
        "precision_accepted": float(np.mean(accepted_tb)) if accepted_tb else None,
        "pct_stamps_traded": len(pnls) / n_stamps if n_stamps else None,
        "regime_breakdown": "unknown (regime tags land in P6)",
    }


def walk_forward(
    panel: pl.DataFrame,
    horizon: int,
    fit_predict,
    *,
    accept: str = "absolute",       # "absolute" (score>=threshold) | "rank" (top-q per fold)
    threshold: float = ACCEPT_THRESHOLD,
    q: float = ACCEPT_QUANTILE,
    target_col: str | None = None,  # training label (default tb_{h} — direction)
    ret_col: str | None = None,     # realized long PnL (default tb_ret_{h})
    ic_col: str | None = None,      # IC measured pred-vs-this (default ret_col)
    label_col: str | None = None,   # binary label for precision stat (default tb_{h})
    score_col: str | None = None,   # precomputed signal (e.g. Kronos) — skip training
) -> dict:
    """Single-name walk-forward with rank-IC + pluggable acceptance/target.

    fit_predict(Xtr, ytr, Xte) -> score array (p_up for classifiers, predicted
    magnitude for regressors). One PnL per non-overlapping entry stamp = mean
    realized ret_col over accepted names at that stamp. Rank-IC is the raw
    ranking power over ALL test rows (before acceptance) — the signal check.

    Returns {pnls, accepted_labels, n_stamps, ic}.
    """
    fcols = dataset.feature_columns(panel)
    target_col = target_col or f"tb_{horizon}"
    ret_col = ret_col or f"tb_ret_{horizon}"
    ic_col = ic_col or ret_col
    label_col = label_col or f"tb_{horizon}"

    pnls: list[float] = []
    accepted_lab: list[int] = []
    n_stamps = 0
    ic_folds: list[float] = []
    for train, test, _label in dataset.quarterly_walk_forward(panel, horizon):
        # precomputed-score mode (Kronos etc): no per-fold training
        needed = [target_col, ret_col] + ([score_col] if score_col else [])
        te = test.drop_nulls(subset=needed)
        if te.is_empty():
            continue
        # non-overlapping entries. Precomputed scores (Kronos) are ALREADY
        # sparse (forecast every >=horizon bars), so re-gathering here would
        # double-subsample to near-zero n — use every scored stamp as an entry
        # (the forecast step guarantees non-overlap). Trained models see dense
        # rows, so keep the gather.
        if not score_col:
            stamps = te["timestamp"].unique().sort()
            entry = set(stamps.gather_every(horizon).to_list())
            te = te.filter(pl.col("timestamp").is_in(entry))
            if te.is_empty():
                continue
        if score_col:
            score = te[score_col].to_numpy().astype(np.float64)
        else:
            tr = train.drop_nulls(subset=[target_col])
            if tr.is_empty() or tr[target_col].n_unique() < 2:
                continue
            Xtr, Xte = _impute(tr.select(fcols).to_numpy(), te.select(fcols).to_numpy())
            ytr = tr[target_col].to_numpy()
            score = np.asarray(fit_predict(Xtr, ytr, Xte), dtype=np.float64)
        te = te.with_columns(pl.Series("score", score))

        # rank-IC over the whole fold's test rows (before acceptance)
        ic_folds.append(_spearman(score, te[ic_col].to_numpy().astype(np.float64)))

        # acceptance
        if accept == "rank":
            cut = float(np.quantile(score, 1.0 - q))
            te_acc = te.with_columns((pl.col("score") >= cut).alias("_acc"))
        else:
            te_acc = te.with_columns((pl.col("score") >= threshold).alias("_acc"))

        for _ts, grp in te_acc.group_by("timestamp", maintain_order=True):
            n_stamps += 1
            acc = grp.filter(pl.col("_acc"))
            if acc.is_empty():
                continue
            pnls.append(float(acc[ret_col].mean()))
            if label_col in acc.columns:
                accepted_lab.extend(int(v) for v in acc[label_col].drop_nulls().to_list())
    return {"pnls": pnls, "accepted_labels": accepted_lab,
            "n_stamps": n_stamps, "ic": _ic_summary(ic_folds)}


def walk_forward_pnls(panel, horizon, fit_predict, threshold=ACCEPT_THRESHOLD):
    """Back-compat 3-tuple wrapper (absolute acceptance, direction target)."""
    r = walk_forward(panel, horizon, fit_predict, accept="absolute", threshold=threshold)
    return r["pnls"], r["accepted_labels"], r["n_stamps"]


def walk_forward_xsect(
    panel: pl.DataFrame,
    horizon: int,
    fit_predict,
    *,
    q: float = 0.34,                # long top third / short bottom third of names
    ret_col: str | None = None,
    target_col: str | None = None,
    score_col: str | None = None,   # precomputed signal (Kronos) — skip training
) -> dict:
    """Cross-sectional long-short: per entry stamp, rank names by predicted
    score, long the top / short the bottom, one net PnL per stamp (costs both
    legs). Rank-IC = per-fold spearman(score, ret) across the name panel.
    """
    fcols = dataset.feature_columns(panel)
    target_col = target_col or f"tb_{horizon}"
    # gross close-to-close return (NOT tb_ret, which is net + long-specific):
    # a long-short nets each leg's gross return then pays per-leg cost once.
    ret_col = ret_col or f"fwd_ret_{horizon}"

    pnls: list[float] = []
    n_stamps = 0
    ic_folds: list[float] = []
    for train, test, _label in dataset.quarterly_walk_forward(panel, horizon):
        needed = [target_col, ret_col, "symbol"] + ([score_col] if score_col else [])
        te = test.drop_nulls(subset=needed)
        if te.is_empty():
            continue
        if not score_col:  # precomputed scores are pre-spaced; don't re-gather
            stamps = te["timestamp"].unique().sort()
            entry = set(stamps.gather_every(horizon).to_list())
            te = te.filter(pl.col("timestamp").is_in(entry))
            if te.is_empty():
                continue
        if score_col:
            score = te[score_col].to_numpy().astype(np.float64)
        else:
            tr = train.drop_nulls(subset=[target_col])
            if tr.is_empty() or tr[target_col].n_unique() < 2:
                continue
            Xtr, Xte = _impute(tr.select(fcols).to_numpy(), te.select(fcols).to_numpy())
            ytr = tr[target_col].to_numpy()
            score = np.asarray(fit_predict(Xtr, ytr, Xte), dtype=np.float64)
        te = te.with_columns(pl.Series("score", score))
        ic_folds.append(_spearman(score, te[ret_col].to_numpy().astype(np.float64)))

        for _ts, grp in te.group_by("timestamp", maintain_order=True):
            if grp.height < 3:  # need enough names for a cross-section
                continue
            n_stamps += 1
            g = grp.sort("score")
            k = max(1, int(round(grp.height * q)))
            shorts = g.head(k)
            longs = g.tail(k)
            gross = float(longs[ret_col].mean()) - float(shorts[ret_col].mean())
            # per-leg round-trip cost, averaged over the 2k names actually traded
            cost = np.mean([2 * config.cost_per_side(s) for s in
                            longs["symbol"].to_list() + shorts["symbol"].to_list()])
            pnls.append(gross - float(cost))
    return {"pnls": pnls, "accepted_labels": [], "n_stamps": n_stamps,
            "ic": _ic_summary(ic_folds)}


def run_battery(
    battery_id: str,
    model_grid,  # iterable of (params_dict, fit_predict callable)
    symbols: list[str],
    timeframe: str = "1Min",
    threshold: float = ACCEPT_THRESHOLD,
    *,
    mode: str = "direction",        # "direction" | "magnitude" | "xsect"
    accept: str = "absolute",       # "absolute" | "rank"
    q: float = ACCEPT_QUANTILE,
    score_col: str | None = None,   # precomputed signal (Kronos) — bypass training
    panel: pl.DataFrame | None = None,  # pre-built panel (e.g. Kronos-joined); else load
) -> dict:
    """Evaluate a pre-registered battery; write scorecard + RESULTS rows.

    mode:
      direction  — target tb_{h}, IC vs realized tb_ret, long-harvest PnL
      magnitude  — target mfe_{h} (regressor), IC vs realized mfe, PnL = long
                   harvest on top-q predicted-excursion bars
      xsect      — cross-sectional long-short across `symbols` per stamp
    Rank-IC is logged for EVERY combo/horizon even when no PnL clears — that is
    the deliverable (does signal exist), not a forced pass.
    """
    reg = registry.load_registration(battery_id)  # raises if unregistered
    family = reg["dataset_family"]
    if panel is None:
        panel = dataset.load_panel(symbols, timeframe)

    results: dict = {
        "battery_id": battery_id, "dataset_family": family, "symbols": symbols,
        "timeframe": timeframe, "feed": config.DEFAULT_FEED, "mode": mode,
        "accept": accept, "q": q, "threshold": threshold,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "combos": [],
    }

    for params, fit_predict in model_grid:
        combo: dict = {"params": params, "horizons": {}, "all_pass": True}
        for h in reg["horizons"]:
            registry.log_trial(battery_id, family, reg["model"], params, h, symbols)
            sc = score_col.format(h=h) if score_col else None
            if mode == "xsect":
                r = walk_forward_xsect(panel, h, fit_predict, q=q, score_col=sc)
            elif mode == "magnitude":
                r = walk_forward(panel, h, fit_predict, accept=accept, q=q,
                                 target_col=f"mfe_{h}", ic_col=f"mfe_{h}",
                                 ret_col=f"tb_ret_{h}", score_col=sc)
            else:
                r = walk_forward(panel, h, fit_predict, accept=accept,
                                 threshold=threshold, q=q, score_col=sc)
            verdict = gate.evaluate_gate(r["pnls"], n_trials=registry.n_trials_for(family))
            verdict["bootstrap"] = confidence(r["pnls"])
            verdict["metrics"] = _metrics(r["pnls"], r["accepted_labels"], r["n_stamps"])
            verdict["ic"] = r["ic"]  # signal-strength diagnostic, always present
            combo["horizons"][str(h)] = verdict
            combo["all_pass"] = combo["all_pass"] and verdict["passes"]
            ic = r["ic"]
            registry.append_result_row(
                battery_id, h, verdict,
                note=f"{reg['model']} {mode}/{accept} IC={ic.get('ic_mean')} t={ic.get('ic_t')}")
        results["combos"].append(combo)

    results["promoted"] = any(c["all_pass"] for c in results["combos"])
    results["status"] = "PASS" if results["promoted"] else "FAIL->SHADOW"
    out = registry.write_scorecard(battery_id, results)
    results["scorecard_path"] = str(out)
    return results
