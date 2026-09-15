"""Pre-registered experiment batteries + the n_trials ledger.

The GPU makes it cheap to test thousands of combos — which makes false
discoveries cheap too. Discipline (CLAUDE.md rule 4, same ethos as
qlib_lab/experiments.py N_TRIALS but computed from an append-only ledger
instead of a hand-maintained constant):

1. register(...) BEFORE any result exists — writes
   journal/experiments/registered/<battery_id>.json. The evaluation harness
   refuses unregistered batteries.
2. Every (model, params, horizon, symbol-set) combo actually executed appends
   one line to journal/experiments/ledger.jsonl.
3. n_trials_for(dataset_family) = cumulative ledger trials against that
   family + the family's seed — the number fed to evaluate_gate(...,
   n_trials=...). Intraday families seed at 0 (genuinely new ground); the
   daily 22-asset family seeds at 60 — the approximate historical trial count
   already spent by macro_gpu_lab + qlib_lab batteries against daily horizons
   on this universe (H01-15, W01-11, V/E/F/L, COT, RF/MLP/shift/surprise).
4. Scorecards land in journal/scorecards/, one human-readable row appended to
   journal/RESULTS.md. Honest FAILs are the expected output.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402

REGISTERED = config.EXPERIMENTS / "registered"
LEDGER = config.EXPERIMENTS / "ledger.jsonl"
RESULTS_MD = config.JOURNAL / "RESULTS.md"

# Trials already spent against a family before this repo existed.
FAMILY_SEEDS = {
    "daily_22": 60,  # macro_gpu_lab + qlib_lab historical batteries
}


class UnregisteredBattery(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def register(
    battery_id: str,
    hypothesis: str,
    dataset_family: str,
    horizons: list[int],
    model: str,
    param_grid: dict,
    n_trials: int,
) -> Path:
    """Pre-register a battery. Refuses to overwrite an existing registration
    (append-only science — a changed hypothesis is a NEW battery id)."""
    REGISTERED.mkdir(parents=True, exist_ok=True)
    path = REGISTERED / f"{battery_id}.json"
    if path.exists():
        raise FileExistsError(
            f"battery {battery_id} already registered — registrations are "
            "append-only; a changed design needs a new battery id"
        )
    rec = {
        "battery_id": battery_id,
        "registered_at": _now(),
        "hypothesis": hypothesis,
        "dataset_family": dataset_family,
        "horizons": horizons,
        "model": model,
        "param_grid": param_grid,
        "n_trials": n_trials,
        "feed": config.DEFAULT_FEED,
    }
    path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return path


def load_registration(battery_id: str) -> dict:
    path = REGISTERED / f"{battery_id}.json"
    if not path.exists():
        raise UnregisteredBattery(
            f"battery {battery_id} is not registered — call register() with a "
            "written hypothesis BEFORE evaluating (CLAUDE.md rule 4)"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def log_trial(battery_id: str, dataset_family: str, model: str,
              params: dict, horizon: int, symbols: list[str]) -> None:
    """Append one executed combo to the ledger (called by the harness)."""
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({
        "ts": _now(), "battery_id": battery_id, "dataset_family": dataset_family,
        "model": model, "params": params, "horizon": horizon,
        "symbols": symbols, "feed": config.DEFAULT_FEED,
    })
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def n_trials_for(dataset_family: str) -> int:
    """Cumulative trials charged against a dataset family (ledger + seed).

    Fed to evaluate_gate(pnls, n_trials=...) so the deflated Sharpe pays for
    every look at the data, not just the current battery's looks.
    Never less than 1.
    """
    n = FAMILY_SEEDS.get(dataset_family, 0)
    if LEDGER.exists():
        with LEDGER.open(encoding="utf-8") as f:
            for line in f:
                try:
                    if json.loads(line).get("dataset_family") == dataset_family:
                        n += 1
                except json.JSONDecodeError:
                    continue  # torn line from a killed run — ignore, never crash
    return max(n, 1)


def append_result_row(battery_id: str, horizon: int, verdict: dict,
                      note: str = "") -> None:
    """One human-readable row per gate verdict in journal/RESULTS.md."""
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_MD.exists():
        RESULTS_MD.write_text(
            "# Results log\n\n"
            "Every gate verdict, honest FAILs included — that is the system "
            "working, not a bug.\n\n"
            "| date | battery | horizon | n | PF | Sharpe | DSR ratio | PBO | n_trials | verdict | note |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    dsr = verdict.get("deflated_sharpe") or {}
    row = ("| {d} | {b} | {h} | {n} | {pf} | {sr} | {ratio} | {pbo} | {nt} | {v} | {note} |\n").format(
        d=_now()[:10], b=battery_id, h=horizon,
        n=verdict.get("n_trades", "-"),
        pf=_fmt(verdict.get("profit_factor")),
        sr=_fmt(verdict.get("sharpe")),
        ratio=_fmt(dsr.get("ratio")),
        pbo=_fmt(verdict.get("pbo")),
        nt=dsr.get("n_trials", "-"),
        v="PASS" if verdict.get("passes") else "FAIL",
        note=note.replace("|", "/"),
    )
    with RESULTS_MD.open("a", encoding="utf-8") as f:
        f.write(row)


def write_scorecard(battery_id: str, payload: dict) -> Path:
    config.SCORECARDS.mkdir(parents=True, exist_ok=True)
    out = config.SCORECARDS / f"{battery_id}_{_now()[:10]}.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out


def _fmt(x) -> str:
    return "-" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)
