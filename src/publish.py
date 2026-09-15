"""Shadow flag-file publisher — journal/flags/alpaca_gpu_state.json.

Adapted from macro_gpu_lab/macro_gpu_lab/publish.py (atomic tmp+replace write,
fail-safe reader, stale-days, ENFORCE env default "no"). No webhook, no
consumer wiring — this is a flag file other repos MAY read, off any hot path,
neutral on stale/missing/corrupt (paper-trading-guardrails rule 2).

State shape:
{
  "as_of": iso, "stale": false, "enforce": "no",
  "status": "SHADOW", "promoted": {battery_id: bool},
  "gate": {battery_id: {horizon: {passes, dsr_ratio, pbo, profit_factor, n}}},
  "signals": {symbol: {horizon: p_up}},          # latest advisory probabilities
  "note": "SHADOW research advisory — no battery has cleared the gate"
}
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

FLAG_PATH = config.FLAGS / "alpaca_gpu_state.json"
FLAG_STALE_DAYS = 4

NEUTRAL = {
    "as_of": None, "stale": True, "enforce": "no", "status": "SHADOW",
    "promoted": {}, "gate": {}, "signals": {},
    "note": "neutral fallback — flag missing/stale/corrupt",
}


def build_state(scorecards: dict[str, dict], signals: dict) -> dict:
    """scorecards: {battery_id: run_battery() result}; signals: {sym: {h: p}}."""
    gate_summary: dict = {}
    promoted: dict = {}
    for bid, res in scorecards.items():
        promoted[bid] = bool(res.get("promoted"))
        best = {}
        for combo in res.get("combos", []):
            for h, v in combo.get("horizons", {}).items():
                cur = best.get(h)
                dsr = (v.get("deflated_sharpe") or {}).get("ratio")
                if cur is None or (dsr is not None and dsr > (cur.get("dsr_ratio") or -1e9)):
                    best[h] = {
                        "passes": v.get("passes"),
                        "dsr_ratio": dsr,
                        "pbo": v.get("pbo"),
                        "profit_factor": v.get("profit_factor"),
                        "n": v.get("n_trades"),
                    }
        gate_summary[bid] = best
    any_promoted = any(promoted.values())
    return {
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stale": False,
        "enforce": os.environ.get("ALPACA_GPU_ENFORCE", "no"),
        "status": "PROMOTED" if any_promoted else "SHADOW",
        "promoted": promoted,
        "gate": gate_summary,
        "signals": signals,
        "note": ("battery cleared the gate — still env-gated, default off"
                 if any_promoted else
                 "SHADOW research advisory — no battery has cleared the gate"),
    }


def publish(state: dict) -> Path:
    FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = FLAG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, FLAG_PATH)
    return FLAG_PATH


def read_state() -> dict:
    """Fail-safe reader: neutral on missing/corrupt/stale. Never raises."""
    try:
        state = json.loads(FLAG_PATH.read_text(encoding="utf-8"))
        as_of = datetime.fromisoformat(state["as_of"])
        age_days = (datetime.now(timezone.utc) - as_of).total_seconds() / 86400
        if age_days > FLAG_STALE_DAYS:
            state["stale"] = True
            state["note"] = f"stale ({age_days:.1f}d old) — treat as neutral"
        return state
    except Exception:  # noqa: BLE001 — neutral fallback by design
        return dict(NEUTRAL)
