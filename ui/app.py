"""Standalone read-only viewer for the alpaca_gpu_lab bench.

Binds 127.0.0.1 only. There is no POST route and nothing here can act.

Data source, in order of preference:
  1. journal/flags/alpaca_gpu_state.json   - the live flag, if you have run the bench
  2. ui/snapshot/alpaca_gpu_state.json     - the committed snapshot shipped with the repo

The fallback is the point: a fresh clone on any machine renders the same bench
results without needing market data, an API key, or a GPU. The page states which
source it is showing and as of when.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template

# Canonical gate thresholds. These mirror macro_gpu_lab.config, which is the one
# lineage the bench itself imports; they are restated here only so the viewer can
# render a snapshot without importing the whole bench. DEFLATED_SHARPE_MIN was
# raised from 0.0 to 1.645 on 2026-07-30.
DEFLATED_SHARPE_MIN = 1.645
PBO_MAX = 0.5

REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_FLAG = REPO_ROOT / "journal" / "flags" / "alpaca_gpu_state.json"
SNAPSHOT = Path(__file__).resolve().parent / "snapshot" / "alpaca_gpu_state.json"

app = Flask(__name__)


# Browsers reject NaN/Infinity in JSON, which left the page stuck on "loading...".
# Emit them as null so the tables render; missing stats already show as "–".
import math as _math
from flask.json.provider import DefaultJSONProvider as _DJP


def _finite(o):
    if isinstance(o, float) and not _math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


class _FiniteJSON(_DJP):
    def dumps(self, obj, **kw):
        return super().dumps(_finite(obj), **kw)


app.json = _FiniteJSON(app)


def _load_state() -> dict:
    """Live flag if present and parseable, else the shipped snapshot.

    Never raises. A missing or corrupt file degrades to a neutral dict rather
    than a 500 - the same fail-safe contract the flag file itself uses.
    """
    for path, source in ((LIVE_FLAG, "live"), (SNAPSHOT, "snapshot")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            data["_source"] = source
            data["_source_path"] = str(path.relative_to(REPO_ROOT))
            return data
    return {
        "_source": "none",
        "_source_path": "",
        "stale": True,
        "status": "SHADOW",
        "enforce": "no",
        "promoted": {},
        "gate": {},
        "note": "no flag file and no snapshot found",
    }


def _flatten_gate(state: dict) -> list[dict]:
    """gate is experiment -> horizon -> verdict; the table wants flat rows.

    `passes` is RECOMPUTED from the stored statistics against the canonical
    thresholds rather than read from the file. The bar was raised from 0.0 to
    1.645 on 2026-07-30 (a `ratio > 0` gate is only a median test, which
    best-of-8 pure noise clears ~45% of the time), and verdicts written before
    that still sit in the flag marked PASS. Trusting the stored field would
    republish a superseded verdict.

    Rows where the stored and recomputed verdicts disagree are marked so the
    page can show the discrepancy instead of hiding it.
    """
    rows = []
    for exp, horizons in (state.get("gate") or {}).items():
        if not isinstance(horizons, dict):
            continue
        for horizon, v in horizons.items():
            if not isinstance(v, dict):
                continue
            dsr, pbo = v.get("dsr_ratio"), v.get("pbo")
            stored = bool(v.get("passes"))
            passes = (
                dsr is not None and pbo is not None
                and dsr > DEFLATED_SHARPE_MIN and pbo < PBO_MAX
            )
            rows.append({
                "experiment": exp,
                "horizon": horizon,
                "passes": passes,
                "stored_passes": stored,
                "superseded": stored and not passes,
                "dsr_ratio": dsr,
                "pbo": pbo,
                "profit_factor": v.get("profit_factor"),
                "n": v.get("n"),
            })
    rows.sort(key=lambda r: (r["experiment"], str(r["horizon"])))
    return rows


@app.route("/")
def page_index():
    return render_template("index.html")


@app.route("/api/alpaca")
def api_alpaca():
    state = _load_state()
    promoted = state.get("promoted") or {}
    rows = _flatten_gate(state)
    state["rows"] = rows
    state["n_experiments"] = len(promoted)
    state["n_promoted"] = sum(1 for v in promoted.values() if v)
    state["n_cells"] = len(rows)
    state["n_pass"] = sum(1 for r in rows if r["passes"])
    state["n_superseded"] = sum(1 for r in rows if r["superseded"])
    state["dsr_min"] = DEFLATED_SHARPE_MIN
    state["pbo_max"] = PBO_MAX
    return jsonify(state)


@app.route("/health")
def health():
    return jsonify({"ok": True, "ts": datetime.now(timezone.utc).isoformat()})


def main() -> None:
    # 127.0.0.1 only. Never 0.0.0.0, never tunnelled.
    app.run(host="127.0.0.1", port=8101, debug=False)


if __name__ == "__main__":
    main()
