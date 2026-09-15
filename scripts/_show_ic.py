"""Print the IC + gate summary for every scorecard (quick read tool)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

for p in sorted(config.SCORECARDS.glob("*.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    mode = d.get("mode", "-")
    print(f"\n{d['battery_id']}  ({d.get('timeframe','?')}, mode={mode}, accept={d.get('accept','-')})")
    for c in d.get("combos", []):
        model = c["params"].get("model", "?")
        for h, v in c["horizons"].items():
            ic = v.get("ic") or {}
            print(f"  {model:12s} h={h:>2s}  IC={ic.get('ic_mean')}  t={ic.get('ic_t')}"
                  f"  hit={ic.get('ic_hit_rate')}  n={v.get('n_trades')}"
                  f"  PF={round(v.get('profit_factor') or 0, 2)}  pass={v.get('passes')}")
