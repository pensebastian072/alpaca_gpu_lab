"""Flag-file contract: atomic publish, fail-safe read, stale detection."""
import json
from datetime import datetime, timedelta, timezone

from src import publish


def test_read_missing_is_neutral(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "FLAG_PATH", tmp_path / "nope.json")
    s = publish.read_state()
    assert s["stale"] is True and s["status"] == "SHADOW" and s["enforce"] == "no"


def test_read_corrupt_is_neutral(tmp_path, monkeypatch):
    p = tmp_path / "flag.json"
    p.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(publish, "FLAG_PATH", p)
    s = publish.read_state()
    assert s["stale"] is True and s["status"] == "SHADOW"


def test_stale_flag_marked(tmp_path, monkeypatch):
    p = tmp_path / "flag.json"
    old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(timespec="seconds")
    p.write_text(json.dumps({"as_of": old, "stale": False, "status": "SHADOW"}),
                 encoding="utf-8")
    monkeypatch.setattr(publish, "FLAG_PATH", p)
    s = publish.read_state()
    assert s["stale"] is True


def test_publish_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "FLAG_PATH", tmp_path / "flag.json")
    fake_result = {
        "promoted": False,
        "combos": [{"horizons": {"5": {
            "passes": False, "pbo": 0.6, "profit_factor": 0.9, "n_trades": 40,
            "deflated_sharpe": {"ratio": -2.0}}}}],
    }
    state = publish.build_state({"B01": fake_result}, signals={"SPY": {"5": 0.51}})
    publish.publish(state)
    back = publish.read_state()
    assert back["status"] == "SHADOW"
    assert back["promoted"] == {"B01": False}
    assert back["stale"] is False
