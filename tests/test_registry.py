"""Registry discipline: pre-registration required, ledger drives n_trials."""
import pytest

from src import config
from src.experiments import registry


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "REGISTERED", tmp_path / "registered")
    monkeypatch.setattr(registry, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(registry, "RESULTS_MD", tmp_path / "RESULTS.md")
    return tmp_path


def test_unregistered_battery_raises(sandbox):
    with pytest.raises(registry.UnregisteredBattery):
        registry.load_registration("B99_never_registered")


def test_register_then_load(sandbox):
    registry.register("B01_test", "TP before SL is predictable from features",
                      "intraday_core7_1min", [5, 15, 30], "logreg",
                      {"C": [1.0]}, n_trials=3)
    rec = registry.load_registration("B01_test")
    assert rec["hypothesis"]
    assert rec["n_trials"] == 3


def test_registrations_append_only(sandbox):
    registry.register("B02_once", "h", "intraday_core7_1min", [5], "rf", {}, 1)
    with pytest.raises(FileExistsError):
        registry.register("B02_once", "changed my mind", "intraday_core7_1min",
                          [5], "rf", {}, 1)


def test_ledger_drives_n_trials(sandbox):
    fam = "intraday_core7_1min"
    assert registry.n_trials_for(fam) == 1  # empty ledger, no seed -> floor 1
    for h in (5, 15, 30):
        registry.log_trial("B01_test", fam, "logreg", {"C": 1.0}, h, ["SPY"])
    assert registry.n_trials_for(fam) == 3


def test_daily_family_carries_historical_seed(sandbox):
    # the daily universe was already mined ~60 times by macro_gpu_lab/qlib_lab
    assert registry.n_trials_for("daily_22") >= 60


def test_results_row_appends(sandbox):
    verdict = {"passes": False, "n_trades": 120, "profit_factor": 0.97,
               "sharpe": -0.02, "pbo": 0.55,
               "deflated_sharpe": {"ratio": -3.1, "n_trials": 6}}
    registry.append_result_row("B01_test", 15, verdict, note="baseline")
    text = registry.RESULTS_MD.read_text(encoding="utf-8")
    assert "B01_test" in text and "FAIL" in text
