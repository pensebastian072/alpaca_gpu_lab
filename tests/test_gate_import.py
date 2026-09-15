"""The canonical gate imports from macro_gpu_lab and behaves honestly.

Mirrors macro_gpu_lab/tests/test_pipeline.py gate tests — the point is not to
re-test the math (that lives upstream) but to prove THIS repo's wiring reaches
the real implementation and that the gate stays conservative.
"""
import numpy as np

from src import config, gate


def test_gate_functions_present():
    for fn in ("evaluate_gate", "pbo_cscv", "deflated_sharpe",
               "purged_kfold", "walk_forward_splits"):
        assert callable(getattr(gate, fn))


def test_gate_rejects_pure_noise():
    rng = np.random.default_rng(1)
    pnls = rng.normal(0, 1, 500).tolist()  # zero-mean noise
    g = gate.evaluate_gate(pnls, n_trials=2)
    assert g["passes"] is False


def test_gate_handles_tiny_input():
    g = gate.evaluate_gate([0.1, -0.2, 0.05], n_trials=2)
    assert g["passes"] is False  # too few for DSR/PBO -> cannot pass


def test_walk_forward_splits_no_label_peek():
    horizon = 30
    for train_idx, test_idx in gate.walk_forward_splits(1000, 5, horizon):
        # gap of at least `horizon` between last train row and first test row
        assert min(test_idx) - max(train_idx) >= horizon


def test_universe_is_22():
    assert len(config.UNIVERSE_22) == 22
    assert set(v["asset_class"] for v in config.UNIVERSE_22.values()) <= {"equity", "crypto"}
    # sign-inverted proxies documented
    assert config.UNIVERSE_22["FXY"]["sign"] == -1
    assert config.UNIVERSE_22["FXC"]["sign"] == -1


def test_cost_model_is_fractional():
    # fx_hermes v3 lesson: costs are fractions of price, never absolute units
    for tier, c in config.COST_PER_SIDE.items():
        assert 0 < c < 0.01, f"{tier} cost {c} not a sane fraction"
