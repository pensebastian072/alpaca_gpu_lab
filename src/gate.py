"""Canonical overfit gate — imported from macro_gpu_lab, NEVER re-ported.

The PBO/Deflated-Sharpe/purged-CV math lives in exactly one lineage:
hq-trading-system/analytics/research_scorecard.py -> copper_brain/validate.py
-> macro_gpu_lab/macro_gpu_lab/validate.py (the shared import target; qlib_lab
imports it the same way). This module wires that package onto sys.path and
re-exports the public API. If the import fails we fail LOUD — a silent
fallback or a vendored copy would fork the gate math, which is exactly the
bug this module exists to prevent.

Gate thresholds (PBO_MAX=0.5, **DEFLATED_SHARPE_MIN=1.645**) live in
macro_gpu_lab.config — not duplicated here. This docstring said 0.0 until
2026-08-07; the bar was raised on 2026-07-30 alongside the DSR unit fix, because
a correctly computed `ratio > 0` is only a MEDIAN test that best-of-8 pure noise
clears 44.8% of the time. Read the value from config, never from this line.

Note: importing macro_gpu_lab.validate executes macro_gpu_lab/config.py, which
mkdirs a few directories inside that repo. Harmless side effect.
"""
from __future__ import annotations

import sys

from src import config

_dir = config.MACRO_GPU_LAB_DIR
if str(_dir) not in sys.path:
    sys.path.insert(0, str(_dir))

try:
    from macro_gpu_lab.validate import (  # noqa: F401
        deflated_sharpe,
        evaluate_gate,
        pbo_cscv,
        profit_factor,
        purged_kfold,
        sharpe,
        walk_forward_splits,
    )
except Exception as e:  # pragma: no cover - exercised only on a broken box
    raise RuntimeError(
        f"canonical gate not importable from {_dir}: {e!r}. "
        "Set MACRO_GPU_LAB_DIR to the macro_gpu_lab repo root. "
        "Do NOT re-port validate.py into this repo."
    ) from e

__all__ = [
    "deflated_sharpe",
    "evaluate_gate",
    "pbo_cscv",
    "profit_factor",
    "purged_kfold",
    "sharpe",
    "walk_forward_splits",
]
