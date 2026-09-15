"""Shared config: paths, the 22-asset research universe, horizons, costs, gate.

Mission (2026-07): standalone GPU alpha-research bench over the 22-asset macro
universe (Alpaca ETF proxies + BTC). Research/advisory only — Alpaca is the
DATA client here, never the trading client. Every model stays SHADOW until the
canonical overfit gate clears (see src/gate.py — imported from macro_gpu_lab,
never re-ported).

The original core-7 minute-bar universe (UNIVERSE + PROXIES) is kept: it is
fully backfilled at 1Min and is where intraday modeling starts while the
22-asset spine backfills.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "market_data"
RAW = DATA / "raw"
FEATURES = DATA / "features"
MODELS = DATA / "models"
BACKTESTS = DATA / "backtests"
SCORES = DATA / "scores"

# Research journal (committed): experiments, scorecards, flags, logs.
JOURNAL = REPO / "journal"
EXPERIMENTS = JOURNAL / "experiments"
SCORECARDS = JOURNAL / "scorecards"
FLAGS = JOURNAL / "flags"
LOGS = JOURNAL / "logs"
DATA_REPORTS = JOURNAL / "data"

# Original first-project universe (Alpaca equities only; 1Min fully backfilled).
UNIVERSE = ["SPY", "QQQ", "IWM", "BND"]
# Cross-market proxies for feature engineering (not primary targets)
PROXIES = ["IEF", "TLT", "VIXY"]  # 7-10y, 20y+ treasuries, VIX short-term
CORE7 = UNIVERSE + PROXIES

# ---------------------------------------------------------------------------
# 22-asset research universe — Alpaca-fetchable ETF proxies mirroring the HQ
# macro brain's forces (hq-trading-system/analytics/macro_config.py UNIVERSE).
# Alpaca serves no indices or FX spot, so tradable proxies stand in:
#   sign=-1 marks proxies that move INVERSELY to the logical macro series
#   (FXY tracks JPY/USD = 1/USDJPY; FXC tracks CAD/USD = 1/USDCAD).
#   VIXY has roll drag — usable as a feature/return series, never as a level.
#   FXI proxies China stress (no liquid CNH ETF; CYB delisted).
# asset_class "crypto" routes through CryptoHistoricalDataClient.
# liquidity tier: "liquid" | "thin" — thin tickers (currency ETFs, CPER, BIL)
# have sparse IEX minute prints; treat them as hourly-granularity context.
# ---------------------------------------------------------------------------
UNIVERSE_22: dict[str, dict] = {
    # rates
    "IEF":    {"ticker": "IEF",     "force": "rates",       "sign": +1, "asset_class": "equity", "tier": "liquid"},
    "TLT":    {"ticker": "TLT",     "force": "rates",       "sign": +1, "asset_class": "equity", "tier": "liquid"},
    "TIP":    {"ticker": "TIP",     "force": "rates",       "sign": +1, "asset_class": "equity", "tier": "liquid"},
    # dollar
    "UUP":    {"ticker": "UUP",     "force": "dollar",      "sign": +1, "asset_class": "equity", "tier": "liquid"},
    "FXE":    {"ticker": "FXE",     "force": "dollar",      "sign": +1, "asset_class": "equity", "tier": "thin"},
    "FXY":    {"ticker": "FXY",     "force": "dollar",      "sign": -1, "asset_class": "equity", "tier": "thin"},  # 1/USDJPY
    "FXB":    {"ticker": "FXB",     "force": "dollar",      "sign": +1, "asset_class": "equity", "tier": "thin"},
    "FXC":    {"ticker": "FXC",     "force": "dollar",      "sign": -1, "asset_class": "equity", "tier": "thin"},  # 1/USDCAD
    # china
    "FXI":    {"ticker": "FXI",     "force": "china",       "sign": +1, "asset_class": "equity", "tier": "liquid"},
    # equities
    "SPY":    {"ticker": "SPY",     "force": "equities",    "sign": +1, "asset_class": "equity", "tier": "liquid"},
    "QQQ":    {"ticker": "QQQ",     "force": "equities",    "sign": +1, "asset_class": "equity", "tier": "liquid"},
    "IWM":    {"ticker": "IWM",     "force": "equities",    "sign": +1, "asset_class": "equity", "tier": "liquid"},
    "EEM":    {"ticker": "EEM",     "force": "equities",    "sign": +1, "asset_class": "equity", "tier": "liquid"},
    "XLF":    {"ticker": "XLF",     "force": "equities",    "sign": +1, "asset_class": "equity", "tier": "liquid"},
    # volatility
    "VIXY":   {"ticker": "VIXY",    "force": "volatility",  "sign": +1, "asset_class": "equity", "tier": "liquid"},
    # commodities
    "GLD":    {"ticker": "GLD",     "force": "commodities", "sign": +1, "asset_class": "equity", "tier": "liquid"},
    "USO":    {"ticker": "USO",     "force": "commodities", "sign": +1, "asset_class": "equity", "tier": "liquid"},
    "CPER":   {"ticker": "CPER",    "force": "commodities", "sign": +1, "asset_class": "equity", "tier": "thin"},
    # credit
    "HYG":    {"ticker": "HYG",     "force": "credit",      "sign": +1, "asset_class": "equity", "tier": "liquid"},
    "LQD":    {"ticker": "LQD",     "force": "credit",      "sign": +1, "asset_class": "equity", "tier": "liquid"},
    # liquidity
    "BIL":    {"ticker": "BIL",     "force": "liquidity",   "sign": +1, "asset_class": "equity", "tier": "thin"},
    # crypto (partition symbol=BTCUSD; Alpaca API symbol "BTC/USD")
    "BTCUSD": {"ticker": "BTC/USD", "force": "crypto",      "sign": +1, "asset_class": "crypto", "tier": "liquid"},
}
assert len(UNIVERSE_22) == 22, "research universe must stay at 22 assets"

EQUITY_22 = [k for k, v in UNIVERSE_22.items() if v["asset_class"] == "equity"]
CRYPTO_22 = [k for k, v in UNIVERSE_22.items() if v["asset_class"] == "crypto"]

DEFAULT_TIMEFRAME = "1Min"
DEFAULT_FEED = "iex"  # free plan; SIP needs subscription (see CLAUDE.md)
DEFAULT_START = "2022-01-01"
# The 22-asset hourly/daily spine reaches further back (cheap at 1Hour/1Day).
SPINE_START = "2016-01-01"

# Label horizons in BARS (5/15/30-minute forward outcomes on 1Min bars;
# same numbers double as 5/15/30-hour horizons on 1Hour bars).
HORIZONS_BARS = [5, 15, 30]

# Round-trip cost model: FRACTION of price per side (spread/2 + slippage).
# Hard-won lesson (fx_hermes v3): costs are fractions, never absolute units.
COST_PER_SIDE = {"liquid": 0.0001, "thin": 0.0003}  # 1bp / 3bp


def cost_per_side(asset: str) -> float:
    """Per-side cost fraction for an asset (default: thin tier if unknown)."""
    tier = UNIVERSE_22.get(asset, {}).get("tier", "thin")
    return COST_PER_SIDE[tier]


# ---------------------------------------------------------------------------
# Holdout lock: 2026 is the final untouched test period. The default dataset
# loader clips everything >= HOLDOUT_START; reading holdout rows requires BOTH
# unlock_holdout=True in code AND this env flag — flipped exactly once at the
# end of the research program, never during iteration.
# ---------------------------------------------------------------------------
HOLDOUT_START = "2026-01-01"
HOLDOUT_UNLOCK = os.environ.get("ALPACA_GPU_HOLDOUT_UNLOCK", "no").lower() in ("yes", "true", "1")

# Canonical overfit gate lives in macro_gpu_lab (see src/gate.py). Thresholds
# (PBO_MAX, DEFLATED_SHARPE_MIN) live there too — never duplicated here.
MACRO_GPU_LAB_DIR = Path(os.environ.get("MACRO_GPU_LAB_DIR", r"C:\Users\<your-user>\macro_gpu_lab"))

# Shadow/enforce flip for the published flag file (default: shadow).
ENFORCE = os.environ.get("ALPACA_GPU_ENFORCE", "no")

SEED = 42
