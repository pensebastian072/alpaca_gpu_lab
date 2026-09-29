# alpaca_gpu_lab

<!-- one-tap-install -->
[![Download ZIP](https://img.shields.io/badge/Download-ZIP-2ea44f?style=for-the-badge&logo=github)](https://github.com/pensebastian072/alpaca_gpu_lab/archive/refs/heads/main.zip)

**Run it on your computer in 3 steps:** 1) [download the ZIP](https://github.com/pensebastian072/alpaca_gpu_lab/archive/refs/heads/main.zip) · 2) unzip it · 3) double-click **`install.bat`** (Windows) or run **`./install.sh`** (macOS/Linux).
The dashboard opens in your browser at `http://127.0.0.1:8101` - it runs only on your machine. Next time use `start.bat` / `./start.sh`.
For the full research stack (large downloads) use `install.bat --full` / `./install.sh --full`.
<!-- one-tap-install -->

Standalone GPU **alpha-research bench** over a 22-asset macro universe (ETF
proxies + BTC via Alpaca). Finds signals, patterns, and cross-asset
relationships (positive and negative correlations) behind the canonical
overfit gate. Research only — no order routing, no TradingView/webhook wiring;
the only output is a shadow flag-file. See `CLAUDE.md` for hard rules.

## Pipeline

```text
Alpaca API (IEX bars + news)  ->  CPU backfill  ->  hive Parquet
                                        |
                                 Polars feature engineering (CPU)
                                 own + cross-asset + regime + news sentiment
                                        |
                                 intraday triple-barrier labels (5/15/30 bars)
                                        |
                                 LogReg/RF baselines -> XGBoost-CUDA -> small
                                 torch temporal models (GPU)
                                        |
                                 quarterly walk-forward + overfit gate
                                 (imported from macro_gpu_lab: PBO / DSR /
                                  purged CV; pre-registered n_trials)
                                        |
                                 journal/scorecards + RESULTS.md
                                 journal/flags/alpaca_gpu_state.json (SHADOW)
```

## Layout

```text
src/
  config.py                 22-asset universe, horizons, costs, holdout lock
  gate.py                   canonical gate import (never re-ported)
  gpu.py                    device/seed helpers + CUDA smoke test
  data/alpaca_backfill.py   historical bars -> partitioned Parquet
  data/coverage.py          spine health report
  features/build.py         polars features + triple-barrier labels
tests/                      pytest (gate wiring, labels, leakage, holdout)
journal/                    committed: experiments, scorecards, flags, RESULTS.md
market_data/                gitignored, regenerable from Alpaca
  raw/bars_<tf>/symbol=X/year=Y/month=M/bars.parquet
  raw/news/  features/ models/ scores/ backtests/
```

## Quickstart

```powershell
# venv + deps (torch cu121, xgboost, sklearn)
pwsh scripts/setup_venv.ps1

# tests + CUDA smoke
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m src.gpu

# smoke: one month of SPY minute bars
.venv\Scripts\python.exe -m src.data.alpaca_backfill --symbols SPY --timeframe 1Min --start 2026-06-01 --end 2026-07-01
```

## Viewer

```bash
pip install flask
python -m ui.app        # http://127.0.0.1:8101
```

A read-only page over the gate verdicts: every experiment × horizon, with its
DSR ratio, PBO, profit factor and sample size, pass or fail marked.

**It works on a fresh clone.** The repo ships a committed snapshot of the bench
flag at `ui/snapshot/alpaca_gpu_state.json`, so you see the real results without
market data, an API key, or a GPU. If you have run the bench yourself, the viewer
prefers your live `journal/flags/alpaca_gpu_state.json`. The header states which
of the two it is rendering, and as of when.

Binds `127.0.0.1` only, exposes no POST route, wired to nothing that acts.

## Status (2026-07-17)

- Data spine complete: 22/22 symbols at 1Min + 1Hour + 1Day (incl. BTCUSD via
  crypto client); 89k news articles 2020->now, all FinBERT-scored.
- Five pre-registered batteries run on the core-7 1Min panel, quarterly
  walk-forward, 2022Q3->2025Q4 (2026 = untouched holdout):

  | battery | model | verdict |
  |---|---|---|
  | B01_baselines_core7 | logreg + RF | FAIL -> SHADOW (all horizons) |
  | B02_xgb_core7 | XGBoost-CUDA, 12-combo grid | FAIL -> SHADOW (0/12) |
  | B03_xgb_cross_asset | + hourly xa_* / daily rg_* context | FAIL -> SHADOW (0/12) |
  | B04_xgb_sentiment | + trailing-60m news sentiment | FAIL -> SHADOW (0/12) |
  | B05_temporal_core7 | 1D-CNN + GRU (seq 60) | FAIL -> SHADOW |

  All honest FAILs — the gate doing its job. Full verdicts in
  `journal/RESULTS.md` + `journal/scorecards/`.
- Flag file `journal/flags/alpaca_gpu_state.json` publishes SHADOW state;
  `AlpacaGpuDaily` task refreshes data/news/flag daily at 17:45.
- P10 (full-22 1Min training + one-shot 2026 holdout read) deliberately
  deferred; the holdout lock stays closed.

## Current state (flag as of 2026-09-13)

The bench has grown past the five batteries above; the shipped snapshot and the
viewer show the current picture:

- **34 experiments**, 102 experiment × horizon cells evaluated.
- **0 promoted**, and **0 of 102 cells pass** the current gate
  (`PBO < 0.5` **and** `Deflated Sharpe ratio > 1.645`). A clean sweep of
  failures.

### A stale-verdict bug worth knowing about

Two cells in the shipped flag file are still **marked `passes: true`**:

| cell | DSR ratio | PBO | profit factor | n |
|---|---:|---:|---:|---:|
| `B06_direction_rank_1Hour` h=5 | 0.373 | 0.290 | 1.049 | 376 |
| `B06_direction_rank_4Hour` h=5 | 0.049 | 0.484 | 1.013 | 83 |

Both **fail** the current bar — 0.373 and 0.049 are nowhere near 1.645. Those
verdicts were computed when the gate asked only for `ratio > 0`, and the bar was
raised on 2026-07-30 without those cells being re-evaluated. The stale `passes`
values are still sitting in the flag.

Why the bar moved is the interesting part: a correctly computed `ratio > 0` is
only a **median test**, which best-of-8 pure noise clears about 45% of the time.
It was never a meaningful gate. 1.645 is the one-sided 5% normal critical value.

The viewer therefore **recomputes every verdict** from the stored statistics
against the current thresholds instead of reading the `passes` field, and tags
any disagreement `superseded`. Reading that field directly would republish a
verdict the project has already retired — which is exactly the class of silent,
measured-half defect this bench keeps finding in itself.
