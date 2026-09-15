# alpaca_gpu_lab — agent guide

Standalone GPU **alpha-research bench** (RTX 3050, torch cu121 + XGBoost CUDA):
Alpaca data -> hive Parquet -> polars features -> intraday triple-barrier labels ->
models behind the canonical overfit gate -> **shadow flag-file**. Mission: find
signals/patterns and cross-asset relationships (positive AND negative correlations)
across the 22-asset macro universe. **Research / advisory only.** Nothing here
places orders, and there is NO TradingView/webhook wiring — output is a flag file
other repos *may* read, gated by `ALPACA_GPU_ENFORCE` (default `no`).

Skills: load `quant-research-gate` before any model/backtest work,
`win-quant-env` before installs/git/PS/Task Scheduler,
`paper-trading-guardrails` before touching flag-file/publish code.

## Hard rules

1. **Data-only Alpaca.** Keys in gitignored `.env` (`APCA_API_KEY_ID`,
   `APCA_API_SECRET_KEY`) are for the market-data + news clients, never the
   trading client. No live execution, no order routing, ever.
2. **The overfit gate is imported, never re-ported.** `src/gate.py` re-exports
   `evaluate_gate`/`pbo_cscv`/`deflated_sharpe`/`purged_kfold`/
   `walk_forward_splits` from `macro_gpu_lab/macro_gpu_lab/validate.py`
   (path override: `MACRO_GPU_LAB_DIR`). If that import breaks, fix the path —
   do not vendor or rewrite the math.
3. **Shadow until the gate clears** (PBO < 0.5 AND **Deflated-Sharpe ratio > 1.645**
   on EVERY horizon). A correct gate usually FAILS the first model — report that
   honestly; never soften thresholds to "pass" something. The bar was raised from
   0.0 on 2026-07-30 with the DSR unit fix: a correct `ratio > 0` is a MEDIAN test
   that best-of-8 noise clears 44.8% of the time. `macro_gpu_lab.config` is
   authoritative. **Nothing on this box has ever cleared it.**
4. **Pre-registration.** Every hypothesis battery is registered in
   `journal/experiments/` BEFORE results exist; every executed (model, params,
   horizon) combo lands in the ledger; the cumulative ledger count is the
   `n_trials` fed to `deflated_sharpe`. No unregistered evaluations.
5. **2026 is the locked holdout.** The dataset loader clips >= `HOLDOUT_START`
   by default; unlocking requires code + env
   (`ALPACA_GPU_HOLDOUT_UNLOCK=yes`) and happens exactly once, at the end.
6. **LLM/sentiment never on any hot path.** News scoring is strictly offline
   batch -> precomputed numeric features; missing news = neutral 0.
7. **Do not re-run the exhausted daily batteries.** macro_gpu_lab (RF/MLP/
   shift/surprise) and qlib_lab (~40 registered batteries: TSMOM, VRP,
   vol-managed, COT, lead-lag miner, ...) all honestly FAIL->SHADOW at daily
   horizons. Near-misses (VRP, vol-managed, net-liquidity, VIX term structure)
   re-enter ONLY as regime-context features for intraday models.

## Data reality

- Alpaca free plan = **IEX feed** (partial volume). Record `feed` in every
  experiment. `--feed sip` only with the subscription.
- No indices/FX spot on Alpaca -> the 22-asset universe uses ETF proxies
  (`src/config.py::UNIVERSE_22`); FXY/FXC are **sign-inverted** vs USDJPY/
  USDCAD; VIXY has roll drag (feature, never a level); BTC via the crypto
  client (partition `symbol=BTCUSD`).
- Thin tickers (FX*, CPER, BIL) = hourly-granularity context assets.
- Storage: Parquet, `raw/bars_<tf>/symbol=X/year=Y/month=M/`. Never commit
  `market_data/` (gitignored, regenerable). `journal/` IS committed.
- **No futures** (no CME MNQ/ES). News API history back to ~2015.

## GPU discipline (RTX 3050, 6 GB)

- GPU earns its keep on **model training + parameter search + news scoring**,
  NOT feature engineering at current sizes — polars on CPU until genuinely at
  millions of rows x many symbols (WSL2/RAPIDS deliberately deferred).
- XGBoost: `tree_method="hist", device="cuda"`, float32, QuantileDMatrix per
  fold; never the whole panel in VRAM.
- The 3050 is shared with Kronos / macro_gpu_lab / the hq LLM sidecar — check
  `nvidia-smi` before launching a training job.


### GPU numeric mode (`gpu.tune_backend()`, added 2026-09-01)

`get_device()` now enables TF32 on CUDA. Measured on this box's RTX 3050 idle at
full clock, 4096x4096 matmul, median of 3: fp32 1.43 -> TF32 2.62 TFLOPS (**1.84x**).
Only `matmul.allow_tf32` actually changes: torch already defaults
`cudnn.allow_tf32` to True, so conv/RNN layers were always running TF32 --
do **not** read this change as "the conv results moved".

| env var | default | effect |
| --- | --- | --- |
| `GPU_TF32` | on | `0` restores the previous precision setting. NOT bit-exact reproduction -- that also needs `cudnn.benchmark` pinned, `torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and the same torch/CUDA/driver build. |
| `GPU_MEM_FRACTION` | 0.92 | `0` lifts the per-process VRAM cap. The cap exists so an oversized allocation raises `OutOfMemoryError` instead of silently spilling into system RAM through the Windows driver's sysmem fallback and running 10-50x slower. It caps against **total** VRAM, not free -- with ollama holding ~5.9 of 6 GB the driver still binds first. |

`confidence()` stamps the returned state under `"tuning"`, so a recorded number
says which numeric mode produced it. Keep that stamp: without it an old row and a
new one are indistinguishable in the ledger.

**Do not re-run a battery merely to compare TF32 against non-TF32.**
`registry.log_trial` appends to the ledger and `n_trials` is cumulative, so a
curiosity re-run permanently tightens the deflated-Sharpe threshold for that
family. Check precision on a scratch script, never through the harness.

## Env (this box)

- uv-managed `.venv` (py3.12); real interpreter `.venv\Scripts\python.exe`
  (bare `python` = Store stub). Installs: `pwsh scripts/setup_venv.ps1`
  (uv `--native-tls`, torch from the cu121 index).
- git: per-commit identity (`git -c user.email=... -c user.name=...`), commit
  to `main`, Norton locks `.git/objects` transiently -> retry.

## Verification

- Tests: `.venv\Scripts\python.exe -m pytest`
- CUDA smoke: `.venv\Scripts\python.exe -m src.gpu`
- Backfill smoke: `.venv\Scripts\python.exe -m src.data.alpaca_backfill --symbols SPY --timeframe 1Min --start 2026-06-01 --end 2026-07-01`
- Coverage report: `.venv\Scripts\python.exe -m src.data.coverage`
