# alpaca_gpu_lab — HANDOFF (read first)

Standalone GPU alpha-research bench over a 22-asset macro universe (Alpaca ETF
proxies + BTC). Research/advisory only, no execution, no TradingView/webhook.
Everything gated by the canonical overfit gate (imported from macro_gpu_lab).
This file is the "start a new chat here" pointer. Full findings write-up:
`docs/RESEARCH_FINDINGS.md` (also ingested into research_rag).

## Where things stand (2026-07-20)

**Phase 1 (P0–P9) — built + committed.** Data spine (22/22 symbols at
1Min/1Hour/1Day incl. BTCUSD), hive parquet, polars features, intraday
triple-barrier labels, experiment registry + n_trials ledger, five batteries
(B01 baselines, B02 xgb-cuda, B03 +cross-asset, B04 +news-sentiment, B05
temporal). **All FAIL→SHADOW** — the correct-gate-fails-first result. 88k
Alpaca news articles FinBERT-scored (news added nothing to direction).

**Phase 2 (chart-focused relaunch) — built + committed, the real findings.**
News decoupled (`enrich`=chart-only, separate `--sentiment`). Harness upgraded
(`src/models/evaluate.py`): rank-IC logging, rank-threshold acceptance,
magnitude/regression target, cross-sectional long-short, precomputed-score
(`score_col`) path. Batteries take `--timeframe` (1/5/15Min/1Hour/4Hour).
Three findings (all still SHADOW — none clear the full gate, but real signal):
- **B06 direction** — weak, nonlinear, short-horizon (1Min xgb rank-IC 0.085
  t=8.8; decays by 15Min; linear flat; PF<1 after cost).
- **B07 magnitude** — THE standout: predicting excursion (mfe) is ~10× more
  predictable than direction. 5Min rank-IC **0.18–0.20, t up to 32**, every
  fold positive, strong at all timeframes. PF<1 only because harvest overlay
  is long-directional — monetize non-directionally.
- **B08 cross-sectional** — 1Hour long-short rank-IC 0.107–0.120 (t~5–6),
  **PF 1.08–1.11 after cost** (first PF>1 with real n); flat intraday-fast.

**Kronos (B09/B10/B11) — SETTLED NEGATIVE, then UNWIRED.** Fully tested through
the gate: zero-shot magnitude 1Hour rank-IC ~0 (vs XGB 0.10 t=4.8 same target),
fine-tuned (`SPY_60m_730d`) no better (IC −0.06/0/−0.07). A generative OHLCV
foundation model does not beat GBDT here. Verdict + numbers in
`docs/RESEARCH_FINDINGS.md` §4; scorecards kept in `journal/scorecards/*kronos*`
as evidence. The wiring was **removed** from this bench (kronos_signals.py,
drivers, forecast data, registrations) 2026-07-22 — Kronos never succeeded at
anything we used it for. The Kronos repo itself is untouched. The general
`score_col` precomputed-signal path stays in `evaluate.py` (reusable for any
external signal). Do not re-add Kronos without a new reason.

## Commits (this repo, main)
P0 4b8f01d · P1+P2 7ecffeb · P3+P4 f02cab0 · P5-P9 0da5bb5 · B01-B03 174629c ·
P7+P9 3aa2bb3 · Phase2 harness+B06 89046bc · Phase2 matrix da2e47d · Kronos
wiring f052761 · Kronos gate fix cf9e176.

## Do NOT redo
- The five Phase-1 batteries + the four Phase-2 timeframe sweeps (results in
  `journal/scorecards/` + `journal/RESULTS.md`; read via `scripts/_show_ic.py`).
- macro_gpu_lab daily 22-asset models + qlib_lab's ~40 factor/lead-lag
  batteries — all exhausted (see docs/RESEARCH_FINDINGS.md).
- Dense per-bar Kronos forecasting — infeasible on the 3050 (memory:
  probe-throughput-before-long-runs).

## Highest-value next work (pick up here)
1. **Monetize magnitude (B07).** Turn the excursion forecast into a
   vol-targeting / position-sizing layer or a straddle-style non-directional
   PnL, then gate THAT. This is the strongest signal we have and it is
   currently un-monetized.
2. **Scale cross-sectional (B08) to the full 22 at 1Hour.** More names → a
   deeper cross-section; 7 names is thin and it already reached PF>1.
3. **Meta-label the XGB signals** (López de Prado) — a second model deciding
   take/skip on B06/B07 entries, gated.

## Env (this box) — read before running
- `.venv\Scripts\python.exe` (bare python = Store stub). Installs: uv
  `--native-tls`, torch cu121 index. git: per-commit identity, commit to main,
  Norton locks `.git/objects` → retry. Sessions bounce often → long runs MUST
  checkpoint granularly. Load skills `win-quant-env`, `quant-research-gate`,
  `paper-trading-guardrails`. Verify: `.venv\Scripts\python.exe -m pytest`.
