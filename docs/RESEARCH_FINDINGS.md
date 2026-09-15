# What the GPU Labs Learned: Predictable Magnitude, Unpredictable Direction

**A synthesis of intraday & daily alpha research across `alpaca_gpu_lab`,
`macro_gpu_lab`, and `qlib_lab` (RTX 3050, ~June–July 2026).**

Research/advisory only. Every claim below is gated by the same overfit gate
(PBO < 0.5 AND Deflated-Sharpe ratio > 0, purged/embargoed walk-forward). No
strategy here is live; none has cleared the gate. This document deliberately
**excludes the TradingView/Pine hand-built strategies** — those are not the
research findings and were never the good signals. Everything below is the
GPU-driven, gate-validated research.

---

## 1. Setup

- **Universe (22):** ETF proxies for the macro forces — rates (IEF/TLT/TIP),
  dollar (UUP/FXE/FXY/FXB/FXC), China (FXI), equities (SPY/QQQ/IWM/EEM/XLF),
  vol (VIXY), commodities (GLD/USO/CPER), credit (HYG/LQD), liquidity (BIL),
  crypto (BTC). Alpaca IEX feed. Core-7 (SPY/QQQ/IWM/BND + IEF/TLT/VIXY) at
  1-minute is the intraday workhorse.
- **Data:** 1Min/1Hour/1Day full (2019–2025 depending on TF), 2026 locked as
  an untouched holdout. 88k Alpaca news articles FinBERT-scored.
- **Targets:** intraday triple-barrier (5/15/30-bar TP/SL in ATR units),
  forward return, max favorable/adverse excursion (MFE/MAE), realized
  barrier-exit PnL net of fractional round-trip costs. Deliberately NO "next
  candle up" target.
- **Validation:** quarterly expanding walk-forward with a horizon gap; PnL
  pooled to one observation per non-overlapping entry (no per-asset
  pseudo-replication); gate deflates Sharpe by the cumulative pre-registered
  trial count. Rank-IC (Spearman of prediction vs. outcome, per fold) is the
  upstream "is there any signal" diagnostic.
- **Models:** logistic regression, RandomForest, XGBoost-CUDA, small torch
  temporal nets (1D-CNN/GRU), and the Kronos OHLCV foundation model.

---

## 2. The headline finding

> **On this universe, *how far* price moves is dramatically more predictable
> than *which way* it moves — and the predictability is strongest at the
> shortest intraday horizons.**

Predicting excursion (MFE, "magnitude") from chart features yields rank-IC
**0.18–0.20 at 5-minute (t up to 32), positive in every walk-forward fold**,
and stays meaningfully positive across all timeframes. Predicting direction
from the *same features* yields at best rank-IC 0.085 (1-minute), decaying to
noise by 15 minutes. Magnitude is ~10× the effect size of direction.

This is the classic volatility-is-forecastable / sign-is-a-coin-flip result,
but measured cleanly on this universe with a deflating gate — and it reframes
where tradeable structure actually lives: **in size/volatility and in
relative-value, not in single-name directional bets.**

---

## 3. What we found (positive results, all still SHADOW)

### 3.1 Magnitude is predictable (alpaca_gpu_lab B07)
XGBoost regressor on chart features → MFE excursion.

| timeframe | rank-IC (h5) | t-stat | folds positive |
|---|---|---|---|
| 5Min  | 0.18–0.20 | up to 32 | 100% |
| 15Min | 0.10 | 7.9 | 100% |
| 1Hour | 0.10 | 4.8 | 90% |
| 4Hour | 0.10 | 2.7 | 80% |

Not yet profitable **because the test overlay was long-directional** (knowing a
big move is coming does not say up or down). The forecast is real; the
monetization must be non-directional (sizing, vol targeting, straddles). This
is the single most promising un-exploited result.

### 3.2 Cross-sectional relative value works at hourly horizons (B08)
Per-stamp long-top / short-bottom across the core names.

| timeframe | horizon | rank-IC | t | PF after cost |
|---|---|---|---|---|
| 1Hour | 15 | 0.107 | 6.0 | **1.08** |
| 1Hour | 30 | 0.120 | 5.0 | **1.11** |
| 4Hour | 15 | 0.135 | 2.8 | 1.68 (small n) |

The **first PF>1 after costs with real trade counts** in the whole program.
Flat/negative intraday-fast (5–15Min). "Which name outperforms over the next
15–30 hours" carries signal that single-name direction does not — this is where
the cross-asset relationship features (rolling corr/beta/absorption/lead-lag,
ported from macro_gpu_lab / hq macro brain) earn their keep. Still fails the
full gate (DSR deflation; PF barely > 1), but it is the closest to an edge.

### 3.3 Direction has weak, nonlinear, short-lived signal (B06)
1-minute XGB rank-IC 0.085 (t=8.8, every fold positive), decaying to ~0.025 by
30 bars and to noise above 15Min. **Linear/RF models see nothing** (IC≈0) — the
signal is nonlinear, which is why the linear Phase-1 baselines uniformly failed.
PF < 1 at every horizon: predictable, but eaten by costs. A textbook
"predictable-but-not-profitable-after-costs" microstructure effect.

---

### 3.4 Variance-risk-premium: two linked findings (qlib_lab vol_desk)
Two separate discoveries from the same weekend, both on the Bollerslev variance
risk premium (VRP = option-implied variance − realized). Both **SHADOW**; the
forward-return one is the same VRP near-miss listed in §4.

- **A richness gauge, not a timer.** The vol_desk publishes a SHADOW advisory
  ticket carrying `vrp_pct` — the percentile of how *overpriced* SPY option
  insurance is right now (recent readings 15–66%). The paired eruption/shift
  detector (`p_eruption_5d/21d`) confirms the flip side of §2: we **cannot
  reliably call *when* vol erupts**, so the desk sizes *how rich* insurance is
  and stays `NO_TRADE` absent an edge — never a pop-timing bet.
- **Expensive insurance → higher next-month return.** When institutions overpay
  for downside protection (high VRP month), forward SPY return is higher than in
  cheap-insurance months: deep-history split **1.97% vs 0.38%** mean forward
  return (Welch t=2.1; n_hi=54 / n_lo=182). PF **2.92** (1995–2015) —
  *replicates* out-of-sample. Still **FAILS** the gate (DSR ratio **−0.004**,
  prob 0.498, n=107, n_trials 20), per §5.3.
  - ⚠️ **Superseded 2026-07-30.** This bullet previously read "PF 5.02 (2016–26)
    … Sharpe 0.40 … DSR −17.3". All three came from the 2026-07-13 scored
    series, which **no longer reproduces**: the re-run gives PF 2.16 and Sharpe
    0.211 at *identical* n (107) and n_trials (20). So the effect is not "real
    and regime-robust but eaten by deflation" — it is a result whose underlying
    series changed. Treat the provenance problem, not the verdict, as the
    finding (research_ledger `Q-GATE-03`).

---

## 4. What did NOT work (the negative-result library — do not repeat)

The gate's job is to fail things honestly, and it did. These are exhausted:

**Daily direction on the 22-asset macro universe (macro_gpu_lab).**
RandomForest, torch MLP, a regime-shift detector, and surprise/eruption
overlays — all FAIL at 5-day and 21-day horizons (Deflated-Sharpe ratios −5 to
−18). Every model has negative Sharpe on raw direction. The surprise-overlay
edge is **regime-specific to ~2016–2026 and degrades on older data** (a 15-year
test was worse than 10-year). Top feature importances: SPY 20-day vol, VIX
20-day change, absorption ratio — i.e. the model leans on *volatility state*,
foreshadowing §2.

**Alpha158 + LightGBM and ~40 pre-registered factor batteries (qlib_lab).**
TSMOM, XS-momentum, reversal, low-vol, BAB, 52-week-high, VRP, absorption-shift,
credit-leads, copper/gold, semis-lead, transports, turn-of-month, DXY/gold,
pair-z-reversion, overnight-SPY, vol-managed SPY, Faber SMA, volume-shock,
skew-XS, Amihud illiquidity, range-breakout, Halloween, PCA-residual,
corr-conditioned momentum, VIX-contango, backwardation, skew-follow,
VVIX-stress, pre-FOMC drift, NFP-day, BTC-funding fade, net-liquidity trend, COT
extreme-fade — **all FAIL the gate.** The recurring near-misses (PF > 1.5 or
Sharpe 0.2–0.4 that die under multiple-testing deflation): **VRP** (PF 5.0,
Sharpe 0.40, DSR −17), **vol-managed SPY** (PF 2.18, Sharpe 0.29),
**net-liquidity trend** (PF 2.46, Sharpe 0.23), **VIX term structure** (PF
1.96), **COT-extreme-fade** (PF 1.66). Note: three of the five survivors are
volatility/liquidity-state signals — again pointing at §2.

**Lead-lag mining (qlib_lab).** 22,800 pairwise tests, Bonferroni-corrected, 25
split-half survivors (e.g. LQD→MUB, EMB→MUB, EWC→INDA, many →INDA at lag 1) —
flagged SHADOW pending review, used only as *features*, never promoted.

**Deep-history replication (qlib_lab, 1995–2015).** VRP, absorption-shift,
copper/gold, turn-of-month, vol-managed, Faber, Halloween all *replicate* (edge
present in old out-of-sample data too, PF > 1.3) — yet still `gate.passes =
false`. DXY/gold is regime-specific. The strongest meta-signal: several effects
are **real but not gate-clearing** — the bottleneck is deflation, not absence.

**Intraday news sentiment (alpaca_gpu_lab B04).** 88k FinBERT-scored Alpaca
articles → per-bar sentiment features. Added nothing to intraday direction.

**Kronos foundation model — zero-shot AND fine-tuned (B09/B10/B11, resolved
2026-07-22).** Graded through the same gate as everything else (precomputed
forecast → rank-IC / acceptance / PBO-DSR). Verdict: **no usable predictive
signal, and fine-tuning does not help.**
- Zero-shot magnitude (B10), 1Hour, healthy n≈105: rank-IC 0.04 / −0.03 / −0.01
  (t ≈ 1.0 / −0.6 / −0.4) — indistinguishable from zero, where a small XGBoost
  (B07) reaches IC 0.10 at t=4.8 on the *same* 1Hour magnitude target. 4Hour
  similarly weak/insignificant.
- Zero-shot direction (B09) is degenerate cross-sectionally (sparse per-symbol
  forecasts rarely align ≥3 names/stamp → n≈2–11): no verdict, but no signal.
- **Fine-tuned** (B11: `SPY_60m_730d`, a full fine-tune on SPY 1-hour),
  magnitude, n≈39: rank-IC −0.06 / 0.00 / −0.07 (t ≈ −1.2 / 0 / −1.0) — **no
  better than, arguably worse than, zero-shot.** PF>1 appears only on
  noise-level n.

The clean takeaway: **a generative OHLCV foundation model — base or fine-tuned —
does not read these charts better than a purpose-trained gradient-boosting model
on engineered features.** Consistent with the earlier copper zero-shot failure.
Autoregressive forecasting is also ~1000× more expensive (≈0.9 s/forecast vs a
batched XGB predict). For this problem, feature-engineering + GBDT dominates.
_Caveat: this tests forecast-derived point signals (predicted close / range),
not Kronos's full predictive distribution, and only SPY for the fine-tune._

---

## 5. Cross-cutting lessons (methodology)

1. **Magnitude ≫ direction.** The most reusable finding. Volatility/excursion
   is forecastable where sign is not; monetize it non-directionally.
2. **Nonlinearity is required.** XGBoost repeatedly sees structure linear/RF
   models miss. Uniform linear-baseline failure ≠ no signal.
3. **The enemy is multiple testing, not absence of edge.** Many effects show
   PF > 1.5 and replicate out-of-sample yet fail the Deflated Sharpe once you
   pay for every look. The pre-registered n_trials ledger is what keeps this
   honest — and what kills most "discoveries."
4. **Acceptance rule matters as much as the model.** A fixed absolute
   probability threshold (p ≥ 0.60) is *vacuous* for calibrated models whose
   outputs sit in a 0.40–0.55 band — it makes verdicts 0-trade non-answers.
   Rank-based acceptance (trade the top decile per fold) is required to even
   ask the question. Rank-IC separates "no signal" from "signal, wrong
   monetization."
5. **Horizon structure.** Intraday predictability concentrates at the shortest
   horizons and decays fast (direction); magnitude persists longer.
   Cross-sectional inverts — it needs hourly+, and is noise intraday-fast.
6. **Costs are the wall for intraday direction.** IC > 0 with PF < 1 is the
   norm; the edge is real and sub-cost. This is exactly where a lower-cost,
   leveraged instrument (e.g. micro futures on the same underlyings) could flip
   the sign — untested, but the natural bridge.
7. **Regime-specificity is pervasive.** Edges present 2016–2026 weaken pre-2016;
   treat any single-regime result as provisional.
8. **Operational:** probe throughput before long GPU runs; checkpoint granularly
   (this box bounces sessions and kills detached jobs).

---

## 6. What we have NOT found (open questions)

- **No standalone strategy clears the full gate.** Everything is SHADOW.
- **No profitable-after-cost intraday directional signal.**
- **Magnitude is un-monetized** — no vol/options/sizing framework built on the
  B07 forecast yet.
- **Cross-sectional hourly is not gate-cleared** — PF barely > 1, DSR fails; it
  has only been tested on 7 names.
- **Kronos is settled — negative.** Zero-shot and fine-tuned both show ~0
  rank-IC (§4); a foundation model does not beat GBDT here. Only the full
  predictive *distribution* (not point forecasts) and non-SPY fine-tunes remain
  untested, but the expected value of chasing them is now low.
- **Full-22 1-minute modeling deferred; the 2026 holdout is untouched.**

---

## 7. Build on top of this (next research)

1. **Monetize magnitude.** Convert the B07 MFE forecast into a vol-targeting /
   dynamic-sizing overlay or a non-directional (straddle-like) PnL; gate that.
   Highest expected value — it is the strongest signal and currently wasted.
2. **Scale the cross-section.** Run B08 on the full 22 assets at 1Hour+; more
   names deepen the ranking. It already reached PF > 1 on 7.
3. ~~Finish + fine-tune Kronos.~~ **DONE — negative (§4).** Zero-shot and
   fine-tuned (`SPY_60m_730d`) both ~0 IC; deprioritize the foundation-model
   route. (Original note retained below for provenance.)
   The fine-tuned models were the real test of "can a foundation model
   read these charts."
4. **Meta-label the XGB signals** (López de Prado): a second model deciding
   take/skip/size on B06/B07 entries, using the magnitude forecast as a
   sizing input — directly attacks the IC>0 / PF<1 gap.
5. **Combine the two edges:** magnitude-sized, cross-sectionally-ranked hourly
   book. Two independent positive signals stacked.
6. **Micro-futures bridge:** revalidate the short-horizon signals on
   lower-cost/leveraged instruments (separate data source; Alpaca has no CME) —
   the cost wall is the only thing between §3.3 and a live edge.

---

## 8. Provenance

- `alpaca_gpu_lab` — intraday bench; scorecards `journal/scorecards/*.json`,
  human log `journal/RESULTS.md`, reader `scripts/_show_ic.py`. Batteries B01–B10.
- `macro_gpu_lab` — daily 22-asset bench; canonical overfit gate
  (`macro_gpu_lab/validate.py`) imported by every repo, never re-ported.
- `qlib_lab` — Alpha158 + factor/lead-lag batteries; scorecards under
  `qlib_lab/journal/`.
- `hq-trading-system/analytics/macro_engine.py` — the cross-asset relationship
  engine (corr/beta/divergence/lead-lag/partial-corr/absorption) whose math the
  intraday cross-asset features extend.

_Generated 2026-07-20 as a research synthesis for the finance-paper RAG
(`research_rag`). Not investment advice; paper/advisory research only._
