# B07 magnitude: why it cannot be monetized, computed before modelling

**Date:** 2026-08-07 · **Verdict:** ABANDONED at the cost-hurdle stage · **Trials charged:** 0

`HANDOFF.md` and `docs/RESEARCH_FINDINGS.md` §7 both name B07 as restart priority #1: the
strongest raw signal on this box, failing only on profit factor because the test overlay was
long-directional, with the un-built fix being "a non-directional monetization". This note
computes the hurdle that monetization would have to clear. It does not clear it. No model was
trained and no trial was charged, which is the point of computing a hurdle first.

## The signal is real — that is not in dispute

`journal/scorecards/B07_magnitude_1Hour_2026-07-18.json`, 1Hour h5:

| | |
|---|---|
| rank-IC | **0.097**, t **4.794** |
| fold hit rate | **0.90** — 90% of walk-forward folds positive |
| PBO | 0.000 |

And the long-directional overlay loses **significantly**, which is the expected result rather
than a disappointment: PF 0.688, expectancy −$0.00057/trade, bootstrap Sharpe CI
[−0.2595, −0.0306] entirely below zero, permutation p 0.9932.

## What the target actually is

`src/features/build.py:219` — `mfe_{h} = (max high over i+1..i+h − close) / ATR`.

Two consequences that were not stated when B07 was called a "magnitude" signal:

1. **It is one-sided.** MFE is the UPSIDE excursion; `mae_{h}` is the downside. B07 predicted
   half the range, not `|move|`.
2. **It is already ATR-normalised.** So an IC of 0.097 is not "predicts volatility" — current
   ATR is divided out. It is predicting excursion *in excess of what current ATR implies*,
   i.e. forecasting a RISE in realised volatility.

A forecast of rising realised vol has exactly one non-directional expression: long volatility.
Which means the hurdle is the variance risk premium.

## The hurdle

SPY vs VIX, 2016-08 → 2026-08, forward 1-week realised vol against contemporaneous implied
(n=2,508):

| | |
|---|---|
| mean implied | 18.61 |
| mean realised | 13.99 |
| **VRP** | **+4.62 vol points** |
| days implied > realised | **83%** |

A long-vol position starts 4.62 vol points underwater. The signal must forecast realised
beating **implied**, not merely beating ATR.

Target for a long-vol trade is `realised − implied`: mean −4.62, sd 8.48, P(>0) = 16.7%.
Under top-decile selection, `E[z | top 10%] = 1.755`, so the conditional mean is
`μ + IC · 1.755 · σ`:

| IC | top-decile lift | conditional mean | |
|---|---|---|---|
| 0.097 *(B07 measured)* | +1.44 | **−3.18** | still loses |
| 0.150 | +2.23 | −2.39 | still loses |
| 0.200 *(B07 best ever, 5Min)* | +2.98 | −1.64 | still loses |
| 0.300 | +4.47 | −0.16 | still loses |
| 0.500 | +7.44 | +2.82 | profitable |

**Break-even IC = 0.310, before paying any option spread.** B07 measured 0.097 — short by
0.213, i.e. it would need to be **3.2x better than it is**. Its best figure anywhere (0.20 at
5Min) still loses, and a 5-minute horizon cannot be expressed in options regardless.

This is an order-of-magnitude hurdle, not a backtest: it assumes top-decile selection, a
normal approximation, and SPY/VIX as the vol pair. Being 3.2x short is not a marginal miss.

## The one constructive finding

The same arithmetic run the other way is favourable. Selling vol on predicted-CALM days
(bottom decile) works WITH the premium instead of against it:

- unconditional premium captured: 4.62 vol points
- bottom decile at IC 0.097: **6.07** vol points — **31% more premium per trade**

Applied to the options desk's own numbers (modeled edge $12.49/trade), that is **+$3.90/trade**
against a **measured spread cost of $27.14/trade**. It closes **14%** of the gap and **does not
flip the book**. Worth knowing, not worth building on its own — the desk's problem was never
signal quality, it was the spread.

## What this closes

B07 stays FAIL. The un-built "non-directional monetization" named as priority #1 is now
costed and should be struck from the restart list rather than left as an open lead. If it is
ever revisited, the number to beat is **IC 0.310**, and the honest first question is whether
any magnitude model on this box has ever come within 3x of it.
