# v11.9 BTC Variance Risk Premium Findings

Date: 2026-07-15

Verdict: `reject_static_short_variance`. This is a normalized variance proxy,
not an account-return backtest. No PaperLive, live, sizing, or leverage
permission changed.

## Result after full-history backfill

The archive was extended to the start of DVOL history, producing 63 monthly,
approximately non-overlapping 30-day labels.

| Candidate | Full n | Full net 1 vol | Validation | Holdout | Full net 2 vol | Worst month | Random-IV percentile |
|---|---:|---:|---:|---:|---:|---:|---:|
| Monthly short variance | 63 | +3.43% | -0.82% | -9.73% | -0.13% | -266.53% | 100% |
| Short only when IV exceeds trailing RV by 5 points | 35 | +7.04% | +43.87% (n=1) | -44.15% (n=5) | +3.54% | -116.52% | 100% |

The unconditional bootstrap interval for net one-vol payoff was -11.65% to
+16.82%. Its normalized maximum drawdown was 266.53%. The rich-IV branch had a
-10.12% to +22.90% interval and 276.76% maximum drawdown.

## What is real

Contemporaneous DVOL is informative: pairing each realized-volatility outcome
with its actual DVOL beat every random-IV permutation. A one-month stale DVOL
strike was materially worse.

That information is not the same as excess return. Development earned a
positive short-variance premium, while both validation and holdout rejected
the unconditional trade. In the recent lower-IV regime, February and June
2026 realized volatility expanded far beyond the initial strike and erased
many ordinary premium months.

## Interpretation

DVOL prices future variance well enough to be useful as a state variable, but
the remaining average premium is unstable and dominated by convex tail loss.
Leverage is specifically contraindicated by this payoff shape.
