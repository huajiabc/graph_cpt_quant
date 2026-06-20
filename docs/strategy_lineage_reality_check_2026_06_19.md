# Strategy Lineage Reality Check - 2026-06-19

This note answers a simple but important question:

Did the later research actually improve the system, or did the earliest version look better?

Short answer: both are partly true.

The earliest MIR1/CIC discoveries had the cleanest historical single-trade edge. Later work did not keep compounding that edge upward. Instead, most later work did three things:

1. Converted a good-looking signal into a capacity-aware basket.
2. Added risk controls that reduce losses but do not create new alpha.
3. Rejected many attractive but non-robust add-ons.

So the later stack is more controlled, but not obviously more profitable in forward/live evidence yet.

## Key Evidence

### 1. Early MIR1 Was Real, But Raw MIR1 Was Not The Final Strategy

In the early 1m execution validation:

- MIR1, 184 trades, net20 expectancy: +1.1020%.
- IR2 reference, 196 trades, net20 expectancy: +0.5012%.

This is the original "wow" phase.

But in the later CIC integration audit, raw MIR1 was weaker under the stricter comparable stack:

- MIR1 reference, 249 trades, net20: +0.4581%.
- CIC1 beta extreme, 135 trades, net20: +1.1948%.
- CIC2 beta broad, 190 trades, net20: +1.0333%.

The important decomposition:

- MIR1 only ex CIC1, 80 trades, net20: -1.0578%.
- MIR1 intersect CIC1, 135 trades, net20: +1.1948%.
- MIR1 only ex CIC2, 13 trades, net20: -1.1825%.
- MIR1 intersect CIC2, 190 trades, net20: +1.0333%.

Conclusion: the early signal was not "MIR1 raw forever." The real object became CIC-filtered MIR1.

## 2. Later Work Improved The Definition, Not The Headline Edge

The good historical long stack became:

- P2 max8 baseline: roughly +10.9% full-period portfolio net20.
- P2 max8 + O6: roughly +12.2%.
- P2 max8 + CP60 + O6: roughly +12.6%.
- P2 max8 + Protect_A cap2 + O6: roughly +13.3%.

So historically, later modules did add incremental improvement.

But the improvement type matters:

- O6 is an additive overflow sleeve, not core signal alpha.
- CP60 is weak-position pruning, not entry alpha.
- Protect_A cap2 reduces CP60 false exits, but forward sample is not enough.
- Failure risk-off improves drawdown in historical replay, but remains shadow.

These are system-management improvements. They do not recreate the simple early feeling of "one clean signal prints money."

## 3. Holdout And Forward Evidence Are Much Less Pretty

The v2.1 holdout period was bad:

- B0 P2 max8 holdout net20: -2.4595%.
- B3 P2 max8 + CP60 + O6 holdout net20: -1.4827%.
- B4 P2 max8 + Protect_A cap2 + O6 holdout net20: -1.4827%.

This is not a win. It is a loss reduction.

The current forward/paper sample is also insufficient and negative:

- Current local primary paper-live trade count: 1.
- Latest primary trade: FARTCOINUSDT, SL, net20 -3.4000%.
- Current v2.4 forward stack sample: 12 trades.
- P2 max8 forward net20: -2.4595%.
- P2 max8 + CP60 forward net20: -1.4827%.

Conclusion: later stack management has reduced drawdown/loss in the weak sample, but it has not yet proven positive forward profitability.

## 4. Why It Feels Like Later Work Got Worse

The feeling is rational for four reasons.

First, the early numbers were mostly single-trade or historical validation metrics. Later numbers include max positions, skipped trades, forward windows, execution stress, component samples, and real data staleness checks.

Second, many later modules were not alpha modules. CP60, Protect_A, failure risk-off, low-coimpulse, and risk envelope work are defensive.

Third, most new information sources failed as deployable selectors:

- Static orderbook ranking failed.
- 15m orderflow ranking failed.
- Cross-exchange A7 was month-concentrated.
- Binance taker-buy fusion was diagnostic only.
- Perp crowding RV failed strict pair cost.
- Narrative sector labels did not beat random sector.
- Token-level on-chain attention lacks enough event overlap.

Fourth, the forward/holdout regime is weaker than the historical discovery window. The later stack has not disproved that problem.

## 5. What Later Work Actually Bought Us

The later work bought four useful things:

1. A better core object:
   - CIC-filtered MIR1 / P2 max8, not MIR1 raw.

2. A capacity interpretation:
   - CIC is a basket/burst alpha, not a top3/top5 selector.

3. A defensive management layer:
   - CP60 reduces weak non-follow-through losses.
   - Protect_A cap2 is the best CP60 refinement candidate.
   - Failure risk-off is a plausible no-long overlay.

4. A clean rejection list:
   - A large number of tempting paths are now explicitly diagnostic/rejected.

That is valuable, but it is not the same as a stronger live-ready alpha.

## 6. Current Honest Status

The current long stack should be treated as:

- Research-to-forward candidate, not proven engine.
- Paper/shadow only.
- Real-live disabled.
- Promotion blocked by insufficient forward sample and execution realism.

Current best classification:

| Component | Status | Reason |
|---|---|---|
| P2 max8 | Core forward baseline | Best historical basket, but forward sample weak |
| O6 | Shadow additive sleeve | Historical lift, no current live triggers |
| CP60 | Shadow weak-position pruning | Reduces losses, needs more CP exits |
| Protect_A cap2 | Research-improved shadow | Offline stable, no forward protected sample yet |
| Failure risk-off | Shadow/diagnostic | Improves historical risk, not active gate |
| Low-coimpulse router | Diagnostic | Explains weak regimes, not action-ready |
| Orderflow/orderbook/crowding/v4/v6 | Diagnostic only | No deployable selector yet |

## 7. Practical Decision

Do not keep expanding new historical rules just to recover the early high.

Instead:

1. Freeze the long stack to a small benchmark set:
   - S0: P2 max8.
   - S3: P2 max8 + CP60 + O6.
   - S5: P2 max8 + Protect_A cap2 + O6.

2. Keep only diagnostic logging for:
   - low-coimpulse score.
   - failure risk-off.
   - orderflow/orderbook.
   - on-chain attention.
   - crowding.

3. Judge only after forward thresholds:
   - core trades >= 100.
   - CP60 exits >= 50.
   - protected exits >= 30.
   - overflow trades >= 30.

4. Add a demotion rule:
   - if core trades >= 100 and P2/S3/S5 net20 remains <= 0, demote the stack from paper candidate to research-only.

## Bottom Line

The earliest work found the alpha seed.

The middle work proved that the seed is not MIR1 raw, but CIC/P2 basket continuation.

The later work has mostly been risk control, rejection, and infrastructure. It has not yet produced a better forward-proven alpha than the original discovery.

So the correct emotional read is:

"We did not waste the work, but we also should stop pretending every added module made the strategy better."

The right next move is not more cleverness. It is freeze, forward sample, and be willing to demote if the live distribution does not confirm the historical one.
