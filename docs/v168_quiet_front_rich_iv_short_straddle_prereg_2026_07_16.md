# v16.8 Quiet Alt Front + Rich IV Short-Straddle Preregistration

Date frozen: 2026-07-16, after v16.7 and its mechanism diagnostics, but before
constructing or inspecting any executable short-straddle return.

## Adaptive disclosure and distinct question

This is explicitly result-informed. v16.7 showed that the continuous alt-minus-BTC
volatility front correlates with the next day's BTC realized-volatility change, but
not with executable long-straddle profit: all four front-gap quartiles lost after
crossing the option spread and paying fees. v16.8 therefore does not relax the
failed long-vol event. It asks a different question: can the alt bucket be used as
a tail-warning filter when harvesting an actual implied-versus-realized volatility
premium?

The single candidate is `OVS1_RICH_IV_QUIET_FRONT_SHORT_STRADDLE`. A historical
pass is hypothesis evidence only because the archive is short and short options
have nonlinear tail and margin risk.

## Frozen shared data and timing

Reuse v16.7's official Binance BTC option hour-0 EOH snapshots, 13-symbol USD-M
hourly price panel, conservative 01:00 UTC availability timestamp, 12-alt fixed
bucket, 30-day shifted causal volatility percentiles, chronological periods,
21--45 DTE nearest-30-day expiry and nearest-spot same-strike call/put selection.
No missing day, option quote or price bar may be filled.

For each eligible daily straddle, define:

- ATM IV as the equal-weight mean of the call and put entry mark IV;
- current BTC realized volatility as the trailing 24-hour hourly-log-return
  volatility annualized by `sqrt(365)`;
- IV-RV spread as ATM IV minus annualized BTC realized volatility;
- quiet front as alt-minus-BTC volatility percentile gap at most zero and alt
  high-vol breadth at most one third.

## Frozen candidate and execution

Enter on every non-overlapping daily observation for which the IV-RV spread is at
least 10 volatility points and the quiet-front condition holds.

1. Sell one selected call and one selected put at their archived bids.
2. The short-option delta is the negative of the recorded call-plus-put delta;
   open the opposite BTC perpetual position at the known entry close so combined
   entry delta is zero.
3. Twenty-four hours later, buy the identical options at their archived asks and
   close the static BTC hedge at the known close.
4. Normalize PnL by entry BTC price. Do not rebalance delta intraday.

Primary option fees are `min(3 bp * BTC price, 10% of option premium)` per option
transaction and primary hedge cost is 8 bp round trip on absolute hedge notional.
Stress fees use 6 bp per option transaction with the same premium cap and 16 bp
hedge round trip. Bid/ask crossing is additional and fully embedded.

No leverage, naked directional hedge, mark-price fill, stop, take-profit, expiry
substitution, stale exit or margin-efficiency assumption is allowed.

## Frozen controls and gates

Controls are the same IV-RV threshold without the alt filter, a one-day-delayed
quiet-front filter, the executable long direction on identical selected dates,
and 2,000 circular shifts of the quiet-front state over the fixed short-straddle
return path. Bootstrap entry days 5,000 times.

Research-follow-up eligibility requires:

- at least 20 complete trades, five validation and five holdout trades;
- positive primary short-straddle mean in development, validation and holdout;
- positive full-sample stress mean and bootstrap 95% lower bound;
- real primary mean at or above the 95th percentile of circular-front controls;
- better primary mean than the IV-RV-only and delayed-front comparators;
- identical-date executable long-straddle mean is negative;
- worst primary trade is no worse than -500 bp of BTC notional;
- mean of the worst 5% of trades is no worse than -300 bp;
- no single positive month contributes over 50% and no single positive trade over
  35% of positive PnL.

Failure rejects this exact rich-IV/quiet-front short-vol rule. Passing cannot
authorize PaperLive, remote changes, leverage or real orders. It would justify
only an independent audit and a longer/forward executable options recorder.
