# v16.9 Weekly Delta-Hedged Quiet-Front Short-Straddle Preregistration

Date frozen: 2026-07-16, after the v16.8 result and an exact algebra audit, but
before constructing any seven-day option return.

## Adaptive disclosure and purpose

v16.8's 24-hour rich-IV/quiet-front short straddle lost in every chronological
period. The same dates also lost in the long direction. Across all 118 executable
daily pairs, long-plus-short gross PnL matched the negative entry-and-exit option
spread to `3.47e-18`, and the average two-sided spread cost was 63.42 bp of BTC
notional before fees. v16.9 is therefore a single execution-horizon follow-up:
amortize one option round trip over seven days while explicitly paying daily BTC
delta-hedge turnover.

The candidate is `OVS2_WEEKLY_RICH_IV_QUIET_FRONT_SHORT_STRADDLE`. This is
result-informed and cannot establish deployable alpha on the short 2023 archive.

## Frozen entry and non-overlap rule

Reuse v16.8's exact data, timing, 12-alt bucket, causal volatility features,
IV-RV spread, quiet-front rule, 21--45 DTE nearest-30-day expiry, ATM strike and
quote-validity rules.

An entry is eligible when IV minus annualized trailing-24h BTC realized volatility
is at least 10 vol points, alt-minus-BTC volatility percentile gap is at most zero,
and alt high-vol breadth is at most one third. Process dates chronologically and
accept the first eligible entry only when no prior seven-day trade is open. Ignore
all intervening signals until the prior position exits. No weekday or threshold
grid is allowed.

## Frozen seven-day execution

1. At entry, sell one call and one put at archived bids.
2. Because the option position is short, hold BTC perpetual quantity equal to the
   recorded call-plus-put delta, making entry delta zero.
3. At each of the next six daily 01:00 UTC snapshots, value the identical contracts
   and change the BTC hedge to that day's recorded call-plus-put delta. The prior
   hedge earns PnL over the completed interval before the rebalance.
4. At day seven, buy back both options at archived asks and close the prior day's
   BTC hedge at the known BTC close. Do not rebalance immediately before closing.
5. Require all eight option snapshots, all seven price intervals, internally valid
   quotes and finite deltas. Any missing day invalidates the whole trade; do not
   substitute another strike or expiry.

Option fees use v16.8's frozen 3 bp primary / 6 bp stress per transaction with the
10%-of-premium cap. Every BTC hedge opening, daily change and final close pays 4 bp
per-side primary or 8 bp stress on absolute traded notional. Report a symmetric
executable long control on identical dates using entry asks, exit bids and the
opposite daily hedge path.

No intraday delta, funding benefit, interest, margin efficiency, leverage, stop,
take-profit, mark fill, stale quote or cross-instrument netting is assumed.

## Frozen controls and gates

Compare with the same weekly IV-RV entry rule without the alt filter, a one-day
delayed quiet-front rule, identical-date executable long straddles, and 2,000
circular shifts of quiet-front state over the eligible seven-day return calendar.
Apply the same chronological non-overlap algorithm separately to every control.
Bootstrap accepted entry days 5,000 times.

Research-follow-up eligibility requires:

- at least ten complete trades, two validation and two holdout trades;
- positive primary mean in development, validation and holdout;
- positive full-sample stress mean and bootstrap 95% lower bound;
- real mean at or above the 95th percentile of circular controls;
- better mean than IV-RV-only and delayed-front controls;
- identical-date long mean is negative;
- worst trade is no worse than -1,000 bp of BTC notional and worst-5% mean no
  worse than -600 bp;
- no positive month or single positive trade contributes more than 50% of total
  positive PnL.

Failure rejects this weekly construction. Passing grants only an independent
audit and a recommendation to obtain longer executable option history or begin
forward recording. It does not authorize PaperLive, remote changes, leverage or
real orders.
