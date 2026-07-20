# v17.4 Deribit Skew-Stress Receiver OCO Preregistration

Status: `PREREGISTERED_OFFLINE_RESEARCH_ONLY`

The v17.3 directional short hypothesis was rejected before this test. No OCO or
future realized-volatility result was inspected before freezing the rules below.

## Revised mechanism

Downside-skew shocks may forecast uncertainty and cross-asset volatility without
forecasting return sign. v17.4 therefore keeps the exact v17.3 completed-day
stress signal and monthly BTC-to-alt receiver graph, but lets subsequent price
behavior determine direction through a symmetric one-cancels-other breakout.

## Signal and receiver bucket

- Reuse v17.3 stress events exactly: downside risk-reversal robust z-score at
  least +1.0, positive ATM-IV innovation, and three-day global cooldown.
- Reuse the monthly top-four lagged absolute-return receiver bucket estimated
  from the preceding 90 days only; require at least three receivers.
- `DOS1_STRESS_RECEIVER_OCO` is the only promotion-eligible candidate.
- `DOS2_RELIEF_RECEIVER_OCO` applies the same execution to v17.3 relief events as
  a non-promotable symmetry diagnostic.

## Causal OCO execution

For every receiver independently:

1. At the signal `feature_time`, form the high/low range from the six completed
   Binance USD-M hourly bars ending at that time.
2. During the next six hourly bars, enter long at the reference high after the
   first single-sided upside touch, or short at the reference low after the first
   single-sided downside touch; cancel the opposite stop.
3. If both stops are touched in the same hourly bar before a unique trigger is
   observed, mark the leg ambiguous and keep that allocation in cash.
4. If neither stop is touched, keep that allocation in cash.
5. Exit every filled leg at the closed hourly price exactly 24 hours after signal
   time, regardless of trigger time.
6. Event return is the sum of filled-leg PnL divided by the original receiver
   count, so unfilled and ambiguous allocations remain zero-return cash.
7. Require at least two uniquely filled receivers for an event to enter the
   candidate sample.

Primary total round-trip cost is 30 bp per filled allocation; stress cost is
50 bp. Costs are multiplied by the filled fraction. No leverage is used.

## Required diagnostics and controls

- Prior and next 24h receiver realized volatility and downside semivariance.
- Same event times with 500 random, size-matched alt buckets; compare against the
  family maximum of DOS1/DOS2.
- Exact signal delayed by one day.
- Reference windows of four and eight hours as fixed sensitivity checks; the
  six-hour rule remains primary.
- Development/validation/holdout and calendar-year decomposition.
- Event bootstrap with 2,000 iterations.

## Candidate gate

DOS1 survives only if all hold:

- at least 25 full-sample, 5 validation, and 8 holdout events;
- at least 60% of receiver allocations uniquely fill and no more than 15% are
  ambiguous;
- primary-cost mean positive in full, validation, and holdout;
- stress-cost full mean positive;
- bootstrap 95% lower bound positive;
- random-bucket family percentile at least 90%;
- six-hour rule beats the one-day delay;
- four- and eight-hour sensitivity means both positive;
- no single positive year contributes more than 50% of positive PnL.

Passing creates an offline research candidate only. No PaperLive, application,
remote, leverage, option-execution, or real-order permission is granted.
