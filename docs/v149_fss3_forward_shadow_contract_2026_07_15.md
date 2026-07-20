# v14.9 FSS3 Forward-Shadow Contract

`FSS3_CURRENT_SIGN_070_TURNOVER_CAP` is frozen as a forward-shadow candidate,
not as a PaperLive, leveraged, or real-order strategy.

## Decision data

At Monday 00:00 UTC, use only settled funding records with timestamps in
`[decision_time - 7d, decision_time)` and closed one-hour price bars strictly
available by the decision time. Monthly membership is fixed for the month.
Each symbol's beta is estimated from the prior 30 days of hourly returns ending
before the month starts, matching the audited research panel.

Fail closed for the week if either side has fewer than four fully valid symbols,
if membership/beta/funding data are incomplete, or if the previous executed
weight state cannot be loaded unambiguously.

## Target and transition

1. Long every symbol with negative seven-day settled funding and short every
   symbol with positive funding; exclude exact zeros.
2. Allocate 0.5 raw gross to each side and equal weight within each side.
3. Add the exact current estimated BTC-beta hedge and normalize gross to one.
4. Starting from the prior executed weights, choose the largest target fraction
   whose full-L1 transition turnover is at most 0.70. At each trial fraction,
   blend alternative-asset weights, recalculate the BTC hedge, and renormalize
   gross to one.
5. Persist the executed weights, data cutoff, membership version, beta snapshot,
   funding snapshot hashes, target fraction, turnover, residual beta, and gross.

Initial opening and mandatory exits are fully recorded. The terminal close used
in research is a reporting convention; forward shadow charges a close only when
the candidate actually exits or is retired, not every Monday.

## Required shadow telemetry

- intended and executed weights, target fraction, cap binding flag and turnover;
- funding cash flow, mark-to-market price PnL, 20bp and 40bp cost shadows;
- data freshness/completeness, residual estimated beta and gross-notional drift;
- weekly drawdown, symbol contribution and realized BTC sensitivity;
- explicit no-signal and fail-closed records.

The seed weight history is
`reports/v14_9_funding_sign_turnover_cap/weekly_weights.parquet`; the frozen
configuration is
`configs/v14_9_fss3_funding_sign_turnover_cap_candidate.yaml`; the independent
audit is
`reports/v14_9_funding_sign_turnover_cap/independent_audit.json`.

No remote host or PaperLive configuration was changed in this research round.
