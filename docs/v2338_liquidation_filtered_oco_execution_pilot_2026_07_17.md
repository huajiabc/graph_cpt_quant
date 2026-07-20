# v23.38 Liquidation-Filtered OCO Execution Pilot

Verdict: `higher_trigger_rate_but_not_cost_robust`.

The retrospective top liquidation quartile changes BTC 0.625-sigma OCO trigger rate from 62.5% in the bottom quartile to 95.8%. Top-quartile mean net is -9.89 bp/decision at 10 bp cost and -19.47 bp/decision at 20 bp cost.

Using alt-only liquidation intensity, the top-quartile result is -15.16 bp/decision primary and -24.74 bp/decision stress.

This is a one-day, retrospectively ranked, overlapping-path mechanism test. It cannot promote a strategy. Any retained execution hypothesis must use a causal expanding threshold in the v23.35 forward ledger and must survive chronological cost-stressed evaluation.
