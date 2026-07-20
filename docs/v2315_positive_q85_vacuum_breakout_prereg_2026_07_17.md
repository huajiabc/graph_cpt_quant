# v23.15 Positive-q85 Vacuum Breakout Preregistration

Status: `FROZEN_BEFORE_SINGLE_Q85_OUTCOME_LOAD`.

q85 is the only interpolation tested between the rejected q80 density extension
and the post-selected q90 tail candidate. No other pressure quantile will be
searched in this round.

## Frozen feature

- Artifact:
  `reports/v23_14_positive_q85_vacuum_breakout_feature_audit/positive_q85_breakout_features.parquet`.
- SHA-256:
  `F14AB2F30433594501EA3FC9284F8E087768BF3B48C58056CB4EFEEEB65BEE28`.
- 75 events across 12 months; development/validation/holdout 28/21/26.
- Positive pressure, 11/16 aligned, 5/16 withdrawing, false transition,
  four-hour cooldown, and 0.625-sigma BTC barriers are frozen.

## Frozen execution and controls

- Reuse the exact v23.12 OCO execution, pessimistic gap/ambiguity handling,
  four-hour exit, and 10/20 bp primary/stress costs.
- Adjacent 0.75-sigma width remains the sole width robustness check.
- Match controls by month, exact UTC hour, nearest trailing sigma, complete path,
  and more than eight hours from every q85 event; require at least five controls.
- Run 1,000 matched paths for all/development/validation/holdout and 5,000
  month-block bootstrap paths, seed `20260717`.

## Frozen gates

The q85 interpolation is supported only if:

1. at least 70 events trigger overall and 20 in each temporal split;
2. primary return is positive overall and in all three temporal splits;
3. full-sample 20 bp stress return is positive;
4. absolute month-bootstrap 95% lower bound is above zero;
5. every event has at least five matched controls and all four matched-random
   percentiles are at least 90;
6. same-bar ambiguity is at most 10%;
7. every leave-one-month-out mean is positive; and
8. adjacent 0.75-sigma return is positive overall and in all three splits.

Failure of any gate rejects q85 and ends pressure-quantile interpolation.
Passing remains research-only with no PaperLive/live permission change.
