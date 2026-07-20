# v23.12 Positive-q80 Vacuum Breakout Preregistration

Status: `FROZEN_BEFORE_Q80_TRIGGER_AND_RETURN_LOAD`.

## Purpose and provenance

The v23.8 positive-q90 0.625-sigma breakout has strong post-selection
robustness but only 53 events and a negative absolute month-bootstrap lower
bound. This test broadens signal density without changing the mechanism:

- positive aggregate book pressure;
- pressure above a shifted trailing-720-hour absolute q80 threshold;
- at least 11/16 symbols aligned;
- at least 5/16 symbols withdrawing one-percent depth;
- false transition and four-hour cooldown;
- symmetric BTC breakout at plus/minus 0.625 causal hourly sigma.

Only the q80 threshold differs from the q90 ancestor. The q80 outcomes have not
been loaded.

## Frozen feature artifact

- Path:
  `reports/v23_11_positive_q80_vacuum_breakout_feature_audit/positive_q80_breakout_features.parquet`.
- SHA-256:
  `163972748E6CC095BD086414CADC8C8A9F7535082B1733A6C7E2933EEF848B93`.
- Events: 89 across 12 months; development/validation/holdout 32/24/33.
- v23.11 feature checks: 14/14 passed.

## Frozen execution

- Reuse the v23.4 first-trigger OCO logic over 16 Bybit 15-minute bars.
- Barriers: `spot * exp(+/- 0.625 * trailing hourly sigma)`.
- Gap fill: long `max(upper, bar open)`, short `min(lower, bar open)`.
- Same-bar dual trigger: choose the lower eventual return.
- Exit: completed BTC close at four hours.
- No trigger: zero return and zero cost.
- Primary/stress round-trip costs: 10/20 bp of traded notional.
- Adjacent 0.75-sigma width is a frozen robustness check, not an alternate chosen
  after results.

## Frozen controls and uncertainty

- Control times must be in the same month and exact UTC hour, have a complete
  path, lie more than eight hours from every q80 event, and rank among the 10
  nearest trailing-sigma contexts; at least five controls are required.
- Draw 1,000 matched paths per all/development/validation/holdout scope.
- Draw 5,000 event-month bootstrap paths.
- Report leave-one-month-out means and trigger ambiguity.
- Seed: `20260717`.

## Frozen gates

The density extension is supported only if:

1. at least 80 events trigger overall and at least 20 in each temporal split;
2. primary mean return per event is positive overall and in development,
   validation, and holdout;
3. full-sample 20 bp stress return is positive;
4. the absolute month-bootstrap 95% lower bound is above zero;
5. all four matched-random percentiles are at least 90;
6. same-bar ambiguity is at most 10% of triggers;
7. every leave-one-month-out mean is positive; and
8. the adjacent 0.75-sigma width is positive overall and in all three temporal
   splits.

Failure of any gate rejects the q80 extension. Passing remains research-only;
it does not authorize PaperLive, live, leverage, remote, application, or order
changes.
