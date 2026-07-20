# v23.4 Book-Vacuum BTC OCO Breakout Preregistration

Status: `FROZEN_BEFORE_TRIGGER_AND_RETURN_LOAD`.

## Question

The v23.1 result found relative post-event volatility alpha but rejected an
unconditional long-option implementation after premium friction. This test asks
whether a direction-agnostic BTC perpetual OCO breakout can monetize the same
information without paying option theta.

## Frozen input

- Feature artifact:
  `reports/v23_3_book_vacuum_oco_breakout_feature_audit/oco_breakout_features.parquet`.
- SHA-256:
  `20E29DC3BCF5E8702E46AE3B21B8F900BD26C42A6E43F7D7B488C847250DD828`.
- Events: all 159 frozen v22.4 book-vacuum events, with development/validation/
  holdout counts 63/47/49.
- At event time `t`, BTC spot is the just-completed Bybit 15-minute close and
  causal hourly sigma is the square root of the mean squared log move over the
  preceding 24 completed hours.
- Primary barriers are `spot * exp(+sigma)` and `spot * exp(-sigma)`.

No post-entry high, low, trigger, direction, fill, or return was used in feature
selection.

## Frozen order and fill logic

- At `t`, place an OCO buy stop at the upper barrier and sell stop at the lower
  barrier.
- Scan exactly 16 Bybit 15-minute bars from `[t, t+4h)`.
- The first bar whose high reaches the upper stop or whose low reaches the lower
  stop determines entry; the other order is cancelled.
- If a bar opens through a stop, use the worse opening price: long fill is
  `max(upper stop, bar open)` and short fill is `min(lower stop, bar open)`.
- If the first triggering bar reaches both barriers and the intrabar order is
  unknowable, calculate both possible fills and choose the direction with the
  lower eventual return. This is a deterministic pessimistic ambiguity rule.
- Exit any triggered position at the completed BTC close at `t+4h`.
- Long gross return is `exit / fill - 1`; short gross return is
  `1 - exit / fill`.
- An untriggered event has zero return and no cost.

## Frozen costs and secondary widths

- Primary round-trip friction: 10 bp of traded notional.
- Stress round-trip friction: 20 bp.
- Secondary barrier widths, reported without promotion authority: 0.75 and 1.25
  times causal hourly sigma, using identical fill and exit rules.
- A reversed-direction control enters the opposite side of the first
  unambiguous breakout, with the same fill convention and cost. Ambiguous bars
  remain pessimistic.

These are research cost assumptions, not a claim about historical queue
position or guaranteed stop execution.

## Frozen matched-time control

Build all eligible non-event hourly BTC contexts in the same sample months. A
control must have a complete 16-bar path, the same calendar month and exact UTC
hour as its event, and lie more than eight hours from every v22.4 event. Rank
controls by absolute log distance in trailing causal hourly sigma and retain the
nearest 10, requiring at least five. Draw one control per event for 1,000
deterministic paths using seed `20260717`.

## Frozen uncertainty and gates

- Resample event months with replacement for 5,000 month-block bootstrap means.
- Primary statistic: mean 4-hour net return per event, including zero for no
  trigger.
- Also report trade rate, mean net return per triggered trade, direction,
  trigger delay, ambiguous-bar fraction, each temporal split, and pressure-sign
  strata.
- The OCO implementation is supported only if:
  1. at least 80 events trigger overall and at least 20 trigger in each temporal
     split;
  2. primary mean net return per event is positive overall and in development,
     validation, and holdout;
  3. the month-block 95% lower bound is above zero;
  4. the event mean is at or above the 90th percentile of matched random-time
     means;
  5. the ambiguous-trigger fraction is no more than 10%; and
  6. primary return exceeds the reversed-direction control.

Failure of any gate rejects the candidate. Passing is still research-only and
does not change PaperLive, live, leverage, remote, application, or order state.
