# v23.35 Liquidation Pre-Event Feature Contract

Status: `frozen_forward_contract_no_outcome_search`.

Conservative causal start: `2026-07-17T04:25:50.864340+00:00`.

## Causal inclusion rule

At decision time `t`, an event is eligible only when `t-window <= event_time < t` and `first_seen_at <= t`. The initial OKX snapshot is not treated as historical data before it became known.

## Frozen feature families

For 5, 15, and 60 minute windows: forced-sell and forced-buy counts and notional, net forced-buy pressure, log imbalance, active-symbol breadth, BTC share, cross-symbol notional HHI, and event-size summaries. Two burst shares compare 5/15 and 15/60 minute notional. No other transformation may be introduced after outcomes are inspected in the first evaluation.

## Forward hypotheses

1. Broad, high-notional liquidation bursts predict continued BTC realized volatility over the next 1 and 4 hours.
2. Forced-buy versus forced-sell imbalance predicts the first BTC excursion side, conditional on a volatility event.
3. Broad low-concentration cascades transmit more strongly to BTC than single-symbol concentrated liquidations.
4. On frozen q90 book-vacuum events, pre-event liquidation intensity is tested only as an OCO trigger/avoidance overlay, not as a reselected base rule.

## Evaluation gates

The hourly volatility panel requires at least 336 decisions across 14 UTC days. The q90 overlay requires at least 30 events and 10 events in each chronological third. Regularized interactions are forbidden below 1,000 hourly decisions; boosted/tree models are forbidden below 2,000 and then require nested walk-forward validation against the frozen linear baseline.

Primary outcomes are next-1h and next-4h BTC log-range/realized absolute movement. Directional secondary outcomes are first upper-versus-lower excursion under symmetric barriers. Costs enter only when translating a validated volatility relation into an executable OCO strategy.

No PaperLive, live, leverage, remote, application, or order state changes are authorized by this contract.
