# v23.24 q90 Broad-Taker Confirmation Preregistration

The v23.23 feature set is frozen at hash
`7C33473808AC4C2CFE3DBB81FD71642C2D6C3814D405D9925EE1D495B6A1C7DD`.
The ancestor q90 rule is post-selected, so this round can only test whether an
independent contemporaneous source improves mechanism attribution.

## Frozen trade

- Buy BTC at the event timestamp when positive q90 book pressure is confirmed
  by taker-buy ratios above one in at least 9 of the 16 frozen symbols.
- Use the worse of the causal hourly entry spot and the next 15-minute bar open.
- Exit at the close of the fourth completed hour.
- Primary round-trip cost is 10 bp and stress cost is 20 bp.
- A 15-minute delayed entry with the same exit is the only latency sensitivity.

## Controls

1. The 27 positive-q90 events without broad taker confirmation.
2. Same-month, same-UTC-hour non-event timestamps that also have at least 9/16
   taker-buy ratios above one. Exclude plus/minus eight hours around every q90
   event; match on causal BTC sigma and median log taker ratio; retain up to ten
   controls and require at least five per event.
3. Short BTC on the same confirmed event timestamps.
4. A within-month permutation of confirmed/unconfirmed q90 labels.

## Conjunctive gates

- At least 24 complete confirmed trades, at least seven per temporal scope.
- Primary and stress return positive in all/development/validation/holdout.
- Absolute month-bootstrap 95% lower bound above zero and every
  leave-one-month-out mean positive.
- Matched-random percentile at least 90 in every scope and no unmatched event.
- Confirmed-minus-unconfirmed primary return positive in every scope, with
  within-month permutation upper-tail p-value at most 0.10.
- The 15-minute delayed primary return positive in every scope.
- Long beats the sign-reversed short, and positive monthly PnL concentration is
  at most 50%.

No live, PaperLive, leverage, remote, application, or order state is changed.
