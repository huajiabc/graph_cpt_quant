# v23.18 q90 Breakout + CM2 Overlay Preregistration

The outcome-free v23.17 mapping is frozen at feature hash
`CF79F9D42324BF5C930292F89D0186D86845B73561FA7E2B8EF6368E24C82046`.
This is a portfolio-integration diagnostic for a post-selected ancestor, not a
new alpha confirmation or deployment authorization.

## Frozen construction

- Preserve the existing 80% FSS3 / 20% TG1 CM2 weekly return unchanged.
- Use only the 53 frozen positive-q90, 0.625-sigma, four-hour OCO events.
- Assign each event return to the CM2 week in which the event exits. This keeps
  Sunday events that cross Monday 00:00 UTC on the correct realization clock.
- Compound non-overlapping event returns within a week to form the satellite
  weekly return. Weeks without an event receive zero satellite return.
- Primary construction: add 10% temporary event notional to CM2 only while an
  event is active. Weekly return is CM2 plus 0.10 times satellite return.
- The event stream retains its existing 10 bp primary and 20 bp stress costs.
- Fixed 5% and 20% overlay weights are reported only as linear capacity
  sensitivities. No allocation, feature, threshold, width, horizon, or cost
  grid is permitted.

## Frozen evidence gates

1. The feature hash and all 53 event mappings must reproduce exactly.
2. Primary and stress incremental return must be positive in all, development,
   validation, and holdout scopes.
3. The month-resampled 95% lower bound of weekly primary increment must exceed
   zero, and every leave-one-month-out primary increment must remain positive.
4. Full-sample satellite/core correlation must be at most 0.30 in absolute
   value; active-week correlation must be at most 0.50.
5. Combined annualized weekly Sharpe must exceed CM2 in every temporal scope.
6. Full-sample downside semideviation and additive maximum drawdown must not
   worsen.
7. The sign-reversed event overlay must underperform the observed overlay.
8. Both 5% and 20% sensitivity scales must retain positive primary increment
   in every temporal scope.
9. No single positive month may contribute more than 50% of total positive
   monthly increment.

Failure of any gate rejects portfolio confirmation. Even a full pass would
only support continued forward shadow because the q90 ancestor was selected
post hoc.

No live, PaperLive, leverage, remote, application, or order state is changed.
