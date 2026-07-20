# v23.21 Alt-First Volatility Ignition Breakout Preregistration

The v23.20 feature set is frozen at hash
`C4F814ADD57330518B98C6ABFA2CCC98A7A4BC7EC3814D2B8DB7F9118A478B6F`.
No BTC path outcome was inspected when the feature rule and gates were frozen.

## Primary payoff

- At each of the 100 event timestamps, place a symmetric BTC OCO at
  plus/minus 0.75 times the causal hourly sigma estimated from the prior 24
  completed hours.
- Use the first 15-minute barrier hit, pessimistic gap fill, and the worse side
  if both barriers hit in one bar. Exit at four hours.
- Charge 10 bp per triggered trade; use 20 bp as stress cost.
- Report 0.625 and 1.0 sigma only as frozen adjacent-width sensitivities. No
  width, horizon, threshold, or feature search is permitted after reveal.

## Causal control

For each event, use up to ten same-month, same-UTC-hour control timestamps that
also have a quiet BTC shock but do not have the broad alt shock. Exclude all
hours within plus/minus eight hours of any event, then rank controls by causal
BTC sigma distance plus contemporaneous BTC shock-z distance. At least four
controls are required for every event. Draw 1,000 matched random paths within
each temporal scope.

## Evidence gates

1. At least 80 total triggers and at least 15 in each development, validation,
   and holdout scope.
2. Primary return must be positive in all four scopes; stress return must also
   be positive in all four scopes.
3. The absolute month bootstrap 95% lower bound and every leave-one-month-out
   mean must exceed zero.
4. The matched-random percentile must be at least 90 in all four scopes, with
   every event having at least four controls.
5. Same-bar ambiguity must not exceed 10%.
6. Both adjacent widths must remain primary-positive in all four scopes.

All gates are conjunctive. A pass remains research-only pending independent
audit and genuinely new forward events. No live, PaperLive, leverage, remote,
application, or order state is changed.
