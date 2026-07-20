# v17.6 Monthly Deribit Surface Coverage-Extension Preregistration

Status: `PREREGISTERED_ROBUSTNESS_AUDIT_ONLY`

v17.5 revealed one non-passing lead: relief-event receiver-minus-insulator spread
was positive in full/development/validation and strong against random pairs, but
had only five holdout events with a negative mean. This audit increases source
coverage without changing its signal, graph, portfolio, horizon, or cost rules.

## Frozen data extension

- Add the last Friday 08:00 UTC BTC option expiry of every calendar month from
  March 2021 through June 2026.
- Keep the exact v17.2 strike lattice, positive-volume/cost filter, inverse-IV
  calculation, 7--45 DTE restriction, and surface quality gates.
- When multiple expiries pass on the same completed UTC day, select the row with
  DTE closest to 30; break ties toward the earlier expiry.
- Recompute innovations only within the same selected expiry and across at most a
  two-day feature gap. Keep the exact v17.3 robust-z and cooldown rule.

## Frozen strategy replay

- Replay v17.5 `DSS1_STRESS_SHORT_RECEIVER_LONG_INSULATOR` and
  `DSS2_RELIEF_LONG_RECEIVER_SHORT_INSULATOR` unchanged.
- Receiver/insulator graph, 24h holding period, unit gross weights, 40/60 bp cost,
  splits, random-pair controls, one-day delay, 8h/48h sensitivities, bootstrap,
  and concentration checks remain unchanged.

## Interpretation ceiling

Because the calendar extension was motivated by an observed quarterly-sample
lead and overlaps the same historical market period, it cannot promote a
strategy. At best DSS2 becomes `RESEARCH_LEAD_FORWARD_WATCH` if all original
v17.5 gates pass with stricter counts of 60 full, 12 validation, and 20 holdout
events. Otherwise it is rejected. Any real promotion would require genuinely
new forward observations.

No PaperLive, application, remote, leverage, or real-order permission changes.
