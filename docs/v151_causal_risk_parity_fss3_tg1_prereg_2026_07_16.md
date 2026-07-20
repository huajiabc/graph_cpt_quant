# v15.1 Causal Risk-Parity FSS3 + TG1 Preregistration

Date frozen: 2026-07-16, before inspecting dynamic FSS3/TG1 allocation returns.

## Motivation and evidence boundary

FSS3 is an independently audited funding-sign cross-sectional candidate. TG1 is
an exact same-coin cross-venue funding-spread sleeve that passed its sample,
three-period, stress, bootstrap, turnover and random-basket gates but failed the
35% positive-month concentration limit. Their observed weekly primary-net
correlation is -0.0692. A descriptive 50/50 combination was inspected while
mapping the evidence and is therefore not promotable here.

The only candidate reuses the exact causal allocation architecture frozen in
v12.9 before this FSS3/TG1 application was conceived:

`RP2_CAUSAL_8W_FSS3_TG1`

## Frozen construction

- Align the exact saved v14.9 FSS3 and v13.2 TG1 weekly artifacts by Monday
  entry time. Do not rebuild, filter, winsorize or relabel either sleeve.
- Use 50/50 capital for the first eight aligned weeks.
- Thereafter, each Monday uses only the prior eight completed weekly primary-net
  sleeve returns. Allocate inverse to trailing sample volatility.
- Clip TG1 capital weight to `[0.25, 0.75]`; FSS3 receives the complement.
- A zero/non-finite trailing volatility retains the preceding allocation.
- Primary allocation cost is 20bp times absolute weekly change in TG1 capital
  weight; stress allocation cost is 40bp. Sleeve returns already contain their
  own initial, transition and terminal costs, so those costs are not duplicated.
- No mean forecast, sign filter, correlation forecast, regime, month, return
  threshold, alternative window, or alternative weight bound is tested.

## Frozen evaluation

- Four-week moving-block bootstrap, 2,000 draws.
- Report price and funding cash-flow attribution, both cost levels, allocation
  turnover, chronological splits, positive-month concentration, leave-one-month
  out, worst week and additive drawdown.
- The two raw sleeves must each retain positive mean primary return and absolute
  correlation no greater than 0.50.

Promotion to forward portfolio shadow requires:

- at least 45 aligned weeks, 11 months, ten validation weeks and ten holdout weeks;
- positive development, validation and holdout primary returns;
- positive full-sample stress return and funding cashflow after stress allocation
  cost;
- four-week bootstrap 95% lower bound above zero;
- positive-month concentration no greater than 35%;
- worst period no worse than -40bp/week;
- every leave-one-month-out mean above zero;
- additive maximum drawdown magnitude at least 25% lower than standalone FSS3.

Passing establishes a portfolio-layer alpha candidate, not a new raw factor.
No PaperLive, leverage, remote-host or real-order permission is granted.
