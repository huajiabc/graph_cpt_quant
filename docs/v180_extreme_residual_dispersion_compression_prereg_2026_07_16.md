# v18.0 Extreme Residual Dispersion Compression Preregistration

Status: frozen before first outcome reveal.

## Distinct hypothesis

The prior 4-hour/12-hour residual models ranked trailing multi-hour residual
momentum. v10.3-v10.6 and v17.8-v17.9 tested target/bucket catch-up or directed
graph propagation. This round instead asks whether an extreme, single-bar,
cross-sectional residual dislocation compresses immediately during the next 15
minutes.

## Data and causal beta

- Binance USD-M completed 15-minute bars in the locally complete liquid panel.
- BTC and XAUT are not selectable alt legs.
- For each calendar month, contemporaneous BTC beta is frozen from only the prior
  30 days with at least 2,000 paired observations.
- Residual at `t` is alt return at `t` minus frozen beta times BTC return at `t`.
- At least 30 finite alt residuals are required.

## Frozen dispersion event

- Cross-sectional dispersion is residual q90 minus residual q10 at the completed
  bar.
- The event threshold is the rolling prior-30-day q97.5 of dispersion, shifted by
  one completed bar, with at least 20 days of history.
- Events use a four-bar / one-hour cooldown.
- The five lowest current residuals form the laggard leg; the five highest form
  the leader leg. Membership is frozen at the completed event bar.
- q95 and q99 thresholds are sensitivity diagnostics only.

## Frozen candidate

`RDC1_EXTREME_RESIDUAL_DISPERSION_COMPRESSION` is half-long the five laggards and
half-short the five leaders. The frozen beta difference is hedged with BTC, and
the return is normalized by total gross exposure including the hedge.

- Primary holding period: one bar / 15 minutes.
- Primary/stress round-trip costs: 30/40 bp.
- Two-bar and four-bar holding periods are sensitivity diagnostics.

## Frozen controls and gates

- Exact reversed trade direction (residual momentum).
- One-bar delayed entry using the original event and original bucket membership.
- 500 deterministic random disjoint five-versus-five rank assignments at the
  same event times, with the same beta hedge and costs.
- Day-block bootstrap with 2,000 iterations.
- At least 100 full-sample, 20 validation, and 25 holdout events.
- Positive primary net mean in development, validation, and holdout.
- Positive full-sample stress net mean and bootstrap 95% lower bound.
- At or above the 95th percentile of random-rank controls.
- Must beat reversed direction and one-bar delay.
- q95, q99, 30-minute, and 60-minute diagnostics must remain positive.
- No single profitable month may contribute more than 35% of total positive
  monthly PnL.

No PaperLive, application scope, leverage, remote host, or real-order permission
may change in this research round.
