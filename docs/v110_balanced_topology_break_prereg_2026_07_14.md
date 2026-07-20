# v11.0 Balanced-Community Topology-Break Preregistration

## Question

Does a sudden loss of synchronization inside a genuinely balanced residual-return community
predict a four-hour market-neutral bucket spread? This is a topology-change test. It is not a
retuning of the rejected v10.9 MST-dispersion rule.

## Frozen graph and timing

- Reuse the v10.8 continuous panel and exclude BTC from community membership.
- At each month start, estimate static BTC betas and the residual correlation matrix from the
  preceding 30 calendar days of hourly observations only; require at least 500 complete rows.
- Form exactly eight communities by deterministic recursive spectral bisection. Always split the
  largest current block at the median of its Fiedler vector. This fixes community sizes near nine
  coins and prevents the giant-component failure seen in v10.9.
- Freeze membership and every normalization statistic for the whole following month.

## Frozen topology-break signal

For each community, standardize residual returns by trailing-history volatility. At every hour,
calculate the average pairwise standardized cross-product and average it over 12 hours. The
community is broken when this coherence score crosses below its trailing-history fifth percentile.
Accept only false-to-true transitions and impose a four-hour cooldown per community.

At entry, rank members by their trailing four-hour residual return and trade both predeclared signs:

- `TBR1_TOPOLOGY_REPAIR`: long the bottom third and short the top third.
- `TBR2_BREAK_CONTINUATION`: long the top third and short the bottom third.

Each sleeve is 0.5 long plus 0.5 short. Select at most three simultaneous communities by the
largest threshold breach. Report 20/30/50 bp round-trip costs.

## Controls and gates

- Development: before 2026-01-01; validation: 2026-01-01 through 2026-03-31; untouched holdout:
  2026-04-01 onward.
- Fifty random monthly partitions preserving the exact balanced community sizes.
- A one-day (24 hourly bars) shifted signal control.
- Two thousand entry-day bootstrap resamples and five chronological slices.
- Promotion requires at least 100 total observations, at least 25 in validation and holdout,
  positive validation and holdout net 20 bp, positive full-sample net 30 bp, at least the 90th
  percentile of the random-family maximum, superiority to the shifted control, positive bootstrap
  lower bound, no negative chronological slice, and no positive month or community above 35% of
  positive PnL.

Failure rejects this exact balanced-community topology-break trading rule. It does not reject
cross-venue flow graphs, graph-aware allocation, or other topology state variables. No PaperLive
permission changes are authorized by this experiment.
