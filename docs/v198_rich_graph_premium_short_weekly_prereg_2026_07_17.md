# v19.8 Rich Graph-Premium Short Weekly Preregistration

Status: frozen after the v19.7 feature-only audit and before calculating any
one-sided portfolio future return.

## Post-hoc status and multiplicity

This hypothesis was motivated by v19.6 sleeve attribution: rich-premium short
contributions were positive while cheap-premium long contributions were
negative. It is explicitly post-hoc. The research family therefore contains all
four directions that could have been selected after inspecting v19.6:

1. global rich-premium short;
2. global cheap-premium long;
3. community rich-premium short;
4. community cheap-premium long.

Only the two rich-premium short portfolios are eligible candidates, but every
random iteration takes the maximum across all four directional variants. The
eligibility percentile is raised to 99%. Passing cannot authorize forward
shadow or PaperLive without natural newly accrued weeks.

## Frozen signal, data, and chronology

- Reuse the exact v19.5 graph-peer premium score: own exact Binance premium-index
  close z-score against shifted prior-30-day moments, minus the frozen monthly
  community median z-score.
- Score at Monday 00:00 UTC uses the completed value at that timestamp, never a
  forward-filled value.
- Global selection takes the eight richest graph-peer scores. Community
  selection takes the richest symbol in every community with at least four
  eligible members.
- Every selected alt is short. Add the causal long BTC hedge that neutralizes
  aggregate prior-30-day BTC beta, then normalize gross notional to one.
- Weights remain fixed for seven days. Binance settled funding PnL includes
  settlements in `(entry, exit]`; settlement exactly at entry is excluded.
- Only complete Monday-to-Monday weeks are included.
- Development is before 2026-01-01, validation is January-February 2026, and
  holdout begins 2026-03-01.

## Frozen candidates

- `RPS1_GLOBAL_RICH_GRAPH_PREMIUM_SHORT_WEEKLY`
- `RPS2_COMMUNITY_RICH_GRAPH_PREMIUM_SHORT_WEEKLY`

The global/community cheap-premium long portfolios are mandatory directional
diagnostics and members of the multiplicity family, not promotable candidates.

## Frozen costs, controls, and diagnostics

- Primary/stress one-way cost: 20/40 bp times exact full L1 turnover.
- Initial opening, every transition, mandatory exit, data gap, and terminal
  close are fully charged. No turnover cap is used.
- Exact reversed weights and one-week delayed score controls.
- Funding-orthogonal graph-premium rich-short diagnostic.
- Parent v19.6 double-sided portfolio comparison.
- 500 deterministic random paths for all four global/community × short/long
  variants, each using its own causal BTC hedge, realized turnover, price PnL,
  funding PnL, and four-direction family maximum.
- Four-week moving-block bootstrap with 2,000 draws.
- Development/validation/holdout, leave-one-month-out, symbol and month
  concentration, short-alt versus BTC-hedge attribution, and FSS3 overlap
  correlation are reported.

## Frozen gates

- At least 40 complete weeks, ten active months, eight validation weeks, and 12
  holdout weeks.
- Positive primary mean in development, validation, and holdout.
- Positive full stress mean, residual price contribution, funding contribution,
  and four-week bootstrap 95% lower bound.
- At or above the four-direction random family-max 99th percentile.
- Beat reversed weights, one-week delay, the matching cheap-long direction, the
  parent v19.6 portfolio, and the matching random-direction mean.
- Funding-orthogonal rich-short diagnostic remains positive after primary cost.
- The selected alt-short sleeve has positive gross contribution and supplies at
  least 50% of all positive short-alt/BTC-hedge gross contribution.
- Mean fully charged turnover is no greater than 0.85; numerical BTC-beta and
  gross-notional constraints hold.
- Absolute primary-return correlation with FSS3 is no greater than 0.60.
- Every leave-one-month-out mean is positive; no single symbol contributes more
  than 25% of positive gross PnL; no profitable month supplies more than 35% of
  positive monthly PnL; worst chronological period mean is at least -40 bp/week.

Passing yields only `posthoc_offline_discovery_requires_natural_forward`.
No PaperLive, live, application, leverage, remote, or real-order scope may
change.
