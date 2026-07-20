# v19.6 Graph-Premium Relative-Value Weekly Preregistration

Status: frozen after the v19.5 feature/turnover audit and before inspecting any
future portfolio return.

## Mechanism and distinction from FSS3

For every symbol, exact Binance premium-index close is standardized against its
own shifted prior-30-day history. The graph score subtracts the contemporaneous
median z-score of the symbol's frozen monthly community. A rich premium relative
to graph peers is shorted and a cheap premium is bought, seeking convergence and
subsequent funding cash flow over one week.

The feature audit found median cross-sectional correlation of only 0.0162
between graph-peer premium and trailing seven-day settled funding. Portfolio
signs align with the existing FSS3 funding-sign strategy about 50% of the time.
This is therefore treated as a distinct information source, while realized
return correlation with FSS3 remains a formal independence gate.

Daily, two-day, and weekly targets all require roughly 1.54-1.68 L1 turnover per
decision and consecutive sleeve Jaccard overlap is roughly 0.1. The signal is
not persistent enough for a turnover cap that carries stale targets. Weekly
Monday decisions are frozen to give convergence and funding enough time to
cover the fully charged turnover cost.

## Frozen data and chronology

- Exact checksummed Binance USD-M 15-minute premium-index close and Binance
  futures close prices.
- Score at Monday 00:00 UTC uses the completed value at that timestamp.
- Own premium z-score uses shifted prior-30-day mean and standard deviation,
  requiring at least 20 days. No forward filling is allowed.
- Frozen monthly communities come from the v13.2 extended balanced membership.
- Monthly BTC beta uses only the preceding 30 days and at least 2,000 paired
  15-minute returns.
- Settled Binance funding at the exact entry timestamp is excluded from every
  score. Portfolio funding PnL includes settlements in `(entry, exit]`.
- Entry is Monday 00:00 UTC and exit is exactly seven days later. Only complete
  weeks are included.
- Development is before 2026-01-01, validation is January-February 2026, and
  holdout begins 2026-03-01.

## Frozen candidates

### `GPRV1_GLOBAL_GRAPH_PEER_PREMIUM_WEEKLY`

Across all eligible alts, long the eight lowest graph-peer premium scores and
short the eight highest. Raw long and short sleeves each carry 0.5 absolute
notional. Add the causal BTC-beta hedge and normalize gross notional to one.

### `GPRV2_COMMUNITY_GRAPH_PEER_PREMIUM_WEEKLY`

Within every community with at least four eligible symbols, long its lowest and
short its highest graph-peer premium score. Weight all community pairs equally,
add the causal BTC-beta hedge, and normalize gross notional to one.

Weights remain fixed during each week. Every transition, initial opening,
mandatory exit, data gap, and terminal close is fully charged. No turnover cap
or target blending is permitted.

## Frozen costs, controls, and diagnostics

- Primary/stress one-way cost: 20/40 bp times exact full L1 turnover.
- Price PnL and settled-funding PnL are reported separately, including the BTC
  hedge's funding cash flow.
- Exact reversed weights and a one-week delayed-score portfolio.
- Global own-premium-z portfolio as the graph-residual control.
- Funding-orthogonal graph-premium score as a global diagnostic; it is not a
  third candidate.
- 500 deterministic random controls. The global null randomly selects equal
  long/short sleeves of the same size. The community null randomly selects two
  distinct members per eligible community. Each path uses its own causal BTC
  hedge, realized turnover, price and funding PnL, and the two-candidate family
  maximum.
- Four-week moving-block bootstrap with 2,000 draws.
- Leave-one-month-out means, long/short sleeve attribution, symbol contribution,
  and overlap correlation with FSS3 are reported.

## Frozen gates

- At least 40 complete weeks, ten active months, eight validation weeks, and 12
  holdout weeks.
- Positive primary mean in development, validation, and holdout.
- Positive full stress mean, price contribution, BTC-residual gross mean, and
  four-week bootstrap 95% lower bound.
- At or above the random family-max 95th percentile.
- Beat reversed direction, one-week delayed score, and the candidate-specific
  graph or random control.
- Global funding-orthogonal diagnostic remains positive after primary cost.
- Both long and short sleeves have positive full-sample gross contribution.
- Mean turnover no greater than 1.75 and numerical BTC-beta/gross constraints
  hold.
- Absolute primary-return correlation with FSS3 no greater than 0.60.
- Every leave-one-month-out mean is positive; no single symbol contributes more
  than 25% of positive gross PnL; no profitable month supplies more than 35% of
  positive monthly PnL; worst chronological period mean is at least -40 bp/week.

Passing creates an offline research candidate only. No PaperLive, live,
application, leverage, remote, or real-order scope may change.
