# v15.4 Binance Book-Depth Imbalance Pre-registration

Date frozen: 2026-07-16, before inspecting any strategy return produced from
the Binance `bookDepth` archives.

## Question

Does persistent, directly displayed near-touch liquidity contain a cross-sectional
24-hour return signal that is not already represented by price, funding, open
interest, or taker-flow features?

The economic direction is frozen as **continuation**: coins with more cumulative
bid than ask notional inside 0.2% are bought; coins with less are sold. A sign-reversed
portfolio is a control and cannot replace the primary candidate.

## Frozen universe and sample

The universe is the top two July-2025 turnover members from each of the eight
August-2025 v11.0 frozen communities. This selection uses only information before
the test sample and is fixed at:

`SOLUSDT, DOGEUSDT, 1000PEPEUSDT, WIFUSDT, ETHUSDT, ENAUSDT, HBARUSDT,
AVAXUSDT, LINKUSDT, ONDOUSDT, XRPUSDT, XLMUSDT, FARTCOINUSDT, WLDUSDT,
SEIUSDT, TIAUSDT`.

- Raw depth request window: 2025-07-01 through 2026-07-14 UTC.
- Return window: the next 24 hours after each daily decision.
- Development: decisions through 2025-12-31.
- Validation: 2026-01-01 through 2026-03-31.
- Holdout: decisions from 2026-04-01 onward.
- A decision is usable only when all 16 symbols have a valid prior-day feature and
  entry/exit prices. Missing days are dropped as a whole; symbols are not silently
  substituted.

## Frozen feature and timing

For each symbol and UTC day, use all valid 30-second snapshots at the `-0.2` and
`+0.2` percentage rows. The daily feature is the median across snapshots of:

`(bid_notional_-0.2 - ask_notional_+0.2) /
 (bid_notional_-0.2 + ask_notional_+0.2)`.

The feature for decision day `D` is computed only from source day `D-1`. The
portfolio is formed at `D 00:00 UTC` and held to `D+1 00:00 UTC`. No same-day
depth observation is available to the decision.

The analogous depth and notional imbalance at 1% and 5%, daily mean, standard
deviation, and snapshot counts are coverage/diagnostic fields only. They cannot
alter the primary ranking or promotion decision.

## Frozen portfolio

Candidate: `BD1_PRIOR_DAY_NEAR_TOUCH_CONTINUATION`.

- Rank the 16 coins descending by the frozen feature.
- Long four and short four.
- Turnover band: retain an existing long while it remains in the top eight and an
  existing short while it remains in the bottom eight; fill vacancies in rank order.
- Start alt weights at +0.125 per long and -0.125 per short.
- Add an exact BTC hedge using causal trailing 30-day hourly betas available at the
  decision; normalize the completed portfolio to gross notional 1.0.
- Primary cost is 20 bp per one-way L1 weight turnover; stress cost is 40 bp.
- Turnover includes BTC hedge changes and the initial opening. The final sample close
  is not charged because each row is a one-day marked portfolio observation, not a
  forced liquidation claim.

## Frozen controls

1. Sign-reversed portfolio with otherwise identical timing and costs.
2. One-day shifted feature (`D-2` information at `D`) with identical construction.
3. 1,000 within-day random rankings with the same 4/4 cardinality, turnover band,
   causal beta hedge, and costs.
4. Report 1% and 5% feature versions as diagnostics only; neither can be promoted
   in this experiment.

## Frozen gates

Promote only to a local `forward-shadow candidate`, never directly to PaperLive,
when every condition is true:

- at least 300 usable decision days and 10 calendar months;
- at least 80 validation and 80 holdout days;
- primary and stress mean net return are positive overall;
- primary mean is positive in development, validation, and holdout;
- 7-day moving-block bootstrap 95% lower bound of primary mean is above zero;
- observed primary mean is at or above the 95th percentile of random rankings;
- the largest positive month contributes no more than 35% of total positive-month PnL;
- mean one-way turnover is no more than 0.50;
- primary mean exceeds both reversed and one-day-shifted controls;
- maximum absolute residual BTC beta and gross-normalization drift are each at most
  `1e-10`.

Any failed gate rejects the candidate without threshold, universe, sign, horizon,
or depth-band tuning in v15.4. PaperLive and remote state remain unchanged.
