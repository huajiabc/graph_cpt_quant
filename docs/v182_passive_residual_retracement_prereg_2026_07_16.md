# v18.2 Passive Residual Retracement Preregistration

Status: frozen before the passive-fill outcome is revealed.

## Fixed research primitive

- Reuse exactly the audited v18.0 q97.5 event timestamps, Bottom5 laggards,
  Top5 leaders, and frozen monthly BTC betas.
- No dispersion threshold, bucket membership, symbol universe, or beta is
  reselected.

## Causal passive-entry protocol

At the completed event close `t`, place ten one-bar good-till-cancelled limits:

- each laggard: buy at `close_t * (1 - 10 bp)`;
- each leader: short at `close_t * (1 + 10 bp)`.

The next completed 15-minute low/high determines fills. A touch fills exactly at
the limit with no favorable price improvement. Orders not touched are cancelled.
All fills are retained in PnL; no event or unbalanced partial fill may be removed
after observing the bar.

Each intended alt leg has absolute weight 10% divided by the original v18.0
gross normalizer. Unfilled allocations remain cash. At the next completed close,
enter a BTC hedge equal to the realized beta exposure of filled alt legs. Exit all
filled alts and the BTC hedge four bars / one hour later. Thus the exit is five
bars after the source event.

## Frozen costs

- Filled alt allocation: 10 bp primary and 20 bp stress total round trip,
  representing passive entry plus market exit and slippage allowance.
- BTC hedge allocation: 8 bp primary and 12 bp stress total round trip.
- Costs scale by realized filled/hedge gross allocation; unfilled cash has no cost.

## Controls

- One-bar delayed order placement with the original event and membership.
- 500 deterministic random disjoint five-buy/five-short assignments at identical
  timestamps, offsets, holding periods, hedge rules, and costs.
- Fixed 5 bp and 20 bp entry-offset sensitivity diagnostics.
- Exact reversed realized alt/hedge PnL direction after fills.
- Day-block bootstrap and development/validation/holdout splits inherited from
  v18.0.

## Gates

- At least 100 source events with at least one fill, 20 validation, and 25 holdout.
- At least 40% fill rate on both laggard and leader intended allocations.
- Primary net mean positive in development, validation, and holdout.
- Full stress net mean and day-block bootstrap 95% lower bound positive.
- At or above the 95th percentile of the 500 random-rank controls.
- Must beat one-bar delay and exact reversed PnL direction.
- Both 5 bp and 20 bp offset diagnostics remain positive after their identical
  cost schedules.
- No single profitable month supplies more than 35% of positive monthly PnL.

A pass is only an offline research candidate because passive fills are bar-based
and historical. No PaperLive, application, leverage, remote, or real-order scope
may change in this round.
