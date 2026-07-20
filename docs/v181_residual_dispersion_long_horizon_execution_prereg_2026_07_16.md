# v18.1 Residual Dispersion Long-Horizon Execution Preregistration

Status: frozen before any outcome beyond the audited v18.0 60-minute horizon is
revealed.

## Fixed input

- Reuse exactly the v18.0 q97.5 dispersion events, source timestamps, Bottom5
  laggards, Top5 leaders, and frozen monthly BTC betas.
- No event threshold, bucket size, symbol membership, or beta is reselected.
- The v18.0 500-random-rank attribution result is inherited; this round tests
  holding/execution feasibility, not another rank-selection family.

## Event-sleeve extension

- Primary holding period: 16 bars / four hours.
- Diagnostics: 8 bars / two hours, 32 bars / eight hours, and 48 bars / twelve
  hours.
- Each sleeve remains half-long laggards and half-short leaders, with the frozen
  BTC beta-difference hedge and gross-exposure normalization.
- Event-sleeve primary/stress round-trip costs remain 30/40 bp.

The four-hour primary horizon is frozen because audited gross compression rose
monotonically from 15 to 30 to 60 minutes. No return beyond 60 minutes has been
inspected before this registration.

## Continuous executable book

At every completed bar, each still-active four-hour sleeve contributes its frozen
normalized weights. The executable target book is the arithmetic mean of all
active sleeves; it is zero when no sleeve is active. This caps intended gross
exposure near one and permits natural netting between overlapping events.

- Bar return uses target weights formed at the prior completed close.
- Turnover is the L1 change in target weights, including entries, exits, and
  changes caused by the active-sleeve average.
- Primary/stress one-way costs are 15/20 bp per unit gross turnover, exactly
  corresponding to 30/40 bp for a full enter-and-exit cycle.

## Gates

- At least 100 four-hour event sleeves, 20 validation, and 25 holdout.
- Four-hour event-sleeve primary net mean positive in development, validation,
  and holdout; full stress net mean and day-block bootstrap lower bound positive.
- Continuous primary and stress cumulative net return positive in development,
  validation, and holdout; day-block bootstrap lower bound positive.
- Two-, eight-, and twelve-hour event-sleeve primary net means positive.
- Four-hour compression must beat exact reversed residual momentum.
- No single profitable month supplies more than 35% of positive continuous
  monthly PnL.

Because the same historical events motivated this holding extension, a pass can
only create a follow-up/forward-observation candidate, not authorize PaperLive,
leverage, remote changes, application scope changes, or real orders.
