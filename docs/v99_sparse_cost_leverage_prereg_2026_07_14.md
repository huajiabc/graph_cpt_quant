# v9.9 Sparse Cost-Aware Residual Trading and Leverage - Pre-Registration

## Scope and permission

This is an offline historical research and risk-stress lane. It cannot change
shadow, P2, paper-live, canary-live, or real-live permissions. Leverage is
evaluated only after the unlevered decision rule is frozen; leverage may not be
used to turn negative expectancy into a candidate.

## Frozen inputs and chronology

- Inputs are the v9.8 monthly walk-forward OOS predictions for the four-hour
  and twelve-hour residual experiments.
- Models: `ridge_residual`, `xgb_residual`, and the negative control
  `xgb_residual_shuffled`.
- January-February 2026 are development only and may select an entry hurdle.
- March-April are untouched validation and May is the final historical holdout.
- The four-hour result is read before the twelve-hour robustness result. No
  parameter may change between horizons.
- The strategy uses only the current model score and current available symbol
  set. Realized return fields are used only by the ledger after selection.

## Frozen sparse cost-aware policy

Books:

- `long_sparse`: up to five beta-neutral long residual slots;
- `long_short_sparse`: up to five long and five short residual slots, with
  unfilled capacity left in cash.

Every filled slot has fixed capacity: 20% in the long book and 10% on each side
of the normalized long-short book. Fewer names therefore reduce gross exposure
instead of concentrating the remaining names.

Entry hurdle candidates, expressed in model-predicted residual-return units:

- 0, 2.5, 5, 10, and 20 basis points.

For a long entry, score must be at least the hurdle. For a short entry, score
must be no greater than the negative hurdle. An incumbent exits to cash when
its score crosses zero. A filled slot is replaced only when the best outside
candidate improves predicted residual return by at least 20 basis points, the
focal full-replacement cost hurdle. Missing symbols exit immediately.

Turnover for a side is `max(entries, exits) / 5`, so entry from cash, exit to
cash, and name replacement are all charged. Long-short turnover is the average
of long- and short-side turnover. Costs are evaluated at 10, 20, 30, and 50
basis points; 20bp is focal.

## Development-only selection

Each model/book combination runs all five active hurdles on January-February.
The active hurdle with highest development net20 is frozen, with ties resolved
by lower turnover and then the higher hurdle.

Cash/off is an explicit development alternative with zero return and zero
turnover. If all active hurdles have development net20 at or below zero, the
honest selected strategy is `off`. The best active hurdle is still carried
forward as `diagnostic_active` so that validation, holdout, and leverage risk
can be measured, but it cannot become a candidate.

Validation starts from cash on 2026-03-01. The frozen policy then runs
continuously through May, so May inherits only positions that would actually
have survived from April.

## Random and negative controls

- `xgb_residual_shuffled` follows the identical development selection process.
- 500 random-score paths permute the focal model scores separately inside every
  timestamp, preserving score distributions, symbol availability, returns,
  sparse capacity, threshold selection, and turnover accounting.
- Random paths also select their hurdle using January-February only before
  March-May is read.

## Unlevered decision gates

A sparse strategy is only `research_candidate_only` if all are true:

1. development selects an active strategy rather than `off`;
2. March-April validation and May holdout net20 are both positive;
3. combined March-May net30 is positive;
4. at least two of the three evaluation months have positive net20;
5. average March-May turnover is at most 10%;
6. it exceeds the matched random-control 90th percentile;
7. a real-label model beats shuffled-label XGBoost;
8. no evaluation month supplies more than 60% of positive net20.

One holdout month still forbids any live promotion.

## Frozen leverage stress

Leverage multipliers are `1x`, `1.5x`, `2x`, `3x`, and `5x`. They are applied
to the frozen March-May `diagnostic_active` ledger; no leverage-specific hurdle
selection is allowed.

Two accounting cases are reported:

- `transaction_only_upper_bound`: leverage multiplies both residual portfolio
  return and transaction cost;
- `carry_stress`: the same result minus 2bp per eight hours per unit of deployed
  gross exposure and leverage. This is a fixed synthetic stress because the
  current historical ledger does not contain complete realized funding and
  borrow cash flows.

For each case report compounded equity multiple, additive return, annualized
volatility, maximum drawdown, worst period, 5% expected shortfall, and counts of
period returns below -20%, -50%, -80%, and -100%. A return at or below -100% is
recorded as ruin. These are research proxies, not exchange-specific liquidation
prices.

Leverage is ineligible regardless of its aggregate return unless the underlying
1x strategy first passes every unlevered gate. For an aggressive research risk
label, 2x additionally requires carry-stressed maximum drawdown no worse than
-30% and no period below -50%; 3x requires drawdown no worse than -45% and no
period below -50%. `5x` is tail-risk stress only and cannot be recommended from
this sample.
