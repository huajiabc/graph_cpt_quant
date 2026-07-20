# v12.8 TG1 + Frozen P2 Orthogonal Combination Preregistration

Date frozen: 2026-07-15, before constructing or inspecting aligned combination
returns.

## Frozen sleeves

1. `TG1_30D_TOP9_HOLD18`: the exact corrected v12.6 weekly primary-net series,
   already including 20 bp one-way realized-turnover costs.
2. `P2_CIC_COMBINED_BASKET_MAX8`: the existing frozen v0.7D2 replay portfolio.
   Only rows with that exact portfolio id and `selected=true` are included. Each
   trade uses its already costed `net_return_20bp` and contributes one eighth of
   sleeve capital. Trades are assigned to the TG1 Monday-to-Monday week by entry
   time and summed; a week with no P2 trade has zero P2 return.

The rejected sparse v11.2 historical strategy is explicitly ineligible as a
combination leg. No trade, week, symbol, or month may be removed.

## Candidate and controls

`CM1_50_50_TG1_P2_MAX8` holds 50% fixed capital in each sleeve. The weight is
not volatility-fit and no alternative weight is evaluated. Combined weekly
return is the arithmetic half-sum of the two already costed sleeve returns.

Report sleeve correlation, contributions, three chronological periods, 2,000
paired week-block bootstrap resamples, active months, positive-month
concentration, and worst-period mean. A one-week circular displacement of P2 is
a diagnostic only and cannot be promoted.

Promotion requires at least 40 aligned weeks, ten months, ten validation weeks,
and eight holdout weeks; positive mean return from each sleeve; absolute sleeve
correlation no greater than 0.50; positive combined return in development,
validation, and holdout; positive paired-bootstrap 95% lower bound; positive-month
concentration no greater than 35%; and worst period no worse than -40 bp/week.

Passing means a forward portfolio-shadow candidate. It does not authorize a new
PaperLive or real-order route. Existing PaperLive processes are unchanged.
