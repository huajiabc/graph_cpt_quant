# v16.5 Fixed Core-Satellite FSS3 + TG1 Pre-registration

Date frozen: 2026-07-16, after the project-wide evidence matrix showed that v15.1
failed because inverse-volatility allocation assigned TG1 an average 69% weight,
and before constructing the fixed-weight weekly portfolio below.

## Scope and disclosure

This is a **portfolio-layer construction candidate**, not a new raw alpha. The 20%
satellite cap is evidence-driven and selected after observing sleeve-level summary
statistics: FSS3 is substantially stronger, while TG1 is positive and has -0.069
weekly correlation with FSS3. No weight grid is allowed. Passing can create only a
forward-shadow portfolio candidate and requires new future observations before any
PaperLive allocation change.

## Frozen construction

Candidate: `CM2_FIXED_80_FSS3_20_TG1`.

- Use the exact independently audited v14.9 FSS3 weekly net-return sleeve.
- Use the exact v13.2 forward-extended TG1 weekly net-return sleeve.
- Align only identical entry time, exit time, period and month labels.
- Allocate 80% capital to FSS3 and 20% to TG1 every week.
- Combine price/funding components and primary/stress net returns linearly.
- No additional allocation-turnover charge is added because weights are constant and
  each underlying sleeve return already includes its own complete position transition
  and trading costs; scaling those sleeve returns scales the underlying costs.
- No leverage is added.

The sample, development/validation/holdout labels and 49 aligned weeks are inherited
unchanged from v15.1.

## Frozen benchmarks and gates

FSS3 standalone is the core benchmark; TG1 standalone is reported as the satellite
benchmark. All gates must pass for forward-shadow portfolio candidacy:

- at least 45 weeks, 11 months, 10 validation weeks and 10 holdout weeks;
- both sleeves have positive mean primary return and absolute correlation at most
  0.25;
- combined primary and stress means are positive;
- combined primary mean is positive in development, validation and holdout;
- four-week moving-block bootstrap 95% lower bound is above zero;
- minimum leave-one-month-out mean is above zero;
- largest positive month contributes at most 35% of positive-month PnL;
- combined mean is at least 75% of FSS3 standalone mean, preventing excessive alpha
  dilution;
- additive maximum drawdown is reduced by at least 15% versus FSS3;
- combined downside semideviation is lower than FSS3 downside semideviation;
- maximum weight drift from 80/20 is at most `1e-12`.

Failure of any gate rejects the construction. The 80/20 weight cannot be replaced by
90/10, 75/25, risk parity, volatility targeting or regime switching in v16.5.
PaperLive and remote state remain unchanged.
