# v12.0 Walk-Forward Variance Direction Preregistration

Date: 2026-07-15

Status: `PREREGISTERED_SECOND_LOOK`.

This is a second-look model motivated by v11.9. It may judge whether a more
complex model has research value, but it cannot promote itself on the reused
history even if it passes.

## Question

Can a small, fully walk-forward model distinguish months when BTC variance is
cheap from months when it is rich, improving on unconditional short variance?

## Frozen sample and labels

- Reuse the v11.9 first-of-month 08:00 UTC panel.
- Target is the normalized gross short-variance payoff over the next 30 days.
- At prediction time, training may include only signals whose entire 30-day
  realized-volatility label has already completed.
- Minimum training history is 24 resolved monthly labels.

## Frozen features

All features are available at signal time:

- current DVOL;
- trailing 30-day realized volatility;
- DVOL minus trailing realized volatility;
- DVOL divided by trailing realized volatility minus one;
- one-month DVOL percentage change;
- one-month trailing-RV percentage change;
- current DVOL z-score against the preceding 12 monthly observations.

No feature selection follows the result.

## Frozen model and action

- Expanding ridge regression with alpha 10.
- Features are standardized using the current training set only.
- Positive predicted short-variance payoff means short variance; negative
  prediction means long variance.
- Every prediction is traded; there is no confidence threshold.
- One- and two-volatility-point replication penalties are direction-aware:
  short positions lower the effective strike and long positions raise it.

## Frozen controls and gate

Compare with always-short variance, a simple sign rule based on
`DVOL - trailing_RV`, and 2,000 random long/short assignments. Use 5,000
monthly bootstrap draws, chronological fifths, worst month, and normalized
maximum drawdown.

The research gate requires at least 24 full predictions, six validation, and
eight holdout predictions; positive one-vol payoff in full, validation, and
holdout; positive two-vol payoff in full and holdout; positive bootstrap lower
bound; 95th random-direction percentile; and improvement over always-short in
both full and holdout.

Because this is a second look and still a variance-payoff proxy, a pass permits
only a new forward recorder/executable options reconstruction, never direct
PaperLive or leverage.
