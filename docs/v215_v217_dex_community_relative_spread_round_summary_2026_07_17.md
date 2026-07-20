# v21.5-v21.7 DEX Community Relative-Spread Round Summary

Verdict: `weak_relative_effect_rejected_not_leverage_worthy`.

This second-stage branch tested the only stable residual from the rejected v21.3
directional propagation study: whether slower community peers outperform faster
peers after a DEX attention event.  It was explicitly marked post-reveal, so the
same history could only assess magnitude, not supply independent promotion proof.

The v21.5 feature audit formed equal-sized, disjoint laggard and leader halves from
returns observed by the feature close.  It passed all feature checks with 274
events, 15 source symbols, 32 communities, and a median pre-entry rank gap of
67.37 bp.  The reveal realized 267 complete events after excluding the vendor
transition and requiring causal beta plus price endpoints.

| Metric | Result |
|---|---:|
| 12-hour gross spread | +9.8349 bp |
| Net at 5 bp round trip | +4.8349 bp |
| Net at 10 bp round trip | -0.1651 bp |
| Net at 20 bp round trip | -10.1651 bp |
| Random-rank percentile | 0.9280 |
| Day-bootstrap lower 95%, net at 20 bp | -21.6665 bp |

Gross results by development/validation/holdout were +10.0858, +14.8778, and
+3.5547 bp.  The effect therefore weakened rather than strengthened in holdout.
An additional 15-minute delay retained +8.5097 bp gross, while the +24-hour
placebo reversed to -10.1875 bp gross.  The timing pattern is compatible with a
short-lived relative response, but its magnitude is below the cost hurdle and its
positive PnL is highly concentrated (month share 95.71%, source share 50.10%).

The independent v21.7 audit passed 26/26 checks, reproducing causal beta, dollar
and beta neutrality, exact PnL, costs, all 500 random-rank paths, bootstrap,
concentration, and rejection.

Conclusion: DEX attention contains a weak community rank signal, not a robust
tradable alpha under the project's execution assumptions.  Leverage would amplify
negative expected net return and estimation risk, so this branch is terminated
rather than promoted or paper-traded.

No live, PaperLive, application, leverage, remote, or order state was read or
changed.
