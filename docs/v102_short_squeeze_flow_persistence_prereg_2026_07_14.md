# v10.2 Short-Squeeze Flow Persistence - Pre-Registration

This is a third-look, path-specific audit selected after reading v10.1 and
v10.1a. It cannot establish independent historical confirmation and cannot
change any live, PaperLive, leverage, or sizing permission.

The candidate is frozen as `path_name == short_squeeze` plus the unchanged
`OF1_CONFIRM_LONG` exact-flow state, held for 240 minutes. No feature, path,
horizon, split, or cost threshold may change.

## Controls and attribution

- raw token-long net10/net20/net30;
- continuous-15m BTC relative gross and two-leg net40;
- candidate versus non-OF1 short-squeeze event lift in each chronological
  segment;
- 2,000 day-block bootstrap iterations for raw net20 and candidate-minus-other
  lift;
- 500 path-specific same-symbol/same-day random 15-minute controls;
- +60-minute shifted-event placebo;
- positive-day contribution concentration.

Because this is a third look, the random-time threshold is the 95th percentile,
not the 90th percentile. A forward-watch clue requires all of the following:

1. at least 50 full, 15 validation, and 12 holdout trades;
2. full, validation, and holdout raw net30 are positive;
3. raw net20 day-bootstrap lower bound is positive;
4. candidate-minus-other raw net20 lift is positive in all three segments and
   its day-bootstrap lower bound is positive;
5. raw net20 exceeds the path-specific random 95th percentile;
6. real timing beats the +60-minute shifted placebo;
7. no positive day contributes more than 35%;
8. continuous-BTC two-leg net40 is non-negative in validation and holdout and
   exceeds the path-specific random 90th percentile.

Passing means only `post_discovery_forward_watch_clue`; it is still ineligible
for PaperLive until genuinely new forward observations accumulate. Failure
closes the single-venue exact-flow persistence branch.
