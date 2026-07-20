# v23.37 Liquidation Mechanism Independent Audit

Verdict: `mechanism_supported_regime_marker_not_standalone_alpha`.

Checks: 8; passed: 8; failed: 0.

The raw 15-minute liquidation-intensity correlation with next-60-minute BTC range is +0.374, and its circular-shift percentile is 95.7%.

However, correlation with the preceding 60-minute range is stronger at +0.593. After controlling for that prior range, the partial rank relation to future range is only +0.111. The one-day evidence therefore supports liquidation flow as a timely high-volatility regime marker, but not as a standalone volatility predictor.

Directional continuation is also unsupported: alt forced-flow continuation is 45.4%, with mean signed return -3.28 bp. The only retained research role is an OCO activation/avoidance overlay or regime covariate, subject to the v23.35 forward gate.

This remains retrospective, non-promotable evidence.
