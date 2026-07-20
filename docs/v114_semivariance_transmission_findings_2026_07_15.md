# v11.4 Sign-Specific Semivariance Transmission Findings

Date: 2026-07-15

Verdict: `reject_semivariance_transmission_family`.

## Results

| Candidate | Observations | Gross 4h | Net 20 bp | Validation net 20 | Holdout net 20 | Random-family percentile |
|---|---:|---:|---:|---:|---:|---:|
| Downside cascade | 220 | +18.91 bp | -1.09 bp | -1.60 bp | -14.75 bp | 78% |
| Upside cascade | 211 | -4.80 bp | -24.80 bp | -39.26 bp | -22.59 bp | 0% |

The downside branch was materially better than the direction-free v11.3 breakout and the upside
branch, but it did not clear the frozen 20 bp cost. Net 30 bp was -11.09 bp. Its entry-day
bootstrap 95% interval was [-31.61, +32.83] bp, three of five chronological slices were negative,
and it did not clear the random-family 90th-percentile gate.

The downside result also weakened through time: development net 20 bp was +9.80 bp, validation
was -1.60 bp, and the holdout label was -14.75 bp. The one-day-shifted result was -14.96 bp, so
timing carries some information, but not enough to establish stable executable alpha.

## Attribution

Downside raw gross return was +18.91 bp, while BTC-residual gross return was only +5.53 bp. Most
of the apparent receiver profit therefore came from continued system-wide market decline rather
than follower-specific downside transmission. The real graph's 78th percentile versus random
family maxima supports the same interpretation.

The upside branch was rejected more decisively. Its raw gross return was negative even before
cost, all five chronological slices were negative, and it ranked at the 0th random-family
percentile.

## Research consequence

Sign separation reveals a real asymmetry: broad downside shock states contain more subsequent
market-direction information than upside states. It does not validate edge-level receiver alpha.
The next distinct question is whether downside shocks distributed across multiple frozen graph
communities lead *systematic* BTC downside before BTC itself has already fallen. That is a
cross-sectional volatility-front timing hypothesis, not another repair of v11.4 edges.

No PaperLive, leverage, or real-order permission changed.
