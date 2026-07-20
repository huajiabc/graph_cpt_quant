# v17.1 BTC Option-Skew Innovation Alpha Preregistration

Date frozen: 2026-07-16, after v17.0's level-skew result, but before computing or
inspecting any innovation-conditioned BTC return.

## Adaptive disclosure and distinct state

v17.0's causal skew-level z-score was directionally degenerate by period: every
development trade was long and every validation trade was short. That indicates
slow surface-baseline drift rather than a stable daily demand shock. v17.1 removes
the level and tests only the one-observation innovation.

The single candidate is `OSD2_25D_SKEW_INNOVATION_FOLLOW_BTC`. It retains the
economic direction: a sudden increase in put-minus-call IV means short BTC; a
sudden decrease means long BTC.

## Frozen construction

Reuse v17.0's exact 25-delta surface, quote, DTE, timestamp, BTC price and period
rules. Define skew innovation as current skew minus the immediately preceding
valid skew observation. Require those two surface timestamps to be at most three
calendar days apart; otherwise the innovation is missing.

Compute innovation z-score against the preceding 30 valid innovations, shifted
one observation, using population standard deviation. The 30-history window may
span at most 45 calendar days. No missing innovation is filled.

Trade when absolute innovation z-score is at least 1.0:

- z >= +1: short BTC for the next calendar day;
- z <= -1: long BTC for the next calendar day.

Entry/exit, consecutive-trade handling, primary 10 bp and stress 20 bp total
round-trip costs are identical to v17.0. No funding benefit, threshold grid,
tenor grid, horizon grid or direction reversal is allowed.

## Controls and gates

Controls are reversed direction, a one-calendar-day delayed signed innovation,
and 2,000 circular shifts over the same BTC-return calendar. Bootstrap entry days
5,000 times.

Research-follow-up eligibility requires at least 25 trades and at least five in
development, validation and holdout; positive primary mean in all three periods;
positive full stress mean and bootstrap lower bound; circular percentile at least
95%; superiority to delayed and reversed controls; short fraction between 25% and
75% in both development and validation; positive month/trade concentration at
most 50%/35%; and worst trade no worse than -500 bp.

Passing allows only an independent audit and longer/forward surface collection.
Failure closes the tested BTC option-skew directional family. No PaperLive,
remote, leverage or real-order permission is granted.
