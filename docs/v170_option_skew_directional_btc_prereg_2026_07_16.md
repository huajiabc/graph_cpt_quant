# v17.0 BTC Option-Skew Directional Alpha Preregistration

Date frozen: 2026-07-16, after rejecting v16.7--v16.9, but before constructing
or inspecting any skew-conditioned BTC perpetual return.

## Distinct hypothesis

Actual option legs were uneconomic at Binance's archived bid/ask width, but the
surface can still be an orthogonal information source. Does a causal extreme in
30-day BTC 25-delta put-versus-call implied-volatility skew predict the next day's
BTC direction strongly enough to trade only the low-cost perpetual?

The single candidate is `OSD1_25D_SKEW_FOLLOW_BTC`. Positive put-minus-call skew
means downside protection is relatively expensive and sets a BTC short; negative
skew sets a BTC long. This directional sign is frozen from the demand-pressure
hypothesis, not chosen from outcome data.

## Frozen surface and causal state

- Use the official Binance BTC option hour-0 snapshots with the conservative
  01:00 UTC availability time and the exact BTC USD-M close known by then.
- Keep expiries 21--45 DTE and choose the expiry closest to 30 days, breaking ties
  toward the earlier expiry.
- Require positive, consistent bid/ask prices and quantities and positive finite
  mark IV.
- Within the chosen expiry, select the call whose delta is closest to +0.25 and
  the put whose delta is closest to -0.25. Require call delta in [0.10, 0.40] and
  put delta in [-0.40, -0.10]. The two legs need not share a strike because they
  are measurements, not a traded spread.
- Daily skew is put mark IV minus call mark IV.
- Compute its z-score against the preceding 30 valid surface observations,
  shifted by one observation, with population standard deviation. Require the
  oldest and newest historical surface observations to span no more than 45
  calendar days. No missing surface is filled.

## Frozen signal and return

Trade only when absolute skew z-score is at least 1.0:

- z >= +1: short BTC;
- z <= -1: long BTC.

Enter the BTC perpetual at the 01:00 UTC close already used by the signal and exit
at the next calendar day's 01:00 UTC close. Consecutive daily trades are allowed
and do not overlap beyond the shared close. Primary total round-trip cost is 10 bp
and stress cost is 20 bp. Do not add funding benefit; report any known funding only
as a future audit item. No threshold, tenor, delta, horizon or direction grid is
allowed.

Chronological labels remain development before 2023-08-01, validation from
2023-08-01 through 2023-09-14, and holdout from 2023-09-15.

## Frozen controls and gates

Controls are the reversed trade direction, a one-calendar-day delayed skew signal,
and 2,000 circular shifts of the complete signed signal over the same BTC-return
calendar. Bootstrap entry days 5,000 times.

Research-follow-up eligibility requires:

- at least 30 trades and at least eight in each chronological period;
- positive primary mean in development, validation and holdout;
- positive full-sample stress mean and bootstrap 95% lower bound;
- real primary mean at or above the 95th percentile of circular controls;
- better primary mean than delayed and reversed controls;
- positive-PnL month and trade concentrations at most 50% and 35%;
- worst primary trade no worse than -500 bp.

Passing grants only an independent audit and a recommendation for longer/forward
surface recording. The archive is old and short; no result can authorize
PaperLive, remote changes, leverage or real orders.
