# v17.5 Deribit Skew Receiver-vs-Insulator Spread Preregistration

Status: `PREREGISTERED_OFFLINE_RESEARCH_ONLY`

v17.3 directional and v17.4 OCO outcomes were rejected before this rule was
frozen. No receiver-minus-insulator spread result was inspected.

## Mechanism

If BTC downside-skew shocks represent a market-wide stress repricing, the coins
with the strongest lagged BTC volatility-transmission edges may underperform the
coins with the weakest edges even when the market's absolute direction is not
forecastable. This test removes the common market direction with a unit-gross
cross-sectional spread.

## Causal graph buckets

- Reuse the v17.3 monthly graph estimated from the prior 90 days.
- Receiver bucket: the existing top four positive forward absolute-return edges,
  ranked by direction advantage and then forward edge; require at least three.
- Insulator bucket: after excluding receivers, take the same number of eligible
  alts with the lowest forward absolute-return correlation, ranked ascending.
- Both buckets are fixed at month start and held equal weight internally.

## Signals and candidates

- Reuse the exact v17.3 stress and relief events with three-day cooldown.
- `DSS1_STRESS_SHORT_RECEIVER_LONG_INSULATOR`: short 50% receiver bucket and
  long 50% insulator bucket for 24h after a stress event.
- `DSS2_RELIEF_LONG_RECEIVER_SHORT_INSULATOR`: reverse the spread after a relief
  event.
- Both candidates are promotion-eligible as one corrected family.
- Primary total round-trip portfolio cost: 40 bp; stress cost: 60 bp.
- No leverage; entry and exit use the exact closed hourly prices at signal time
  and 24 hours later.

## Controls

- 500 random disjoint bucket-pair assignments of matching sizes at the same
  event times; use the candidate-family maximum.
- Signal delayed by one day.
- Fixed 8h and 48h holding sensitivities, charged the same 40 bp primary cost.
- Development, validation, holdout, and calendar-year decomposition.
- Event bootstrap with 2,000 iterations.

## Candidate gate

Each candidate must independently pass all:

- at least 25 full, 5 validation, and 8 holdout events;
- primary-cost mean positive in full, validation, and holdout;
- full stress-cost mean positive;
- bootstrap 95% lower bound positive;
- random-pair family percentile at least 90%;
- beats one-day delay;
- 8h and 48h sensitivities both positive after primary cost;
- no single positive year exceeds 50% of total positive PnL.

Passing creates an offline research candidate only. No PaperLive, application,
remote, leverage, or real-order permission changes.
