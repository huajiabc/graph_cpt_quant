# v20.1 Reference-Price Transmission Preregistration

Status: frozen after the v19.9 data audit and v20.0 feature-only audit, and
before inspecting any candidate future return.

## Mechanism and orthogonality

The Binance index price represents the external reference basket, mark price is
the fair-value layer used by the perpetual, and the futures close is the last
traded execution layer. The experiment tests two direct completed-bar states:

1. the reference index has moved but mark has not fully followed after removing
   official premium-index innovation; trade receiver catch-up; and
2. the last traded futures price has moved beyond mark; fade the execution-price
   overshoot.

The reference gap is first residualized on official premium innovation using a
shifted prior-30-day rolling regression with at least 20 days of history. The
remaining score is again residualized against the contemporaneous premium
cross section. The v20.0 audit found median absolute cross-sectional correlation
of 5.65e-17 and maximum absolute cross-sectional correlation of 3.07e-15. The
median absolute per-symbol time-series correlation is 0.0345 and its maximum is
0.1276. This is treated as incremental to premium, not as a second copy of it.

## Frozen source events and receiver buckets

- Global source: completed BTC index-return absolute value at or above its
  shifted prior-30-day q90, followed by a four-bar source cooldown.
- Community source: absolute median standardized index return at or above 2.0
  inside each frozen monthly graph community, with a four-bar cooldown applied
  independently per community.
- A receiver must have at least 0.25 standard deviations of completed-bar return
  alignment with the source direction.
- Global reference-lag and both trade-overshoot candidates require receiver
  score at or above 1.5. Community reference-lag uses 1.0 because its final
  double-orthogonal score is standardized globally before restriction to
  smaller graph communities. This threshold choice was made only from v20.0
  coverage, without future-return inspection.
- Rank descending by score, retain at most eight receivers, and require at least
  three. No missing receiver is substituted after the feature time.

Frozen feature coverage is:

| Candidate | Events >=3 | Development | Validation | Holdout | Active months |
|---|---:|---:|---:|---:|---:|
| Global reference catch-up | 986 | 508 | 211 | 267 | 12 |
| Global trade overshoot fade | 619 | 297 | 158 | 164 | 12 |
| Community reference catch-up | 227 | 74 | 74 | 79 | 11 |
| Community trade overshoot fade | 360 | 179 | 98 | 83 | 11 |

## Frozen candidates

1. `RPT1_GLOBAL_REFERENCE_RESIDUAL_CATCHUP`
2. `RPT2_GLOBAL_TRADE_OVERSHOOT_FADE`
3. `RPT3_COMMUNITY_REFERENCE_RESIDUAL_CATCHUP`
4. `RPT4_COMMUNITY_TRADE_OVERSHOOT_FADE`

For catch-up, every receiver is traded in the source direction. For overshoot
fade, every receiver is traded opposite the source direction. Receiver raw
weights are equal. Add a BTC hedge using monthly betas estimated only from the
preceding 30 days with at least 2,000 paired observations, then normalize total
absolute notional to one.

The primary holding period is one 15-minute bar. Entry uses the futures close at
the completed feature timestamp (the next bar's opening boundary) and exit uses
the next completed close. Primary/stress round-trip book costs are 20/40 bp.
Gross break-even cost is reported separately.

## Frozen controls and diagnostics

- Exact reversed trade direction.
- One-bar delayed entry.
- Two- and four-bar holding-period diagnostics.
- Positive- and negative-source sides reported separately.
- Development is before 2026-01-01 UTC; validation is January-February 2026;
  holdout begins 2026-03-01.
- 500 deterministic same-event, same-community where applicable, equal-size
  random receiver controls. Each iteration records the maximum mean across the
  four-candidate family.
- Day-block bootstrap with 2,000 iterations.
- Monthly and symbol contribution concentration, exact beta residual, and gross
  notional drift diagnostics.

## Frozen gates

- Global candidates: at least 500 full, 100 validation, and 100 holdout events.
- Community candidates: at least 150 full, 40 validation, and 40 holdout events.
- At least ten active months.
- Positive primary mean in development, validation, and holdout.
- Positive full stress mean and day-block bootstrap 95% lower bound.
- At or above the random four-candidate family-max 95th percentile.
- Beat reversed direction and one-bar delayed entry.
- Two- and four-bar holding diagnostics remain positive after primary cost.
- Both positive- and negative-source full-sample primary means are positive.
- Maximum absolute BTC-beta residual and gross-notional drift are at most 1e-10.
- No single profitable month supplies more than 35% of positive monthly PnL and
  no symbol supplies more than 25% of positive gross contribution.

Passing creates an offline research candidate only. It does not authorize
PaperLive, live, application, leverage, remote, or real-order changes.
