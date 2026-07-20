# v22.5 Alt-Book Vacuum Pressure to BTC Preregistration

Date frozen: 2026-07-17, after v22.4 feature-only audit and before any future
price, return, volatility, PnL, or turnover outcome was constructed or read.

## Hypothesis

When standardized one-percent book imbalance points in the same direction
across most of a 16-alt universe while displayed depth is simultaneously being
withdrawn across several names, the cross-sectional book state represents a
systematic pressure front rather than an isolated coin signal. The pressure
direction should propagate into BTC over the next four hours, while the broad
depth vacuum should also precede higher BTC realized volatility.

This differs from prior book studies: v15.5/v15.9 ranked individual coins and
v16.1 multiplied a coin's own prior return by its own withdrawal. No earlier
test aggregated synchronized multi-coin book direction and withdrawal breadth
as a causal BTC propagation signal.

## Frozen feature input

- Candidate events:
  `reports/v22_4_alt_book_vacuum_pressure_feature_audit/candidate_feature_events.parquet`.
- SHA256:
  `A6495D01FD26E05D1762531590A886176CCFDF3AFAE870559072A0132C07D43F`.
- v22.4 passed 16/16 checks and contains no outcome field.
- Exactly 159 feature events across 11 months: 63 development, 47 validation,
  49 holdout; 53 long and 106 short.
- Each event is a false transition with four-hour cooldown, aggregate absolute
  pressure above its shifted trailing-720-hour q90, at least 11/16 symbols in
  the pressure direction, and at least 5/16 symbols in their own shifted
  trailing q20 depth-withdrawal state.

## Frozen primary candidate

`DVB1_ALT_BOOK_VACUUM_PRESSURE_TO_BTC`

- The completed book-feature hour ends at decision/entry time `t`.
- Enter BTC perpetual at the exact Bybit hourly close at `t` in the sign of the
  aggregate alt-book pressure; exit at the exact close at `t+4h`.
- Primary gross return is
  `direction * (BTC_close[t+4h] / BTC_close[t] - 1)`.
- Primary and stress round-trip costs are fixed at 10bp and 20bp per event.
- Missing exact entry or exit marks drop the event; no nearest-time fill is
  allowed. Formal coverage must retain at least 150 events, 45 in every period,
  and 15 of each direction in every period.

## Secondary mechanism views

- One-hour BTC signed return at the same 10bp cost.
- Equal-weight signed return of the same 16-alt universe over four hours at
  20bp primary and 30bp stress cost. This tests whether pressure remains local
  to the source bucket; it cannot rescue a failed BTC endpoint.
- BTC future four-hour hourly realized variance divided by the preceding four
  completed hours. Volatility expansion is a mechanism condition, not a
  substitute for positive tradable return.

No beta hedge or overlapping portfolio construction is used; the feature's
four-hour cooldown matches the primary holding horizon.

## Frozen controls and inference

- Exact reversed direction on the same event times.
- One-hour delayed entry (`t+1h` to `t+5h`) with unchanged direction and cost.
- A feature-only no-vacuum comparator: aggregate pressure still breaches q90
  with at least 11/16 aligned symbols, but fewer than 5/16 withdrawal states;
  false transitions and the same four-hour cooldown are applied. Its 307
  events and direction are frozen without reading outcomes.
- 1,000 random-time paths. For every real event, sample a non-event bucket hour
  from the same calendar month and same pressure direction, preserving the
  real path's month and long/short counts; apply identical pricing and cost.
- 2,000 entry-day block-bootstrap draws of the primary event return.
- Report development/validation/holdout, long/short, monthly and entry-day
  concentration, delayed/reversed/comparator results, random percentile,
  volatility expansion, one-hour decay, and the exact cost frontier.

## Frozen gates

At most a new research candidate requires all of:

- the exact formal coverage floors above and at least 11 active months;
- positive BTC primary and stress mean after cost overall, in development,
  validation, holdout, long events, and short events;
- positive one-hour gross and four-hour gross return;
- day-block bootstrap 95% lower bound above zero;
- random-time percentile at least 95;
- primary mean above reversed, one-hour delayed, and no-vacuum comparator means;
- future BTC variance ratio above one overall, in validation, and in holdout;
- no single positive month above 35% and no single positive entry day above
  20% of total positive primary PnL.

The secondary alt bucket cannot rescue the BTC candidate. Passing remains
retrospective research only and grants no PaperLive, live, leverage, remote,
application, or order permission.
