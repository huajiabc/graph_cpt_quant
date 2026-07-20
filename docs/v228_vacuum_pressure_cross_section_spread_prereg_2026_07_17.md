# v22.8 Vacuum-Pressure Cross-Section Spread Preregistration

Date frozen: 2026-07-17, after v22.7 feature-only audit and before inspecting
any Top4-minus-Bottom4 future return.

## Hypothesis

The v22.4 broad liquidity-vacuum state forecasts larger future market moves but
not their common direction. Its untested monetization route is cross-sectional:
within each event, alts with the strongest standardized one-percent bid-side
imbalance should outperform those with the weakest imbalance as the volatile
state resolves.

## Frozen feature input

- `reports/v22_7_vacuum_pressure_cross_section_feature_audit/ranked_symbol_features.parquet`
- SHA256:
  `5B7D351886C63DA3178B81C101BF24A47CA90085651A6654B414226674DD546E`
- v22.7 passed 11/11 checks: exactly 159 events/11 months, four long and
  four short names per event, raw 0.5 notional per side, strict score ordering,
  exact feature/entry time, and four-hour event cooldown.
- No beta, price, future return, PnL, or outcome entered the rank construction.

## Frozen candidate

`DVS1_VACUUM_PRESSURE_TOP4_MINUS_BOTTOM4`

- At event time `t`, long the four highest causal `imbalance_z` names and short
  the four lowest, with raw weights `+/-0.125`.
- At each calendar month start, estimate every alt's BTC beta from exact Bybit
  hourly close returns in `[month_start-30d, month_start)`, requiring at least
  500 paired observations. No target-month price enters beta estimation.
- Add the exact BTC hedge `-sum(raw_alt_weight * beta)` and normalize total
  absolute alt-plus-BTC gross to one.
- Enter at the exact Bybit hourly close at `t` and exit at `t+4h`. Missing exact
  marks or any missing selected beta drop the event; no nearest-time fill.
- Primary/stress total round-trip costs are fixed at 30/40bp per completed
  event. The four-hour feature cooldown prevents primary-event overlap.

## Secondary views and controls

- One-hour and eight-hour beta-neutral returns use the same entry weights; they
  describe horizon shape and cannot rescue the four-hour endpoint.
- Raw dollar-neutral Top4-minus-Bottom4 return after 20bp is diagnostic.
- Exact reversed rank direction on the same event times.
- One-hour delayed entry (`t+1h` to `t+5h`) with the same frozen weights and
  30bp cost.
- 1,000 random-rank paths: independently permute all 16 symbols within every
  event, assign four random longs/four random shorts, apply the same monthly
  betas, hedge, normalization, prices and 30bp cost.
- 2,000 entry-day cluster-bootstrap draws of primary event return.
- Report chronological periods, long/short/hedge attribution, monthly and day
  concentration, random percentile, delayed/reversed controls and cost frontier.

## Frozen gates

Research candidacy requires all of:

- at least 150 events, 11 months, and 45 events per chronological period;
- positive four-hour gross, 30bp primary and 40bp stress mean overall, with
  positive primary mean in development, validation and holdout;
- positive raw dollar-neutral mean after 20bp;
- entry-day bootstrap 95% lower bound above zero;
- random-rank percentile at least 95;
- primary mean above reversed and one-hour-delayed controls;
- largest positive month at most 35% and positive entry day at most 20% of
  total positive primary PnL;
- maximum absolute residual BTC beta and gross drift at most `1e-12`.

Passing remains a second-generation retrospective research result and cannot
change PaperLive, live, leverage, remote, application, or order permissions.
