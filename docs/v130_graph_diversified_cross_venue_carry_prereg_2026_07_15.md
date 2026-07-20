# v13.0 Graph-Diversified Cross-Venue Carry Preregistration

Date frozen: 2026-07-15, before inspecting any v13.0 return.

## Mechanism

TG1's concentration attribution shows that multiple dominant funding-spread
contracts can occupy the same frozen price-behavior community. `GC1_30D_COMMUNITY_TOP1_HOLD2`
therefore reuses the unchanged v12.5 panel and 30-day funding-spread score but
permits at most one contract per frozen monthly community.

Within each community, select its highest positive score. At later weekly
decisions retain the previously held symbol if it is still in that current
community, remains positive, and ranks in its top two; otherwise replace it with
the current top positive symbol. Trade only when at least six communities have a
positive candidate. Every active community receives fixed 1/8 pair-notional;
missing-community capital remains cash. No global fill, basis filter, volatility
gate, or realized-return cap is allowed.

## Costs, controls, and gates

- Primary/stress costs are 20/40 bp one-way times realized symbol-weight
  turnover, including initial entry and terminal close.
- 2,000 week bootstrap draws use primary net returns.
- 200 random monthly 8-by-9 partitions run the exact same top-1/hold-2 strategy,
  minimum-six breadth, fixed 1/8 weights, and costs.
- Promotion requires at least 40 weeks, ten months, ten validation weeks, eight
  holdout weeks, mean invested exposure at least 75%, positive primary net in all
  periods, positive stress net and funding contribution, positive bootstrap
  lower bound, random-partition percentile at least 90, positive-month
  concentration no greater than 35%, and worst period no worse than -40 bp/week.

Passing means forward graph-carry shadow candidacy. PaperLive is unchanged.
