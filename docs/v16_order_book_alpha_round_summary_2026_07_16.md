# v16 Order-Book Alpha Research Round Summary

## Outcome

No new order-book candidate qualifies for forward shadow. The round nevertheless
closes three previously untested hypotheses with auditable evidence and leaves a
reusable 30-second/hourly data asset.

| Study | Frozen question | Primary result | Decision |
|---|---|---:|---|
| v15.4 | prior-day ±0.2% depth continuation | field did not exist for enough history | rejected before returns |
| v15.5 | prior-day 1% depth continuation, 24h | -10.12 bp/day net; bootstrap upper -1.31 bp | rejected |
| v15.7 | bucket source shock into fragile receiver, 24h | +0.74 bp/day gross, -14.92 bp net; turnover 0.783 | rejected |
| v15.9 | prior-hour 1% depth continuation, next 1h | -0.18 bp/hour gross, -4.42 bp net; all periods negative | rejected |

The v15.5, v15.7 and v15.9 conclusions each have a separate independent audit.
Raw-feature, timing, source/receiver, turnover, cost, BTC beta and gross-normalization
checks pass at numerical tolerances around `1e-12` or tighter.

## Reusable data produced

- 6,056 valid official Binance USD-M daily `bookDepth` archives for the fixed
  16-symbol universe, with raw ZIP retention and SHA-256 manifests.
- 145,342 causal symbol-hour feature rows from strict `[H-60m, H)` windows.
- 9,055 complete 16-symbol trade hours aligned to Bybit entry/exit marks and rolling
  720-hour betas.
- Deterministic raw replay audits for 80 daily samples and 80 hourly samples.

## Interpretation

Displayed cumulative depth **imbalance direction** is not supported as a standalone
linear-contract alpha here. Daily continuation is significantly negative; reversing
it is unstable across periods. Same-bucket shock transmission has almost no gross
edge and excessive turnover. Hourly continuation has approximately zero gross edge
before costs. Random-ranking percentiles are not persuasive because real depth ranks
are persistent and therefore have much lower turnover than random ranks.

The result does not say that the order book contains no information. It narrows the
next test to information not represented by signed imbalance:

1. changes in total displayed depth (withdrawal/replenishment), normalized by recent
   traded notional;
2. cross-venue depth or mid-price divergence once the Bybit tape has enough continuous
   paired history;
3. execution alpha from passive fill probability, which requires a separate fill and
   queue model rather than lowering the current cost assumption.

No rejected sign, 5% diagnostic, or post-hoc subgroup was promoted. The existing
v14.9 FSS3 forward-shadow candidate, PaperLive configurations and remote processes
were not changed.
