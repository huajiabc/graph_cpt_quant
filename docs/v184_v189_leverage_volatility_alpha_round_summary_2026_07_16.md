# v18.4-v18.9 Leverage and Volatility Alpha Round Summary

## Outcome

The BTC-inclusive Binance USD-M metrics archive is usable, but no candidate in
this round passes transaction costs, split stability, bootstrap, and controls.
No candidate is eligible for PaperLive or live use.

| Round | Candidate/mechanism | Events | Mean gross | Mean primary net | Control result | Verdict |
|---|---|---:|---:|---:|---|---|
| v18.5 | BTC build directed residual bucket | 189 | -0.17 bp | -30.17 bp | family percentile 1.8% | reject |
| v18.5 | BTC unwind directed residual bucket | 242 | +0.46 bp | -29.54 bp | family percentile 11.4% | reject |
| v18.6 | BTC build continuation | 199 | +2.63 bp | -7.37 bp | family percentile 78.8% | reject |
| v18.6 | BTC unwind reversal | 256 | +5.37 bp | -4.63 bp | family percentile 97.8% | reject; retain sub-cost primitive |
| v18.7 | residual overshoot reversal bucket | 190 | -1.04 bp | -31.04 bp | delayed entry is better | reject |
| v18.7 | synchronized stress reversal bucket | 237 | -1.31 bp | -31.31 bp | delayed entry is better | reject |
| v18.8 | BTC top-trader absorption reversal | 164 | +2.49 bp | -7.51 bp | worse than non-absorbing complement | reject |
| v18.8 | alt top-trader absorption bucket | 140 | -2.97 bp | -32.97 bp | family percentile 0.0% | reject |
| v18.9 | high-breadth unwind cascade continuation | 275 | -1.05 bp | -11.05 bp | random percentile 60.0% | reject |

Primary costs are 10 bp for direct BTC books and 30 bp for beta-hedged alt
buckets. All values are from preregistered first reveals.

## What was learned

1. BTC taker-confirmed OI unwind contains a small reversal-timing effect. Its
   +5.37 bp gross mean is unusual relative to circular time controls, but it is
   below even the direct BTC cost assumption and is not positive in every split.
2. Monthly directed receiver membership does not attribute that effect to alts.
3. Event-bar residual overshoot, simultaneous OI/taker stress, and top-trader
   opposition do not identify a profitable receiver bucket.
4. Cross-asset volatility breadth strongly describes the source bar: the high
   regime has a median 28 of 45 alts beyond one prior-volatility standard
   deviation. It does not predict next-bar continuation; the source move is
   already incorporated by the completed event close.
5. Filtering on BTC top-trader opposition makes the direct reversal weaker, not
   stronger. It should not be interpreted as informed absorption at this horizon.

## Frozen negative boundaries

- Do not tune monthly correlation edges, receiver bucket size, or stress-score
  weights on these outcomes.
- Do not promote the v18.6 unwind reversal merely because its random-control
  percentile is high; cost and split gates remain binding.
- Do not add leverage to any rejected candidate. Leverage scales the negative
  after-cost expectancy and does not solve the cost hurdle.
- Do not infer tradable volatility exposure from realized breadth without an
  instrument and execution model that can monetize variance.

## Next defensible data branch

The current five-minute metrics are position and ratio snapshots. The next
round should add information that changes the payoff mechanism rather than
another receiver rank:

1. Binance mark/index premium and funding histories for basis pressure and
   crowded carry state;
2. liquidation or forced-order-flow history where coverage is auditable;
3. directionally separated long-liquidation and short-cover events, frozen
   before their future returns are inspected;
4. only if a gross edge clears costs, a separate passive execution/fill study.

Every research round in this summary has an independent audit, and no live,
PaperLive, application, leverage, remote, or real-order scope changed.
