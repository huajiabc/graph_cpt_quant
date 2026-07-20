# v18.3 Short-Cycle Price and Volatility Round Summary

Date: 2026-07-16

## Verdict

No new standalone alpha candidate is eligible. No PaperLive, live, application,
leverage, remote, or real-order scope changed.

The round did identify one audited research primitive: after an extreme 15-minute
cross-sectional BTC-beta-neutral dispersion event, the true Bottom5-versus-Top5
rank spread compresses more than random rank assignments. The effect is too small
and too unstable in validation to monetize after realistic turnover.

## Completed branches

| Version | Mechanism | Main evidence | Verdict | Independent audit |
|---|---|---|---|---:|
| v17.8 | BTC confirmed-flow shock -> current laggard catch-up | BFR1 net -28.96 bp; BFR2 net -28.37 bp | reject | 22/22 |
| v17.9 | direct graph receiver propagation / receiver-insulator spread | BRP1/2/3 net -25.44/-31.33/-29.92 bp; spread gross +0.08 bp | reject | 26/26 |
| v18.0 | extreme residual-dispersion compression | gross +2.54 bp, net30 -27.46 bp; random-rank percentile 100%; holdout gross +4.99 bp | structural primitive only; standalone reject | 22/22 |
| v18.1 | 2h/4h/8h/12h and continuous four-hour execution | 4h gross +1.58 bp; continuous gross +10.70%, turnover 786.93, primary net -107.34% | reject | 22/22 |
| v18.2 | 10 bp passive retracement entry with all partial fills retained | gross +4.92 bp, realized cost 9.19 bp, net -4.28 bp; random-rank percentile 99.6% | reject | 24/24 |

## What the evidence says

1. Monthly price lead-lag rankings do not forecast executable 15-60 minute
   receiver returns. Direct receivers, BTC-neutral receivers, and graph spreads
   were all approximately random and far below cost.
2. Cross-sectional residual extremes contain a real rank-specific compression
   effect. It survives development and holdout in gross terms, beats exact
   reversed direction, beats a one-bar delay, and exceeds every v18.0 random-rank
   draw.
3. The effect is only a few basis points. Longer holding does not accumulate it
   monotonically, and a continuous book pays prohibitive turnover.
4. Passive fills narrow the gap but do not close it. Fixed 5/10/20 bp offsets all
   remain around -4 bp net under the same cost schedule; validation is the weakest
   split. Optimizing the offset or declaring sub-10 bp costs would be specification
   search, not new evidence.

## Frozen negative boundaries

- Do not rerun BTC shock laggard catch-up, direct receiver propagation, or
  receiver-insulator price spreads with another nearby threshold.
- Do not extend residual-dispersion holding beyond twelve hours expecting costs to
  disappear.
- Do not tune passive offsets on these same outcomes or drop unbalanced partial
  fills after observing them.
- Do not promote, lever, or attach the v18.0 primitive to PaperLive as a new sleeve.

## Defensible continuation

The v18.0 state can be tested only where it does not create a fresh high-turnover
book: as a causally lagged entry/exit/risk overlay on an independently valid
strategy, with the base strategy and overlay rule frozen before outcomes. The
current fixed FSS3/TG1 weekly portfolio supplies only 49 aligned weeks and the
15-minute panel ends in early June, so any such overlay will be low-power and can
authorize forward observation only.

For a new standalone entry alpha, the next material information addition should be
actual synchronized microstructure: BTC-inclusive five-minute taker/OI/top-trader
metrics, liquidation tape, or forward order-book/passive-fill recording. The
current price-only panel has reached an economic rather than modeling boundary.
