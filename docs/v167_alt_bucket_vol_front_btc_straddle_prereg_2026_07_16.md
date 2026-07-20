# v16.7 Alt-Bucket Volatility Front -> BTC Straddle Preregistration

Date frozen: 2026-07-16, after confirming the official Binance archive schema
and coverage, but before downloading the full option sample or inspecting any
candidate return.

## Scope and scientific question

Prior price-only tests showed that receiver volatility often expands, but a
direction-free volatility forecast could not be monetized with perpetual-futures
breakouts. v16.7 changes the traded instrument rather than re-encoding the same
return graph: does a synchronized volatility front across major altcoins lead a
still-quiet BTC strongly enough to profit from an actual BTC long straddle?

This is one fixed candidate, `OVT1_ALT_VOL_FRONT_LONG_BTC_STRADDLE`. It is an
options-data feasibility and alpha test, not a graph-attribution claim. The
available Binance option archive is short, so a pass can grant only a research
follow-up recommendation.

## Frozen data and chronology

- Official Binance `BTCUSDT` option `EOHSummary` daily archives from the complete
  available interval, currently 2023-05-18 through 2023-10-23. The archive exposes
  hourly option bid/ask, IV, Greeks, volume and open interest.
- Treat a row labelled `date=D, hour=H` conservatively as available only at
  `D + H + 1 hour`. The daily decision uses hour 0 and is therefore timestamped
  01:00 UTC.
- Official Binance USD-M one-hour perpetual klines. A feature or hedge price is
  usable only when its bar close is no later than the option snapshot time.
- Fixed alt bucket: `ETHUSDT`, `BNBUSDT`, `SOLUSDT`, `XRPUSDT`, `DOGEUSDT`,
  `ADAUSDT`, `LTCUSDT`, `LINKUSDT`, `AVAXUSDT`, `DOTUSDT`, `BCHUSDT`, and
  `MATICUSDT`. BTC is excluded from the bucket and used separately.
- Require at least ten valid alt symbols at every decision.
- Development decisions end before 2023-08-01; validation is 2023-08-01 through
  2023-09-14; holdout label begins 2023-09-15. These are chronological labels,
  not a claim of a pristine modern-market holdout.

## Frozen causal volatility front

For BTC and every alt, compute the square root of the sum of squared hourly log
returns over the 24 completed hours ending at the decision time. At each daily
decision, convert each current 24-hour realized volatility to its percentile
among that symbol's preceding 30 daily decision values, shifted by one day.
Require all 30 historical values.

Define:

- alt high-vol breadth: fraction of valid alts whose own volatility percentile is
  at least 80%;
- alt bucket percentile: median current percentile across valid alts;
- front gap: alt bucket percentile minus BTC's current percentile.

The event is active when alt high-vol breadth is at least one third, BTC's own
volatility percentile is at most 70%, and front gap is at least 25 percentage
points. Accept only false-to-true transitions and impose a 24-hour cooldown.
There is no threshold, universe, holding-period, or option-tenor grid.

## Frozen option construction and executable return

At the event's hour-0 snapshot:

1. keep expiries with 21 to 45 calendar days remaining and choose the expiry
   closest to 30 days;
2. among strikes having both a call and put with positive, internally consistent
   bid/ask prices and positive bid/ask quantities, choose the strike closest to
   the known BTC perpetual close;
3. buy one call and one put at their recorded asks;
4. use the sum of their entry deltas to open an equal-and-opposite BTC perpetual
   hedge at the known close;
5. at the next day's hour-0 snapshot, sell the identical options at their bids
   and close the static hedge at the corresponding BTC close.

The primary gross payoff is option PnL plus static hedge PnL, divided by entry
BTC price. Bid/ask crossing is embedded. Primary fees add, for each option
transaction, `min(3 bp * BTC price, 10% of option premium)` and 8 bp round trip
on absolute hedge notional. Stress fees use 6 bp per option transaction with the
same premium cap and 16 bp hedge round trip. Premium return and unhedged straddle
return are diagnostics only.

No delta rebalancing, expiry substitution, stale exit, mark-price fill, stop,
take-profit, leverage or early exit is allowed.

## Frozen controls and gates

Report all eligible daily straddles, the BTC-compression-only subset, and a
one-day-delayed alt-front control. Build 2,000 circular date-shift controls that
preserve the complete BTC/options path while shifting the alt front relative to
it. Bootstrap entry days 5,000 times.

Research follow-up eligibility requires all of:

- at least 20 complete trades, five validation trades and five holdout trades;
- positive primary mean in development, validation and holdout;
- positive full-sample stress mean;
- primary bootstrap 95% lower bound above zero;
- real primary mean at or above the 95th percentile of circular-shift controls;
- better primary mean than both the delayed-front and BTC-compression controls;
- no single month supplies more than 50% of positive PnL and no single trade more
  than 35%;
- option-pair selection and price-bar timestamp audits pass with no stale fills.

Failure rejects this exact alt-front/ATM-straddle construction. Passing does not
authorize PaperLive, remote changes, leverage or real orders; it only justifies
seeking a longer executable options history or starting a forward recorder.
