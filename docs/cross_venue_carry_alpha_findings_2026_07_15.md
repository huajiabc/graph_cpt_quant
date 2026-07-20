# Cross-Venue Carry Alpha Findings

Date: 2026-07-15

## Data and mechanism progression

This round turned funding from a crowding feature into an actual cash-flow
return, then removed direction risk in stages:

1. v12.3: long low-funding / short high-funding Bybit perpetuals.
2. v12.4: long Binance spot / short high-funding Bybit perpetual.
3. v12.5: long low-funding Bybit perpetual / short same-coin Binance USD-M.
4. v12.6: exact same-coin carry with top-9/hold-18 turnover governance.
5. v12.7-v13.0: breadth, portfolio combination, risk parity, and graph
   diversification controls.
6. v13.2: unchanged TG1 extended with six genuinely new June/July weeks and a
   causally generated July 2026 membership.

The research box now contains 71 Binance USD-M contract histories, 72 Bybit
histories, exact dynamic settlement frequencies, and recent data through
2026-07-15. MNT has no Binance USD-M contract. TON stopped producing complete
Bybit history during June and was causally excluded from the July graph. July
membership therefore contains 71 symbols in community sizes `8,9,9,9,9,9,9,9`.

## Verdict table

| Version | Mechanism | Strongest result | Verdict |
|---|---|---|---|
| v12.3 | Bybit cross-sectional funding carry | FC1 net60 +74.81 bp/week; null 98.8%; all periods positive | Near-candidate; bootstrap lower -52.11 bp |
| v12.4 | Spot/perp high-positive funding | Gross -7.52 bp/week for focal 7d rank | Reject |
| v12.5 | Bybit/Binance same-coin funding spread | 30d gross +29.51 bp; realized-turnover net +19.28 bp | Reject fixed weekly cost gate |
| v12.6 | TG1 top-9/hold-18 | Primary +21.20 bp; stress +14.79; bootstrap lower +5.25; null 100%; all periods positive | Near-candidate; month concentration 42.30% |
| v12.7 | Broader top-12/hold-24 | Primary +17.05 bp; bootstrap lower +3.77 | Reject; concentration 43.33% |
| v12.8 | Fixed 50/50 TG1 + frozen P2 | Correlation 0.185; concentration 24.39%; bootstrap lower +0.41 | Reject; validation -1.05 bp |
| v12.9 | Causal 8w inverse-vol combination | Bootstrap lower +2.98 bp; concentration 23.18% | Reject; validation -3.06 bp |
| v13.0 | One name per frozen graph community | Real partition only 39th percentile of random partitions | Reject |
| v13.1 | Backward July 2025 extension | No frozen July membership existed | Coverage-blocked; no verdict |
| v13.2 | Exact TG1 plus six new forward weeks | 49 weeks; primary +18.20; stress +11.85; bootstrap lower +3.72; null 100%; all periods positive | Near-candidate; concentration 41.52% |

## What can and cannot be concluded

There is real evidence for persistent same-coin cross-venue funding spread. The
v13.2 carry sleeve is delta-neutral by construction, survives realistic realized
turnover costs, has a positive bootstrap lower bound, ranks above all fixed-cost
random positive-spread baskets, and stayed positive in development, validation,
and the expanded holdout. Six genuinely new weeks were mixed and reduced average
return, but did not reverse the carry.

It is not yet a promoted tradable alpha. November/December 2025 still supply
41.52% of positive monthly PnL versus the frozen 35% limit. Wider buckets,
community constraints, and combinations did not repair this without breaking
other gates. The graph communities do not explain the carry: the real graph
partition underperformed most random partitions in v13.0.

The defensible state is therefore `strongest_near_candidate_waiting_for_natural_forward_time`.
Do not add another sample-informed filter or leverage overlay. Continue recording
cross-venue tape and append naturally completed TG1 weeks under the exact v13.2
rule. PaperLive remains unchanged.

## Subsequent independent result

Later same-day research changed the broader project conclusion without changing TG1 itself.
The equal-weight, all-negative-Bybit-funding, BTC-beta-neutral v14.0 portfolio passed every
frozen candidate gate and its independent audit. It does not use the cross-venue TG1 spread and
does not repair or promote TG1; it is a separate negative-funding state candidate. See
`docs/negative_funding_state_alpha_findings_2026_07_15.md`.
