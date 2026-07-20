# v13.2 Exact TG1 Forward Temporal-Extension Preregistration

Date frozen: 2026-07-15, before collecting or inspecting the cross-venue
portfolio returns after 2026-06-04.

## New as-of data and membership

- Append Bybit linear 15-minute prices/funding and Binance USD-M one-hour
  prices/funding through 2026-07-15 from their public APIs.
- Keep the existing frozen June 2026 v11.0 membership.
- At 2026-07-01, form July membership with the exact v11.0 causal method:
  preceding 30 calendar days of hourly Bybit returns, static BTC beta residuals,
  at least 500 complete observations, and deterministic recursive spectral
  bisection into eight balanced communities. Only data strictly before July 1
  may enter this membership.
- July membership is used as an eligibility universe only. No community filter
  or community weight enters TG1.

## Exact strategy rerun

Rerun the corrected `TG1_30D_TOP9_HOLD18` from 2025-08-04 through the last
complete Monday-to-Monday label. The score, Bybit-long/Binance-short direction,
positive-spread entry rule, top-18 hold band, equal weights, and primary/stress
20/40 bp one-way realized-turnover costs are unchanged. All newly complete June
and July weeks are included regardless of sign.

The same 1,000 fixed-cost-path random positive-spread baskets, 2,000 bootstrap,
three-period gates, 35% month-concentration limit, turnover limit, and minimum
sample rules apply. Passing means a forward carry-shadow candidate. Existing
PaperLive strategies and processes remain unchanged.
