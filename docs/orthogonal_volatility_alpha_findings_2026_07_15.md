# Orthogonal Volatility Alpha Exploration — Consolidated Findings

Date: 2026-07-15

## Data acquired

- 73 Bybit symbols times 8,761 hourly long/short account-ratio observations:
  639,553 rows total.
- 46,537 hourly BTC DVOL rows and 46,537 hourly ETH DVOL rows from the start of
  the available history through 2026-07-15.
- 47,089 hourly BTC perpetual bars.
- Monthly BTCDVOL futures archives from May 2023 onward where the historical
  instrument actually existed; missing contracts were not synthesized.
- Existing OI, funding, price, and exact frozen 8-by-9 community assets were
  reused with strict timestamp alignment.
- 6,854,302 five-minute Binance USD-M large-trader/account/taker/OI metrics
  across 71 of the 72 frozen community symbols. Seventy symbols cover all 339
  requested days; XAUT starts on 2026-03-26 and MNT is unavailable.

The account-ratio source passed the data-age gate with 100% crowding-z coverage
in all 11 formal community months. The DVOL source supported 63 completed
monthly forward-variance labels.

## Four independent verdicts

| Family | Strongest informative result | Trade verdict |
|---|---|---|
| Direct DVOL futures basis | Validation positive, but only 20 executable primary contracts and holdout negative | Reject |
| Crowding + OI unwind propagation | Stress state abundant; frozen community followers net negative | Reject |
| Static short variance | Actual DVOL pairing beats random IV, but validation/holdout and tail risk fail | Reject |
| Purged walk-forward ridge | Slightly mitigates holdout short-vol loss; still negative and random-like | Reject |
| Top-trader divergence rotation | 1,838 decisions over 11 months; best gross +1.32 bp and best realized-turnover net -5.84 bp | Reject |

## Main conclusion

The new orthogonal data change the scientific conclusion but not the deployment
decision:

1. **Volatility is priced, not absent.** DVOL contains substantial information
   about the next 30 days of realized variance.
2. **The residual is not a stable premium.** Static short variance worked in
   the earlier high-IV development regime and failed in the recent low-IV
   regime with severe convex losses.
3. **Graph topology still does not explain transmission.** Crowding and OI
   identify stressed leaders, but frozen price communities do not select
   profitable followers beyond random partitions.
4. **Model complexity is not the bottleneck.** A purged expanding ridge failed;
   the effective independent volatility sample is too small to justify boosted
   trees, transformers, or GNNs on the same primitives.
5. **Large-trader positioning is not a missing directional key.** Position vs
   crowd, position vs top accounts, taker-confirmed divergence, and frozen
   community rotation all failed chronological and cost gates. The real frozen
   communities ranked at only the 46th percentile of random nine-symbol
   partitions.

No v11.7-v12.1 candidate is eligible for PaperLive or leverage. The existing
remote v11.2 PaperLive remains unchanged.

## Defensible next data action

Historical liquidation is not exposed by the tested Bybit history interfaces;
the official all-liquidation feed is a forward WebSocket stream. Actual option
skew, bid/ask, delta-hedging cost, and DVOL futures executable depth also need
forward recording. These are data-collection tasks, not reasons to tune another
historical threshold on the same outcomes.
