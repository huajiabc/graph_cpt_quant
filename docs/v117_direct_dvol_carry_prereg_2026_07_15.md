# v11.7 Direct DVOL Carry Preregistration

Date: 2026-07-15

Status before outcome inspection: `PREREGISTERED`.

## Question

Can a directly tradable volatility instrument earn a stable return from the
gap between monthly BTCDVOL futures and the contemporaneous 30-day BTC DVOL
index? This is a different claim from predicting absolute futures returns.

## Frozen data contract

- Deribit public hourly BTC DVOL index.
- Deribit public hourly TradingView bars for monthly
  `BTCDVOL_USDC-DDMMMYY` futures.
- Contract expiration is the last Wednesday of each month at 08:00 UTC.
- Research window begins with the first May 2023 contract and ends at the last
  completed contract present in the archive.
- A futures bar is executable only when its reported hourly volume is positive.
- Every signal uses only the latest DVOL and futures close at or before signal
  time. Entry is the open of the first later positive-volume futures bar, no
  more than 24 hours after the signal.

## Frozen primary strategy

For each monthly contract:

1. Target signal time is expiration minus 14 calendar days at 08:00 UTC.
2. Use the last positive-volume futures bar no more than 48 hours before that
   target and the latest non-stale DVOL close no more than two hours old.
3. Basis in index points is futures close minus DVOL close.
4. Short the future when basis is positive and long it when basis is negative.
5. Enter on the next positive-volume bar and hold to the reported contract
   settlement price at expiration.

`DVC1_ALL_BASIS_CONVERGENCE` trades every covered non-zero basis.
`DVC2_ABS2_BASIS_CONVERGENCE` is the only gated secondary candidate and
requires absolute signal basis of at least two DVOL points.

Return is direction times settlement minus entry, divided by entry. Primary
net cost is 10 bp: 5 bp taker entry plus 5 bp delivery. Stress costs are 30 bp
and 50 bp to represent unobserved spread and thin-book slippage. No leverage is
evaluated unless the unlevered candidate passes.

## Frozen splits and controls

- Development: expirations through 2024-12-31.
- Validation: 2025-01-01 through 2025-06-30.
- Holdout: expirations from 2025-07-01 onward.
- Timing controls: otherwise identical entries at 7 and 21 days before expiry.
- Direction control: reverse every basis-convergence direction.
- Random control: 2,000 contract-level random sign assignments.
- Uncertainty: 5,000 contract bootstrap draws and leave-one-contract-out audit.
- Concentration: maximum positive-contract contribution.

## Promotion gate

Both candidates are rejected unless all of the following hold:

- at least six validation and six holdout contracts;
- positive net10 in full, validation, and holdout;
- positive net30 in full and holdout;
- bootstrap 95% lower bound for net10 above zero;
- real result at or above the 95th random-sign percentile;
- convergence direction beats reversed direction;
- no single positive contract contributes more than 35% of positive PnL.

Historical bar coverage alone cannot authorize PaperLive. A passing result
still requires a live spread/volume recorder because historical bid/ask is not
available from this endpoint.
