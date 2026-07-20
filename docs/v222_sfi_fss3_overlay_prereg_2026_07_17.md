# v22.2 SFI-on-FSS3 Overlay Preregistration

Date frozen: 2026-07-17, after the feature-only v22.1 audit and before any
v22.2 return, PnL, turnover, or outcome was constructed or inspected.

## Hypothesis

The weekly funding-sign spread FSS3 identifies the structural long and short
sides. A causal spot-minus-perpetual taker-flow inventory score may improve
coin selection *within* those already selected sides: favor spot-led demand on
the long side and perpetual-led selling on the short side. This is an overlay,
not a new standalone strategy, and introduces no extra decision or trade cycle.

## Frozen feature input

- Input:
  `reports/v22_1_sfi_fss3_overlay_feature_audit/weekly_symbol_overlay_features.parquet`.
- SHA256:
  `D1C10E394B0DF5D3202FB4342C38933ABB2D2137D8EB706FCC9570AE0F027D04`.
- v22.1 passed 13/13 causal and structural checks before outcome reveal.
- The feature covers 35 weeks and nine months: 16 development, 11 validation,
  and eight holdout weeks.
- Every source snapshot is 12--36 hours before the unchanged Monday 00:00 UTC
  FSS3 entry and contains at least 30 eligible SFI symbols.

## Frozen candidate

`SFO1_FSS3_WITH_CAUSAL_SFI_RANK_TILT`

- Preserve every FSS3 funding-sign name and its side.
- Within the long side, rank benefit by increasing spot-minus-perpetual flow;
  within the short side, rank benefit by decreasing spot-minus-perpetual flow.
- Use the sole frozen rank tilt of 0.50, producing multipliers in `[0.75, 1.25]`.
  There is no tilt grid, threshold search, or outcome-conditioned fallback.
- Missing SFI names receive multiplier 1.0. Renormalize each raw side to 0.5.
- Apply the original prior-month BTC-beta hedge and normalize total gross to one.
- Rebuild all 49 FSS3 weeks in chronological order. Outside the 35 feature
  weeks, use the unchanged FSS3 target. This preserves execution path state.
- Use the frozen v14.9 transition rule: full-L1 turnover at most 0.70 on
  cap-applicable weekly transitions. Charge initial entry, terminal close,
  gaps, and mandatory exits in full.
- Primary/stress costs remain 20/40bp times each path's own realized full-L1
  turnover.
- Form the portfolio overlay only by replacing FSS3 with overlay-FSS3 inside
  the already frozen CM2 weights: 80% FSS3 and 20% TG1. No sleeve weight,
  timing, or TG1 return is changed, and no extra allocation cycle is assumed.

## Frozen controls and inference

- Zero-tilt reconstruction must reproduce the saved v14.9 FSS3 weekly gross,
  component, turnover, primary, and stress returns within `1e-12`; otherwise
  the experiment is invalid.
- A reversed-rank overlay uses the same multiplier distribution and execution
  but inverts SFI ranks within each funding-sign side. It is diagnostic only.
- Run 1,000 within-week, within-side random SFI-rank permutations. Each random
  path retains the active weeks, names, signs, multiplier multiset, beta hedge,
  cap, and its own realized transaction costs.
- Run 2,000 paired four-week moving-block bootstrap draws on active-week
  overlay-minus-baseline primary returns.
- Report full-path and active-window results, chronological periods, monthly
  contribution concentration, leave-one-month-out increments, price/funding
  attribution, turnover, cap binding, gross/beta residuals, and drawdown.

## Frozen promotion gates

The overlay is promotable only if all of the following hold:

- exactly 49 reconstructed path weeks and 35 active feature weeks, with the
  frozen 16/11/8 chronological split and nine active months;
- zero-tilt baseline reconstruction error at most `1e-12`;
- positive active-week mean primary and stress incremental return for FSS3 and
  fixed CM2, and positive primary increment in development, validation, and
  holdout separately;
- paired block-bootstrap 95% lower bound above zero and random-permutation
  percentile at least 95;
- observed active-week primary increment exceeds the reversed-rank control;
- mean full-path turnover no more than baseline plus 0.10, every cap-applicable
  transition no more than 0.70, and maximum cap breach at most `1e-10`;
- maximum absolute residual BTC beta and gross-notional drift at most `1e-12`;
- no single positive active month contributes more than 35% of total positive
  monthly increment, every leave-one-month-out mean increment is positive, and
  fixed-CM2 active-window maximum drawdown is no worse than baseline by more
  than two percentage points.

Failure means rejection as a strategy overlay. A weak positive but statistically
unresolved increment may be retained only as research evidence; it does not
grant forward-shadow, PaperLive, leverage, remote-host, application, or order
permission.
