# v9.7 Direct Cross-Sectional ML Alpha - Findings

## Decision

The direct bar-level model lane is rejected under both frozen horizons.
There is no present case for adding model complexity to the existing
price/volume/funding/OI feature set.

Status: `reject_direct_alpha_after_cost`. No selector, shadow, paper-live,
canary-live, or real-live permission changes are allowed. P2 remains unchanged.

## Why this test was materially different

The prior v2.2 meta-router used only 191 P2/CIC events. That sample was large
enough for a small logistic diagnostic but too small to justify a more flexible
model. v9.7 instead used the full dynamic monthly Top30 bar panel and directly
predicted same-timestamp relative future return.

- Four-hour dataset: 60,270 prepared rows, 2,009 non-overlapping decision
  timestamps, and 906 monthly walk-forward OOS timestamps.
- Twelve-hour dataset: 20,070 prepared rows, 669 non-overlapping decision
  timestamps, and 302 monthly walk-forward OOS timestamps.
- All features were as-of; labels and current-universe survivorship fields were
  blocked.

## Four-hour result

| model | mean OOS IC | gross excess | net10 excess | net20 excess | break-even full-turnover cost |
|---|---:|---:|---:|---:|---:|
| momentum rank | -0.0266 | +11.63% | -60.77% | -133.17% | 1.61 bp |
| Ridge | +0.0281 | -9.03% | -78.69% | -148.35% | negative |
| shallow XGBoost | +0.0314 | -5.03% | -73.65% | -142.27% | negative |
| XGBoost without funding/OI | +0.0375 | +24.66% | -41.88% | -108.42% | 3.71 bp |
| shuffled-label XGBoost | -0.0010 | -23.41% | -95.01% | -166.61% | negative |

The XGBoost IC lift over Ridge was only 0.0033. More importantly, score
quintiles were not monotonic: the highest XGBoost score bucket had average
relative return of -1.25 bp while the second bucket had +3.90 bp. The model
did not isolate a tradable upper tail.

The focal XGBoost random-Top5 percentile was 65.5%, far below the frozen 90%
gate. All five OOS months were negative after 20bp. The only ablation with
positive gross excess removed funding/OI, but its 3.71bp break-even cost is far
below the 10-20bp execution range.

Positive XGBoost symbol contribution was also concentrated: HYPE and NEAR
provided 86.8% of all positive symbol contribution, and only five symbols had
positive net20 contribution before the total negative tail was considered.

## Twelve-hour result

| model | mean OOS IC | gross excess | net10 excess | net20 excess | break-even full-turnover cost |
|---|---:|---:|---:|---:|---:|
| momentum rank | -0.0376 | +27.35% | +3.59% | -20.17% | 11.51 bp |
| Ridge | +0.0094 | -21.98% | -44.68% | -67.38% | negative |
| shallow XGBoost | +0.0040 | -11.43% | -33.61% | -55.79% | negative |
| shuffled-label XGBoost | +0.0205 | +12.59% | -11.13% | -34.85% | 5.31 bp |

The longer horizon reduced the number of rebalances but did not make the model
relationship more stable. Real-label XGBoost underperformed shuffled-label
XGBoost, had a 45th-percentile random-control result, and was negative in both
validation and holdout. Its highest score quintile again had negative average
relative return (-4.93 bp).

Momentum had some gross twelve-hour continuation and a positive May slice, but
validation was negative and the full result failed at 20bp. It is not a new
alpha candidate.

## Post-hoc regime clue

The diagnostic regime table found one non-global pocket: twelve-hour decisions
when BTC four-hour return was non-negative and BTC volatility percentile was
below 75.

This pocket was discovered after the global result and is explicitly post-hoc.

| model in BTC-up/low-vol pocket | net20 excess | random percentile | validation | May holdout |
|---|---:|---:|---:|---:|
| momentum rank | +0.28% | 89.2% | +1.83% | +2.69% |
| Ridge | -18.66% | 60.0% | -7.37% | -1.07% |
| shallow XGBoost | +11.77% | 96.6% | -0.38% | -2.43% |
| shuffled-label XGBoost | -20.96% | 53.0% | -19.56% | -1.40% |
| XGBoost without funding/OI | +29.56% | 100.0% | +29.04% | -5.95% |

The no-funding/OI variant was positive from January through April, then turned
negative in the sole complete holdout month. April supplied about 56% of its
full positive net20, breaching the 35% concentration rule. Therefore this is a
future-only hypothesis, not a historical candidate and not a shadow. It may be
revisited only on untouched post-2026-06 data with the regime and feature
ablation frozen exactly as written here.

## Assessment of more complex models

More complexity is not currently justified:

1. Ridge already captured the available linear relation; shallow trees added
   only tiny IC at four hours and lost IC at twelve hours.
2. The economic problem is not underfitting. Highest-score buckets were
   non-monotonic, gross Top5 excess was weak or negative, and turnover was
   roughly 70-80% per rebalance.
3. The twelve-hour shuffled-label model beat the real-label model. A higher
   capacity learner would have more ways to exploit the same instability.
4. The only historically interesting conditional pocket failed the final
   holdout and month-concentration gates.
5. The P2 event-level dataset has only 191 examples, which is unsuitable for
   boosted-tree, neural, or stacking promotion claims.

Do not run neural networks, Transformers, symbol embeddings, stacking, Optuna,
or broad hyperparameter search on this dataset. The next model class, if later
earned, should be a constrained learning-to-rank tree model rather than a deep
network.

## What would reopen model complexity

Complex modeling becomes reasonable only after at least one of these changes:

- synchronized cross-venue orderflow supplies a genuinely new information
  block with the preregistered 90-day coverage;
- a frozen shallow model beats Ridge in at least two complete validation months
  and two untouched holdout months, remains positive at 30bp, clears random
  P90, and has monotonic score buckets;
- P2/meta-router forward labels reach several hundred independent events with
  adequate class and month breadth.

Until then, the better use of effort is new information and lower-turnover
economic structure, not a more expressive estimator.
