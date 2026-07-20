# v23.26-v23.33 Multisource and Volatility-Model Round Summary

## Bottom line

This round did not find a promotable alpha. It materially narrows the search space:
adding derivatives state, quadratic interactions, continuous cross-coin volatility
transmission, or exhaustive single-feature tails does not turn the fixed 0.75-sigma
BTC OCO payoff into a robust positive-expectancy strategy on the 159-event set.

No live, PaperLive, leverage, remote, application, or order state was changed.

## v23.26-v23.28: multisource payoff model

- Built an outcome-free 19-feature matrix for all 159 events from book pressure,
  causal BTC volatility/return, 16-alt taker flow, open interest, top-trader state,
  BTC derivatives state, and UTC time.
- One validation event lacks the exact XLM five-minute observation. It uses the
  other 15 exact-time alts; no stale or future fill was used.
- Froze a 19-column linear ridge and a 209-column degree-two interaction ridge.
  Development predicts validation; development plus validation predicts holdout.
- Interaction ridge result:
  - validation: 18 trades, -6.88 bp/trade primary;
  - holdout: 17 trades, +7.53 bp/trade primary but -2.47 bp/trade stress;
  - combined: +0.04 bp/opportunity primary, -3.60 bp/opportunity stress;
  - random-label percentile 78.12;
  - month-bootstrap 5th percentile -8.37 bp/opportunity;
  - only 1/11 gates passed.
- The linear baseline was less bad: +4.21 and +0.36 bp/opportunity in validation
  and holdout under primary cost, but holdout stress was -3.31 bp/opportunity. The
  interaction expansion therefore reduced rather than added usable information.
- Independent sklearn reproduction passed 13/13 checks and upheld
  `rejected_no_incremental_complex_model_alpha`.

Interpretation: 63 development events cannot support 209 effective columns, even
with ridge shrinkage. More importantly, the failure is not just overfitting: the
19-column linear model's small primary-cost edge disappears under the frozen stress
cost, so the economic margin is inadequate.

## v23.29-v23.31: direct continuous transmission score

- Built 17 outcome-free price-volatility features for all 159 events, using 16 alts
  plus BTC and strictly prior 30-day histories:
  - standardized cross-sectional shocks and dispersion;
  - 4h-versus-24h volatility acceleration;
  - BTC-beta residual shock breadth;
  - causal one-hour alt-to-BTC volatility-edge advantage;
  - top-leader shock score and BTC receiver gap.
- Feature audit retained 159/159 events with full 16-alt price coverage and at least
  720 observed prior hours at every event.
- Froze an outcome-free equal-weight score from receiver gap, volatility
  acceleration, residual shock breadth, and directed-edge strength; selected the
  top 30% of the training feature distribution.
- Result:
  - validation: 12 trades, -28.70 bp/trade primary;
  - holdout: 15 trades, -5.00 bp/trade primary;
  - combined: -4.37 bp/opportunity primary;
  - random same-count percentile 38.26;
  - month-bootstrap 5th percentile -15.45 bp/opportunity;
  - only 1/10 gates passed.
- Independent reproduction passed 12/12 checks and upheld
  `rejected_direct_volatility_transmission_selector`.

Interpretation: a strong, measurable alt-to-BTC volatility front is not itself a
tradable long-vol edge under this payoff. At an hourly decision boundary it likely
mixes already-realized movement with subsequent whipsaw; propagation strength does
not imply profitable continuation after the first OCO touch.

## v23.32-v23.33: sparse 34-candidate search with search correction

- Allowed each of the 17 volatility features to contribute only its high 30% or low
  30% tail: exactly 34 candidates, one winning rule per temporal fit.
- Feature/orientation selection used training returns only. The 1,000-label null
  repeated the entire 34-candidate search, so the comparison includes mining
  freedom rather than treating the winner as prespecified.
- Development selected low `leader_shock_breadth`; development plus validation
  selected low `alt_btc_abs_z_gap`. The rule therefore failed temporal stability.
- Result:
  - validation: 21 trades, -0.09 bp/trade primary;
  - holdout: 12 trades, -24.36 bp/trade primary;
  - combined: -3.06 bp/opportunity primary;
  - full-search random-label percentile 48.15;
  - month-bootstrap 5th percentile -10.33 bp/opportunity;
  - 2/10 gates passed.
- Independent reproduction passed 12/12 checks and upheld
  `rejected_sparse_volatility_tail_selector`.

Interpretation: even the best training tail is unstable and indistinguishable from
the result of searching noise. The selectors reduce losses relative to always
trading, but loss avoidance is not positive alpha.

## Research boundary after this round

The evidence now argues against three nearby extensions:

1. increasing model complexity on the same 159 hourly events;
2. treating broad alt volatility transmission as a direct BTC long-vol trigger;
3. mining fixed high/low tails of these 17 features without new data.

The next useful price-behavior work needs a changed information/payoff resolution,
not more tuning of these matrices. The two defensible directions are:

1. Intrahour sequencing: use five-minute or finer cross-coin lead order to separate
   a genuinely early volatility front from an already-completed common shock.
2. Receiver payoff relocation: test whether the transmission is monetized in
   lagging alt/bucket receivers rather than BTC OCO continuation, with a frozen
   cross-sectional long/short payoff and full cost model.

Separately, the isolated q90 forward-shadow collector remains the only live-adjacent
candidate. It should continue accumulating untouched daily observations; this round
does not change its status.
