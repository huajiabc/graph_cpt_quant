# nameless short-side work progress report

Date: 2026-06-18
Scope: review of `origin/main` through commit `838ad2d`

## Executive Summary

nameless largely followed the requested direction: he did not keep squeezing old short motifs, and he did not promote the most attractive single-month result. The work shifted from "find a short strategy" to three healthier deliverables:

1. a short-side closure inventory,
2. a global concentration/verdict framework,
3. a continuous Binance CVD data layer plus monthly OOS replay harness.

The headline result is disciplined: no standalone short alpha is currently validated. The closest candidate, `A1_imb10bp_h24`, is real enough to keep watching but fails promotion because 2025-10 contributes 63.7% of total alpha, bootstrap CI crosses zero, and early walk-forward is negative. The current status is diagnostic / OOS-waiting, not paper-live.

## Requested Work vs Delivered Work

| Your instruction | Delivered artifact | Status | Notes |
|---|---|---|---|
| Short Research Closure / Inventory | `docs/short_research_closure.md` | Done | Covers rejected, diagnostic-only, data-blocked, and OOS-waiting buckets. Explicitly forbids re-litigating old motifs. |
| Global gate5 / concentration validator | `src/pressure_graph/validation/gate_checks.py`, `docs/candidate_verdict_schema.md`, `tests/test_gate_checks.py` | Mostly done | Best-month-share, leave-one-month, symbol concentration, bootstrap, walk-forward, random p90 are implemented. OOS harness uses it. Some older report modules still carry local gate logic and are not fully migrated. |
| Continuous Binance CVD backfill | `src/pressure_graph/binance_continuous_cvd.py`, `scripts/v7s_continuous_cvd_backfill.py`, `tests/test_binance_continuous_cvd.py` | Done as framework, partially run | Implements symbol-day continuous aggTrades to 1m/5m/15m CVD shards, coverage report, quality audit. D2 rerun used a partial backfill and exposed Binance `IncompleteRead` retry hardening need. |
| A1 Regime Autopsy, not upgrade | `scripts/v7s_a1_regime_autopsy.py`, `docs/v7s_short_alpha_findings.md` | Done | Correctly branded as autopsy, not strategy upgrade. Shows 2025-10 regime: BTC_up, high BTC vol, extreme OI, extreme volume, uncrowded funding. |
| Monthly OOS replay | `scripts/v7s_monthly_oos_replay.py` | Done as harness | Registry includes A1, D, E, v3.4, v6S. Latest commit adds explicit A1 promotion and auto-demote protocol. |
| OKX / Hyperliquid deferred | Closure docs and v7S inventory | Followed | Hyperliquid A3 and liquidation tape are marked data-blocked; no premature exchange expansion. |
| New short direction allowed only with data discipline | D2 CVD pair scaffold and rerun | Done and falsified | D2 initially looked strong on 3 months, then failed on 6-month rerun. Correctly left as `no_value`. |

## Short Closure Inventory

The closure document is the strongest part of the handoff. It does exactly what you wanted: it turns the short research lane into a termination list.

### Rejected directions

- Motif-led naked short: rejected.
- CIC-failure short: rejected.
- Crowded-stall short: demoted from candidate to `risk_off_only` / diagnostic.
- Relative-value beta short and pair hedge: rejected.
- A1 unfiltered short: rejected.
- Strict CIC-failure-confirmed short: rejected.

### Still useful but not tradable

- `A1_imb10bp_h24`: diagnostic-only / OOS-waiting.
- F3 / F5 failure risk layer: retained as long-side shadow overlay, not a short.
- Crowded / funding diagnostics: useful for regime analysis.
- CVD / orderflow windows: useful as data layer, not standalone alpha.

### Data-blocked

- Breakdown-bar sell flow for stricter CIC-failure shorts.
- Liquidation continuation.
- Direction C v2 with CVD divergence / taker exhaustion.
- Hyperliquid lead-lag.

This is aligned with your intention: future work should not keep revisiting already-failed motif variants.

## Global Gate and Verdict Layer

The key methodological bug was addressed. `gate_checks.py` now makes `best_month_share <= 35%` a first-class gate rather than relying only on capped monthly net.

Implemented metrics include:

- `best_month_share`
- `month_cap35_net`
- `leave_one_month_min`
- `best_symbol_share`
- `leave_one_symbol_min`
- `bootstrap_ci_lo / bootstrap_ci_hi / bootstrap_p_positive`
- `walk_forward_min_net / walk_forward_delta`
- `random_p50 / random_p75 / random_p90`
- canonical `final_verdict`

This directly prevents the A1 mistake: a candidate can no longer look promotable just because `month_cap35_net` remains positive while one month carries most of the return.

Remaining gap: this is not yet fully wired into every historical report. `scripts/v7s_monthly_oos_replay.py` uses the global evaluator, and the validator has tests, but some report modules such as `v7s_short_alpha.py` still retain local gate logic. That is acceptable for this stage, but the next cleanup should migrate all candidate-summary emitters to the shared evaluator.

## Continuous Binance CVD Data Layer

The P1 data-layer work is real, not just a placeholder.

Delivered:

- continuous Binance UM aggTrades backfill driver,
- 1m / 5m / 15m aggregation,
- taker buy ratio,
- CVD delta by volume and turnover,
- buy-sell imbalance,
- large buy / sell counts and turnover,
- per-symbol/month parquet shards,
- coverage report,
- quality audit.

This satisfies the spirit of your instruction: move orderflow from CIC-anchored event windows to symbol-day continuous coverage.

Known issue: longer backfill runs hit Binance `IncompleteRead` throttling. The docs note that the existing downloader retries `URLError`, `OSError`, and `HTTPError`, but not `IncompleteRead`. That should be hardened before large unattended backfills.

## A1 Regime Autopsy

A1 was handled correctly. It was not promoted.

Best-looking A1 cell:

- `A1_imb10bp_h24`
- N = 56
- gross +1.49%
- net20 +1.18%
- net30 +0.98%
- win 75%
- max symbol share 14.2%

Why it was rejected:

- best month share = 63.7%, above the 35% limit,
- bootstrap 95% CI = [-0.40%, +2.52%], crosses zero,
- first walk-forward bucket = -1.19%,
- 2025-10 dominates the return profile.

Regime read:

2025-10 is interpretable rather than random-looking:

- BTC_up,
- high BTC volatility,
- extreme OI,
- extreme volume,
- uncrowded funding.

The correct conclusion is: A1 may describe a real cross-exchange info-propagation regime, but it is not a strategy candidate yet. It needs future OOS months plus shuffled-regime controls before any shadow/live discussion.

## Monthly OOS Replay

`scripts/v7s_monthly_oos_replay.py` matches your requested P3.

It:

- isolates target month,
- compares baseline vs with-new-month verdict,
- reports target-month contribution,
- recomputes gate checks through the global evaluator,
- records verdict changes,
- keeps a registry of A1, D, E, v3.4, and v6S candidates.

The latest nameless commit added the most important governance rule:

Promotion of `A1_imb10bp_h24` requires all three in one replay:

1. ex-target-month mean net20 still > 0,
2. shuffled-regime control p < 0.05,
3. target-month net20 > 0.

Auto-demote:

- 3 consecutive `newly_no_value` replays,
- or 3 negative target-month replay outcomes,
- then drop A1 from registry and close it.

This is exactly the kind of process discipline the short lane needed.

## New Short Alpha Search Results

No validated standalone short alpha emerged.

### Direction A: Cross-exchange downside lead-lag

This is the only direction with a real signal shape.

The A1 filter matters: A0 no-filter loses, while A1 with Binance sell impulse + Bybit lag improves. But the best cell is non-stationary and single-month concentrated, so it remains diagnostic.

Verdict: `no_value`, OOS-waiting.

### Direction D: Relative-value pair

Old Direction D was rejected across 30 cells. Pair hedges made performance worse than naked short because beta and hedge legs are too correlated and hedge cost eats the signal.

Verdict: rejected.

### Direction D2: CVD-confirmed relative-value pair

D2 was the sensible new direction you allowed: beta overextended + beta CVD weakening + hedge CVD stable/strong + failed follow-through.

Initial 3-month sample looked promising:

- `D2_eth_cvd_pair h24`: net20 +269 bps, bootstrap CI positive.

But after 6-month backfill:

- N grew to 102,
- h24 net20 fell to -9 bps,
- CI became [-1.62%, +1.25%],
- p(>0) = 0.465,
- h4 became significantly negative.

Verdict: refuted / no_value.

This is an excellent example of the new evaluation discipline doing its job.

## Overall Assessment

He followed the assignment well.

The work is useful because it closes more doors than it opens:

- it prevents repeated motif sweeps,
- it upgrades the evaluation standard,
- it establishes an orderflow data path,
- it formalizes monthly OOS monitoring,
- it keeps promising but concentrated results out of live.

The best "new alpha" is not tradable alpha. It is a regime clue:

`A1 / D2 both light up around the same 2025-10 cross-exchange info-propagation regime.`

That suggests the next real opportunity may be a regime detector or flow-propagation detector, not another short entry rule.

## Recommended Next Steps

1. Harden Binance continuous CVD downloader retry logic for `IncompleteRead`.
2. Finish broader CVD backfill before any new D2 / A1 strategy search.
3. Migrate all candidate-summary reports to `evaluate_candidate_verdict`.
4. Keep A1 in monthly OOS replay, but do not promote it.
5. Treat F3 / F5 as long-side shadow overlays only.
6. Do not reopen motif-led naked short, CIC-failure short, crowded-stall short, or pair hedge D unless a new data source changes the premise.

## Final Verdict

Status of nameless work:

`Good research hygiene, no validated short alpha, strong process upgrade.`

The project is now safer because it has a closure list, a stricter global verdict gate, and an OOS replay protocol. The short lane should stay closed for execution purposes until continuous CVD and new OOS months provide genuinely distributed evidence.
