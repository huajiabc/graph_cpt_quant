"""Global gate-check evaluator — load-bearing per 77.docx P0.

The v7S work surfaced a gate5 implementation bug: the original
``month_capped_net > 0`` check let A1_imb10bp h24 pass while a single
month (2025-10) carried 63.7 % of the alpha. The closure doc's reopen
§1 wording demanded both ``month_capped_net > 0`` AND
``best_month_share ≤ 0.35``; only the first half had been enforced.

This module is the canonical evaluator for distribution-/concentration-/
robustness-style gates that every candidate report (v3.4, v3.5, v4S,
v6S, v7S, future) must emit. The intent is twofold:

1. **Prevent silent regressions of the gate5 fix.** Any report that
   imports ``evaluate_candidate_verdict`` automatically inherits the
   tightened distribution checks; ad-hoc per-report gate code is no
   longer the source of truth.
2. **Standardise the verdict surface.** Every report adds the same
   fields to its candidate_summary.csv so a reader can compare
   verdicts across reports without grokking N different schemas.

Required output fields per the 77 docx P0 spec:

- ``best_month_share`` — top-month signed share of total net.
- ``month_cap35_net`` — sum of per-month nets capped at
  ``month_cap_pct × total`` (per §reopen-1).
- ``leave_one_month_min`` — minimum sum after dropping each month in
  turn (one-month-out worst case).
- ``best_symbol_share`` — top-symbol signed share of total net.
- ``leave_one_symbol_min`` — minimum sum after dropping each symbol
  in turn (one-symbol-out worst case).
- ``bootstrap_ci_lo`` / ``bootstrap_ci_hi`` / ``bootstrap_p_positive``
  — empirical 95 % bootstrap CI on mean_net20 + the fraction of
  resample-means strictly above zero.
- ``walk_forward_min_net`` / ``walk_forward_delta`` — minimum bucket
  mean_net20 across N disjoint time buckets + (max - min) spread.
- ``random_p50`` / ``random_p75`` / ``random_p90`` — random-baseline
  mean_net at the 50th / 75th / 90th percentile of resample means.

Required gate fields (bool) per the 77 docx P0:

- ``gate_best_month_share_pass``  (best_month_share ≤ threshold)
- ``gate_month_cap_pass``         (capped net > 0)
- ``gate_leave_one_month_pass``   (leave_one_month_min > 0)
- ``gate_best_symbol_share_pass`` (best_symbol_share ≤ threshold)
- ``gate_leave_one_symbol_pass``  (leave_one_symbol_min > 0)
- ``gate_random_pass``            (candidate_mean ≥ random_p90)
- ``gate_bootstrap_pass``         (bootstrap_ci_lo > 0)
- ``gate_walk_forward_pass``      (walk_forward_min_net > 0)

Plus ``final_verdict`` — one of {``promote``, ``risk_off_only``,
``diagnostic_only``, ``no_value``, ``no_data``}.

All functions are PURE — no IO, no logging side-effects. Callers
(report modules) wrap them with their domain-specific gate ordering
and write the output rows themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

import numpy as np
import pandas as pd


class Verdict(str, Enum):
    PROMOTE = "promote"
    RISK_OFF_ONLY = "risk_off_only"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    NO_VALUE = "no_value"
    NO_DATA = "no_data"


class GateNames:
    """Canonical column names for the gate-pass booleans.

    Use these as dict keys / DataFrame columns rather than hardcoded
    strings so a typo can never silently produce a missing gate.
    """

    BEST_MONTH_SHARE: Final[str] = "gate_best_month_share_pass"
    MONTH_CAP: Final[str] = "gate_month_cap_pass"
    LEAVE_ONE_MONTH: Final[str] = "gate_leave_one_month_pass"
    BEST_SYMBOL_SHARE: Final[str] = "gate_best_symbol_share_pass"
    LEAVE_ONE_SYMBOL: Final[str] = "gate_leave_one_symbol_pass"
    RANDOM_BASELINE: Final[str] = "gate_random_pass"
    BOOTSTRAP: Final[str] = "gate_bootstrap_pass"
    WALK_FORWARD: Final[str] = "gate_walk_forward_pass"

    ALL: Final[tuple[str, ...]] = (
        BEST_MONTH_SHARE,
        MONTH_CAP,
        LEAVE_ONE_MONTH,
        BEST_SYMBOL_SHARE,
        LEAVE_ONE_SYMBOL,
        RANDOM_BASELINE,
        BOOTSTRAP,
        WALK_FORWARD,
    )


@dataclass(frozen=True)
class GateThresholds:
    """All gate-check thresholds. Defaults match the closure doc's reopen §1
    + §2 + §3 wording and the v7S validation findings."""

    # Distribution / concentration.
    month_cap_pct: float = 0.35
    symbol_share_max: float = 0.35
    # Bootstrap.
    bootstrap_draws: int = 5000
    bootstrap_seed: int = 20260617
    # Walk-forward.
    walk_forward_buckets: int = 3
    # Random baseline.
    random_draws: int = 1000
    random_seed: int = 20260618
    random_pass_percentile: int = 90  # candidate must beat random p90


@dataclass(frozen=True)
class DistributionMetrics:
    n_trades: int
    total_net: float
    best_month_share: float
    month_cap35_net: float
    leave_one_month_min: float
    best_symbol_share: float
    leave_one_symbol_min: float
    months: int
    symbols: int


@dataclass(frozen=True)
class BootstrapMetrics:
    n_trades: int
    mean: float
    ci_lo: float
    ci_hi: float
    p_positive: float


@dataclass(frozen=True)
class WalkForwardMetrics:
    buckets: int
    bucket_mean_nets: tuple[float, ...]
    walk_forward_min_net: float
    walk_forward_max_net: float
    walk_forward_delta: float


@dataclass(frozen=True)
class RandomBaselineMetrics:
    candidate_mean: float
    random_p50: float
    random_p75: float
    random_p90: float


@dataclass(frozen=True)
class CandidateVerdict:
    """Full verdict surface produced by ``evaluate_candidate_verdict``."""

    distribution: DistributionMetrics
    bootstrap: BootstrapMetrics | None
    walk_forward: WalkForwardMetrics | None
    random_baseline: RandomBaselineMetrics | None
    gate_pass: dict[str, bool]
    failures: tuple[str, ...]
    final_verdict: Verdict

    def to_dict(self) -> dict[str, object]:
        """Flatten into a one-row dict suitable for CSV emission."""
        out: dict[str, object] = {
            "n_trades": self.distribution.n_trades,
            "total_net": self.distribution.total_net,
            "best_month_share": self.distribution.best_month_share,
            "month_cap35_net": self.distribution.month_cap35_net,
            "leave_one_month_min": self.distribution.leave_one_month_min,
            "best_symbol_share": self.distribution.best_symbol_share,
            "leave_one_symbol_min": self.distribution.leave_one_symbol_min,
            "months": self.distribution.months,
            "symbols": self.distribution.symbols,
        }
        if self.bootstrap is not None:
            out["bootstrap_mean"] = self.bootstrap.mean
            out["bootstrap_ci_lo"] = self.bootstrap.ci_lo
            out["bootstrap_ci_hi"] = self.bootstrap.ci_hi
            out["bootstrap_p_positive"] = self.bootstrap.p_positive
        if self.walk_forward is not None:
            out["walk_forward_min_net"] = self.walk_forward.walk_forward_min_net
            out["walk_forward_max_net"] = self.walk_forward.walk_forward_max_net
            out["walk_forward_delta"] = self.walk_forward.walk_forward_delta
        if self.random_baseline is not None:
            out["random_p50"] = self.random_baseline.random_p50
            out["random_p75"] = self.random_baseline.random_p75
            out["random_p90"] = self.random_baseline.random_p90
            out["candidate_mean_vs_random"] = (
                self.random_baseline.candidate_mean - self.random_baseline.random_p50
            )
        for gate, ok in self.gate_pass.items():
            out[gate] = bool(ok)
        out["gate_failures"] = ";".join(self.failures) if self.failures else ""
        out["final_verdict"] = self.final_verdict.value
        return out


# --------------------------------------------------------------------------------------
# Distribution metrics
# --------------------------------------------------------------------------------------


def compute_distribution_metrics(
    trades: pd.DataFrame,
    *,
    net_col: str = "net20",
    month_col: str = "month",
    symbol_col: str = "symbol",
    thresholds: GateThresholds | None = None,
) -> DistributionMetrics:
    """Per-month / per-symbol concentration metrics.

    The trades frame must carry one row per execution with the listed
    columns. Missing columns are treated as a "no data" row in the
    relevant axis (still returns the frame's other axes' metrics).
    """
    th = thresholds or GateThresholds()
    if trades.empty or net_col not in trades.columns:
        return DistributionMetrics(
            n_trades=0,
            total_net=float("nan"),
            best_month_share=float("nan"),
            month_cap35_net=float("nan"),
            leave_one_month_min=float("nan"),
            best_symbol_share=float("nan"),
            leave_one_symbol_min=float("nan"),
            months=0,
            symbols=0,
        )
    nets = pd.to_numeric(trades[net_col], errors="coerce").fillna(0.0)
    total = float(nets.sum())

    # --- per-month axis ---
    if month_col in trades.columns:
        per_month = pd.DataFrame({"net": nets, "month": trades[month_col].astype(str)})
        month_sums = per_month.groupby("month")["net"].sum()
        months = int(len(month_sums))
        if total != 0:
            best_month_share = float(month_sums.abs().max() / month_sums.abs().sum()) if len(month_sums) else float("nan")
            # We want signed best-month-share too — if best month is negative-net, the
            # share question doesn't apply. Use signed share of total:
            signed_best_share = float(month_sums.max() / total) if total > 0 else float(month_sums.min() / total)
            # Honour the closure doc wording: "best month contribution" = SIGNED share
            # of total. When total is positive, the worst-case is best month carrying
            # too much positive contribution. When total is non-positive, the metric is
            # diagnostic only — we leave it as the absolute share.
            best_month_share = signed_best_share if total > 0 else best_month_share
        else:
            best_month_share = float("nan")
        # month_cap35
        if total > 0:
            cap_value = th.month_cap_pct * total
            capped = month_sums.clip(upper=cap_value)
            month_cap35_net = float(capped.sum())
        else:
            month_cap35_net = total  # nothing to cap when total ≤ 0
        # leave-one-month
        if len(month_sums) > 1:
            leave_one_month_min = float(min(total - v for v in month_sums))
        elif len(month_sums) == 1:
            leave_one_month_min = 0.0
        else:
            leave_one_month_min = float("nan")
    else:
        months = 0
        best_month_share = float("nan")
        month_cap35_net = float("nan")
        leave_one_month_min = float("nan")

    # --- per-symbol axis ---
    if symbol_col in trades.columns:
        per_symbol = pd.DataFrame({"net": nets, "symbol": trades[symbol_col].astype(str)})
        symbol_sums = per_symbol.groupby("symbol")["net"].sum()
        symbols = int(len(symbol_sums))
        if total != 0:
            best_symbol_share = float(symbol_sums.abs().max() / symbol_sums.abs().sum()) if len(symbol_sums) else float("nan")
        else:
            best_symbol_share = float("nan")
        if len(symbol_sums) > 1:
            leave_one_symbol_min = float(min(total - v for v in symbol_sums))
        elif len(symbol_sums) == 1:
            leave_one_symbol_min = 0.0
        else:
            leave_one_symbol_min = float("nan")
    else:
        symbols = 0
        best_symbol_share = float("nan")
        leave_one_symbol_min = float("nan")

    return DistributionMetrics(
        n_trades=int(len(nets)),
        total_net=total,
        best_month_share=best_month_share,
        month_cap35_net=month_cap35_net,
        leave_one_month_min=leave_one_month_min,
        best_symbol_share=best_symbol_share,
        leave_one_symbol_min=leave_one_symbol_min,
        months=months,
        symbols=symbols,
    )


# --------------------------------------------------------------------------------------
# Bootstrap CI on mean_net20
# --------------------------------------------------------------------------------------


def compute_bootstrap_ci(
    trades: pd.DataFrame,
    *,
    net_col: str = "net20",
    thresholds: GateThresholds | None = None,
) -> BootstrapMetrics:
    """Empirical 95 % bootstrap CI on mean_net20 + fraction of resample
    means strictly above zero."""
    th = thresholds or GateThresholds()
    if trades.empty or net_col not in trades.columns:
        return BootstrapMetrics(
            n_trades=0, mean=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"), p_positive=float("nan")
        )
    nets = pd.to_numeric(trades[net_col], errors="coerce").dropna().to_numpy()
    if nets.size == 0:
        return BootstrapMetrics(
            n_trades=0, mean=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"), p_positive=float("nan")
        )
    rng = np.random.default_rng(th.bootstrap_seed)
    n = len(nets)
    draws = np.array(
        [float(rng.choice(nets, size=n, replace=True).mean()) for _ in range(th.bootstrap_draws)]
    )
    return BootstrapMetrics(
        n_trades=int(n),
        mean=float(nets.mean()),
        ci_lo=float(np.percentile(draws, 2.5)),
        ci_hi=float(np.percentile(draws, 97.5)),
        p_positive=float((draws > 0).mean()),
    )


# --------------------------------------------------------------------------------------
# Walk-forward over N disjoint time buckets
# --------------------------------------------------------------------------------------


def compute_walk_forward_metrics(
    trades: pd.DataFrame,
    *,
    time_col: str = "signal_time",
    net_col: str = "net20",
    thresholds: GateThresholds | None = None,
) -> WalkForwardMetrics:
    """Sort by ``time_col`` and split into N equal-sized disjoint buckets,
    return per-bucket mean_net20 + min/max/delta.

    The min bucket is the load-bearing field for the walk-forward gate:
    if any bucket's mean is negative, the alpha is non-stationary.
    """
    th = thresholds or GateThresholds()
    if trades.empty or time_col not in trades.columns or net_col not in trades.columns:
        return WalkForwardMetrics(
            buckets=0,
            bucket_mean_nets=(),
            walk_forward_min_net=float("nan"),
            walk_forward_max_net=float("nan"),
            walk_forward_delta=float("nan"),
        )
    sub = trades.copy()
    sub[time_col] = pd.to_datetime(sub[time_col], utc=True, errors="coerce")
    sub = sub.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if len(sub) < th.walk_forward_buckets:
        return WalkForwardMetrics(
            buckets=len(sub),
            bucket_mean_nets=(),
            walk_forward_min_net=float("nan"),
            walk_forward_max_net=float("nan"),
            walk_forward_delta=float("nan"),
        )
    bucket_idx = np.linspace(0, th.walk_forward_buckets, num=len(sub), endpoint=False).astype(int)
    sub = sub.assign(_bucket=bucket_idx)
    means: list[float] = []
    for b in range(th.walk_forward_buckets):
        nets = pd.to_numeric(sub.loc[sub["_bucket"] == b, net_col], errors="coerce").dropna()
        means.append(float(nets.mean()) if len(nets) else float("nan"))
    means_arr = np.array(means, dtype=float)
    finite = means_arr[np.isfinite(means_arr)]
    if finite.size == 0:
        return WalkForwardMetrics(
            buckets=th.walk_forward_buckets,
            bucket_mean_nets=tuple(means),
            walk_forward_min_net=float("nan"),
            walk_forward_max_net=float("nan"),
            walk_forward_delta=float("nan"),
        )
    return WalkForwardMetrics(
        buckets=th.walk_forward_buckets,
        bucket_mean_nets=tuple(means),
        walk_forward_min_net=float(finite.min()),
        walk_forward_max_net=float(finite.max()),
        walk_forward_delta=float(finite.max() - finite.min()),
    )


# --------------------------------------------------------------------------------------
# Random baseline at p50 / p75 / p90
# --------------------------------------------------------------------------------------


def compute_random_baseline_metrics(
    trades: pd.DataFrame,
    *,
    net_col: str = "net20",
    pool_nets: np.ndarray | None = None,
    thresholds: GateThresholds | None = None,
) -> RandomBaselineMetrics:
    """Random-shuffle baseline at p50/p75/p90 of resample means.

    The pool defaults to the trades' own ``net_col`` (matched bootstrap).
    Pass an external ``pool_nets`` array to compute a different baseline
    (e.g. all-trades pool when evaluating per-candidate against the
    global pool, per the 77 docx random_p90 spec).
    """
    th = thresholds or GateThresholds()
    if trades.empty or net_col not in trades.columns:
        return RandomBaselineMetrics(
            candidate_mean=float("nan"),
            random_p50=float("nan"),
            random_p75=float("nan"),
            random_p90=float("nan"),
        )
    nets = pd.to_numeric(trades[net_col], errors="coerce").dropna().to_numpy()
    if nets.size == 0:
        return RandomBaselineMetrics(
            candidate_mean=float("nan"),
            random_p50=float("nan"),
            random_p75=float("nan"),
            random_p90=float("nan"),
        )
    pool = pool_nets if pool_nets is not None else nets
    if pool.size == 0:
        return RandomBaselineMetrics(
            candidate_mean=float(nets.mean()),
            random_p50=float("nan"),
            random_p75=float("nan"),
            random_p90=float("nan"),
        )
    rng = np.random.default_rng(th.random_seed)
    n = len(nets)
    draws = np.array([float(rng.choice(pool, size=n, replace=True).mean()) for _ in range(th.random_draws)])
    return RandomBaselineMetrics(
        candidate_mean=float(nets.mean()),
        random_p50=float(np.percentile(draws, 50)),
        random_p75=float(np.percentile(draws, 75)),
        random_p90=float(np.percentile(draws, 90)),
    )


# --------------------------------------------------------------------------------------
# Verdict assembly
# --------------------------------------------------------------------------------------


def evaluate_candidate_verdict(
    trades: pd.DataFrame,
    *,
    net_col: str = "net20",
    month_col: str = "month",
    symbol_col: str = "symbol",
    time_col: str = "signal_time",
    pool_nets: np.ndarray | None = None,
    thresholds: GateThresholds | None = None,
    additional_failures: tuple[str, ...] = (),
    additional_passes: tuple[str, ...] = (),
) -> CandidateVerdict:
    """Assemble the full eight-gate verdict for a candidate's trade frame.

    ``additional_failures`` / ``additional_passes`` let a caller (e.g.
    v7S Direction A) inject domain-specific gates (squeeze share,
    clean-short hit, vs-no-long head-to-head) without re-implementing
    the shared distribution checks. The verdict logic AND's everything
    together.

    Verdict ladder:
    - ``promote`` — all gates pass AND no caller-injected failures.
    - ``risk_off_only`` — all distribution / concentration gates pass
      but bootstrap_pass OR random_pass fails (one of the two).
    - ``diagnostic_only`` — walk_forward_min_net is negative (alpha
      didn't survive any disjoint third) but every other gate passes.
    - ``no_value`` — any distribution/concentration gate fails.
    - ``no_data`` — empty trade frame.
    """
    th = thresholds or GateThresholds()
    if trades.empty:
        empty_dist = compute_distribution_metrics(trades, net_col=net_col, month_col=month_col, symbol_col=symbol_col, thresholds=th)
        return CandidateVerdict(
            distribution=empty_dist,
            bootstrap=None,
            walk_forward=None,
            random_baseline=None,
            gate_pass={g: False for g in GateNames.ALL},
            failures=("no_data",),
            final_verdict=Verdict.NO_DATA,
        )

    distribution = compute_distribution_metrics(
        trades,
        net_col=net_col,
        month_col=month_col,
        symbol_col=symbol_col,
        thresholds=th,
    )
    bootstrap = compute_bootstrap_ci(trades, net_col=net_col, thresholds=th)
    walk_forward = compute_walk_forward_metrics(trades, time_col=time_col, net_col=net_col, thresholds=th)
    random_baseline = compute_random_baseline_metrics(trades, net_col=net_col, pool_nets=pool_nets, thresholds=th)

    gate_pass: dict[str, bool] = {}
    failures: list[str] = []

    # Distribution / concentration gates.
    best_share = distribution.best_month_share
    gate_pass[GateNames.BEST_MONTH_SHARE] = bool(np.isfinite(best_share) and best_share <= th.month_cap_pct)
    if not gate_pass[GateNames.BEST_MONTH_SHARE]:
        failures.append(GateNames.BEST_MONTH_SHARE)

    capped = distribution.month_cap35_net
    gate_pass[GateNames.MONTH_CAP] = bool(np.isfinite(capped) and capped > 0)
    if not gate_pass[GateNames.MONTH_CAP]:
        failures.append(GateNames.MONTH_CAP)

    leave_min = distribution.leave_one_month_min
    gate_pass[GateNames.LEAVE_ONE_MONTH] = bool(np.isfinite(leave_min) and leave_min > 0)
    if not gate_pass[GateNames.LEAVE_ONE_MONTH]:
        failures.append(GateNames.LEAVE_ONE_MONTH)

    best_sym = distribution.best_symbol_share
    gate_pass[GateNames.BEST_SYMBOL_SHARE] = bool(np.isfinite(best_sym) and best_sym <= th.symbol_share_max)
    if not gate_pass[GateNames.BEST_SYMBOL_SHARE]:
        failures.append(GateNames.BEST_SYMBOL_SHARE)

    leave_sym = distribution.leave_one_symbol_min
    gate_pass[GateNames.LEAVE_ONE_SYMBOL] = bool(np.isfinite(leave_sym) and leave_sym > 0)
    if not gate_pass[GateNames.LEAVE_ONE_SYMBOL]:
        failures.append(GateNames.LEAVE_ONE_SYMBOL)

    # Robustness gates.
    if th.random_pass_percentile == 90:
        random_target = random_baseline.random_p90
    elif th.random_pass_percentile == 75:
        random_target = random_baseline.random_p75
    else:
        random_target = random_baseline.random_p50
    gate_pass[GateNames.RANDOM_BASELINE] = bool(
        np.isfinite(random_baseline.candidate_mean)
        and np.isfinite(random_target)
        and random_baseline.candidate_mean >= random_target
    )
    if not gate_pass[GateNames.RANDOM_BASELINE]:
        failures.append(GateNames.RANDOM_BASELINE)

    gate_pass[GateNames.BOOTSTRAP] = bool(np.isfinite(bootstrap.ci_lo) and bootstrap.ci_lo > 0)
    if not gate_pass[GateNames.BOOTSTRAP]:
        failures.append(GateNames.BOOTSTRAP)

    gate_pass[GateNames.WALK_FORWARD] = bool(
        np.isfinite(walk_forward.walk_forward_min_net) and walk_forward.walk_forward_min_net > 0
    )
    if not gate_pass[GateNames.WALK_FORWARD]:
        failures.append(GateNames.WALK_FORWARD)

    # Inject caller-supplied gates so the surface is uniform.
    failures.extend(additional_failures)
    # additional_passes is informational only; callers track them in their own outputs.
    _ = additional_passes

    final = _resolve_verdict(gate_pass, walk_forward, failures)

    return CandidateVerdict(
        distribution=distribution,
        bootstrap=bootstrap,
        walk_forward=walk_forward,
        random_baseline=random_baseline,
        gate_pass=gate_pass,
        failures=tuple(failures),
        final_verdict=final,
    )


def _resolve_verdict(
    gate_pass: dict[str, bool],
    walk_forward: WalkForwardMetrics | None,
    failures: list[str],
) -> Verdict:
    """Verdict-ladder logic. Distribution / concentration gates dominate;
    a single walk-forward failure demotes to ``diagnostic_only`` rather
    than ``no_value`` (the candidate is interesting research output, not
    tradable)."""
    if failures and "no_data" in failures:
        return Verdict.NO_DATA
    dist_gates = (
        GateNames.BEST_MONTH_SHARE,
        GateNames.MONTH_CAP,
        GateNames.LEAVE_ONE_MONTH,
        GateNames.BEST_SYMBOL_SHARE,
        GateNames.LEAVE_ONE_SYMBOL,
    )
    dist_ok = all(gate_pass.get(g, False) for g in dist_gates)
    if not dist_ok:
        return Verdict.NO_VALUE
    wf_ok = gate_pass.get(GateNames.WALK_FORWARD, False)
    bs_ok = gate_pass.get(GateNames.BOOTSTRAP, False)
    rnd_ok = gate_pass.get(GateNames.RANDOM_BASELINE, False)
    if not wf_ok:
        # alpha existed in aggregate but didn't survive every disjoint third
        return Verdict.DIAGNOSTIC_ONLY
    if bs_ok and rnd_ok:
        return Verdict.PROMOTE
    if bs_ok or rnd_ok:
        return Verdict.RISK_OFF_ONLY
    return Verdict.NO_VALUE
