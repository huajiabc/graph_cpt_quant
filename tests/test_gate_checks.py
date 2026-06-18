"""Tests for validation/gate_checks.py — the global verdict evaluator.

The key bug this module exists to prevent is the v7S A1_imb10bp h24
PROMOTE-vs-NO_VALUE flip: a single-month-concentrated sample that
passes ``month_capped_net > 0`` but violates the closure doc's
``best_month_share ≤ 0.35`` rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.validation.gate_checks import (
    CandidateVerdict,
    GateNames,
    GateThresholds,
    Verdict,
    compute_bootstrap_ci,
    compute_distribution_metrics,
    compute_random_baseline_metrics,
    compute_walk_forward_metrics,
    evaluate_candidate_verdict,
)


def _make_trades(
    *,
    n_per_month: dict[str, int],
    net_per_month: dict[str, float],
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    syms = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "AAVEUSDT", "ARBUSDT"]
    base = pd.Timestamp("2026-01-01", tz="UTC")
    for month, n in n_per_month.items():
        net = net_per_month.get(month, 0.0)
        per_trade = net / max(n, 1)
        for i in range(n):
            sig_time = base + pd.Timedelta(days=int(month.split("-")[1]) * 20 + i)
            rows.append(
                {
                    "symbol": syms[i % len(syms)],
                    "month": month,
                    "signal_time": sig_time,
                    "net20": per_trade,
                }
            )
    return pd.DataFrame(rows)


class TestDistributionMetrics:
    def test_well_distributed_passes_concentration_checks(self) -> None:
        trades = _make_trades(
            n_per_month={"2026-01": 30, "2026-02": 30, "2026-03": 30},
            net_per_month={"2026-01": 0.3, "2026-02": 0.3, "2026-03": 0.3},
        )
        dm = compute_distribution_metrics(trades)
        assert dm.months == 3
        assert dm.symbols >= 3
        # Each month carries roughly 1/3 of total — well under 35 %.
        assert dm.best_month_share <= 0.36
        assert dm.month_cap35_net > 0
        assert dm.leave_one_month_min > 0

    def test_single_month_dominance_flags_concentration(self) -> None:
        # Mimic the A1_imb10bp pattern: best month = 64 % of total.
        trades = _make_trades(
            n_per_month={"2026-01": 10, "2026-02": 30, "2026-03": 10},
            net_per_month={"2026-01": -0.05, "2026-02": 0.50, "2026-03": 0.05},
        )
        dm = compute_distribution_metrics(trades)
        # total = 0.50 - 0.05 + 0.05 = 0.50. Best month (Feb) carries 0.50/0.50 = 100 % of total
        # since the others nearly cancel. Even at less extreme settings,
        # best_month_share well above 35 %.
        assert dm.best_month_share > 0.35


class TestBootstrap:
    def test_bootstrap_ci_strictly_positive_for_strong_alpha(self) -> None:
        trades = pd.DataFrame(
            {
                "net20": np.full(200, 0.05),  # constant +5 % returns
                "signal_time": pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC"),
                "month": ["2026-01"] * 100 + ["2026-02"] * 100,
                "symbol": ["BTCUSDT"] * 200,
            }
        )
        thr = GateThresholds(bootstrap_draws=200, bootstrap_seed=1)
        bs = compute_bootstrap_ci(trades, thresholds=thr)
        assert bs.ci_lo > 0
        assert bs.p_positive == 1.0

    def test_bootstrap_ci_straddles_zero_for_noisy_sample(self) -> None:
        # Sample mean designed to be near zero: alternating ±0.05 + small noise.
        rng = np.random.default_rng(7)
        base = np.array([0.05, -0.05] * 15)
        noise = rng.normal(0.0, 0.005, size=30)
        trades = pd.DataFrame(
            {
                "net20": base + noise,
                "signal_time": pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC"),
                "month": ["2026-01"] * 30,
                "symbol": ["BTCUSDT"] * 30,
            }
        )
        thr = GateThresholds(bootstrap_draws=400, bootstrap_seed=2)
        bs = compute_bootstrap_ci(trades, thresholds=thr)
        # With ±0.05 mean-zero samples, CI should straddle zero.
        assert bs.ci_lo < 0 < bs.ci_hi


class TestWalkForward:
    def test_bucket_means_match_expected_split(self) -> None:
        trades = pd.DataFrame(
            {
                "signal_time": pd.date_range("2026-01-01", periods=30, freq="d", tz="UTC"),
                "net20": [0.01] * 10 + [-0.02] * 10 + [0.03] * 10,
                "month": ["2026-01"] * 10 + ["2026-02"] * 10 + ["2026-03"] * 10,
                "symbol": ["BTCUSDT"] * 30,
            }
        )
        thr = GateThresholds(walk_forward_buckets=3)
        wf = compute_walk_forward_metrics(trades, thresholds=thr)
        assert wf.buckets == 3
        assert wf.bucket_mean_nets[0] > 0
        assert wf.bucket_mean_nets[1] < 0
        assert wf.bucket_mean_nets[2] > 0
        assert wf.walk_forward_min_net < 0  # bucket 1


class TestRandomBaseline:
    def test_candidate_mean_above_random_p90(self) -> None:
        rng = np.random.default_rng(13)
        pool = rng.normal(0.0, 0.01, size=2000)
        candidate_net = 0.02
        trades = pd.DataFrame(
            {
                "net20": [candidate_net] * 30,
                "signal_time": pd.date_range("2026-01-01", periods=30, freq="d", tz="UTC"),
                "month": ["2026-01"] * 30,
                "symbol": ["BTCUSDT"] * 30,
            }
        )
        thr = GateThresholds(random_draws=200, random_seed=14)
        rb = compute_random_baseline_metrics(trades, pool_nets=pool, thresholds=thr)
        assert rb.candidate_mean > rb.random_p90


class TestEvaluateCandidateVerdict:
    def test_no_data_returns_no_data_verdict(self) -> None:
        verdict = evaluate_candidate_verdict(pd.DataFrame())
        assert verdict.final_verdict is Verdict.NO_DATA

    def test_single_month_concentration_returns_no_value(self) -> None:
        # A1_imb10bp pattern reproduction at the verdict level.
        rng = np.random.default_rng(101)
        # 56 trades; 36 in Oct 2025 (concentrated), 20 spread across other months.
        n_per_month = {
            "2025-07": 12, "2025-09": 3, "2025-10": 13, "2025-11": 2,
            "2025-12": 3, "2026-01": 11, "2026-03": 3, "2026-05": 8, "2026-06": 1,
        }
        net_per_month = {
            "2025-07": -0.174, "2025-09": -0.078, "2025-10": 0.419, "2025-11": -0.0004,
            "2025-12": 0.129, "2026-01": 0.298, "2026-03": -0.081, "2026-05": 0.157, "2026-06": -0.011,
        }
        trades = _make_trades(n_per_month=n_per_month, net_per_month=net_per_month)
        verdict = evaluate_candidate_verdict(trades)
        # 2025-10 carries ~64 % of total — best_month_share gate must fail.
        assert verdict.gate_pass[GateNames.BEST_MONTH_SHARE] is False
        assert verdict.final_verdict is Verdict.NO_VALUE

    def test_well_distributed_strong_alpha_returns_promote(self) -> None:
        # Construct a fairly distributed positive sample with strong CI.
        n_per_month = {"2026-01": 40, "2026-02": 40, "2026-03": 40, "2026-04": 40}
        net_per_month = {k: 0.50 for k in n_per_month}
        trades = _make_trades(n_per_month=n_per_month, net_per_month=net_per_month)
        # Pool nets close to zero so random_p90 stays well below candidate mean.
        rng = np.random.default_rng(2)
        pool = rng.normal(0.0, 0.005, size=2000)
        thr = GateThresholds(
            bootstrap_draws=300, bootstrap_seed=3, random_draws=200, random_seed=4
        )
        verdict = evaluate_candidate_verdict(trades, pool_nets=pool, thresholds=thr)
        assert verdict.final_verdict is Verdict.PROMOTE

    def test_walk_forward_loss_downgrades_to_diagnostic_only(self) -> None:
        # 9 months × 10 trades each = 90. Months 1-2 negative, months 3-9 positive
        # — distribution clean (best month share ≤ 35%), but the time-ordered
        # first third (= months 1-3) has negative aggregate mean.
        # Pool nets shifted positive so random_p90 stays low → random gate passes.
        rows: list[dict[str, object]] = []
        symbols = ["BTC", "ETH", "SOL", "BNB", "ARB", "OP", "DOGE", "AAVE", "LDO", "SUI"]
        base = pd.Timestamp("2026-01-01", tz="UTC")
        for month_i in range(1, 10):
            net_per_trade = -0.02 if month_i <= 2 else 0.02
            for trade_j in range(10):
                rows.append(
                    {
                        "symbol": symbols[trade_j % len(symbols)],
                        "month": f"2026-{month_i:02d}",
                        "signal_time": base + pd.Timedelta(days=(month_i - 1) * 30 + trade_j),
                        "net20": net_per_trade,
                    }
                )
        trades = pd.DataFrame(rows)
        # External pool with mean ≈ 0 so candidate beats random p90 cleanly.
        rng = np.random.default_rng(9)
        pool = rng.normal(0.0, 0.003, size=2000)
        thr = GateThresholds(
            bootstrap_draws=200,
            bootstrap_seed=5,
            random_draws=200,
            random_seed=6,
        )
        verdict = evaluate_candidate_verdict(trades, pool_nets=pool, thresholds=thr)
        # Sanity: distribution gates all clean.
        assert verdict.gate_pass[GateNames.BEST_MONTH_SHARE] is True
        assert verdict.gate_pass[GateNames.MONTH_CAP] is True
        # Walk-forward FAILS because the first third spans months 1-3 and
        # months 1-2's negative contribution dominates.
        assert verdict.gate_pass[GateNames.WALK_FORWARD] is False
        # The candidate is interesting research output, NOT tradable — diagnostic_only.
        assert verdict.final_verdict is Verdict.DIAGNOSTIC_ONLY

    def test_to_dict_contains_all_gate_columns(self) -> None:
        trades = _make_trades(
            n_per_month={"2026-01": 30, "2026-02": 30, "2026-03": 30},
            net_per_month={"2026-01": 0.3, "2026-02": 0.3, "2026-03": 0.3},
        )
        verdict = evaluate_candidate_verdict(trades)
        d = verdict.to_dict()
        for g in GateNames.ALL:
            assert g in d, f"verdict.to_dict() missing gate column {g}"
        assert "final_verdict" in d
        assert "gate_failures" in d
        assert d["final_verdict"] in {v.value for v in Verdict}
