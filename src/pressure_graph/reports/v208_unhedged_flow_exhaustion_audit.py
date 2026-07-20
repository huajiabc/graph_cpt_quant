"""Independent audit for the v20.7 unhedged flow-exhaustion diagnostic."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v184_btc_inclusive_metrics_audit import (
    KLINE_ROOT,
    METRICS_ROOT,
    load_v184_exact_panels,
)
from pressure_graph.reports.v185_btc_leverage_flow_graph import BTC
from pressure_graph.reports.v206_aggtrade_flow_exhaustion_audit import parse_mapping
from pressure_graph.reports.v207_unhedged_flow_exhaustion import (
    CANDIDATE,
    REPORT_ROOT as V207_REPORT_ROOT,
    V207Config,
)


REPORT_ROOT = Path("reports/v20_8_unhedged_flow_exhaustion_audit")
FINDINGS_PATH = Path("docs/v208_unhedged_flow_exhaustion_audit_2026_07_17.md")


def audit_v208(
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
    source_root: Path = V207_REPORT_ROOT,
    cfg: V207Config = V207Config(),
) -> pd.DataFrame:
    events = pd.read_parquet(source_root / "candidate_events.parquet")
    delayed = pd.read_parquet(source_root / "delayed_candidate_events.parquet")
    summary = pd.read_csv(source_root / "period_summary.csv")
    horizons = pd.read_csv(source_root / "holding_horizon_summary.csv")
    random = pd.read_parquet(source_root / "random_controls.parquet")
    bootstrap = pd.read_csv(source_root / "bootstrap_summary.csv").iloc[0]
    gates = pd.read_csv(source_root / "candidate_gates.csv")
    outcome = pd.read_csv(source_root / "candidate_outcome.csv").iloc[0]
    close, _ = load_v184_exact_panels(metrics_root, kline_root)
    maximum_contribution_error = 0.0
    maximum_receiver_error = 0.0
    maximum_btc_error = 0.0
    maximum_pair_error = 0.0
    maximum_cost_error = 0.0
    maximum_notional_error = 0.0
    causal = True
    no_btc_weight = True
    for item in events.itertuples(index=False):
        weights = parse_mapping(item.weights)
        stored = parse_mapping(item.symbol_contributions)
        entry = pd.Timestamp(item.entry_time)
        exit_time = pd.Timestamp(item.exit_time)
        causal &= entry == pd.Timestamp(item.feature_time)
        causal &= exit_time == entry + pd.Timedelta(minutes=15)
        no_btc_weight &= BTC not in weights
        future = close.loc[exit_time].div(close.loc[entry]).sub(1.0)
        contributions = {
            symbol: weight * float(future[symbol])
            for symbol, weight in weights.items()
        }
        maximum_contribution_error = max(
            maximum_contribution_error,
            max(abs(contributions[symbol] - stored[symbol]) for symbol in weights),
        )
        receiver = float(sum(contributions.values()))
        btc = float(-item.source_sign * future[BTC])
        pair = receiver - btc
        maximum_receiver_error = max(
            maximum_receiver_error, abs(receiver - item.receiver_gross_return)
        )
        maximum_btc_error = max(
            maximum_btc_error, abs(btc - item.btc_control_gross_return)
        )
        maximum_pair_error = max(
            maximum_pair_error, abs(pair - item.paired_receiver_minus_btc_return)
        )
        maximum_cost_error = max(
            maximum_cost_error,
            abs(
                item.receiver_primary_net_return
                - (receiver - cfg.primary_round_trip_cost)
            ),
            abs(
                item.receiver_stress_net_return
                - (receiver - cfg.stress_round_trip_cost)
            ),
        )
        maximum_notional_error = max(
            maximum_notional_error,
            abs(sum(abs(weight) for weight in weights.values()) - 1.0),
        )
    maximum_summary_error = 0.0
    for item in summary.itertuples(index=False):
        local = events if item.scope == "all" else events[events["period"].eq(item.scope)]
        maximum_summary_error = max(
            maximum_summary_error,
            abs(
                float(local["receiver_gross_return"].mean() * 10_000)
                - item.mean_receiver_gross_bp
            ),
            abs(
                float(local["paired_receiver_minus_btc_return"].mean() * 10_000)
                - item.mean_receiver_minus_btc_bp
            ),
        )
    observed = float(events["receiver_gross_return"].mean())
    percentile = float(random["mean_receiver_gross_return"].le(observed).mean())
    rng = np.random.default_rng(cfg.seed + 100)
    net = events["receiver_primary_net_return"].to_numpy(dtype=float)
    paired = events["paired_receiver_minus_btc_return"].to_numpy(dtype=float)
    net_means: list[float] = []
    paired_means: list[float] = []
    for _ in range(cfg.bootstrap_iterations):
        indices = rng.integers(0, len(events), size=len(events))
        net_means.append(float(net[indices].mean()))
        paired_means.append(float(paired[indices].mean()))
    bootstrap_error = max(
        abs(
            float(np.quantile(net_means, 0.025) * 10_000)
            - bootstrap["lower_95_receiver_primary_net_bp"]
        ),
        abs(
            float(np.quantile(paired_means, 0.025) * 10_000)
            - bootstrap["lower_95_receiver_minus_btc_bp"]
        ),
    )
    checks = {
        "candidate_name_frozen": events["candidate"].eq(CANDIDATE).all(),
        "event_count_53": len(events) == 53,
        "period_counts_30_15_8": events.groupby("period").size().to_dict()
        == {"development": 30, "validation": 15, "holdout": 8},
        "event_keys_unique": events["source_event_id"].nunique() == len(events),
        "entry_exit_timing_causal": causal,
        "no_btc_weight_in_receiver_book": no_btc_weight,
        "receiver_gross_notional_one": maximum_notional_error < 1e-9,
        "symbol_contributions_reproduced": maximum_contribution_error < 1e-12,
        "receiver_gross_reproduced": maximum_receiver_error < 1e-12,
        "btc_control_reproduced": maximum_btc_error < 1e-12,
        "paired_excess_reproduced": maximum_pair_error < 1e-12,
        "cost_charges_reproduced": maximum_cost_error < 1e-12,
        "period_summary_reproduced": maximum_summary_error < 1e-10,
        "delayed_count_53": len(delayed) == 53,
        "horizon_rows_complete": len(horizons) == 8,
        "random_control_rows_500": len(random) == cfg.random_iterations,
        "random_percentile_reproduced": abs(
            percentile - outcome["random_control_percentile"]
        )
        < 1e-12,
        "bootstrap_intervals_reproduced": bootstrap_error < 1e-10,
        "gate_outcome_consistent": bool(gates["passed"].all())
        == bool(outcome["eligible_for_natural_forward_observation"]),
        "diagnostic_rejected": not bool(
            outcome["eligible_for_natural_forward_observation"]
        ),
    }
    return pd.DataFrame(
        {"check": list(checks), "passed": list(checks.values())}
    )


def write_v208_audit(
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    checks = audit_v208()
    root = ensure_dir(report_root)
    checks_path = root / "independent_audit_checks.csv"
    checks.to_csv(checks_path, index=False)
    verdict = (
        "audit_pass_v207_diagnostic_rejection_reproduced"
        if bool(checks["passed"].all())
        else "audit_failed"
    )
    text = [
        "# v20.8 Unhedged Flow-Exhaustion Independent Audit",
        "",
        f"Verdict: `{verdict}`.",
        "",
        f"Passed {int(checks['passed'].sum())}/{len(checks)} independent checks.",
        "",
        "The audit independently reloaded official close prices and reproduced "
        "the receiver book, BTC-only control, paired excess, costs, random-event "
        "percentile, bootstrap intervals, and diagnostic rejection.",
        "",
        "No live, PaperLive, application, leverage, remote, or order state changed.",
        "",
    ]
    ensure_dir(findings_path.parent)
    findings_path.write_text("\n".join(text), encoding="utf-8")
    return {"checks": checks_path, "findings": findings_path}


__all__ = ["audit_v208", "write_v208_audit"]
