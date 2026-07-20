"""Independent fill and PnL audit for v18.2 passive retracement."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v180_extreme_residual_dispersion_compression import (
    REPORT_ROOT as V180_REPORT_ROOT,
)
from pressure_graph.reports.v182_passive_residual_retracement import (
    REPORT_ROOT,
    V182Config,
)


AUDIT_ROOT = Path("reports/v18_2_passive_residual_retracement_audit")
FINDINGS_PATH = Path(
    "docs/v182_passive_residual_retracement_audit_2026_07_16.md"
)


def _check(rows: list[dict[str, object]], name: str, passed: bool, value: object) -> None:
    rows.append({"check": name, "passed": bool(passed), "value": value})


def audit_v182(
    report_root: Path = REPORT_ROOT,
    v180_report_root: Path = V180_REPORT_ROOT,
    cfg: V182Config = V182Config(),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    source_signals = pd.read_parquet(v180_report_root / "dispersion_signals.parquet")
    fixed_signals = pd.read_parquet(report_root / "fixed_v180_signals.parquet")
    events = pd.read_parquet(report_root / "candidate_events.parquet")
    legs = pd.read_parquet(report_root / "passive_fill_legs.parquet")
    delayed = pd.read_parquet(report_root / "delayed_candidate_events.parquet")
    summary = pd.read_csv(report_root / "period_summary.csv")
    sensitivity = pd.read_csv(report_root / "entry_offset_sensitivity.csv")
    random = pd.read_parquet(report_root / "random_rank_controls.parquet")
    gates = pd.read_csv(report_root / "candidate_gates.csv")
    outcome = pd.read_csv(report_root / "candidate_outcome.csv")
    for frame in (source_signals, fixed_signals):
        frame["source_feature_time"] = pd.to_datetime(
            frame["source_feature_time"], utc=True
        )
    for frame in (events, delayed):
        for column in (
            "source_feature_time",
            "reference_time",
            "fill_time",
            "exit_time",
        ):
            frame[column] = pd.to_datetime(frame[column], utc=True)
    for column in ("source_feature_time", "reference_time", "fill_time", "exit_time"):
        legs[column] = pd.to_datetime(legs[column], utc=True)

    source_keys = set(source_signals["source_feature_time"])
    fixed_keys = set(fixed_signals["source_feature_time"])
    _check(rows, "fixed_signals_exact", source_keys == fixed_keys, len(source_keys))
    _check(rows, "all_source_events_retained", len(events) == len(fixed_signals), len(events))
    _check(
        rows,
        "reference_time_exact",
        events["reference_time"].eq(events["source_feature_time"]).all(),
        int(events["reference_time"].ne(events["source_feature_time"]).sum()),
    )
    _check(
        rows,
        "fill_time_exact_next_bar",
        (events["fill_time"] - events["reference_time"])
        .eq(pd.Timedelta(minutes=15))
        .all(),
        len(events),
    )
    _check(
        rows,
        "exit_time_exact_four_bars_after_fill",
        (events["exit_time"] - events["fill_time"])
        .eq(pd.Timedelta(minutes=60))
        .all(),
        len(events),
    )
    legs_per_event = legs.groupby("source_feature_time").size()
    _check(
        rows,
        "ten_intended_legs_per_event",
        legs_per_event.eq(10).all(),
        f"{legs_per_event.min()}:{legs_per_event.max()}",
    )
    buy = legs["side"].eq("laggard_buy")
    expected_limit = legs["reference_close"] * (
        1.0 + legs["entry_offset"] * buy.map({True: -1.0, False: 1.0})
    )
    limit_error = (legs["limit_price"] - expected_limit).abs().max()
    _check(rows, "limit_price_formula_exact", limit_error <= 1e-12, limit_error)
    expected_fill = (
        buy & legs["fill_bar_low"].le(legs["limit_price"])
    ) | (~buy & legs["fill_bar_high"].ge(legs["limit_price"]))
    _check(
        rows,
        "touch_fill_rule_exact",
        legs["filled"].astype(bool).eq(expected_fill).all(),
        int(legs["filled"].astype(bool).ne(expected_fill).sum()),
    )
    filled_counts = legs.groupby("source_feature_time")["filled"].sum().astype(int)
    event_counts = events.set_index("source_feature_time")["filled_count"].astype(int)
    _check(
        rows,
        "partial_fills_all_retained",
        filled_counts.equals(event_counts),
        int((filled_counts - event_counts).abs().sum()),
    )
    leg_gross = legs.groupby("source_feature_time")["gross_contribution"].sum()
    event_alt_gross = events.set_index("source_feature_time")["alt_gross_return"]
    _check(
        rows,
        "alt_gross_matches_filled_legs",
        (leg_gross - event_alt_gross).abs().max() <= 1e-12,
        float((leg_gross - event_alt_gross).abs().max()),
    )
    filled_legs = legs[legs["filled"].astype(bool)].copy()
    filled_legs["beta_exposure"] = (
        filled_legs["signed_weight"] * filled_legs["btc_beta"]
    )
    beta_exposure = filled_legs.groupby("source_feature_time")["beta_exposure"].sum()
    hedge = events.set_index("source_feature_time")["btc_hedge_weight"]
    hedge_error = (hedge.add(beta_exposure.reindex(hedge.index, fill_value=0.0))).abs().max()
    _check(rows, "filled_beta_hedge_exact", hedge_error <= 1e-12, hedge_error)
    gross_error = (
        events["gross_return"]
        - (events["alt_gross_return"] + events["btc_gross_return"])
    ).abs().max()
    _check(rows, "event_gross_identity", gross_error <= 1e-12, gross_error)
    primary_cost_expected = (
        events["filled_alt_allocation"] * cfg.alt_primary_cost
        + events["btc_hedge_weight"].abs() * cfg.btc_primary_cost
    )
    stress_cost_expected = (
        events["filled_alt_allocation"] * cfg.alt_stress_cost
        + events["btc_hedge_weight"].abs() * cfg.btc_stress_cost
    )
    cost_error = max(
        (events["primary_cost_return"] - primary_cost_expected).abs().max(),
        (events["stress_cost_return"] - stress_cost_expected).abs().max(),
    )
    _check(rows, "realized_allocation_costs_exact", cost_error <= 1e-12, cost_error)
    net_error = max(
        (
            events["primary_net_return"]
            - (events["gross_return"] - events["primary_cost_return"])
        ).abs().max(),
        (
            events["stress_net_return"]
            - (events["gross_return"] - events["stress_cost_return"])
        ).abs().max(),
    )
    _check(rows, "net_return_identities", net_error <= 1e-12, net_error)
    reported = summary[summary["scope"].eq("all")]
    actual_mean = float(events["primary_net_return"].mean() * 10_000)
    reported_mean = float(reported["mean_primary_net_bp"].iloc[0])
    _check(
        rows,
        "summary_exact",
        math.isclose(actual_mean, reported_mean, rel_tol=0, abs_tol=1e-10),
        actual_mean - reported_mean,
    )
    _check(
        rows,
        "delayed_orders_exact_one_bar",
        (delayed["reference_time"] - delayed["source_feature_time"])
        .eq(pd.Timedelta(minutes=15))
        .all(),
        len(delayed),
    )
    _check(
        rows,
        "sensitivity_offsets_exact",
        set(sensitivity["entry_offset_bp"].astype(float)) == {5.0, 20.0},
        "|".join(map(str, sorted(sensitivity["entry_offset_bp"].unique()))),
    )
    _check(
        rows,
        "random_iterations_complete",
        random["iteration"].nunique() == cfg.random_iterations,
        int(random["iteration"].nunique()),
    )
    _check(
        rows,
        "random_source_counts_match",
        random["source_events"].eq(len(events)).all(),
        f"{random['source_events'].min()}:{random['source_events'].max()}",
    )
    real_mean_return = float(events["primary_net_return"].mean())
    percentile = float(random["mean_primary_net_return"].le(real_mean_return).mean())
    reported_percentile = float(outcome["random_rank_percentile"].iloc[0])
    _check(
        rows,
        "random_percentile_exact",
        math.isclose(percentile, reported_percentile, rel_tol=0, abs_tol=1e-12),
        percentile - reported_percentile,
    )
    _check(rows, "all_gate_rows_ineligible", not gates["eligible"].any(), len(gates))
    _check(rows, "candidate_rejected", not outcome["eligible"].any(), len(outcome))
    _check(
        rows,
        "rejection_verdict_exact",
        outcome["verdict"].eq("reject_passive_residual_retracement").all(),
        str(outcome["verdict"].iloc[0]),
    )
    config_hits = [
        path.name
        for path in Path("configs").glob("*")
        if "v182" in path.name.lower() or "passive_residual" in path.name.lower()
    ]
    _check(rows, "no_live_config_created", not config_hits, "|".join(config_hits))
    audit = pd.DataFrame(rows)
    audit["round_verdict"] = (
        "audit_pass_passive_candidate_rejected"
        if audit["passed"].all()
        else "audit_failure_requires_investigation"
    )
    return audit


def _write_findings(audit: pd.DataFrame, path: Path) -> None:
    failed = audit[~audit["passed"]]
    text = [
        "# v18.2 Passive Residual Retracement Independent Audit",
        "",
        f"Verdict: `{audit['round_verdict'].iloc[0]}`.",
        "",
        f"Checks: {len(audit)}; passed: {int(audit['passed'].sum())}; failed: {len(failed)}.",
        "",
        failed.to_markdown(index=False) if not failed.empty else "No failed checks.",
        "",
        "Passive execution improves the cost gap but remains negative in every",
        "chronological split. No live or application scope changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v182_audit(
    report_root: Path = AUDIT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    audit = audit_v182()
    root = ensure_dir(report_root)
    outputs = {"audit": root / "audit_checks.csv", "findings": findings_path}
    audit.to_csv(outputs["audit"], index=False)
    _write_findings(audit, findings_path)
    return outputs


__all__ = ["audit_v182", "write_v182_audit"]
