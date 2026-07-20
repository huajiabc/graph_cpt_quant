"""Independent audit for the v18.1 holding/execution extension."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v180_extreme_residual_dispersion_compression import (
    REPORT_ROOT as V180_REPORT_ROOT,
)
from pressure_graph.reports.v181_residual_dispersion_long_horizon_execution import (
    REPORT_ROOT,
    V181Config,
)


AUDIT_ROOT = Path("reports/v18_1_residual_dispersion_long_horizon_execution_audit")
FINDINGS_PATH = Path(
    "docs/v181_residual_dispersion_long_horizon_execution_audit_2026_07_16.md"
)


def _check(rows: list[dict[str, object]], name: str, passed: bool, value: object) -> None:
    rows.append({"check": name, "passed": bool(passed), "value": value})


def audit_v181(
    report_root: Path = REPORT_ROOT,
    v180_report_root: Path = V180_REPORT_ROOT,
    cfg: V181Config = V181Config(),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    source_signals = pd.read_parquet(v180_report_root / "dispersion_signals.parquet")
    fixed_signals = pd.read_parquet(report_root / "fixed_v180_signals.parquet")
    events = pd.read_parquet(report_root / "long_horizon_events.parquet")
    event_summary = pd.read_csv(report_root / "event_horizon_summary.csv")
    book = pd.read_parquet(report_root / "continuous_four_hour_book.parquet")
    sleeves = pd.read_parquet(report_root / "continuous_four_hour_sleeves.parquet")
    continuous_summary = pd.read_csv(report_root / "continuous_summary.csv")
    gates = pd.read_csv(report_root / "candidate_gates.csv")
    outcome = pd.read_csv(report_root / "candidate_outcome.csv")
    for frame in (source_signals, fixed_signals):
        frame["source_feature_time"] = pd.to_datetime(
            frame["source_feature_time"], utc=True
        )
    for column in ("source_feature_time", "entry_time", "exit_time"):
        events[column] = pd.to_datetime(events[column], utc=True)
    for column in ("source_feature_time", "exit_time"):
        sleeves[column] = pd.to_datetime(sleeves[column], utc=True)
    book["bar_time"] = pd.to_datetime(book["bar_time"], utc=True)

    source_keys = set(source_signals["source_feature_time"])
    fixed_keys = set(fixed_signals["source_feature_time"])
    _check(
        rows,
        "fixed_signals_exact_subset",
        fixed_keys.issubset(source_keys),
        len(fixed_keys - source_keys),
    )
    expected_horizons = {8, 16, 32, 48}
    _check(
        rows,
        "frozen_horizons_exact",
        set(events["holding_bars"].astype(int)) == expected_horizons,
        "|".join(map(str, sorted(events["holding_bars"].unique()))),
    )
    holding_error = (
        (events["exit_time"] - events["entry_time"])
        - pd.to_timedelta(events["holding_bars"] * 15, unit="min")
    ).abs().max()
    _check(
        rows,
        "event_holding_durations_exact",
        holding_error == pd.Timedelta(0),
        str(holding_error),
    )
    expected_gross = (
        0.5
        * (
            events["mean_laggard_future_return"]
            - events["mean_leader_future_return"]
        )
        - events["spread_beta"] * events["btc_future_return"]
    ) / (1 + events["spread_beta"].abs())
    gross_error = (events["gross_return"] - expected_gross).abs().max()
    _check(rows, "event_gross_formula_exact", gross_error <= 1e-12, gross_error)
    primary_error = (
        events["primary_net_return"] - (events["gross_return"] - cfg.primary_cost)
    ).abs().max()
    stress_error = (
        events["stress_net_return"] - (events["gross_return"] - cfg.stress_cost)
    ).abs().max()
    _check(
        rows,
        "event_cost_formulas_exact",
        primary_error <= 1e-12 and stress_error <= 1e-12,
        max(primary_error, stress_error),
    )
    for horizon in sorted(expected_horizons):
        sample = events[events["holding_bars"].eq(horizon)]
        reported = event_summary[
            event_summary["holding_bars"].eq(horizon)
            & event_summary["scope"].eq("all")
        ]
        actual = float(sample["primary_net_return"].mean() * 10_000)
        value = float(reported["mean_primary_net_bp"].iloc[0])
        _check(
            rows,
            f"horizon_{horizon}_summary_exact",
            math.isclose(actual, value, rel_tol=0, abs_tol=1e-10),
            actual - value,
        )

    _check(
        rows,
        "continuous_sleeves_match_primary_events",
        len(sleeves) == int(events["holding_bars"].eq(cfg.primary_holding_bars).sum()),
        len(sleeves),
    )
    _check(
        rows,
        "sleeves_unit_gross_exposure",
        sleeves["gross_exposure"].sub(1.0).abs().max() <= 1e-12,
        float(sleeves["gross_exposure"].sub(1.0).abs().max()),
    )
    _check(
        rows,
        "continuous_cost_formula_exact",
        (
            book["primary_cost_return"]
            - book["turnover"] * cfg.continuous_primary_one_way_cost
        ).abs().max()
        <= 1e-12
        and (
            book["stress_cost_return"]
            - book["turnover"] * cfg.continuous_stress_one_way_cost
        ).abs().max()
        <= 1e-12,
        float(book["turnover"].sum()),
    )
    _check(
        rows,
        "continuous_primary_identity",
        (
            book["primary_net_return"]
            - (book["gross_return"] - book["primary_cost_return"])
        ).abs().max()
        <= 1e-12,
        float(book["primary_net_return"].sum()),
    )
    _check(
        rows,
        "continuous_stress_identity",
        (
            book["stress_net_return"]
            - (book["gross_return"] - book["stress_cost_return"])
        ).abs().max()
        <= 1e-12,
        float(book["stress_net_return"].sum()),
    )
    for scope in ("all", "development", "validation", "holdout"):
        sample = book if scope == "all" else book[book["period"].eq(scope)]
        reported = continuous_summary[continuous_summary["scope"].eq(scope)]
        actual = float(sample["primary_net_return"].sum() * 100)
        value = float(reported["primary_net_sum_pct"].iloc[0])
        _check(
            rows,
            f"continuous_{scope}_summary_exact",
            math.isclose(actual, value, rel_tol=0, abs_tol=1e-10),
            actual - value,
        )
    _check(rows, "all_gate_rows_ineligible", not gates["eligible"].any(), len(gates))
    _check(rows, "candidate_rejected", not outcome["eligible"].any(), len(outcome))
    _check(
        rows,
        "rejection_verdict_exact",
        outcome["verdict"].eq("reject_long_horizon_execution_extension").all(),
        str(outcome["verdict"].iloc[0]),
    )
    config_hits = [
        path.name
        for path in Path("configs").glob("*")
        if "v181" in path.name.lower() or "long_horizon_execution" in path.name.lower()
    ]
    _check(rows, "no_live_config_created", not config_hits, "|".join(config_hits))
    audit = pd.DataFrame(rows)
    audit["round_verdict"] = (
        "audit_pass_extension_rejected"
        if audit["passed"].all()
        else "audit_failure_requires_investigation"
    )
    return audit


def _write_findings(audit: pd.DataFrame, path: Path) -> None:
    failed = audit[~audit["passed"]]
    text = [
        "# v18.1 Long-Horizon Execution Independent Audit",
        "",
        f"Verdict: `{audit['round_verdict'].iloc[0]}`.",
        "",
        f"Checks: {len(audit)}; passed: {int(audit['passed'].sum())}; failed: {len(failed)}.",
        "",
        failed.to_markdown(index=False) if not failed.empty else "No failed checks.",
        "",
        "The audited v18.0 structure does not survive longer holding or actual",
        "continuous-book turnover. No live or application scope changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v181_audit(
    report_root: Path = AUDIT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    audit = audit_v181()
    root = ensure_dir(report_root)
    outputs = {"audit": root / "audit_checks.csv", "findings": findings_path}
    audit.to_csv(outputs["audit"], index=False)
    _write_findings(audit, findings_path)
    return outputs


__all__ = ["audit_v181", "write_v181_audit"]
