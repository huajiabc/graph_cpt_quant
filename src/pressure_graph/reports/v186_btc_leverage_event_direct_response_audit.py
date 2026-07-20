"""Independent audit for the v18.6 direct BTC event response."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v184_btc_inclusive_metrics_audit import (
    load_v184_exact_panels,
)
from pressure_graph.reports.v185_btc_leverage_flow_graph import BTC
from pressure_graph.reports.v186_btc_leverage_event_direct_response import (
    BUILD_CANDIDATE,
    CANDIDATES,
    REPORT_ROOT,
    UNWIND_CANDIDATE,
    V186Config,
)


AUDIT_ROOT = Path("reports/v18_6_btc_leverage_event_direct_response_audit")
FINDINGS_PATH = Path(
    "docs/v186_btc_leverage_event_direct_response_audit_2026_07_16.md"
)
V185_REPORT_ROOT = Path("reports/v18_5_btc_leverage_flow_graph")


def _check(rows: list[dict[str, object]], name: str, passed: bool, value: object) -> None:
    rows.append({"check": name, "passed": bool(passed), "value": value})


def audit_v186(
    report_root: Path = REPORT_ROOT,
    cfg: V186Config = V186Config(),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    signals = pd.read_parquet(report_root / "source_signals.parquet")
    prior_signals = pd.read_parquet(V185_REPORT_ROOT / "source_signals.parquet")
    events = pd.read_parquet(report_root / "candidate_events.parquet")
    delayed = pd.read_parquet(report_root / "delayed_candidate_events.parquet")
    summary = pd.read_csv(report_root / "period_summary.csv")
    random = pd.read_parquet(report_root / "random_circular_controls.parquet")
    gates = pd.read_csv(report_root / "candidate_gates.csv")
    outcome = pd.read_csv(report_root / "candidate_outcome.csv")
    close, _ = load_v184_exact_panels()

    time_columns = ("feature_time", "source_feature_time")
    for frame in (signals, prior_signals):
        for column in time_columns:
            frame[column] = pd.to_datetime(frame[column], utc=True)
    for frame in (events, delayed):
        for column in (*time_columns, "entry_time", "exit_time"):
            frame[column] = pd.to_datetime(frame[column], utc=True)

    shared_columns = sorted(set(signals.columns) & set(prior_signals.columns))
    left = signals[shared_columns].sort_values(["feature_time", "kind"])
    right = prior_signals[shared_columns].sort_values(["feature_time", "kind"])
    same_sources = left.reset_index(drop=True).equals(right.reset_index(drop=True))
    _check(
        rows,
        "source_signals_exact_v185_freeze",
        same_sources,
        len(signals),
    )
    _check(
        rows,
        "source_keys_unique_by_kind",
        not signals.duplicated(["source_feature_time", "kind"]).any(),
        len(signals),
    )
    _check(
        rows,
        "all_frozen_sources_have_direct_events",
        set(zip(signals["source_feature_time"], signals["kind"], strict=False))
        == set(zip(events["source_feature_time"], events["kind"], strict=False)),
        len(events),
    )
    _check(
        rows,
        "event_holding_exact_30m",
        (events["exit_time"] - events["entry_time"])
        .eq(pd.Timedelta(minutes=30))
        .all(),
        len(events),
    )
    _check(
        rows,
        "entry_is_completed_source_close",
        events["entry_time"].eq(events["source_feature_time"]).all(),
        int(events["entry_time"].ne(events["source_feature_time"]).sum()),
    )

    expected_candidate = events["kind"].map(
        {"build": BUILD_CANDIDATE, "unwind": UNWIND_CANDIDATE}
    )
    expected_direction = events["source_sign"] * events["kind"].map(
        {"build": 1.0, "unwind": -1.0}
    )
    _check(
        rows,
        "candidate_kind_mapping_exact",
        events["candidate"].eq(expected_candidate).all(),
        int(events["candidate"].ne(expected_candidate).sum()),
    )
    _check(
        rows,
        "trade_direction_mapping_exact",
        events["trade_direction"].eq(expected_direction).all(),
        int(events["trade_direction"].ne(expected_direction).sum()),
    )

    entry = pd.Series(
        [close.at[timestamp, BTC] for timestamp in events["entry_time"]],
        index=events.index,
        dtype=float,
    )
    exit_price = pd.Series(
        [close.at[timestamp, BTC] for timestamp in events["exit_time"]],
        index=events.index,
        dtype=float,
    )
    expected_underlying = exit_price / entry - 1.0
    expected_gross = expected_direction * expected_underlying
    return_error = max(
        (events["btc_underlying_return"] - expected_underlying).abs().max(),
        (events["gross_return"] - expected_gross).abs().max(),
    )
    _check(
        rows,
        "direct_return_formula_exact",
        return_error <= 1e-12,
        return_error,
    )
    cost_error = max(
        (
            events["primary_net_return"]
            - (events["gross_return"] - cfg.primary_cost)
        ).abs().max(),
        (
            events["stress_net_return"]
            - (events["gross_return"] - cfg.stress_cost)
        ).abs().max(),
        (
            events["reversed_primary_net_return"]
            - (-events["gross_return"] - cfg.primary_cost)
        ).abs().max(),
    )
    _check(rows, "cost_and_reverse_formulas_exact", cost_error <= 1e-12, cost_error)

    for candidate, sample in events.groupby("candidate", sort=True):
        reported = summary[
            summary["candidate"].eq(candidate) & summary["scope"].eq("all")
        ].iloc[0]
        actual_gross = float(sample["gross_return"].mean() * 10_000)
        actual_net = float(sample["primary_net_return"].mean() * 10_000)
        error = max(
            abs(actual_gross - float(reported["mean_gross_bp"])),
            abs(actual_net - float(reported["mean_primary_net_bp"])),
        )
        _check(rows, f"{candidate}_summary_exact", error <= 1e-10, error)
        source_keys = set(sample["source_feature_time"])
        delayed_keys = set(
            delayed.loc[
                delayed["candidate"].eq(candidate), "source_feature_time"
            ]
        )
        _check(
            rows,
            f"{candidate}_delay_same_sources",
            source_keys == delayed_keys,
            len(source_keys.symmetric_difference(delayed_keys)),
        )

    _check(
        rows,
        "random_iterations_complete",
        random["iteration"].nunique() == cfg.random_iterations,
        int(random["iteration"].nunique()),
    )
    actual_counts = events.groupby("candidate").size().to_dict()
    random_candidates = random[random["candidate"].isin(CANDIDATES)]
    count_ok = all(
        sample["events"].eq(actual_counts[candidate]).all()
        for candidate, sample in random_candidates.groupby("candidate")
    )
    _check(
        rows,
        "circular_controls_preserve_event_counts",
        count_ok,
        int(random_candidates["events"].min()),
    )
    random_wide = random.pivot(
        index="iteration", columns="candidate", values="mean_primary_net_return"
    )
    family_error = (
        random_wide["FAMILY_MAX"] - random_wide[list(CANDIDATES)].max(axis=1)
    ).abs().max()
    _check(rows, "random_family_max_exact", family_error <= 1e-12, family_error)
    family = random_wide["FAMILY_MAX"]
    percentile_errors = []
    for outcome_row in outcome.itertuples(index=False):
        real_mean = float(
            events.loc[
                events["candidate"].eq(outcome_row.candidate),
                "primary_net_return",
            ].mean()
        )
        actual_percentile = float(family.le(real_mean).mean())
        percentile_errors.append(
            abs(actual_percentile - float(outcome_row.random_family_percentile))
        )
    _check(
        rows,
        "outcome_random_percentiles_exact",
        max(percentile_errors) <= 1e-12,
        max(percentile_errors),
    )

    _check(rows, "all_gate_rows_ineligible", not gates["eligible"].any(), len(gates))
    _check(rows, "no_candidate_eligible", not outcome["eligible"].any(), len(outcome))
    _check(
        rows,
        "rejection_verdict_exact",
        outcome["verdict"].eq("reject_btc_leverage_event_direct_response").all(),
        "|".join(outcome["verdict"].astype(str)),
    )
    _check(
        rows,
        "unwind_gross_positive_but_below_primary_cost",
        float(
            outcome.loc[
                outcome["candidate"].eq(UNWIND_CANDIDATE), "mean_gross_bp"
            ].iloc[0]
        )
        > 0
        and float(
            outcome.loc[
                outcome["candidate"].eq(UNWIND_CANDIDATE), "mean_primary_net_bp"
            ].iloc[0]
        )
        < 0,
        float(
            outcome.loc[
                outcome["candidate"].eq(UNWIND_CANDIDATE), "mean_gross_bp"
            ].iloc[0]
        ),
    )
    config_hits = [
        path.name
        for path in Path("configs").glob("*")
        if "v186" in path.name.lower()
        or "leverage_event_direct" in path.name.lower()
    ]
    _check(rows, "no_live_config_created", not config_hits, "|".join(config_hits))
    audit = pd.DataFrame(rows)
    audit["round_verdict"] = np.where(
        audit["passed"].all(),
        "audit_pass_direct_alpha_rejected_cost_boundary_retained",
        "audit_failure_requires_investigation",
    )
    return audit


def _write_findings(audit: pd.DataFrame, path: Path) -> None:
    failed = audit[~audit["passed"]]
    text = [
        "# v18.6 BTC Direct Leverage-Event Response Independent Audit",
        "",
        f"Verdict: `{audit['round_verdict'].iloc[0]}`.",
        "",
        f"Checks: {len(audit)}; passed: {int(audit['passed'].sum())}; failed: {len(failed)}.",
        "",
        failed.to_markdown(index=False) if not failed.empty else "No failed checks.",
        "",
        "The unwind-reversal timing effect is retained as a sub-cost research primitive,",
        "not as a tradable candidate. No live or application scope changed.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v186_audit(
    report_root: Path = AUDIT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    audit = audit_v186()
    root = ensure_dir(report_root)
    outputs = {"audit": root / "audit_checks.csv", "findings": findings_path}
    audit.to_csv(outputs["audit"], index=False)
    _write_findings(audit, findings_path)
    return outputs


__all__ = ["audit_v186", "write_v186_audit"]
