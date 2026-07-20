"""Independent audit for the v18.0 residual-dispersion compression round."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v178_btc_confirmed_flow_laggard import (
    BTC,
    load_v178_market_data,
)
from pressure_graph.reports.v180_extreme_residual_dispersion_compression import (
    CANDIDATE,
    REPORT_ROOT,
    V180Config,
)


AUDIT_ROOT = Path("reports/v18_0_extreme_residual_dispersion_compression_audit")
FINDINGS_PATH = Path(
    "docs/v180_extreme_residual_dispersion_compression_audit_2026_07_16.md"
)


def _check(rows: list[dict[str, object]], name: str, passed: bool, value: object) -> None:
    rows.append({"check": name, "passed": bool(passed), "value": value})


def audit_v180(
    report_root: Path = REPORT_ROOT,
    cfg: V180Config = V180Config(),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    graph = pd.read_parquet(report_root / "monthly_btc_beta_graph.parquet")
    signals = pd.read_parquet(report_root / "dispersion_signals.parquet")
    events = pd.read_parquet(report_root / "candidate_events.parquet")
    delayed = pd.read_parquet(report_root / "delayed_candidate_events.parquet")
    summary = pd.read_csv(report_root / "period_summary.csv")
    random = pd.read_parquet(report_root / "random_rank_controls.parquet")
    gates = pd.read_csv(report_root / "candidate_gates.csv")
    outcome = pd.read_csv(report_root / "candidate_outcome.csv")
    close, _ = load_v178_market_data()
    returns = close.pct_change(fill_method=None)

    graph["graph_month"] = pd.to_datetime(graph["graph_month"], utc=True)
    for column in ("feature_time", "source_feature_time", "graph_month"):
        signals[column] = pd.to_datetime(signals[column], utc=True)
    for frame in (events, delayed):
        for column in (
            "feature_time",
            "source_feature_time",
            "graph_month",
            "entry_time",
            "exit_time",
        ):
            frame[column] = pd.to_datetime(frame[column], utc=True)

    _check(rows, "signals_unique", signals["feature_time"].is_unique, len(signals))
    _check(
        rows,
        "signals_completed_bar_time",
        signals["feature_time"].eq(signals["source_feature_time"]).all(),
        int(signals["feature_time"].ne(signals["source_feature_time"]).sum()),
    )
    _check(
        rows,
        "signals_cross_section_minimum",
        signals["cross_section"].ge(cfg.min_cross_section).all(),
        int(signals["cross_section"].min()),
    )
    _check(
        rows,
        "signals_exceed_prior_threshold",
        signals["dispersion"].ge(signals["dispersion_threshold"]).all(),
        float((signals["dispersion"] - signals["dispersion_threshold"]).min()),
    )

    beta_map = {
        pd.Timestamp(month): local.set_index("receiver")["btc_beta"].astype(float)
        for month, local in graph.groupby("graph_month", sort=True)
    }
    dispersion_parts: list[pd.Series] = []
    for month, beta in beta_map.items():
        end = month + pd.offsets.MonthBegin(1)
        local_returns = returns[(returns.index >= month) & (returns.index < end)]
        names = [name for name in beta.index.astype(str) if name in local_returns]
        residual = local_returns[names].sub(
            local_returns[BTC].to_numpy()[:, None] * beta.reindex(names).to_numpy(),
            axis=0,
        )
        dispersion_parts.append(
            residual.quantile(cfg.dispersion_upper_quantile, axis=1)
            - residual.quantile(cfg.dispersion_lower_quantile, axis=1)
        )
    dispersion = pd.concat(dispersion_parts).sort_index()
    expected_threshold = (
        dispersion.shift(1)
        .rolling(
            cfg.dispersion_lookback_bars,
            min_periods=cfg.dispersion_min_bars,
        )
        .quantile(cfg.dispersion_quantile)
        .reindex(signals["feature_time"])
    )
    actual_threshold = signals.set_index("feature_time")["dispersion_threshold"]
    threshold_error = (actual_threshold - expected_threshold).abs().max()
    _check(
        rows,
        "threshold_exact_prior_window",
        threshold_error <= 1e-12,
        float(threshold_error),
    )

    bucket_errors = 0
    disjoint_errors = 0
    for signal in signals.itertuples(index=False):
        beta = beta_map[pd.Timestamp(signal.graph_month)]
        names = [name for name in beta.index.astype(str) if name in returns]
        current = returns.reindex(
            index=[signal.feature_time], columns=[BTC, *names]
        ).iloc[0]
        residual = (current[names] - beta.reindex(names) * float(current[BTC])).dropna()
        expected_laggards = residual.nsmallest(cfg.bucket_size).index.astype(str).tolist()
        expected_leaders = residual.nlargest(cfg.bucket_size).index.astype(str).tolist()
        actual_laggards = str(signal.laggards).split("|")
        actual_leaders = str(signal.leaders).split("|")
        bucket_errors += int(
            expected_laggards != actual_laggards or expected_leaders != actual_leaders
        )
        disjoint_errors += int(bool(set(actual_laggards) & set(actual_leaders)))
    _check(rows, "extreme_bucket_ranks_exact", bucket_errors == 0, bucket_errors)
    _check(rows, "event_buckets_disjoint", disjoint_errors == 0, disjoint_errors)

    _check(
        rows,
        "primary_holding_exact_15m",
        (events["exit_time"] - events["entry_time"])
        .eq(pd.Timedelta(minutes=15))
        .all(),
        len(events),
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
    _check(rows, "gross_formula_exact", gross_error <= 1e-12, float(gross_error))
    primary_error = (
        events["primary_net_return"] - (events["gross_return"] - cfg.primary_cost)
    ).abs().max()
    stress_error = (
        events["stress_net_return"] - (events["gross_return"] - cfg.stress_cost)
    ).abs().max()
    _check(
        rows,
        "cost_formulas_exact",
        primary_error <= 1e-12 and stress_error <= 1e-12,
        float(max(primary_error, stress_error)),
    )
    reported = summary[summary["scope"].eq("all")]
    actual_mean = float(events["primary_net_return"].mean() * 10_000)
    reported_mean = float(reported["mean_primary_net_bp"].iloc[0])
    _check(
        rows,
        "summary_exact",
        math.isclose(actual_mean, reported_mean, rel_tol=0, abs_tol=1e-10),
        actual_mean - reported_mean,
    )

    main_keys = set(events["source_feature_time"])
    delayed_keys = set(delayed["source_feature_time"])
    _check(
        rows,
        "delay_same_source_events",
        main_keys == delayed_keys,
        len(main_keys.symmetric_difference(delayed_keys)),
    )
    _check(
        rows,
        "delay_exact_one_bar",
        (delayed["entry_time"] - delayed["source_feature_time"])
        .eq(pd.Timedelta(minutes=15))
        .all(),
        len(delayed),
    )
    main_membership = events.set_index("source_feature_time")[["laggards", "leaders"]]
    delayed_membership = delayed.set_index("source_feature_time")[["laggards", "leaders"]]
    _check(
        rows,
        "delay_membership_frozen",
        main_membership.equals(delayed_membership),
        len(main_membership.compare(delayed_membership)),
    )
    _check(
        rows,
        "random_iterations_complete",
        random["iteration"].nunique() == cfg.random_iterations,
        int(random["iteration"].nunique()),
    )
    _check(
        rows,
        "random_event_counts_match",
        random["events"].eq(len(events)).all(),
        f"{random['events'].min()}:{random['events'].max()}",
    )
    real_mean = float(events["primary_net_return"].mean())
    actual_percentile = float(random["mean_primary_net_return"].le(real_mean).mean())
    reported_percentile = float(outcome["random_rank_percentile"].iloc[0])
    _check(
        rows,
        "random_percentile_exact",
        math.isclose(
            actual_percentile, reported_percentile, rel_tol=0, abs_tol=1e-12
        ),
        actual_percentile - reported_percentile,
    )
    _check(rows, "all_gate_rows_ineligible", not gates["eligible"].any(), len(gates))
    _check(rows, "candidate_rejected", not outcome["eligible"].any(), len(outcome))
    _check(
        rows,
        "rejection_verdict_exact",
        outcome["verdict"]
        .eq("reject_extreme_residual_dispersion_compression")
        .all(),
        str(outcome["verdict"].iloc[0]),
    )
    config_hits = [
        path.name
        for path in Path("configs").glob("*")
        if "v180" in path.name.lower() or "residual_dispersion" in path.name.lower()
    ]
    _check(rows, "no_live_config_created", not config_hits, "|".join(config_hits))
    _check(rows, "candidate_name_exact", events["candidate"].eq(CANDIDATE).all(), len(events))
    audit = pd.DataFrame(rows)
    audit["round_verdict"] = (
        "audit_pass_structure_positive_alpha_rejected"
        if audit["passed"].all()
        else "audit_failure_requires_investigation"
    )
    return audit


def _write_findings(audit: pd.DataFrame, path: Path) -> None:
    failed = audit[~audit["passed"]]
    text = [
        "# v18.0 Extreme Residual Dispersion Independent Audit",
        "",
        f"Verdict: `{audit['round_verdict'].iloc[0]}`.",
        "",
        f"Checks: {len(audit)}; passed: {int(audit['passed'].sum())}; failed: {len(failed)}.",
        "",
        failed.to_markdown(index=False) if not failed.empty else "No failed checks.",
        "",
        "The rank-specific gross compression effect is accepted as a research",
        "primitive, but the standalone candidate remains rejected after costs.",
        "No live or application scope changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v180_audit(
    report_root: Path = AUDIT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    audit = audit_v180()
    root = ensure_dir(report_root)
    outputs = {"audit": root / "audit_checks.csv", "findings": findings_path}
    audit.to_csv(outputs["audit"], index=False)
    _write_findings(audit, findings_path)
    return outputs


__all__ = ["audit_v180", "write_v180_audit"]
