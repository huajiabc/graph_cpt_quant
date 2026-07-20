"""Independent audit for the v18.5 leverage-flow directed graph."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v184_btc_inclusive_metrics_audit import (
    load_v184_exact_panels,
)
from pressure_graph.reports.v185_btc_leverage_flow_graph import (
    BTC,
    CANDIDATES,
    REPORT_ROOT,
    V185Config,
)


AUDIT_ROOT = Path("reports/v18_5_btc_leverage_flow_graph_audit")
FINDINGS_PATH = Path("docs/v185_btc_leverage_flow_graph_audit_2026_07_16.md")


def _check(rows: list[dict[str, object]], name: str, passed: bool, value: object) -> None:
    rows.append({"check": name, "passed": bool(passed), "value": value})


def audit_v185(
    report_root: Path = REPORT_ROOT,
    cfg: V185Config = V185Config(),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    signals = pd.read_parquet(report_root / "source_signals.parquet")
    graph = pd.read_parquet(report_root / "monthly_leverage_flow_graph.parquet")
    events = pd.read_parquet(report_root / "candidate_events.parquet")
    summary = pd.read_csv(report_root / "period_summary.csv")
    delayed = pd.read_parquet(report_root / "delayed_candidate_events.parquet")
    random = pd.read_parquet(report_root / "random_receiver_controls.parquet")
    gates = pd.read_csv(report_root / "candidate_gates.csv")
    outcome = pd.read_csv(report_root / "candidate_outcome.csv")
    close, panels = load_v184_exact_panels()
    for column in ("feature_time", "source_feature_time"):
        signals[column] = pd.to_datetime(signals[column], utc=True)
    graph["graph_month"] = pd.to_datetime(graph["graph_month"], utc=True)
    for frame in (events, delayed):
        for column in (
            "feature_time",
            "source_feature_time",
            "entry_time",
            "exit_time",
            "graph_month",
        ):
            frame[column] = pd.to_datetime(frame[column], utc=True)

    _check(
        rows,
        "source_keys_unique_by_kind",
        not signals.duplicated(["feature_time", "kind"]).any(),
        len(signals),
    )
    _check(
        rows,
        "source_exact_completed_time",
        signals["feature_time"].eq(signals["source_feature_time"]).all(),
        int(signals["feature_time"].ne(signals["source_feature_time"]).sum()),
    )
    _check(
        rows,
        "source_breadth_minimum_40",
        signals["metric_breadth"].ge(cfg.min_metric_breadth).all(),
        int(signals["metric_breadth"].min()),
    )
    btc_return = close[BTC].pct_change(fill_method=None)
    btc_flow = np.log(
        panels["sum_taker_long_short_vol_ratio"][BTC].where(
            panels["sum_taker_long_short_vol_ratio"][BTC].gt(0)
        )
    )
    btc_oi = np.log(
        panels["sum_open_interest"][BTC].where(
            panels["sum_open_interest"][BTC].gt(0)
        )
    ).diff()
    return_threshold = (
        btc_return.abs()
        .shift(1)
        .rolling(cfg.source_lookback_bars, min_periods=cfg.source_min_bars)
        .quantile(cfg.source_return_quantile)
    )
    flow_threshold = (
        btc_flow.abs()
        .shift(1)
        .rolling(cfg.source_lookback_bars, min_periods=cfg.source_min_bars)
        .quantile(cfg.source_flow_quantile)
    )
    oi_high = (
        btc_oi.shift(1)
        .rolling(cfg.source_lookback_bars, min_periods=cfg.source_min_bars)
        .quantile(cfg.source_oi_tail_quantile)
    )
    oi_low = (
        btc_oi.shift(1)
        .rolling(cfg.source_lookback_bars, min_periods=cfg.source_min_bars)
        .quantile(1 - cfg.source_oi_tail_quantile)
    )
    indexed = signals.set_index("feature_time")
    return_error = (
        indexed["return_threshold"] - return_threshold.reindex(indexed.index)
    ).abs().max()
    flow_error = (
        indexed["flow_threshold"] - flow_threshold.reindex(indexed.index)
    ).abs().max()
    expected_oi = pd.Series(index=indexed.index, dtype=float)
    build_mask = indexed["kind"].eq("build")
    expected_oi.loc[build_mask] = oi_high.reindex(indexed.index[build_mask])
    expected_oi.loc[~build_mask] = oi_low.reindex(indexed.index[~build_mask])
    oi_error = (indexed["oi_threshold"] - expected_oi).abs().max()
    _check(
        rows,
        "source_thresholds_exact_prior_window",
        max(return_error, flow_error, oi_error) <= 1e-12,
        float(max(return_error, flow_error, oi_error)),
    )
    source_eligible = (
        indexed["btc_return_15m"].abs().ge(indexed["return_threshold"])
        & (
            np.sign(indexed["btc_return_15m"]) * indexed["btc_flow"]
        ).ge(indexed["flow_threshold"])
        & indexed["metric_breadth"].ge(cfg.min_metric_breadth)
    )
    source_eligible &= np.where(
        build_mask,
        indexed["btc_oi_change"].ge(indexed["oi_threshold"]),
        indexed["btc_oi_change"].le(indexed["oi_threshold"]),
    )
    _check(
        rows,
        "all_sources_meet_frozen_rules",
        bool(source_eligible.all()),
        int((~source_eligible).sum()),
    )

    _check(
        rows,
        "graph_month_normalized",
        graph["graph_month"].dt.day.eq(1).all()
        and graph["graph_month"].dt.hour.eq(0).all()
        and graph["graph_month"].dt.minute.eq(0).all(),
        int(graph["graph_month"].dt.minute.ne(0).sum()),
    )
    _check(
        rows,
        "graph_sample_minimum",
        graph["forward_samples"].ge(cfg.graph_min_samples).all()
        and graph["reverse_samples"].ge(cfg.graph_min_samples).all(),
        int(min(graph["forward_samples"].min(), graph["reverse_samples"].min())),
    )
    selected = graph[graph["selected"].astype(bool)]
    selected_counts = selected.groupby(["graph_month", "kind"]).size()
    _check(
        rows,
        "selected_bucket_cap_eight",
        selected_counts.le(cfg.receiver_bucket_size).all(),
        int(selected_counts.max()),
    )
    _check(
        rows,
        "selected_direction_advantage_positive",
        selected["direction_advantage"].gt(0).all(),
        float(selected["direction_advantage"].min()),
    )
    _check(
        rows,
        "selected_edge_sign_nonzero",
        selected["edge_sign"].ne(0).all(),
        int(selected["edge_sign"].eq(0).sum()),
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
        "event_receiver_minimum",
        events["receiver_count"].ge(cfg.min_receiver_bucket).all(),
        int(events["receiver_count"].min()),
    )
    gross_errors: list[float] = []
    for event in events.itertuples(index=False):
        local = graph[
            graph["graph_month"].eq(event.graph_month)
            & graph["kind"].eq(event.kind)
        ].set_index("receiver")
        receivers = str(event.receivers).split("|")
        future = close.loc[event.exit_time, [BTC, *receivers]].div(
            close.loc[event.entry_time, [BTC, *receivers]]
        ).sub(1.0)
        weights = (
            float(event.source_sign)
            * local.reindex(receivers)["edge_sign"]
            / len(receivers)
        )
        hedge = -float(
            (weights * local.reindex(receivers)["btc_beta"]).sum()
        )
        normalizer = float(weights.abs().sum() + abs(hedge))
        expected = float(
            ((weights * future[receivers]).sum() + hedge * future[BTC])
            / normalizer
        )
        gross_errors.append(abs(expected - float(event.gross_return)))
    _check(
        rows,
        "event_gross_formula_exact",
        max(gross_errors) <= 1e-12,
        max(gross_errors),
    )
    net_error = max(
        (
            events["primary_net_return"]
            - (events["gross_return"] - cfg.primary_cost)
        ).abs().max(),
        (
            events["stress_net_return"]
            - (events["gross_return"] - cfg.stress_cost)
        ).abs().max(),
    )
    _check(rows, "event_cost_formulas_exact", net_error <= 1e-12, net_error)
    for candidate, sample in events.groupby("candidate", sort=True):
        reported = summary[
            summary["candidate"].eq(candidate) & summary["scope"].eq("all")
        ]
        actual = float(sample["primary_net_return"].mean() * 10_000)
        value = float(reported["mean_primary_net_bp"].iloc[0])
        _check(
            rows,
            f"{candidate}_summary_exact",
            math.isclose(actual, value, rel_tol=0, abs_tol=1e-10),
            actual - value,
        )
        real_keys = set(sample["source_feature_time"])
        delayed_keys = set(
            delayed.loc[delayed["candidate"].eq(candidate), "source_feature_time"]
        )
        _check(
            rows,
            f"{candidate}_delay_same_sources",
            real_keys == delayed_keys,
            len(real_keys.symmetric_difference(delayed_keys)),
        )
    _check(
        rows,
        "random_iterations_complete",
        random["iteration"].nunique() == cfg.random_iterations,
        int(random["iteration"].nunique()),
    )
    random_wide = random.pivot(
        index="iteration", columns="candidate", values="mean_primary_net_return"
    )
    expected_family = random_wide[list(CANDIDATES)].max(axis=1)
    family_error = (random_wide["FAMILY_MAX"] - expected_family).abs().max()
    _check(rows, "random_family_max_exact", family_error <= 1e-12, family_error)
    _check(rows, "all_gate_rows_ineligible", not gates["eligible"].any(), len(gates))
    _check(rows, "no_candidate_eligible", not outcome["eligible"].any(), len(outcome))
    _check(
        rows,
        "rejection_verdict_exact",
        outcome["verdict"].eq("reject_btc_leverage_flow_graph").all(),
        "|".join(outcome["verdict"].astype(str)),
    )
    config_hits = [
        path.name
        for path in Path("configs").glob("*")
        if "v185" in path.name.lower() or "leverage_flow_graph" in path.name.lower()
    ]
    _check(rows, "no_live_config_created", not config_hits, "|".join(config_hits))
    audit = pd.DataFrame(rows)
    audit["round_verdict"] = (
        "audit_pass_graph_alpha_rejected"
        if audit["passed"].all()
        else "audit_failure_requires_investigation"
    )
    return audit


def _write_findings(audit: pd.DataFrame, path: Path) -> None:
    failed = audit[~audit["passed"]]
    text = [
        "# v18.5 BTC Leverage-Flow Graph Independent Audit",
        "",
        f"Verdict: `{audit['round_verdict'].iloc[0]}`.",
        "",
        f"Checks: {len(audit)}; passed: {int(audit['passed'].sum())}; failed: {len(failed)}.",
        "",
        failed.to_markdown(index=False) if not failed.empty else "No failed checks.",
        "",
        "The BTC-inclusive metrics archive is accepted, but both directed graph",
        "candidates remain rejected. No live or application scope changed.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v185_audit(
    report_root: Path = AUDIT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    audit = audit_v185()
    root = ensure_dir(report_root)
    outputs = {"audit": root / "audit_checks.csv", "findings": findings_path}
    audit.to_csv(outputs["audit"], index=False)
    _write_findings(audit, findings_path)
    return outputs


__all__ = ["audit_v185", "write_v185_audit"]
