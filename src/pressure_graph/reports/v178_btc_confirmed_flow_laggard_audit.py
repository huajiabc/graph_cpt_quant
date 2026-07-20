"""Independent consistency audit for the v17.8 BTC receiver round."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v178_btc_confirmed_flow_laggard import (
    BTC,
    CANDIDATES,
    KLINE_ROOT,
    NEUTRAL_CANDIDATE,
    RAW_CANDIDATE,
    REPORT_ROOT,
    V178Config,
)


AUDIT_ROOT = Path("reports/v17_8_btc_confirmed_flow_laggard_audit")
FINDINGS_PATH = Path("docs/v178_btc_confirmed_flow_laggard_audit_2026_07_16.md")


def _check(rows: list[dict[str, object]], name: str, passed: bool, value: object) -> None:
    rows.append({"check": name, "passed": bool(passed), "value": value})


def audit_v178(
    report_root: Path = REPORT_ROOT,
    kline_root: Path = KLINE_ROOT,
    cfg: V178Config = V178Config(),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    signals = pd.read_parquet(report_root / "btc_source_signals.parquet")
    graph = pd.read_parquet(report_root / "monthly_btc_receiver_graph.parquet")
    events = pd.read_parquet(report_root / "candidate_events.parquet")
    selected = pd.read_parquet(report_root / "selected_laggards.parquet")
    summary = pd.read_csv(report_root / "period_summary.csv")
    random = pd.read_parquet(report_root / "random_receiver_controls.parquet")
    gates = pd.read_csv(report_root / "candidate_gates.csv")
    outcome = pd.read_csv(report_root / "candidate_outcome.csv")

    for column in ("feature_time", "source_feature_time"):
        signals[column] = pd.to_datetime(signals[column], utc=True)
    graph["graph_month"] = pd.to_datetime(graph["graph_month"], utc=True)
    for column in ("entry_time", "exit_time"):
        events[column] = pd.to_datetime(events[column], utc=True)
    selected["entry_time"] = pd.to_datetime(selected["entry_time"], utc=True)

    _check(rows, "signals_unique", signals["feature_time"].is_unique, len(signals))
    _check(
        rows,
        "signals_completed_bar_time",
        signals["feature_time"].eq(signals["source_feature_time"]).all(),
        int(signals["feature_time"].ne(signals["source_feature_time"]).sum()),
    )
    eligible = (
        signals["btc_return_15m"].abs().ge(signals["return_threshold"])
        & signals["signed_flow"].ge(signals["flow_threshold"])
        & signals["turnover"].ge(signals["turnover_threshold"])
    )
    _check(rows, "signals_meet_frozen_thresholds", eligible.all(), int((~eligible).sum()))

    btc = pd.read_parquet(
        kline_root / f"{BTC}.parquet",
        columns=["bar_close_time", "close", "turnover", "taker_buy_quote"],
    )
    btc["bar_close_time"] = pd.to_datetime(btc["bar_close_time"], utc=True)
    btc = btc.drop_duplicates("bar_close_time", keep="last").set_index("bar_close_time")
    btc_return = pd.to_numeric(btc["close"], errors="coerce").pct_change(fill_method=None)
    imbalance = 2 * btc["taker_buy_quote"] / btc["turnover"] - 1
    expected = pd.DataFrame(
        {
            "return_threshold": btc_return.abs()
            .shift(1)
            .rolling(cfg.source_lookback_bars, min_periods=cfg.source_min_bars)
            .quantile(cfg.source_return_quantile),
            "flow_threshold": imbalance.abs()
            .shift(1)
            .rolling(cfg.source_lookback_bars, min_periods=cfg.source_min_bars)
            .quantile(cfg.source_flow_quantile),
            "turnover_threshold": btc["turnover"]
            .shift(1)
            .rolling(cfg.source_lookback_bars, min_periods=cfg.source_min_bars)
            .quantile(cfg.source_turnover_quantile),
        }
    ).reindex(signals["feature_time"])
    actual = signals.set_index("feature_time")[expected.columns]
    threshold_error = (actual - expected).abs().max().max()
    _check(
        rows,
        "thresholds_exact_prior_window",
        bool(threshold_error <= 1e-12),
        float(threshold_error),
    )

    normalized_month = graph["graph_month"].dt.floor("D")
    _check(
        rows,
        "graph_month_normalized",
        graph["graph_month"].eq(normalized_month).all()
        and graph["graph_month"].dt.day.eq(1).all(),
        int(graph["graph_month"].dt.minute.ne(0).sum()),
    )
    _check(
        rows,
        "graph_minimum_history",
        graph["sample_n"].ge(cfg.graph_min_samples).all(),
        int(graph["sample_n"].min()),
    )
    selected_graph = graph[graph["selected"].astype(bool)]
    selected_count = selected_graph.groupby("graph_month").size()
    _check(
        rows,
        "graph_receiver_pool_cap",
        selected_count.le(cfg.receiver_pool_size).all(),
        int(selected_count.max()),
    )
    _check(
        rows,
        "selected_graph_forward_sign_positive",
        selected_graph["signed_forward_correlation"].gt(0).all(),
        float(selected_graph["signed_forward_correlation"].min()),
    )

    per_entry = events.groupby("entry_time")["candidate"].agg(list)
    _check(
        rows,
        "two_candidates_per_event",
        per_entry.map(lambda values: set(values) == set(CANDIDATES)).all(),
        len(per_entry),
    )
    holding = events["exit_time"] - events["entry_time"]
    _check(
        rows,
        "holding_exact_30_minutes",
        holding.eq(pd.Timedelta(minutes=30)).all(),
        str(holding.unique().tolist()),
    )
    _check(
        rows,
        "laggard_count_frozen_range",
        events["laggard_count"].between(cfg.min_laggards, cfg.max_laggards).all(),
        f"{events['laggard_count'].min()}:{events['laggard_count'].max()}",
    )
    _check(
        rows,
        "selected_directed_residual_nonpositive",
        selected["directed_residual_15m"].le(1e-15).all(),
        float(selected["directed_residual_15m"].max()),
    )

    raw = events[events["candidate"].eq(RAW_CANDIDATE)]
    raw_expected = raw["direction"] * raw["mean_laggard_future_return"]
    raw_error = (raw["gross_return"] - raw_expected).abs().max()
    _check(rows, "raw_gross_formula_exact", raw_error <= 1e-12, float(raw_error))
    neutral = events[events["candidate"].eq(NEUTRAL_CANDIDATE)]
    neutral_expected = neutral["direction"] * (
        neutral["mean_laggard_future_return"]
        - neutral["mean_laggard_beta"] * neutral["btc_future_return"]
    ) / (1 + neutral["mean_laggard_beta"].abs())
    neutral_error = (neutral["gross_return"] - neutral_expected).abs().max()
    _check(
        rows,
        "neutral_gross_formula_exact",
        neutral_error <= 1e-12,
        float(neutral_error),
    )
    for candidate, sample in events.groupby("candidate", sort=True):
        reported = summary[
            summary["candidate"].eq(candidate) & summary["scope"].eq("all")
        ]
        actual_mean = float(sample["primary_net_return"].mean() * 10_000)
        reported_mean = float(reported["mean_primary_net_bp"].iloc[0])
        _check(
            rows,
            f"{candidate}_summary_exact",
            math.isclose(actual_mean, reported_mean, rel_tol=0, abs_tol=1e-10),
            actual_mean - reported_mean,
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
    candidate_max = random_wide[list(CANDIDATES)].max(axis=1)
    family_error = (random_wide["FAMILY_MAX"] - candidate_max).abs().max()
    _check(
        rows,
        "random_family_max_exact",
        family_error <= 1e-12,
        float(family_error),
    )
    _check(rows, "all_gate_rows_ineligible", not gates["eligible"].any(), len(gates))
    _check(rows, "no_candidate_eligible", not outcome["eligible"].any(), len(outcome))
    _check(
        rows,
        "rejection_verdict_exact",
        outcome["verdict"].eq("reject_btc_confirmed_flow_laggard").all(),
        "|".join(outcome["verdict"].astype(str)),
    )
    config_hits = [
        path.name
        for path in Path("configs").glob("*")
        if "v178" in path.name.lower() or "btc_confirmed_flow_laggard" in path.name.lower()
    ]
    _check(rows, "no_live_config_created", not config_hits, "|".join(config_hits))
    audit = pd.DataFrame(rows)
    audit["round_verdict"] = (
        "audit_pass_alpha_rejected"
        if audit["passed"].all()
        else "audit_failure_requires_investigation"
    )
    return audit


def _write_findings(audit: pd.DataFrame, path: Path) -> None:
    failed = audit[~audit["passed"]]
    verdict = str(audit["round_verdict"].iloc[0])
    text = [
        "# v17.8 BTC Confirmed-Flow Laggard Independent Audit",
        "",
        f"Verdict: `{verdict}`.",
        "",
        f"Checks: {len(audit)}; passed: {int(audit['passed'].sum())}; failed: {len(failed)}.",
        "",
        failed.to_markdown(index=False) if not failed.empty else "No failed checks.",
        "",
        "The catch-up candidates remain rejected. No live, PaperLive, application,",
        "leverage, remote, or real-order permission changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v178_audit(
    report_root: Path = AUDIT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    audit = audit_v178()
    root = ensure_dir(report_root)
    outputs = {"audit": root / "audit_checks.csv", "findings": findings_path}
    audit.to_csv(outputs["audit"], index=False)
    _write_findings(audit, findings_path)
    return outputs


__all__ = ["audit_v178", "write_v178_audit"]
