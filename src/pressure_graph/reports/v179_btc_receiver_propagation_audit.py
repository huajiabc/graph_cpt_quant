"""Independent consistency audit for the v17.9 propagation round."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v179_btc_receiver_propagation import (
    CANDIDATES,
    NEUTRAL_CANDIDATE,
    RAW_CANDIDATE,
    REPORT_ROOT,
    SPREAD_CANDIDATE,
    V179Config,
)


AUDIT_ROOT = Path("reports/v17_9_btc_receiver_propagation_audit")
FINDINGS_PATH = Path("docs/v179_btc_receiver_propagation_audit_2026_07_16.md")


def _check(rows: list[dict[str, object]], name: str, passed: bool, value: object) -> None:
    rows.append({"check": name, "passed": bool(passed), "value": value})


def audit_v179(
    report_root: Path = REPORT_ROOT,
    cfg: V179Config = V179Config(),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    signals = pd.read_parquet(report_root / "btc_source_signals.parquet")
    graph = pd.read_parquet(report_root / "monthly_btc_receiver_graph.parquet")
    assignments = pd.read_parquet(report_root / "monthly_bucket_assignments.parquet")
    events = pd.read_parquet(report_root / "candidate_events.parquet")
    summary = pd.read_csv(report_root / "period_summary.csv")
    rank_reversed = pd.read_parquet(
        report_root / "rank_reversed_candidate_events.parquet"
    )
    random = pd.read_parquet(report_root / "random_bucket_controls.parquet")
    gates = pd.read_csv(report_root / "candidate_gates.csv")
    outcome = pd.read_csv(report_root / "candidate_outcome.csv")

    for column in ("feature_time", "source_feature_time"):
        signals[column] = pd.to_datetime(signals[column], utc=True)
    graph["graph_month"] = pd.to_datetime(graph["graph_month"], utc=True)
    assignments["graph_month"] = pd.to_datetime(
        assignments["graph_month"], utc=True
    )
    for frame in (events, rank_reversed):
        for column in ("entry_time", "exit_time"):
            frame[column] = pd.to_datetime(frame[column], utc=True)

    _check(rows, "signals_unique_200", signals["feature_time"].is_unique, len(signals))
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
        "graph_minimum_history",
        graph["sample_n"].ge(cfg.graph_min_samples).all(),
        int(graph["sample_n"].min()),
    )

    receiver_assignments = assignments[assignments["bucket"].eq("receiver")]
    _check(
        rows,
        "receiver_edges_strictly_positive",
        receiver_assignments["signed_forward_correlation"].gt(0).all(),
        float(receiver_assignments["signed_forward_correlation"].min()),
    )
    bucket_cap = assignments.groupby(["graph_month", "bucket"]).size()
    _check(
        rows,
        "bucket_size_cap_eight",
        bucket_cap.le(cfg.receiver_bucket_size).all(),
        int(bucket_cap.max()),
    )
    overlaps = []
    for month in assignments["graph_month"].unique():
        local = assignments[assignments["graph_month"].eq(month)]
        receiver_names = set(local.loc[local["bucket"].eq("receiver"), "receiver"])
        insulator_names = set(local.loc[local["bucket"].eq("insulator"), "receiver"])
        overlaps.extend(receiver_names & insulator_names)
    _check(rows, "buckets_disjoint", not overlaps, "|".join(sorted(set(overlaps))))

    ranking_exact = True
    for month, local_graph in graph.groupby("graph_month", sort=True):
        local_assignments = assignments[assignments["graph_month"].eq(month)]
        expected_receivers = (
            local_graph[local_graph["signed_forward_correlation"].gt(0)]
            .sort_values(
                ["receiver_score", "signed_forward_correlation"], ascending=False
            )
            .head(cfg.receiver_bucket_size)["receiver"]
            .astype(str)
            .tolist()
        )
        actual_receivers = (
            local_assignments[local_assignments["bucket"].eq("receiver")]
            .sort_values("bucket_rank")["receiver"]
            .astype(str)
            .tolist()
        )
        remaining = local_graph[
            ~local_graph["receiver"].astype(str).isin(expected_receivers)
        ]
        expected_insulators = (
            remaining.sort_values(
                ["receiver_score", "signed_forward_correlation"], ascending=True
            )
            .head(cfg.insulator_bucket_size)["receiver"]
            .astype(str)
            .tolist()
        )
        actual_insulators = (
            local_assignments[local_assignments["bucket"].eq("insulator")]
            .sort_values("bucket_rank")["receiver"]
            .astype(str)
            .tolist()
        )
        ranking_exact &= (
            expected_receivers == actual_receivers
            and expected_insulators == actual_insulators
        )
    _check(rows, "monthly_bucket_ranking_exact", ranking_exact, len(assignments))

    _check(
        rows,
        "primary_holding_exact_15m",
        (events["exit_time"] - events["entry_time"])
        .eq(pd.Timedelta(minutes=15))
        .all(),
        len(events),
    )
    _check(
        rows,
        "event_bucket_minimum_five",
        events["receiver_count"].ge(cfg.min_bucket_size).all()
        and events["insulator_count"].ge(cfg.min_bucket_size).all(),
        f"{events['receiver_count'].min()}:{events['insulator_count'].min()}",
    )
    raw = events[events["candidate"].eq(RAW_CANDIDATE)]
    raw_expected = raw["direction"] * raw["mean_receiver_future_return"]
    raw_error = (raw["gross_return"] - raw_expected).abs().max()
    _check(rows, "raw_formula_exact", raw_error <= 1e-12, float(raw_error))
    neutral = events[events["candidate"].eq(NEUTRAL_CANDIDATE)]
    neutral_expected = neutral["direction"] * (
        neutral["mean_receiver_future_return"]
        - neutral["mean_receiver_beta"] * neutral["btc_future_return"]
    ) / (1 + neutral["mean_receiver_beta"].abs())
    neutral_error = (neutral["gross_return"] - neutral_expected).abs().max()
    _check(
        rows, "neutral_formula_exact", neutral_error <= 1e-12, float(neutral_error)
    )
    spread = events[events["candidate"].eq(SPREAD_CANDIDATE)]
    spread_expected = spread["direction"] * (
        0.5
        * (
            spread["mean_receiver_future_return"]
            - spread["mean_insulator_future_return"]
        )
        - spread["spread_beta"] * spread["btc_future_return"]
    ) / (1 + spread["spread_beta"].abs())
    spread_error = (spread["gross_return"] - spread_expected).abs().max()
    _check(
        rows, "spread_formula_exact", spread_error <= 1e-12, float(spread_error)
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
        real_keys = set(zip(sample["entry_time"], sample["candidate"], strict=False))
        reversed_sample = rank_reversed[
            rank_reversed["candidate"].eq(candidate)
        ]
        reversed_keys = set(
            zip(
                reversed_sample["entry_time"],
                reversed_sample["candidate"],
                strict=False,
            )
        )
        _check(
            rows,
            f"{candidate}_rank_reversal_same_events",
            real_keys == reversed_keys,
            len(real_keys.symmetric_difference(reversed_keys)),
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
    family_expected = random_wide[list(CANDIDATES)].max(axis=1)
    family_error = (random_wide["FAMILY_MAX"] - family_expected).abs().max()
    _check(
        rows, "random_family_max_exact", family_error <= 1e-12, float(family_error)
    )
    _check(rows, "all_gate_rows_ineligible", not gates["eligible"].any(), len(gates))
    _check(rows, "no_candidate_eligible", not outcome["eligible"].any(), len(outcome))
    _check(
        rows,
        "rejection_verdict_exact",
        outcome["verdict"].eq("reject_btc_receiver_propagation").all(),
        "|".join(outcome["verdict"].astype(str)),
    )
    config_hits = [
        path.name
        for path in Path("configs").glob("*")
        if "v179" in path.name.lower() or "receiver_propagation" in path.name.lower()
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
    text = [
        "# v17.9 BTC Receiver Propagation Independent Audit",
        "",
        f"Verdict: `{audit['round_verdict'].iloc[0]}`.",
        "",
        f"Checks: {len(audit)}; passed: {int(audit['passed'].sum())}; failed: {len(failed)}.",
        "",
        failed.to_markdown(index=False) if not failed.empty else "No failed checks.",
        "",
        "All three propagation candidates remain rejected. No live or application",
        "scope changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v179_audit(
    report_root: Path = AUDIT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    audit = audit_v179()
    root = ensure_dir(report_root)
    outputs = {"audit": root / "audit_checks.csv", "findings": findings_path}
    audit.to_csv(outputs["audit"], index=False)
    _write_findings(audit, findings_path)
    return outputs


__all__ = ["audit_v179", "write_v179_audit"]
