"""Independent audit for the v23.23-v23.24 broad taker confirmation branch."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v2319_q90_breakout_cm2_overlay_audit import (
    _frames_equal,
    _maximum_numeric_error,
)
from pressure_graph.reports.v2323_q90_broad_taker_confirmation_feature_audit import (
    REPORT_ROOT as V2323_ROOT,
    V2323Config,
    audit_v2323,
    build_v2323_features,
    feature_hash_v2323,
    load_v2323_inputs,
    summarize_v2323,
)
from pressure_graph.reports.v2324_q90_broad_taker_confirmation import (
    CANDIDATE,
    CONTROL,
    FROZEN_FEATURE_HASH,
    REPORT_ROOT as V2324_ROOT,
    V2324Config,
    build_v2324_bootstrap,
    build_v2324_control_pools,
    build_v2324_control_universe,
    build_v2324_label_permutation,
    build_v2324_leaveout,
    build_v2324_random_by_scope,
    decide_v2324,
    load_v2324_inputs,
    price_v2324_long,
    summarize_v2324,
)


REPORT_ROOT = Path("reports/v23_25_q90_broad_taker_confirmation_audit")
FINDINGS_PATH = Path(
    "docs/v2325_q90_broad_taker_confirmation_audit_2026_07_17.md"
)


@dataclass(frozen=True)
class V2325Config:
    v2323_root: Path = V2323_ROOT
    v2324_root: Path = V2324_ROOT
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    tolerance: float = 1e-12


def run_v2325_audit(cfg: V2325Config = V2325Config()) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cfg = V2323Config(report_root=cfg.v2323_root)
    positive, metric_series = load_v2323_inputs(feature_cfg)
    rebuilt_context, rebuilt_features = build_v2323_features(
        positive,
        metric_series,
        feature_cfg,
    )
    rebuilt_feature_summary = summarize_v2323(rebuilt_features)
    rebuilt_feature_checks = audit_v2323(
        positive,
        rebuilt_context,
        rebuilt_features,
        rebuilt_feature_summary,
        feature_cfg,
    )
    saved_context = pd.read_parquet(
        cfg.v2323_root / "positive_q90_taker_context.parquet"
    )
    saved_features = pd.read_parquet(
        cfg.v2323_root / "broad_taker_confirmed_features.parquet"
    )
    saved_feature_summary = pd.read_csv(cfg.v2323_root / "feature_coverage_summary.csv")
    saved_feature_checks = pd.read_csv(cfg.v2323_root / "data_quality_checks.csv")

    reveal_cfg = V2324Config(v2323_root=cfg.v2323_root, report_root=cfg.v2324_root)
    confirmed, context, metrics, bars = load_v2324_inputs(reveal_cfg)
    unconfirmed_features = context.loc[
        context["taker_buy_symbol_count"].lt(reveal_cfg.minimum_buy_symbols)
    ].copy()
    rebuilt_outcomes = price_v2324_long(confirmed, bars, reveal_cfg)
    rebuilt_delayed = price_v2324_long(
        confirmed,
        bars,
        reveal_cfg,
        delay_minutes=reveal_cfg.delay_minutes,
        candidate=f"{CANDIDATE}_DELAYED_15M",
    )
    rebuilt_unconfirmed = price_v2324_long(
        unconfirmed_features,
        bars,
        reveal_cfg,
        candidate=f"{CANDIDATE}_UNCONFIRMED_Q90",
    )
    rebuilt_universe = build_v2324_control_universe(
        context,
        metrics,
        bars,
        reveal_cfg,
    )
    rebuilt_pools = build_v2324_control_pools(
        confirmed,
        rebuilt_universe,
        bars,
        reveal_cfg,
    )
    rebuilt_controls = price_v2324_long(
        rebuilt_universe,
        bars,
        reveal_cfg,
        candidate=CONTROL,
    )
    rebuilt_summary = summarize_v2324(rebuilt_outcomes, label="confirmed")
    rebuilt_delayed_summary = summarize_v2324(rebuilt_delayed, label="delay_15m")
    rebuilt_unconfirmed_summary = summarize_v2324(
        rebuilt_unconfirmed,
        label="unconfirmed",
    )
    rebuilt_random, rebuilt_random_summary = build_v2324_random_by_scope(
        rebuilt_outcomes,
        rebuilt_controls,
        rebuilt_pools,
        reveal_cfg,
    )
    rebuilt_bootstrap = build_v2324_bootstrap(rebuilt_outcomes, reveal_cfg)
    rebuilt_leaveout = build_v2324_leaveout(rebuilt_outcomes)
    rebuilt_permutations, rebuilt_permutation_p = build_v2324_label_permutation(
        rebuilt_outcomes,
        rebuilt_unconfirmed,
        reveal_cfg,
    )
    rebuilt_gates, rebuilt_verdict = decide_v2324(
        rebuilt_summary,
        rebuilt_delayed_summary,
        rebuilt_unconfirmed_summary,
        rebuilt_random_summary,
        rebuilt_bootstrap,
        rebuilt_leaveout,
        rebuilt_outcomes,
        rebuilt_permutation_p,
    )
    saved = {
        "outcomes": pd.read_parquet(cfg.v2324_root / "confirmed_event_outcomes.parquet"),
        "delayed": pd.read_parquet(cfg.v2324_root / "delayed_event_outcomes.parquet"),
        "unconfirmed": pd.read_parquet(
            cfg.v2324_root / "unconfirmed_event_outcomes.parquet"
        ),
        "universe": pd.read_parquet(cfg.v2324_root / "control_universe.parquet"),
        "pools": pd.read_parquet(cfg.v2324_root / "matched_control_pools.parquet"),
        "controls": pd.read_parquet(cfg.v2324_root / "control_outcomes.parquet"),
        "summary": pd.read_csv(cfg.v2324_root / "result_summary.csv"),
        "delayed_summary": pd.read_csv(cfg.v2324_root / "delayed_summary.csv"),
        "unconfirmed_summary": pd.read_csv(
            cfg.v2324_root / "unconfirmed_summary.csv"
        ),
        "random": pd.read_parquet(cfg.v2324_root / "matched_random_paths.parquet"),
        "random_summary": pd.read_csv(cfg.v2324_root / "matched_random_summary.csv"),
        "bootstrap": pd.read_parquet(cfg.v2324_root / "absolute_month_bootstrap.parquet"),
        "leaveout": pd.read_csv(cfg.v2324_root / "leave_one_month_out.csv"),
        "permutations": pd.read_parquet(
            cfg.v2324_root / "within_month_label_permutations.parquet"
        ),
        "gates": pd.read_csv(cfg.v2324_root / "evidence_gates.csv"),
    }
    comparisons = {
        "feature_context": (rebuilt_context, saved_context),
        "feature_events": (rebuilt_features, saved_features),
        "feature_summary": (rebuilt_feature_summary, saved_feature_summary),
        "feature_checks": (rebuilt_feature_checks, saved_feature_checks),
        "outcomes": (rebuilt_outcomes, saved["outcomes"]),
        "delayed": (rebuilt_delayed, saved["delayed"]),
        "unconfirmed": (rebuilt_unconfirmed, saved["unconfirmed"]),
        "control_universe": (rebuilt_universe, saved["universe"]),
        "control_pools": (rebuilt_pools, saved["pools"]),
        "controls": (rebuilt_controls, saved["controls"]),
        "summary": (rebuilt_summary, saved["summary"]),
        "delayed_summary": (rebuilt_delayed_summary, saved["delayed_summary"]),
        "unconfirmed_summary": (
            rebuilt_unconfirmed_summary,
            saved["unconfirmed_summary"],
        ),
        "random_paths": (rebuilt_random, saved["random"]),
        "random_summary": (rebuilt_random_summary, saved["random_summary"]),
        "bootstrap": (rebuilt_bootstrap, saved["bootstrap"]),
        "leaveout": (rebuilt_leaveout, saved["leaveout"]),
        "permutations": (rebuilt_permutations, saved["permutations"]),
        "gates": (rebuilt_gates, saved["gates"]),
    }
    diagnostics = pd.DataFrame(
        [
            {
                "artifact": name,
                "rows_rebuilt": len(left),
                "rows_saved": len(right),
                "maximum_numeric_error": _maximum_numeric_error(left, right),
                "exact_within_tolerance": _frames_equal(left, right, cfg.tolerance),
            }
            for name, (left, right) in comparisons.items()
        ]
    )
    metadata = json.loads((cfg.v2324_root / "metadata.json").read_text(encoding="utf-8"))
    failed = set(saved["gates"].loc[~saved["gates"]["passed"], "gate"])
    expected_failed = set(saved["gates"]["gate"]) - {
        "minimum_events_and_period_coverage"
    }
    checks = {
        "feature_audit_rebuilds": rebuilt_feature_checks["passed"].all(),
        "feature_hash_exact": feature_hash_v2323(rebuilt_features)
        == FROZEN_FEATURE_HASH,
        "all_artifacts_exact_within_tolerance": diagnostics[
            "exact_within_tolerance"
        ].all(),
        "all_numeric_errors_within_tolerance": diagnostics[
            "maximum_numeric_error"
        ].le(cfg.tolerance).all(),
        "exactly_26_confirmed_and_27_unconfirmed": len(rebuilt_outcomes) == 26
        and len(rebuilt_unconfirmed) == 27,
        "exactly_two_unmatched_controls": int(
            rebuilt_random_summary.loc[
                rebuilt_random_summary["scope"].eq("all"), "unmatched_events"
            ].iloc[0]
        )
        == 2,
        "failed_gate_set_exact": failed == expected_failed,
        "permutation_p_exact": abs(
            float(metadata["permutation_p"]) - rebuilt_permutation_p
        )
        <= cfg.tolerance,
        "rejection_verdict_exact": rebuilt_verdict
        == "broad_taker_confirmation_rejected"
        and metadata["verdict"] == rebuilt_verdict
        and not metadata["all_gates_passed"],
        "full_sample_gross_negative": float(
            rebuilt_summary.loc[
                rebuilt_summary["scope"].eq("all"), "mean_gross_return_bp"
            ].iloc[0]
        )
        < 0,
        "confirmation_permutation_not_significant": rebuilt_permutation_p > 0.10,
    }
    audit_checks = pd.DataFrame(
        [{"check": check, "passed": bool(passed)} for check, passed in checks.items()]
    )
    return audit_checks, diagnostics


def write_v2325_q90_broad_taker_confirmation_audit(
    cfg: V2325Config = V2325Config(),
) -> dict[str, Path]:
    checks, diagnostics = run_v2325_audit(cfg)
    root = ensure_dir(cfg.report_root)
    paths = {
        "checks": root / "audit_checks.csv",
        "diagnostics": root / "audit_diagnostics.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    checks.to_csv(paths["checks"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)
    passed = bool(checks["passed"].all())
    paths["metadata"].write_text(
        json.dumps(
            {
                "audit_passed": passed,
                "checks_passed": int(checks["passed"].sum()),
                "checks_total": len(checks),
                "tolerance": cfg.tolerance,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.25 q90 Broad-Taker Confirmation Independent Audit",
                "",
                f"Verdict: `{'audit_passed' if passed else 'audit_failed'}`.",
                "",
                checks.to_markdown(index=False),
                "",
                diagnostics.to_markdown(index=False, floatfmt=".4g"),
                "",
                "The audit confirms that broad taker buying does not explain q90 as",
                "directional long alpha and reproduces the rejection exactly.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = ["V2325Config", "run_v2325_audit", "write_v2325_q90_broad_taker_confirmation_audit"]
