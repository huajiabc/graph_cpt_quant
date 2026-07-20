"""Independent audit of v23.20-v23.21 alt-first ignition research."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v2319_q90_breakout_cm2_overlay_audit import (
    _frames_equal,
    _maximum_numeric_error,
)
from pressure_graph.reports.v2320_alt_first_volatility_ignition_feature_audit import (
    REPORT_ROOT as V2320_ROOT,
    V2320Config,
    audit_v2320,
    build_v2320_features,
    feature_hash_v2320,
    load_v2320_inputs,
    summarize_v2320,
)
from pressure_graph.reports.v2321_alt_first_volatility_ignition_breakout import (
    CANDIDATE,
    CONTROL,
    FROZEN_FEATURE_HASH,
    REPORT_ROOT as V2321_ROOT,
    V2321Config,
    _simulation_config,
    build_v2321_control_pools,
    build_v2321_control_universe,
    build_v2321_leave_one_month_out,
    build_v2321_random_by_scope,
    decide_v2321,
    load_v2321_inputs,
    summarize_v2321,
)
from pressure_graph.reports.v234_book_vacuum_oco_breakout import (
    build_v234_month_bootstrap,
    simulate_v234_oco,
)


REPORT_ROOT = Path("reports/v23_22_alt_first_volatility_ignition_audit")
FINDINGS_PATH = Path(
    "docs/v2322_alt_first_volatility_ignition_audit_2026_07_17.md"
)


@dataclass(frozen=True)
class V2322Config:
    v2320_root: Path = V2320_ROOT
    v2321_root: Path = V2321_ROOT
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    tolerance: float = 1e-12


def run_v2322_audit(cfg: V2322Config = V2322Config()) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cfg = V2320Config(report_root=cfg.v2320_root)
    prices, bars = load_v2320_inputs(feature_cfg)
    rebuilt_states, rebuilt_features = build_v2320_features(prices, bars, feature_cfg)
    rebuilt_feature_summary = summarize_v2320(rebuilt_features)
    rebuilt_feature_checks = audit_v2320(
        rebuilt_states,
        rebuilt_features,
        rebuilt_feature_summary,
        feature_cfg,
    )
    saved_states = pd.read_parquet(cfg.v2320_root / "hourly_ignition_states.parquet")
    saved_features = pd.read_parquet(cfg.v2320_root / "alt_first_ignition_features.parquet")
    saved_feature_summary = pd.read_csv(cfg.v2320_root / "feature_coverage_summary.csv")
    saved_feature_checks = pd.read_csv(cfg.v2320_root / "data_quality_checks.csv")

    reveal_cfg = V2321Config(v2320_root=cfg.v2320_root, report_root=cfg.v2321_root)
    features, states, bars = load_v2321_inputs(reveal_cfg)
    sim_cfg = _simulation_config(reveal_cfg)
    rebuilt_universe = build_v2321_control_universe(features, states, bars, reveal_cfg)
    rebuilt_pools = build_v2321_control_pools(features, rebuilt_universe, reveal_cfg)
    rebuilt_outcomes = simulate_v234_oco(
        features,
        bars,
        sim_cfg,
        sigma_multiple=reveal_cfg.primary_sigma_multiple,
        candidate=CANDIDATE,
    )
    rebuilt_controls = simulate_v234_oco(
        rebuilt_universe,
        bars,
        sim_cfg,
        sigma_multiple=reveal_cfg.primary_sigma_multiple,
        candidate=CONTROL,
    )
    variant_frames = []
    variant_summaries = []
    for multiple in (
        reveal_cfg.primary_sigma_multiple,
        *reveal_cfg.adjacent_sigma_multiples,
    ):
        local = simulate_v234_oco(
            features,
            bars,
            sim_cfg,
            sigma_multiple=multiple,
            candidate=CANDIDATE,
        )
        variant_frames.append(local)
        local_summary = summarize_v2321(local, variant=f"{multiple:g}sigma")
        local_summary["sigma_multiple"] = multiple
        variant_summaries.append(local_summary)
    rebuilt_variants = pd.concat(variant_frames, ignore_index=True)
    rebuilt_variant_summary = pd.concat(variant_summaries, ignore_index=True)
    rebuilt_summary = summarize_v2321(rebuilt_outcomes, variant="0.75sigma")
    rebuilt_random, rebuilt_random_summary = build_v2321_random_by_scope(
        rebuilt_outcomes,
        rebuilt_controls,
        rebuilt_pools,
        reveal_cfg,
    )
    rebuilt_bootstrap = build_v234_month_bootstrap(rebuilt_outcomes, sim_cfg)
    rebuilt_leaveout = build_v2321_leave_one_month_out(rebuilt_outcomes)
    rebuilt_gates, rebuilt_verdict = decide_v2321(
        rebuilt_summary,
        rebuilt_random_summary,
        rebuilt_variant_summary,
        rebuilt_bootstrap,
        rebuilt_leaveout,
    )
    saved = {
        "control_universe": pd.read_parquet(cfg.v2321_root / "control_universe.parquet"),
        "control_pools": pd.read_parquet(cfg.v2321_root / "matched_control_pools.parquet"),
        "outcomes": pd.read_parquet(cfg.v2321_root / "event_outcomes.parquet"),
        "controls": pd.read_parquet(cfg.v2321_root / "control_outcomes.parquet"),
        "variants": pd.read_parquet(cfg.v2321_root / "barrier_variant_outcomes.parquet"),
        "summary": pd.read_csv(cfg.v2321_root / "result_summary.csv"),
        "variant_summary": pd.read_csv(cfg.v2321_root / "barrier_variant_summary.csv"),
        "random": pd.read_parquet(cfg.v2321_root / "matched_random_paths.parquet"),
        "random_summary": pd.read_csv(cfg.v2321_root / "matched_random_summary.csv"),
        "bootstrap": pd.read_parquet(cfg.v2321_root / "absolute_month_bootstrap.parquet"),
        "leaveout": pd.read_csv(cfg.v2321_root / "leave_one_month_out.csv"),
        "gates": pd.read_csv(cfg.v2321_root / "evidence_gates.csv"),
    }
    comparisons = {
        "feature_states": (rebuilt_states, saved_states),
        "feature_events": (rebuilt_features, saved_features),
        "feature_summary": (rebuilt_feature_summary, saved_feature_summary),
        "feature_checks": (rebuilt_feature_checks, saved_feature_checks),
        "control_universe": (rebuilt_universe, saved["control_universe"]),
        "control_pools": (rebuilt_pools, saved["control_pools"]),
        "outcomes": (rebuilt_outcomes, saved["outcomes"]),
        "controls": (rebuilt_controls, saved["controls"]),
        "variants": (rebuilt_variants, saved["variants"]),
        "summary": (rebuilt_summary, saved["summary"]),
        "variant_summary": (rebuilt_variant_summary, saved["variant_summary"]),
        "random_paths": (rebuilt_random, saved["random"]),
        "random_summary": (rebuilt_random_summary, saved["random_summary"]),
        "bootstrap": (rebuilt_bootstrap, saved["bootstrap"]),
        "leaveout": (rebuilt_leaveout, saved["leaveout"]),
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
    metadata = json.loads((cfg.v2321_root / "metadata.json").read_text(encoding="utf-8"))
    expected_failed = {
        "primary_positive_all_scopes",
        "stress_positive_all_scopes",
        "absolute_month_bootstrap_lower_above_zero",
        "matched_random_percentile_at_least_90_all_scopes",
        "leave_one_month_out_minimum_above_zero",
        "adjacent_widths_positive_all_scopes",
    }
    actual_failed = set(saved["gates"].loc[~saved["gates"]["passed"], "gate"])
    checks = {
        "feature_audit_rebuilds": rebuilt_feature_checks["passed"].all(),
        "feature_hash_exact": feature_hash_v2320(rebuilt_features)
        == FROZEN_FEATURE_HASH,
        "all_artifacts_exact_within_tolerance": diagnostics[
            "exact_within_tolerance"
        ].all(),
        "all_numeric_errors_within_tolerance": diagnostics[
            "maximum_numeric_error"
        ].le(cfg.tolerance).all(),
        "exactly_100_events": len(rebuilt_outcomes) == 100,
        "all_events_have_controls": rebuilt_random_summary[
            "unmatched_events"
        ].eq(0).all(),
        "failed_gate_set_exact": actual_failed == expected_failed,
        "rejection_verdict_exact": rebuilt_verdict
        == "alt_first_ignition_breakout_rejected"
        and metadata["verdict"] == rebuilt_verdict
        and not metadata["all_gates_passed"],
        "gross_result_negative_all_scopes": rebuilt_summary[
            "mean_gross_return_bp"
        ].lt(0).all(),
        "random_percentile_below_90_all_scopes": rebuilt_random_summary[
            "matched_random_percentile"
        ].lt(90).all(),
        "primary_width_frozen": np.isclose(
            rebuilt_outcomes["sigma_multiple"], reveal_cfg.primary_sigma_multiple
        ).all(),
    }
    audit_checks = pd.DataFrame(
        [{"check": check, "passed": bool(passed)} for check, passed in checks.items()]
    )
    return audit_checks, diagnostics


def write_v2322_alt_first_volatility_ignition_audit(
    cfg: V2322Config = V2322Config(),
) -> dict[str, Path]:
    checks, diagnostics = run_v2322_audit(cfg)
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
                "# v23.22 Alt-First Volatility Ignition Independent Audit",
                "",
                f"Verdict: `{'audit_passed' if passed else 'audit_failed'}`.",
                "",
                checks.to_markdown(index=False),
                "",
                diagnostics.to_markdown(index=False, floatfmt=".4g"),
                "",
                "The audit validates the negative gross result, weak matched-control",
                "rank, all saved paths, and the rejection decision.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = ["V2322Config", "run_v2322_audit", "write_v2322_alt_first_volatility_ignition_audit"]
