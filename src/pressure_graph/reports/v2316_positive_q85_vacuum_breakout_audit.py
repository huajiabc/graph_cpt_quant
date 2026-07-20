"""Audit the sole q85 interpolation and its rejection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v234_book_vacuum_oco_breakout import (
    V234Config,
    build_v234_month_bootstrap,
    load_v234_inputs,
    simulate_v234_oco,
)
from pressure_graph.reports.v238_positive_pressure_narrow_breakout_robustness import (
    V238Config,
    _leave_one_month_out,
    _matched_random_by_scope,
    summarize_v238_periods,
)
from pressure_graph.reports.v2312_positive_q80_vacuum_breakout import decide_v2312
from pressure_graph.reports.v2313_positive_q80_vacuum_breakout_audit import (
    _error,
    _utc,
)
from pressure_graph.reports.v2315_positive_q85_vacuum_breakout import (
    FEATURE_SHA256,
    V2315Config,
    _decision_config,
)


V2314_ROOT = Path("reports/v23_14_positive_q85_vacuum_breakout_feature_audit")
V2315_ROOT = Path("reports/v23_15_positive_q85_vacuum_breakout")
REPORT_ROOT = Path("reports/v23_16_positive_q85_vacuum_breakout_audit")
FINDINGS_PATH = Path(
    "docs/v2316_positive_q85_vacuum_breakout_audit_2026_07_17.md"
)


@dataclass(frozen=True)
class V2316Config:
    v2314_root: Path = V2314_ROOT
    v2315_root: Path = V2315_ROOT
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    tolerance: float = 1e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def run_v2316_audit(
    cfg: V2316Config = V2316Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_path = cfg.v2314_root / "positive_q85_breakout_features.parquet"
    features = _utc(pd.read_parquet(feature_path))
    _, _, bars = load_v234_inputs(V234Config())
    sim_cfg = V234Config()
    primary = simulate_v234_oco(features, bars, sim_cfg, sigma_multiple=0.625)
    adjacent = simulate_v234_oco(features, bars, sim_cfg, sigma_multiple=0.75)
    saved_primary = _utc(
        pd.read_parquet(cfg.v2315_root / "primary_event_outcomes.parquet")
    )
    saved_adjacent = _utc(
        pd.read_parquet(cfg.v2315_root / "adjacent_event_outcomes.parquet")
    )
    fields = [
        "upper_stop_price",
        "lower_stop_price",
        "exit_spot",
        "fill_price",
        "gross_return",
        "primary_net_return",
        "stress_net_return",
    ]
    primary_error = _error(primary, saved_primary, ["entry_time"], fields)
    adjacent_error = _error(adjacent, saved_adjacent, ["entry_time"], fields)
    universe = _utc(
        pd.read_parquet(cfg.v2315_root / "causal_control_universe.parquet")
    )
    pools = _utc(pd.read_parquet(cfg.v2315_root / "matched_control_pools.parquet"))
    controls = simulate_v234_oco(universe, bars, sim_cfg, sigma_multiple=0.625)
    saved_controls = _utc(
        pd.read_parquet(cfg.v2315_root / "control_outcomes.parquet")
    )
    control_error = _error(controls, saved_controls, ["entry_time"], fields)
    random_paths, random_summary = _matched_random_by_scope(
        primary,
        controls,
        pools,
        replace(V238Config(), random_iterations=1000, seed=20260717),
    )
    saved_random = pd.read_parquet(cfg.v2315_root / "matched_random_paths.parquet")
    random_error = _error(
        random_paths,
        saved_random,
        ["scope", "iteration"],
        [
            "event_mean_primary_net",
            "control_mean_primary_net",
            "event_minus_control",
        ],
    )
    bootstrap = build_v234_month_bootstrap(primary, sim_cfg)
    saved_bootstrap = pd.read_parquet(
        cfg.v2315_root / "month_block_bootstrap.parquet"
    )
    bootstrap_error = _error(
        bootstrap,
        saved_bootstrap,
        ["iteration"],
        ["mean_primary_net_return"],
    )
    leave = _leave_one_month_out(primary)
    saved_leave = pd.read_csv(cfg.v2315_root / "leave_one_month_out.csv")
    leave_error = _error(
        leave,
        saved_leave,
        ["excluded_month"],
        ["remaining_events", "mean_primary_net_return_bp"],
    )
    primary_summary = summarize_v238_periods(primary, label="q85_0.625sigma")
    adjacent_summary = summarize_v238_periods(adjacent, label="q85_0.75sigma")
    saved_summary = pd.read_csv(cfg.v2315_root / "primary_summary.csv")
    summary_error = _error(
        primary_summary,
        saved_summary,
        ["variant", "scope"],
        [
            "events",
            "triggered_trades",
            "ambiguous_trades",
            "mean_primary_net_return_bp",
            "mean_stress_net_return_bp",
        ],
    )
    decision_cfg = _decision_config(V2315Config())
    decision, _ = decide_v2312(
        primary_summary,
        adjacent_summary,
        random_summary,
        bootstrap,
        leave,
        decision_cfg,
    )
    saved_decision = pd.read_csv(cfg.v2315_root / "decision_gates.csv")
    decision_error = float(
        np.max(
            np.abs(
                decision["observed"].to_numpy(dtype=float)
                - saved_decision["observed"].to_numpy(dtype=float)
            )
        )
    )
    diagnostics = {
        "primary_maximum_error": primary_error,
        "adjacent_maximum_error": adjacent_error,
        "control_maximum_error": control_error,
        "random_maximum_error": random_error,
        "bootstrap_maximum_error": bootstrap_error,
        "leave_one_out_maximum_error": leave_error,
        "summary_maximum_error": summary_error,
        "decision_maximum_error": decision_error,
        "matched_events": pools["event_time"].nunique(),
    }
    checks = {
        "v2314_feature_audit_passed": bool(
            pd.read_csv(cfg.v2314_root / "data_quality_checks.csv")["passed"].all()
        ),
        "feature_hash_matches_preregistration": _sha256(feature_path)
        == FEATURE_SHA256,
        "all_75_q85_events_preserved": len(primary) == 75,
        "event_and_control_paths_exact": max(
            primary_error, adjacent_error, control_error
        )
        <= cfg.tolerance,
        "all_75_events_have_matched_controls": pools["event_time"].nunique() == 75,
        "all_random_paths_exact": random_error <= cfg.tolerance,
        "all_month_bootstraps_exact": bootstrap_error <= cfg.tolerance,
        "leave_one_month_out_exact": leave_error <= cfg.tolerance,
        "summary_and_decision_exact": max(summary_error, decision_error)
        <= cfg.tolerance,
        "absolute_month_gate_fails": not bool(
            saved_decision.loc[
                saved_decision["gate"].eq(
                    "absolute_month_bootstrap_lower_above_zero"
                ),
                "passed",
            ].iloc[0]
        ),
        "holdout_random_gate_fails": float(
            random_summary.loc[
                random_summary["scope"].eq("holdout"),
                "matched_random_percentile",
            ].iloc[0]
        )
        < 90.0,
        "adjacent_width_gate_fails": not bool(
            saved_decision.loc[
                saved_decision["gate"].eq(
                    "adjacent_width_positive_all_temporal_scopes"
                ),
                "passed",
            ].iloc[0]
        ),
        "q85_verdict_is_rejection": (
            "Verdict: `positive_q85_interpolation_rejected`."
            in (
                cfg.v2315_root.parent.parent
                / "docs/v2315_positive_q85_vacuum_breakout_findings_2026_07_17.md"
            ).read_text(encoding="utf-8")
        ),
    }
    return (
        pd.DataFrame({"check": list(checks), "passed": list(checks.values())}),
        pd.DataFrame(
            {"diagnostic": list(diagnostics), "value": list(diagnostics.values())}
        ),
    )


def write_v2316_positive_q85_vacuum_breakout_audit(
    cfg: V2316Config = V2316Config(),
) -> dict[str, Path]:
    checks, diagnostics = run_v2316_audit(cfg)
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
                "validated_verdict": (
                    "positive_q85_interpolation_rejected"
                    if passed
                    else "audit_failed"
                ),
                "permissions_changed": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "audit_pass_validates_q85_rejection" if passed else "audit_failed"
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.16 Positive-q85 Interpolation Audit",
                "",
                f"Verdict: `{verdict}`.",
                "",
                f"Audit checks: {int(checks['passed'].sum())}/{len(checks)} passed.",
                "",
                "All 75 q85 paths, matched controls, 4,000 random paths, 5,000",
                "month bootstraps, leave-one-month-out values, and rejection were",
                "replayed. No further pressure-quantile interpolation is warranted.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "V2316Config",
    "run_v2316_audit",
    "write_v2316_positive_q85_vacuum_breakout_audit",
]
