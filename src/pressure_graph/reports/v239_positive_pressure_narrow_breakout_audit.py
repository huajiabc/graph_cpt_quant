"""Artifact audit for the v23.8 post-selection robustness result."""

from __future__ import annotations

import json
from dataclasses import dataclass
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
    _month_sign_bootstrap,
    _within_month_sign_permutation,
    classify_v238,
    simulate_v238_latency_horizon,
    summarize_v238_periods,
)


V234_ROOT = Path("reports/v23_4_book_vacuum_oco_breakout")
V238_ROOT = Path("reports/v23_8_positive_pressure_narrow_breakout_robustness")
REPORT_ROOT = Path("reports/v23_9_positive_pressure_narrow_breakout_audit")
FINDINGS_PATH = Path(
    "docs/v239_positive_pressure_narrow_breakout_audit_2026_07_17.md"
)


@dataclass(frozen=True)
class V239Config:
    v234_root: Path = V234_ROOT
    v238_root: Path = V238_ROOT
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    tolerance: float = 1e-12


def _utc(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if column.endswith("_time"):
            output[column] = pd.to_datetime(output[column], utc=True, errors="coerce")
    return output


def _numeric_error(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: list[str],
    fields: list[str],
) -> float:
    merged = left[[*keys, *fields]].merge(
        right[[*keys, *fields]],
        on=keys,
        suffixes=("_audit", "_saved"),
        validate="one_to_one",
    )
    return max(
        float(
            np.nanmax(
                np.abs(
                    merged[f"{field}_audit"].to_numpy(dtype=float)
                    - merged[f"{field}_saved"].to_numpy(dtype=float)
                )
            )
        )
        for field in fields
    )


def run_v239_audit(
    cfg: V239Config = V239Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis_cfg = V238Config()
    features, _, bars = load_v234_inputs(V234Config())
    positive_features = features[features["signal_direction"].eq(1)]
    primary = simulate_v234_oco(
        positive_features,
        bars,
        V234Config(),
        sigma_multiple=analysis_cfg.primary_sigma_multiple,
    )
    adjacent = simulate_v234_oco(
        positive_features,
        bars,
        V234Config(),
        sigma_multiple=analysis_cfg.adjacent_sigma_multiple,
    )
    saved_primary = _utc(
        pd.read_parquet(cfg.v238_root / "primary_positive_pressure_outcomes.parquet")
    )
    saved_adjacent = _utc(
        pd.read_parquet(cfg.v238_root / "adjacent_width_outcomes.parquet")
    )
    outcome_fields = [
        "upper_stop_price",
        "lower_stop_price",
        "exit_spot",
        "fill_price",
        "gross_return",
        "primary_net_return",
        "stress_net_return",
    ]
    primary_error = _numeric_error(
        primary, saved_primary, ["entry_time"], outcome_fields
    )
    adjacent_error = _numeric_error(
        adjacent, saved_adjacent, ["entry_time"], outcome_fields
    )

    sensitivity_saved = _utc(
        pd.read_parquet(cfg.v238_root / "latency_horizon_outcomes.parquet")
    )
    sensitivity = []
    for horizon in (3, 4, 6):
        for delay in (0, 15, 30):
            sensitivity.append(
                simulate_v238_latency_horizon(
                    positive_features,
                    bars,
                    analysis_cfg,
                    horizon_hours=horizon,
                    activation_delay_minutes=delay,
                )
            )
    sensitivity_audit = pd.concat(sensitivity, ignore_index=True)
    sensitivity_error = _numeric_error(
        sensitivity_audit,
        sensitivity_saved,
        ["entry_time", "horizon_hours", "activation_delay_minutes"],
        ["gross_return", "primary_net_return", "stress_net_return"],
    )

    universe = _utc(pd.read_parquet(cfg.v234_root / "causal_control_universe.parquet"))
    pools = _utc(pd.read_parquet(cfg.v234_root / "matched_control_pools.parquet"))
    positive_pools = pools[pools["event_time"].isin(set(primary["entry_time"]))]
    controls = simulate_v234_oco(
        universe,
        bars,
        V234Config(),
        sigma_multiple=analysis_cfg.primary_sigma_multiple,
    )
    random_audit, random_summary = _matched_random_by_scope(
        primary, controls, positive_pools, analysis_cfg
    )
    random_saved = pd.read_parquet(cfg.v238_root / "matched_random_paths.parquet")
    random_fields = [
        "event_mean_primary_net",
        "control_mean_primary_net",
        "event_minus_control",
    ]
    random_error = _numeric_error(
        random_audit,
        random_saved,
        ["scope", "iteration"],
        random_fields,
    )

    all_primary = simulate_v234_oco(
        features,
        bars,
        V234Config(),
        sigma_multiple=analysis_cfg.primary_sigma_multiple,
    )
    permutation, p_upper = _within_month_sign_permutation(
        all_primary, analysis_cfg
    )
    permutation_saved = pd.read_parquet(
        cfg.v238_root / "within_month_sign_permutations.parquet"
    )
    permutation_error = _numeric_error(
        permutation,
        permutation_saved,
        ["iteration"],
        ["permuted_sign_difference_bp"],
    )
    sign_bootstrap = _month_sign_bootstrap(all_primary, analysis_cfg)
    sign_saved = pd.read_parquet(
        cfg.v238_root / "month_sign_difference_bootstrap.parquet"
    )
    sign_error = _numeric_error(
        sign_bootstrap,
        sign_saved,
        ["iteration"],
        ["sign_difference_bp"],
    )
    absolute_bootstrap = build_v234_month_bootstrap(
        primary,
        V234Config(
            bootstrap_iterations=analysis_cfg.bootstrap_iterations,
            seed=analysis_cfg.seed,
        ),
    )
    absolute_saved = pd.read_parquet(
        cfg.v238_root / "absolute_month_bootstrap.parquet"
    )
    absolute_error = _numeric_error(
        absolute_bootstrap,
        absolute_saved,
        ["iteration"],
        ["mean_primary_net_return"],
    )
    leave_one_out = _leave_one_month_out(primary)
    leave_saved = pd.read_csv(cfg.v238_root / "leave_one_month_out.csv")
    leave_error = _numeric_error(
        leave_one_out,
        leave_saved,
        ["excluded_month"],
        ["remaining_events", "mean_primary_net_return_bp"],
    )

    base_summary = pd.concat(
        [
            summarize_v238_periods(primary, label="0.625sigma_primary"),
            summarize_v238_periods(adjacent, label="0.75sigma_adjacent"),
        ],
        ignore_index=True,
    )
    saved_summary = pd.read_csv(cfg.v238_root / "base_summary.csv")
    summary_error = _numeric_error(
        base_summary,
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
    saved_gates = pd.read_csv(cfg.v238_root / "evidence_gates.csv")
    verdict = classify_v238(saved_gates)
    diagnostics = {
        "primary_maximum_error": primary_error,
        "adjacent_maximum_error": adjacent_error,
        "sensitivity_maximum_error": sensitivity_error,
        "random_maximum_error": random_error,
        "permutation_maximum_error": permutation_error,
        "sign_bootstrap_maximum_error": sign_error,
        "absolute_bootstrap_maximum_error": absolute_error,
        "leave_one_out_maximum_error": leave_error,
        "summary_maximum_error": summary_error,
        "permutation_upper_p": p_upper,
        "minimum_matched_random_percentile": float(
            random_summary["matched_random_percentile"].min()
        ),
    }
    checks = {
        "exactly_53_positive_pressure_events": len(primary) == 53,
        "primary_and_adjacent_paths_exact": max(primary_error, adjacent_error)
        <= cfg.tolerance,
        "zero_primary_and_adjacent_ambiguities": not bool(
            primary["ambiguous_trigger"].any()
            or adjacent["ambiguous_trigger"].any()
        ),
        "all_latency_horizon_paths_exact": sensitivity_error <= cfg.tolerance,
        "all_scope_random_paths_exact": random_error <= cfg.tolerance,
        "all_permutations_exact": permutation_error <= cfg.tolerance,
        "all_sign_bootstraps_exact": sign_error <= cfg.tolerance,
        "all_absolute_bootstraps_exact": absolute_error <= cfg.tolerance,
        "leave_one_month_out_exact": leave_error <= cfg.tolerance,
        "base_summary_exact": summary_error <= cfg.tolerance,
        "nine_structural_gates_pass": int(saved_gates["passed"].sum()) == 9,
        "absolute_month_gate_fails": not bool(
            saved_gates.loc[
                saved_gates["gate"].eq(
                    "absolute_month_bootstrap_lower_above_zero"
                ),
                "passed",
            ].iloc[0]
        ),
        "verdict_is_forward_shadow_only": verdict
        == "forward_shadow_candidate_not_statistically_confirmed",
    }
    return (
        pd.DataFrame({"check": list(checks), "passed": list(checks.values())}),
        pd.DataFrame(
            {"diagnostic": list(diagnostics), "value": list(diagnostics.values())}
        ),
    )


def write_v239_positive_pressure_narrow_breakout_audit(
    cfg: V239Config = V239Config(),
) -> dict[str, Path]:
    checks, diagnostics = run_v239_audit(cfg)
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
                "validated_status": (
                    "forward_shadow_candidate_not_statistically_confirmed"
                    if passed
                    else "audit_failed"
                ),
                "permissions_changed": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "audit_pass_validates_forward_shadow_only" if passed else "audit_failed"
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.9 Positive-Pressure Narrow-Breakout Audit",
                "",
                f"Verdict: `{verdict}`.",
                "",
                f"Audit checks: {int(checks['passed'].sum())}/{len(checks)} passed.",
                "",
                "Primary and adjacent widths, latency/horizon paths, matched random",
                "paths, sign permutations, both month bootstraps, leave-one-month-out",
                "and the forward-shadow-only status were replayed.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "V239Config",
    "run_v239_audit",
    "write_v239_positive_pressure_narrow_breakout_audit",
]
