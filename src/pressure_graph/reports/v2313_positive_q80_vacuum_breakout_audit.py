"""Audit the rejected v23.12 positive-q80 density extension."""

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
from pressure_graph.reports.v2312_positive_q80_vacuum_breakout import (
    FEATURE_SHA256,
    V2312Config,
    decide_v2312,
)


V2311_ROOT = Path("reports/v23_11_positive_q80_vacuum_breakout_feature_audit")
V2312_ROOT = Path("reports/v23_12_positive_q80_vacuum_breakout")
REPORT_ROOT = Path("reports/v23_13_positive_q80_vacuum_breakout_audit")
FINDINGS_PATH = Path(
    "docs/v2313_positive_q80_vacuum_breakout_audit_2026_07_17.md"
)


@dataclass(frozen=True)
class V2313Config:
    v2311_root: Path = V2311_ROOT
    v2312_root: Path = V2312_ROOT
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    tolerance: float = 1e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _utc(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if column.endswith("_time"):
            output[column] = pd.to_datetime(output[column], utc=True, errors="coerce")
    return output


def _error(
    audit: pd.DataFrame,
    saved: pd.DataFrame,
    keys: list[str],
    fields: list[str],
) -> float:
    merged = audit[[*keys, *fields]].merge(
        saved[[*keys, *fields]],
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


def run_v2313_audit(
    cfg: V2313Config = V2313Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_path = cfg.v2311_root / "positive_q80_breakout_features.parquet"
    features = _utc(pd.read_parquet(feature_path))
    _, _, bars = load_v234_inputs(V234Config())
    sim_cfg = V234Config()
    primary = simulate_v234_oco(
        features, bars, sim_cfg, sigma_multiple=0.625
    )
    adjacent = simulate_v234_oco(
        features, bars, sim_cfg, sigma_multiple=0.75
    )
    saved_primary = _utc(
        pd.read_parquet(cfg.v2312_root / "primary_event_outcomes.parquet")
    )
    saved_adjacent = _utc(
        pd.read_parquet(cfg.v2312_root / "adjacent_event_outcomes.parquet")
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
        pd.read_parquet(cfg.v2312_root / "causal_control_universe.parquet")
    )
    pools = _utc(pd.read_parquet(cfg.v2312_root / "matched_control_pools.parquet"))
    controls = simulate_v234_oco(
        universe, bars, sim_cfg, sigma_multiple=0.625
    )
    saved_controls = _utc(
        pd.read_parquet(cfg.v2312_root / "control_outcomes.parquet")
    )
    control_error = _error(controls, saved_controls, ["entry_time"], fields)
    random_paths, random_summary = _matched_random_by_scope(
        primary,
        controls,
        pools,
        replace(V238Config(), random_iterations=1000, seed=20260717),
    )
    saved_random = pd.read_parquet(cfg.v2312_root / "matched_random_paths.parquet")
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
        cfg.v2312_root / "month_block_bootstrap.parquet"
    )
    bootstrap_error = _error(
        bootstrap,
        saved_bootstrap,
        ["iteration"],
        ["mean_primary_net_return"],
    )
    leave = _leave_one_month_out(primary)
    saved_leave = pd.read_csv(cfg.v2312_root / "leave_one_month_out.csv")
    leave_error = _error(
        leave,
        saved_leave,
        ["excluded_month"],
        ["remaining_events", "mean_primary_net_return_bp"],
    )
    primary_summary = summarize_v238_periods(primary, label="q80_0.625sigma")
    adjacent_summary = summarize_v238_periods(adjacent, label="q80_0.75sigma")
    saved_primary_summary = pd.read_csv(cfg.v2312_root / "primary_summary.csv")
    summary_error = _error(
        primary_summary,
        saved_primary_summary,
        ["variant", "scope"],
        [
            "events",
            "triggered_trades",
            "ambiguous_trades",
            "mean_primary_net_return_bp",
            "mean_stress_net_return_bp",
        ],
    )
    decision, verdict = decide_v2312(
        primary_summary,
        adjacent_summary,
        random_summary,
        bootstrap,
        leave,
        V2312Config(),
    )
    saved_decision = pd.read_csv(cfg.v2312_root / "decision_gates.csv")
    decision_error = float(
        np.max(
            np.abs(
                decision["observed"].to_numpy(dtype=float)
                - saved_decision["observed"].to_numpy(dtype=float)
            )
        )
    )
    matched_counts = pools.groupby("event_time").size()
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
        "unmatched_events": len(primary) - pools["event_time"].nunique(),
    }
    checks = {
        "v2311_feature_audit_passed": bool(
            pd.read_csv(cfg.v2311_root / "data_quality_checks.csv")["passed"].all()
        ),
        "feature_hash_matches_preregistration": _sha256(feature_path)
        == FEATURE_SHA256,
        "all_89_event_keys_preserved": len(primary) == 89
        and set(primary["entry_time"]) == set(features["entry_time"]),
        "primary_adjacent_and_control_paths_exact": max(
            primary_error, adjacent_error, control_error
        )
        <= cfg.tolerance,
        "matched_pools_have_frozen_5_to_10_controls": matched_counts.between(
            5, 10
        ).all(),
        "exactly_two_events_are_unmatched": pools["event_time"].nunique() == 87,
        "all_random_paths_exact": random_error <= cfg.tolerance,
        "all_month_bootstraps_exact": bootstrap_error <= cfg.tolerance,
        "leave_one_month_out_exact": leave_error <= cfg.tolerance,
        "summary_and_decision_exact": max(summary_error, decision_error)
        <= cfg.tolerance,
        "unmatched_coverage_gate_fails": not bool(
            saved_decision.loc[
                saved_decision["gate"].eq(
                    "every_event_has_at_least_five_matched_controls"
                ),
                "passed",
            ].iloc[0]
        ),
        "economic_and_temporal_gates_fail": not bool(
            saved_decision.loc[
                saved_decision["gate"].eq(
                    "primary_positive_all_temporal_scopes"
                ),
                "passed",
            ].iloc[0]
        ),
        "verdict_recomputed_as_rejection": verdict
        == "positive_q80_density_extension_rejected",
    }
    return (
        pd.DataFrame({"check": list(checks), "passed": list(checks.values())}),
        pd.DataFrame(
            {"diagnostic": list(diagnostics), "value": list(diagnostics.values())}
        ),
    )


def write_v2313_positive_q80_vacuum_breakout_audit(
    cfg: V2313Config = V2313Config(),
) -> dict[str, Path]:
    checks, diagnostics = run_v2313_audit(cfg)
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
                    "positive_q80_density_extension_rejected"
                    if passed
                    else "audit_failed"
                ),
                "permissions_changed": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "audit_pass_validates_rejection" if passed else "audit_failed"
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.13 Positive-q80 Breakout Audit",
                "",
                f"Verdict: `{verdict}`.",
                "",
                f"Audit checks: {int(checks['passed'].sum())}/{len(checks)} passed.",
                "",
                "All q80 event/control paths, matched subsets, 4,000 random paths,",
                "5,000 month bootstraps, leave-one-month-out values, the two unmatched",
                "events, and the rejection were replayed.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "V2313Config",
    "run_v2313_audit",
    "write_v2313_positive_q80_vacuum_breakout_audit",
]
