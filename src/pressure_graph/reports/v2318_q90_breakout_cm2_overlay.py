"""Frozen portfolio-layer reveal for the q90 breakout over CM2."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v2317_q90_breakout_cm2_feature_audit import (
    CM2_PATH,
    REPORT_ROOT as V2317_ROOT,
    feature_hash_v2317,
)


OUTCOME_PATH = Path(
    "reports/v23_8_positive_pressure_narrow_breakout_robustness/"
    "primary_positive_pressure_outcomes.parquet"
)
REPORT_ROOT = Path("reports/v23_18_q90_breakout_cm2_overlay")
FINDINGS_PATH = Path("docs/v2318_q90_breakout_cm2_overlay_findings_2026_07_17.md")
CANDIDATE = "CM3_CM2_PLUS_10PCT_Q90_BREAKOUT_OVERLAY"
FROZEN_FEATURE_HASH = "CF79F9D42324BF5C930292F89D0186D86845B73561FA7E2B8EF6368E24C82046"


@dataclass(frozen=True)
class V2318Config:
    v2317_root: Path = V2317_ROOT
    outcome_path: Path = OUTCOME_PATH
    cm2_path: Path = CM2_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    overlay_weight: float = 0.10
    sensitivity_weights: tuple[float, ...] = (0.05, 0.20)
    bootstrap_iterations: int = 10_000
    seed: int = 20260717


def _utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="raise")


def _compound(series: pd.Series) -> float:
    return float(np.prod(1.0 + series.to_numpy(dtype=float)) - 1.0)


def _annualized_sharpe(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    volatility = float(values.std(ddof=1))
    if not np.isfinite(volatility) or volatility <= 0:
        return np.nan
    return float(values.mean() / volatility * np.sqrt(52.0))


def _downside_semideviation(series: pd.Series) -> float:
    values = np.minimum(pd.to_numeric(series, errors="coerce").to_numpy(float), 0.0)
    return float(np.sqrt(np.mean(np.square(values))))


def _additive_max_drawdown(series: pd.Series) -> float:
    cumulative = pd.to_numeric(series, errors="coerce").cumsum()
    drawdown = cumulative - cumulative.cummax()
    return float(drawdown.min())


def load_v2318_inputs(
    cfg: V2318Config = V2318Config(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping = pd.read_parquet(cfg.v2317_root / "q90_event_week_mapping.parquet")
    outcomes = pd.read_parquet(
        cfg.outcome_path,
        columns=[
            "entry_time",
            "triggered",
            "ambiguous_trigger",
            "primary_net_return",
            "stress_net_return",
            "reversed_primary_net_return",
        ],
    )
    core = pd.read_parquet(
        cfg.cm2_path,
        columns=[
            "entry_time",
            "exit_time",
            "month_start",
            "period",
            "primary_net_return",
            "stress_net_return",
        ],
    )
    for frame, columns in (
        (
            mapping,
            (
                "feature_time",
                "event_entry_time",
                "event_exit_time",
                "signal_week_entry_time",
                "signal_week_exit_time",
                "portfolio_entry_time",
                "portfolio_exit_time",
                "portfolio_month_start",
            ),
        ),
        (outcomes, ("entry_time",)),
        (core, ("entry_time", "exit_time", "month_start")),
    ):
        for column in columns:
            frame[column] = _utc(frame[column])
    return mapping, outcomes, core


def build_v2318_panel(
    mapping: pd.DataFrame,
    outcomes: pd.DataFrame,
    core: pd.DataFrame,
    cfg: V2318Config = V2318Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_outcomes = mapping.merge(
        outcomes.rename(columns={"entry_time": "event_entry_time"}),
        on="event_entry_time",
        how="left",
        validate="one_to_one",
    )
    if event_outcomes["primary_net_return"].isna().any():
        raise ValueError("not every frozen event has an outcome")
    weekly = (
        event_outcomes.groupby("portfolio_entry_time", observed=True)
        .agg(
            event_count=("event_entry_time", "size"),
            satellite_primary_return=("primary_net_return", _compound),
            satellite_stress_return=("stress_net_return", _compound),
            satellite_reversed_return=("reversed_primary_net_return", _compound),
            ambiguous_events=("ambiguous_trigger", "sum"),
            triggered_events=("triggered", "sum"),
        )
        .reset_index()
    )
    panel = core.rename(
        columns={
            "entry_time": "portfolio_entry_time",
            "exit_time": "portfolio_exit_time",
            "month_start": "portfolio_month_start",
            "primary_net_return": "core_primary_return",
            "stress_net_return": "core_stress_return",
        }
    ).merge(weekly, on="portfolio_entry_time", how="left", validate="one_to_one")
    panel["event_count"] = panel["event_count"].fillna(0).astype(int)
    panel["ambiguous_events"] = panel["ambiguous_events"].fillna(0).astype(int)
    panel["triggered_events"] = panel["triggered_events"].fillna(0).astype(int)
    for column in (
        "satellite_primary_return",
        "satellite_stress_return",
        "satellite_reversed_return",
    ):
        panel[column] = panel[column].fillna(0.0)
    panel["candidate"] = CANDIDATE
    panel["overlay_weight"] = cfg.overlay_weight
    panel["primary_increment"] = (
        cfg.overlay_weight * panel["satellite_primary_return"]
    )
    panel["stress_increment"] = cfg.overlay_weight * panel["satellite_stress_return"]
    panel["reversed_increment"] = (
        cfg.overlay_weight * panel["satellite_reversed_return"]
    )
    panel["combined_primary_return"] = (
        panel["core_primary_return"] + panel["primary_increment"]
    )
    panel["combined_stress_return"] = (
        panel["core_stress_return"] + panel["stress_increment"]
    )
    panel["active_hours"] = panel["event_count"] * 4
    return event_outcomes, panel


def summarize_v2318(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ("all", "development", "validation", "holdout"):
        local = panel if scope == "all" else panel.loc[panel["period"].eq(scope)]
        active = local.loc[local["event_count"].gt(0)]
        downside_active = active.loc[active["core_primary_return"].lt(0)]
        correlation = float(
            local[["satellite_primary_return", "core_primary_return"]]
            .corr()
            .iloc[0, 1]
        )
        active_correlation = float(
            active[["satellite_primary_return", "core_primary_return"]]
            .corr()
            .iloc[0, 1]
        )
        rows.append(
            {
                "scope": scope,
                "weeks": len(local),
                "active_weeks": len(active),
                "events": int(local["event_count"].sum()),
                "mean_core_primary_bp": float(local["core_primary_return"].mean() * 10_000),
                "mean_satellite_primary_bp": float(
                    local["satellite_primary_return"].mean() * 10_000
                ),
                "mean_primary_increment_bp": float(local["primary_increment"].mean() * 10_000),
                "mean_stress_increment_bp": float(local["stress_increment"].mean() * 10_000),
                "mean_combined_primary_bp": float(
                    local["combined_primary_return"].mean() * 10_000
                ),
                "mean_combined_stress_bp": float(
                    local["combined_stress_return"].mean() * 10_000
                ),
                "core_annualized_sharpe": _annualized_sharpe(local["core_primary_return"]),
                "combined_annualized_sharpe": _annualized_sharpe(
                    local["combined_primary_return"]
                ),
                "satellite_core_correlation": correlation,
                "active_satellite_core_correlation": active_correlation,
                "core_downside_semideviation_bp": _downside_semideviation(
                    local["core_primary_return"]
                )
                * 10_000,
                "combined_downside_semideviation_bp": _downside_semideviation(
                    local["combined_primary_return"]
                )
                * 10_000,
                "core_additive_max_drawdown_bp": _additive_max_drawdown(
                    local["core_primary_return"]
                )
                * 10_000,
                "combined_additive_max_drawdown_bp": _additive_max_drawdown(
                    local["combined_primary_return"]
                )
                * 10_000,
                "core_negative_active_weeks": len(downside_active),
                "downside_active_satellite_mean_bp": float(
                    downside_active["satellite_primary_return"].mean() * 10_000
                )
                if len(downside_active)
                else np.nan,
                "mean_reversed_increment_bp": float(
                    local["reversed_increment"].mean() * 10_000
                ),
            }
        )
    return pd.DataFrame(rows)


def build_v2318_sensitivity(
    panel: pd.DataFrame,
    cfg: V2318Config = V2318Config(),
) -> pd.DataFrame:
    rows = []
    for weight in (*cfg.sensitivity_weights, cfg.overlay_weight):
        for scope in ("all", "development", "validation", "holdout"):
            local = panel if scope == "all" else panel.loc[panel["period"].eq(scope)]
            combined = local["core_primary_return"] + weight * local[
                "satellite_primary_return"
            ]
            rows.append(
                {
                    "overlay_weight": weight,
                    "scope": scope,
                    "mean_primary_increment_bp": float(
                        weight * local["satellite_primary_return"].mean() * 10_000
                    ),
                    "combined_annualized_sharpe": _annualized_sharpe(combined),
                }
            )
    return pd.DataFrame(rows).sort_values(["overlay_weight", "scope"]).reset_index(
        drop=True
    )


def build_v2318_month_bootstrap(
    panel: pd.DataFrame,
    cfg: V2318Config = V2318Config(),
) -> pd.DataFrame:
    grouped = [
        group["primary_increment"].to_numpy(float)
        for _, group in panel.groupby("portfolio_month_start", observed=True)
    ]
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for iteration in range(cfg.bootstrap_iterations):
        choices = rng.integers(0, len(grouped), size=len(grouped))
        sample = np.concatenate([grouped[index] for index in choices])
        rows.append(
            {"iteration": iteration, "mean_primary_increment": float(sample.mean())}
        )
    return pd.DataFrame(rows)


def build_v2318_leave_one_month_out(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month in panel["portfolio_month_start"].drop_duplicates().sort_values():
        local = panel.loc[panel["portfolio_month_start"].ne(month)]
        rows.append(
            {
                "excluded_month": month,
                "mean_primary_increment_bp": float(
                    local["primary_increment"].mean() * 10_000
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_v2318_gates(
    mapping: pd.DataFrame,
    event_outcomes: pd.DataFrame,
    panel: pd.DataFrame,
    summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    bootstrap: pd.DataFrame,
    leave_one_out: pd.DataFrame,
) -> pd.DataFrame:
    all_row = summary.loc[summary["scope"].eq("all")].iloc[0]
    monthly = panel.groupby("portfolio_month_start", observed=True)[
        "primary_increment"
    ].sum()
    positive = monthly.loc[monthly.gt(0)]
    concentration = (
        float(positive.max() / positive.sum()) if positive.sum() > 0 else np.inf
    )
    bootstrap_low = float(bootstrap["mean_primary_increment"].quantile(0.025) * 10_000)
    sharpe_improvements = (
        summary["combined_annualized_sharpe"] - summary["core_annualized_sharpe"]
    )
    rows = [
        ("feature_hash_exact", feature_hash_v2317(mapping) == FROZEN_FEATURE_HASH, 0.0),
        (
            "all_53_events_matched_and_triggered",
            len(event_outcomes) == 53
            and event_outcomes["primary_net_return"].notna().all()
            and int(event_outcomes["triggered"].sum()) == 53,
            float(len(event_outcomes)),
        ),
        (
            "primary_increment_positive_all_scopes",
            summary["mean_primary_increment_bp"].gt(0).all(),
            float(summary["mean_primary_increment_bp"].min()),
        ),
        (
            "stress_increment_positive_all_scopes",
            summary["mean_stress_increment_bp"].gt(0).all(),
            float(summary["mean_stress_increment_bp"].min()),
        ),
        (
            "absolute_month_bootstrap_lower_above_zero",
            bootstrap_low > 0,
            bootstrap_low,
        ),
        (
            "leave_one_month_out_minimum_above_zero",
            leave_one_out["mean_primary_increment_bp"].min() > 0,
            float(leave_one_out["mean_primary_increment_bp"].min()),
        ),
        (
            "satellite_core_correlation_abs_at_most_030",
            abs(all_row["satellite_core_correlation"]) <= 0.30,
            float(all_row["satellite_core_correlation"]),
        ),
        (
            "active_satellite_core_correlation_abs_at_most_050",
            abs(all_row["active_satellite_core_correlation"]) <= 0.50,
            float(all_row["active_satellite_core_correlation"]),
        ),
        (
            "combined_sharpe_improves_all_scopes",
            sharpe_improvements.gt(0).all(),
            float(sharpe_improvements.min()),
        ),
        (
            "full_downside_semideviation_not_worse",
            all_row["combined_downside_semideviation_bp"]
            <= all_row["core_downside_semideviation_bp"],
            float(
                all_row["core_downside_semideviation_bp"]
                - all_row["combined_downside_semideviation_bp"]
            ),
        ),
        (
            "full_additive_drawdown_not_worse",
            all_row["combined_additive_max_drawdown_bp"]
            >= all_row["core_additive_max_drawdown_bp"],
            float(
                all_row["combined_additive_max_drawdown_bp"]
                - all_row["core_additive_max_drawdown_bp"]
            ),
        ),
        (
            "observed_overlay_beats_sign_reversed",
            all_row["mean_primary_increment_bp"]
            > all_row["mean_reversed_increment_bp"],
            float(
                all_row["mean_primary_increment_bp"]
                - all_row["mean_reversed_increment_bp"]
            ),
        ),
        (
            "fixed_sensitivities_positive_all_scopes",
            sensitivity["mean_primary_increment_bp"].gt(0).all(),
            float(sensitivity["mean_primary_increment_bp"].min()),
        ),
        (
            "positive_month_increment_concentration_at_most_050",
            concentration <= 0.50,
            concentration,
        ),
    ]
    return pd.DataFrame(
        [
            {"gate": gate, "passed": bool(passed), "observed": observed}
            for gate, passed, observed in rows
        ]
    )


def write_v2318_q90_breakout_cm2_overlay(
    cfg: V2318Config = V2318Config(),
) -> dict[str, Path]:
    mapping, outcomes, core = load_v2318_inputs(cfg)
    event_outcomes, panel = build_v2318_panel(mapping, outcomes, core, cfg)
    summary = summarize_v2318(panel)
    sensitivity = build_v2318_sensitivity(panel, cfg)
    bootstrap = build_v2318_month_bootstrap(panel, cfg)
    leave_one_out = build_v2318_leave_one_month_out(panel)
    gates = evaluate_v2318_gates(
        mapping,
        event_outcomes,
        panel,
        summary,
        sensitivity,
        bootstrap,
        leave_one_out,
    )
    root = ensure_dir(cfg.report_root)
    paths = {
        "event_outcomes": root / "mapped_event_outcomes.parquet",
        "portfolio": root / "weekly_portfolio.parquet",
        "summary": root / "summary.csv",
        "sensitivity": root / "allocation_sensitivity.csv",
        "bootstrap": root / "month_bootstrap.parquet",
        "leave_one_out": root / "leave_one_month_out.csv",
        "gates": root / "evidence_gates.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    event_outcomes.to_parquet(paths["event_outcomes"], index=False)
    panel.to_parquet(paths["portfolio"], index=False)
    summary.to_csv(paths["summary"], index=False)
    sensitivity.to_csv(paths["sensitivity"], index=False)
    bootstrap.to_parquet(paths["bootstrap"], index=False)
    leave_one_out.to_csv(paths["leave_one_out"], index=False)
    gates.to_csv(paths["gates"], index=False)
    passed = bool(gates["passed"].all())
    verdict = (
        "portfolio_overlay_research_pass_forward_shadow_only"
        if passed
        else "q90_cm2_portfolio_confirmation_rejected"
    )
    metadata = {
        "candidate": CANDIDATE,
        "verdict": verdict,
        "all_gates_passed": passed,
        "feature_hash": feature_hash_v2317(mapping),
        "config": {
            **asdict(cfg),
            "v2317_root": str(cfg.v2317_root),
            "outcome_path": str(cfg.outcome_path),
            "cm2_path": str(cfg.cm2_path),
            "report_root": str(cfg.report_root),
            "findings_path": str(cfg.findings_path),
        },
        "scope": "research_only_post_selected_ancestor",
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.18 q90 Breakout + CM2 Overlay Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                gates.to_markdown(index=False, floatfmt=".4f"),
                "",
                "The fixed 10% overlay is temporary notional during four-hour events;",
                "CM2 itself is unchanged. The 5% and 20% rows are frozen scale",
                "sensitivities, not an allocation search. The ancestor remains post-selected.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "V2318Config",
    "build_v2318_leave_one_month_out",
    "build_v2318_month_bootstrap",
    "build_v2318_panel",
    "build_v2318_sensitivity",
    "evaluate_v2318_gates",
    "load_v2318_inputs",
    "summarize_v2318",
    "write_v2318_q90_breakout_cm2_overlay",
]
