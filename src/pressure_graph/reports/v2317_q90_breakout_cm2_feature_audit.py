"""Outcome-free mapping of frozen q90 breakout events into the CM2 calendar."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


FEATURE_PATH = Path(
    "reports/v22_4_alt_book_vacuum_pressure_feature_audit/"
    "candidate_feature_events.parquet"
)
CM2_PATH = Path("reports/v16_5_fixed_core_satellite_fss3_tg1/weekly_portfolio.parquet")
REPORT_ROOT = Path("reports/v23_17_q90_breakout_cm2_feature_audit")
FINDINGS_PATH = Path("docs/v2317_q90_breakout_cm2_feature_audit_2026_07_17.md")
CANDIDATE = "CM3_CM2_PLUS_Q90_BREAKOUT_OVERLAY"


@dataclass(frozen=True)
class V2317Config:
    feature_path: Path = FEATURE_PATH
    cm2_path: Path = CM2_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    signal_direction: int = 1
    horizon_hours: int = 4
    primary_overlay_weight: float = 0.10
    sensitivity_overlay_weights: tuple[float, ...] = (0.05, 0.20)
    minimum_events: int = 50
    minimum_period_events: int = 15
    minimum_active_months: int = 10


def _utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="raise")


def load_v2317_inputs(
    cfg: V2317Config = V2317Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = [
        "candidate",
        "feature_time",
        "entry_time",
        "entry_month",
        "period",
        "signal_direction",
        "bucket_pressure",
        "prior_abs_pressure_threshold",
        "covered_symbols",
        "directional_symbol_count",
        "directional_breadth",
        "withdrawing_symbol_count",
        "withdrawal_breadth",
    ]
    features = pd.read_parquet(cfg.feature_path, columns=feature_columns)
    calendar = pd.read_parquet(
        cfg.cm2_path,
        columns=["entry_time", "exit_time", "month_start", "period", "candidate"],
    )
    for frame, columns in (
        (features, ("feature_time", "entry_time")),
        (calendar, ("entry_time", "exit_time", "month_start")),
    ):
        for column in columns:
            frame[column] = _utc(frame[column])
    return features, calendar


def build_v2317_event_mapping(
    features: pd.DataFrame,
    calendar: pd.DataFrame,
    cfg: V2317Config = V2317Config(),
) -> pd.DataFrame:
    selected = features.loc[features["signal_direction"].eq(cfg.signal_direction)].copy()
    selected = selected.sort_values("entry_time").reset_index(drop=True)
    selected["event_exit_time"] = selected["entry_time"] + pd.Timedelta(
        hours=cfg.horizon_hours
    )
    rows: list[dict[str, object]] = []
    for event in selected.itertuples(index=False):
        signal_matches = calendar.loc[
            calendar["entry_time"].le(event.entry_time)
            & calendar["exit_time"].gt(event.entry_time)
        ]
        realization_matches = calendar.loc[
            calendar["entry_time"].le(event.event_exit_time)
            & calendar["exit_time"].gt(event.event_exit_time)
        ]
        if len(signal_matches) != 1 or len(realization_matches) != 1:
            raise ValueError(
                f"event {event.entry_time} maps to {len(signal_matches)} signal "
                f"and {len(realization_matches)} realization CM2 weeks"
            )
        signal_week = signal_matches.iloc[0]
        realization_week = realization_matches.iloc[0]
        rows.append(
            {
                "candidate": CANDIDATE,
                "feature_time": event.feature_time,
                "event_entry_time": event.entry_time,
                "event_exit_time": event.event_exit_time,
                "entry_month": event.entry_month,
                "feature_period": event.period,
                "signal_direction": event.signal_direction,
                "bucket_pressure": event.bucket_pressure,
                "prior_abs_pressure_threshold": event.prior_abs_pressure_threshold,
                "covered_symbols": event.covered_symbols,
                "directional_symbol_count": event.directional_symbol_count,
                "directional_breadth": event.directional_breadth,
                "withdrawing_symbol_count": event.withdrawing_symbol_count,
                "withdrawal_breadth": event.withdrawal_breadth,
                "signal_week_entry_time": signal_week.entry_time,
                "signal_week_exit_time": signal_week.exit_time,
                "signal_week_period": signal_week.period,
                "portfolio_entry_time": realization_week.entry_time,
                "portfolio_exit_time": realization_week.exit_time,
                "portfolio_month_start": realization_week.month_start,
                "portfolio_period": realization_week.period,
                "portfolio_candidate": realization_week.candidate,
                "crosses_calendar_week": signal_week.entry_time
                != realization_week.entry_time,
                "horizon_hours": cfg.horizon_hours,
                "primary_overlay_weight": cfg.primary_overlay_weight,
            }
        )
    return pd.DataFrame(rows)


def summarize_v2317(
    mapping: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for scope in ("all", "development", "validation", "holdout"):
        local = (
            mapping
            if scope == "all"
            else mapping.loc[mapping["portfolio_period"].eq(scope)]
        )
        weekly = local.groupby("portfolio_entry_time", observed=True).size()
        rows.append(
            {
                "scope": scope,
                "events": len(local),
                "active_weeks": local["portfolio_entry_time"].nunique(),
                "calendar_weeks": (
                    len(calendar)
                    if scope == "all"
                    else int(calendar["period"].eq(scope).sum())
                ),
                "active_months": local["entry_month"].nunique(),
                "mean_events_per_active_week": float(weekly.mean()),
                "maximum_events_per_week": int(weekly.max()),
            }
        )
    return pd.DataFrame(rows)


def audit_v2317(
    mapping: pd.DataFrame,
    calendar: pd.DataFrame,
    summary: pd.DataFrame,
    cfg: V2317Config = V2317Config(),
) -> pd.DataFrame:
    period_counts = mapping["portfolio_period"].value_counts()
    gaps = mapping["event_entry_time"].sort_values().diff().dropna()
    forbidden = {
        "entry_spot",
        "exit_spot",
        "triggered",
        "fill_price",
        "gross_return",
        "primary_net_return",
        "stress_net_return",
    }
    checks = {
        "exactly_one_positive_direction": mapping["signal_direction"].eq(1).all(),
        "minimum_event_count": len(mapping) >= cfg.minimum_events,
        "minimum_period_event_count": all(
            int(period_counts.get(period, 0)) >= cfg.minimum_period_events
            for period in ("development", "validation", "holdout")
        ),
        "minimum_active_month_count": mapping["entry_month"].nunique()
        >= cfg.minimum_active_months,
        "all_events_map_to_calendar": mapping["portfolio_entry_time"].notna().all(),
        "event_entry_inside_signal_week": (
            mapping["event_entry_time"].ge(mapping["signal_week_entry_time"])
            & mapping["event_entry_time"].lt(mapping["signal_week_exit_time"])
        ).all(),
        "event_exit_inside_realization_week": (
            mapping["event_exit_time"].ge(mapping["portfolio_entry_time"])
            & mapping["event_exit_time"].lt(mapping["portfolio_exit_time"])
        ).all(),
        "feature_and_signal_period_match": mapping["feature_period"].eq(
            mapping["signal_week_period"]
        ).all(),
        "exact_four_hour_horizon": (
            mapping["event_exit_time"] - mapping["event_entry_time"]
        ).eq(pd.Timedelta(hours=cfg.horizon_hours)).all(),
        "no_overlapping_events": gaps.ge(pd.Timedelta(hours=cfg.horizon_hours)).all(),
        "calendar_has_49_unique_weeks": len(calendar) == 49
        and calendar["entry_time"].nunique() == 49,
        "frozen_primary_overlay_weight": np.isclose(
            mapping["primary_overlay_weight"], cfg.primary_overlay_weight
        ).all(),
        "outcome_columns_absent": forbidden.isdisjoint(mapping.columns),
        "summary_reconciles": int(summary.loc[summary["scope"].eq("all"), "events"].iloc[0])
        == len(mapping),
    }
    return pd.DataFrame(
        [{"check": check, "passed": bool(passed)} for check, passed in checks.items()]
    )


def feature_hash_v2317(mapping: pd.DataFrame) -> str:
    frozen = mapping.sort_values("event_entry_time").reset_index(drop=True).copy()
    payload = frozen.to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%S%z")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def write_v2317_q90_breakout_cm2_feature_audit(
    cfg: V2317Config = V2317Config(),
) -> dict[str, Path]:
    features, calendar = load_v2317_inputs(cfg)
    mapping = build_v2317_event_mapping(features, calendar, cfg)
    summary = summarize_v2317(mapping, calendar)
    checks = audit_v2317(mapping, calendar, summary, cfg)
    mapping_hash = feature_hash_v2317(mapping)
    root = ensure_dir(cfg.report_root)
    paths = {
        "mapping": root / "q90_event_week_mapping.parquet",
        "summary": root / "feature_coverage_summary.csv",
        "checks": root / "data_quality_checks.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    mapping.to_parquet(paths["mapping"], index=False)
    summary.to_csv(paths["summary"], index=False)
    checks.to_csv(paths["checks"], index=False)
    metadata = {
        "candidate": CANDIDATE,
        "feature_hash": mapping_hash,
        "all_checks_passed": bool(checks["passed"].all()),
        "config": {
            **asdict(cfg),
            "feature_path": str(cfg.feature_path),
            "cm2_path": str(cfg.cm2_path),
            "report_root": str(cfg.report_root),
            "findings_path": str(cfg.findings_path),
        },
        "outcomes_loaded": False,
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    verdict = (
        "feature_viable_freeze_q90_cm2_overlay"
        if metadata["all_checks_passed"]
        else "feature_audit_failed"
    )
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.17 q90 Breakout + CM2 Feature Audit",
                "",
                f"Verdict: `{verdict}`.",
                "",
                f"Feature hash: `{mapping_hash}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "The frozen positive-q90 event times were mapped into the existing",
                "49-week CM2 calendar using only calendar fields. Returns are assigned",
                "to the week in which the four-hour event exits; signal and realization",
                "weeks are both retained when an event crosses Monday 00:00 UTC.",
                "The primary overlay",
                "weight is fixed at 10%; 5% and 20% are sensitivity scales only.",
                "No trigger, fill, BTC path, or return column was loaded.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "V2317Config",
    "audit_v2317",
    "build_v2317_event_mapping",
    "feature_hash_v2317",
    "load_v2317_inputs",
    "summarize_v2317",
    "write_v2317_q90_breakout_cm2_feature_audit",
]
