"""Monthly-expiry coverage extension for the frozen v17.5 spread rules."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v173_deribit_skew_receiver_bucket import (
    KLINE_ROOT,
    V173Config,
    build_monthly_receiver_graph,
    build_surface_signals,
    hourly_log_returns,
    load_v173_prices,
)
from pressure_graph.reports.v175_deribit_skew_receiver_insulator_spread import (
    CANDIDATES,
    RELIEF_CANDIDATE,
    V175Config,
    _random_controls,
    audit_v175,
    build_v175_events,
    summarize_v175,
)


SURFACE_PATH = Path("data/external/deribit_monthly_option_trades/daily_trade_surface.parquet")
REPORT_ROOT = Path("reports/v17_6_monthly_deribit_surface_extension")
FINDINGS_PATH = Path("docs/v176_monthly_deribit_surface_extension_findings_2026_07_16.md")


def select_nearest_30d_surface(surface: pd.DataFrame) -> pd.DataFrame:
    frame = surface[surface["quality_pass"].eq(True)].copy()
    frame["feature_time"] = pd.to_datetime(
        frame["feature_time"], utc=True, errors="coerce"
    )
    frame["expiration_time"] = pd.to_datetime(
        frame["expiration_time"], utc=True, errors="coerce"
    )
    frame["distance_to_30d"] = (pd.to_numeric(frame["dte"], errors="coerce") - 30).abs()
    return (
        frame.sort_values(["feature_time", "distance_to_30d", "expiration_time"])
        .drop_duplicates("feature_time", keep="first")
        .sort_values("feature_time")
        .reset_index(drop=True)
    )


def _write_findings(
    outcome: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
) -> None:
    verdict = (
        "research_lead_forward_watch"
        if bool(outcome["forward_watch"].any())
        else "reject_monthly_surface_extension"
    )
    text = [
        "# v17.6 Monthly Deribit Surface Extension Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        outcome.to_markdown(index=False, floatfmt=".4f"),
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "This is an overlapping historical coverage extension motivated by v17.5.",
        "Even a pass can create only a forward-watch research lead, never a candidate.",
        "The signal, graph, spread, horizon, and costs are unchanged. No live permission",
        "changes are authorized.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v176_monthly_deribit_surface_extension(
    surface_path: Path = SURFACE_PATH,
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    signal_cfg: V173Config = V173Config(),
    cfg: V175Config = V175Config(seed=17_600),
) -> dict[str, Path]:
    raw_surface = pd.read_parquet(surface_path)
    surface = select_nearest_30d_surface(raw_surface)
    prices = load_v173_prices(kline_root)
    returns = hourly_log_returns(prices)
    graph = build_monthly_receiver_graph(
        returns,
        surface["feature_time"].min(),
        surface["feature_time"].max(),
        signal_cfg,
    )
    signals = build_surface_signals(surface, signal_cfg)
    events = build_v175_events(signals, graph, prices, cfg)
    summary = summarize_v175(events)

    delayed_signals = build_surface_signals(surface, signal_cfg, shift_days=1)
    delayed_events = build_v175_events(delayed_signals, graph, prices, cfg)
    delayed_summary = summarize_v175(delayed_events)

    sensitivity_frames: list[pd.DataFrame] = []
    for holding_hours in (8, 48):
        local_events = build_v175_events(
            signals, graph, prices, cfg, holding_hours=holding_hours
        )
        local_summary = summarize_v175(local_events)
        local_summary["holding_hours"] = holding_hours
        sensitivity_frames.append(local_summary)
    sensitivity = pd.concat(sensitivity_frames, ignore_index=True)
    random_controls = _random_controls(signals, graph, prices, cfg)
    gates, original_outcome = audit_v175(
        events,
        summary,
        delayed_summary,
        sensitivity,
        random_controls,
        cfg,
    )
    strict_rows: list[dict[str, object]] = []
    outcomes: list[dict[str, object]] = []
    count_requirements = {"all": 60, "validation": 12, "holdout": 20}
    for candidate in CANDIDATES:
        candidate_summary = summary[summary["candidate"].eq(candidate)].set_index("scope")
        strict_pass = True
        for scope, required in count_requirements.items():
            value = int(candidate_summary.loc[scope, "events"])
            passed = value >= required
            strict_pass &= passed
            strict_rows.append(
                {
                    "candidate": candidate,
                    "check": f"extension_{scope}_events_{required}",
                    "passed": passed,
                    "value": float(value),
                    "eligible": False,
                }
            )
        original = original_outcome[original_outcome["candidate"].eq(candidate)].iloc[0]
        original_gate_pass = bool(original["eligible"])
        forward_watch = bool(original_gate_pass and strict_pass)
        payload = original.to_dict()
        payload["original_gate_pass"] = original_gate_pass
        payload["strict_count_pass"] = strict_pass
        payload["forward_watch"] = forward_watch
        payload["eligible"] = False
        payload["verdict"] = (
            "research_lead_forward_watch"
            if forward_watch
            else "reject_monthly_surface_extension"
        )
        outcomes.append(payload)
    gates = pd.concat([gates, pd.DataFrame(strict_rows)], ignore_index=True)
    gates["eligible"] = False
    outcome = pd.DataFrame(outcomes)
    outcome["lead_of_interest"] = outcome["candidate"].eq(RELIEF_CANDIDATE)
    root = ensure_dir(report_root)
    paths = {
        "selected_surface": root / "nearest_30d_daily_surface.parquet",
        "signals": root / "surface_signals.parquet",
        "events": root / "candidate_events.parquet",
        "summary": root / "period_summary.csv",
        "delayed_events": root / "delayed_candidate_events.parquet",
        "delayed_summary": root / "delayed_period_summary.csv",
        "sensitivity": root / "holding_sensitivity.csv",
        "random_controls": root / "random_bucket_pair_controls.parquet",
        "gates": root / "robustness_gates.csv",
        "outcome": root / "robustness_outcome.csv",
        "findings": findings_path,
    }
    surface.to_parquet(paths["selected_surface"], index=False)
    signals.to_parquet(paths["signals"], index=False)
    events.to_parquet(paths["events"], index=False)
    summary.to_csv(paths["summary"], index=False)
    delayed_events.to_parquet(paths["delayed_events"], index=False)
    delayed_summary.to_csv(paths["delayed_summary"], index=False)
    sensitivity.to_csv(paths["sensitivity"], index=False)
    random_controls.to_parquet(paths["random_controls"], index=False)
    gates.to_csv(paths["gates"], index=False)
    outcome.to_csv(paths["outcome"], index=False)
    _write_findings(outcome, summary, findings_path)
    return paths


__all__ = [
    "select_nearest_30d_surface",
    "write_v176_monthly_deribit_surface_extension",
]
