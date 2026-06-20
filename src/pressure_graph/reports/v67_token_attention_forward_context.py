"""v6.7 Token Attention Forward Context Ledger.

This report promotes the v6.6 token/pool attention finding into a forward
evaluation ledger.  It records token-attention context for P2/CIC/O6 trades
and produces counterfactual diagnostics only.  It does not create a selector,
gate, sizing rule, shadow portfolio, or live permission.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v66_token_attention_attribution import (
    V66Config,
    _attach_prior_flags,
    _control_distribution,
    _fusion_summary,
    _leave_one_month,
    _modules,
    _prepare_p2_trades,
    _read_mapping,
    _read_market_days,
    _read_token_events,
)


REPORT_ROOT = Path("reports/v6_7_token_attention_forward_context")
DEFAULT_V66_REPORT_ROOT = Path("reports/v6_6_token_attention_attribution")


@dataclass(frozen=True)
class V67Config:
    report_root: Path = REPORT_ROOT
    v66_report_root: Path = DEFAULT_V66_REPORT_ROOT
    v66: V66Config = V66Config()
    main_prior_window: str = "24h"
    min_p2_prior_trades: int = 100
    min_o6_prior_trades: int = 30
    random_p75_threshold: float = 0.75
    random_p90_threshold: float = 0.90


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce")


def _bool(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    series = frame[col]
    if series.dtype == object:
        return series.astype(str).str.lower().isin(["true", "1", "yes"])
    return series.fillna(False).astype(bool)


def _parse_window(value: str) -> pd.Timedelta:
    return pd.Timedelta(value)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _join_unique(values: pd.Series, limit: int = 5) -> str:
    unique = [str(value) for value in values.dropna().astype(str).unique() if str(value)]
    return "|".join(unique[:limit])


def _module_tags(row: pd.Series) -> str:
    tags = ["P2_all"]
    if bool(row.get("is_cic1", False)):
        tags.append("CIC1")
    if bool(row.get("is_cic2", False)):
        tags.append("CIC2")
    if bool(row.get("is_o6", False)):
        tags.append("O6_late9")
    return "|".join(tags)


def _events_by_symbol(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if events.empty:
        return {}
    return {
        str(symbol): group.sort_values("event_available_time").reset_index(drop=True)
        for symbol, group in events.groupby("cex_symbol", sort=False)
    }


def _window_event_detail(
    symbol: str,
    entry_time: pd.Timestamp,
    events_by_symbol: dict[str, pd.DataFrame],
    window: str,
) -> dict[str, Any]:
    local = events_by_symbol.get(symbol, pd.DataFrame())
    if local.empty or pd.isna(entry_time):
        return {
            f"token_event_types_{window}": "",
            f"token_event_sources_{window}": "",
            f"token_event_max_zscore_{window}": np.nan,
            f"token_event_max_percentile_{window}": np.nan,
            f"token_event_latest_available_time_{window}": "",
            f"token_event_age_minutes_{window}": np.nan,
        }
    start = entry_time - _parse_window(window)
    prior = local[local["event_available_time"].le(entry_time) & local["event_available_time"].gt(start)]
    if prior.empty:
        return {
            f"token_event_types_{window}": "",
            f"token_event_sources_{window}": "",
            f"token_event_max_zscore_{window}": np.nan,
            f"token_event_max_percentile_{window}": np.nan,
            f"token_event_latest_available_time_{window}": "",
            f"token_event_age_minutes_{window}": np.nan,
        }
    latest = pd.to_datetime(prior["event_available_time"], utc=True, errors="coerce").max()
    return {
        f"token_event_types_{window}": _join_unique(prior.get("event_type", pd.Series(dtype=str))),
        f"token_event_sources_{window}": _join_unique(prior.get("source", pd.Series(dtype=str))),
        f"token_event_max_zscore_{window}": float(_num(prior, "zscore").max()) if _num(prior, "zscore").notna().any() else np.nan,
        f"token_event_max_percentile_{window}": float(_num(prior, "percentile").max()) if _num(prior, "percentile").notna().any() else np.nan,
        f"token_event_latest_available_time_{window}": latest.isoformat() if pd.notna(latest) else "",
        f"token_event_age_minutes_{window}": float((entry_time - latest).total_seconds() / 60.0) if pd.notna(latest) else np.nan,
    }


def _context_ledger(
    scored: pd.DataFrame,
    events: pd.DataFrame,
    market_days: set[pd.Timestamp],
    cfg: V67Config,
) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    event_groups = _events_by_symbol(events)
    out = scored.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    out["entry_day"] = out["entry_time"].dt.floor("D")
    out["market_attention_day"] = out["entry_day"].isin(market_days)
    out["module_tags"] = out.apply(_module_tags, axis=1)
    out["net20_later"] = _num(out, "net20")
    out["live_action_allowed"] = False
    out["recommended_use"] = "forward_counterfactual_diagnostic_only"

    for window in cfg.v66.prior_windows:
        prior_col = f"token_prior_{window}"
        out[f"token_attention_context_{window}"] = np.where(
            _bool(out, prior_col),
            "token_prior",
            "no_token_prior",
        )
        details: list[dict[str, Any]] = []
        for row in out[["symbol", "entry_time"]].itertuples(index=False):
            details.append(_window_event_detail(str(row.symbol), pd.Timestamp(row.entry_time), event_groups, window))
        detail_frame = pd.DataFrame(details, index=out.index)
        out = pd.concat([out, detail_frame], axis=1)

    keep = [
        "trade_id",
        "signal_id",
        "symbol",
        "candidate",
        "module_tags",
        "entry_time",
        "signal_time",
        "month",
        "net20_later",
        "market_attention_day",
        "burst_count_so_far",
        "same_timestamp_peer_count",
        "volume_impulse_density",
        "cluster_impulse_density",
        "is_cic1",
        "is_cic2",
        "is_o6",
        "live_action_allowed",
        "recommended_use",
    ]
    for window in cfg.v66.prior_windows:
        keep.extend(
            [
                f"token_prior_{window}",
                f"token_prior_{window}_count",
                f"token_attention_context_{window}",
                f"token_event_types_{window}",
                f"token_event_sources_{window}",
                f"token_event_max_zscore_{window}",
                f"token_event_max_percentile_{window}",
                f"token_event_latest_available_time_{window}",
                f"token_event_age_minutes_{window}",
            ]
        )
    return out[[col for col in keep if col in out.columns]].copy()


def build_token_attention_context_for_trades(
    trades: pd.DataFrame,
    cfg: V67Config | None = None,
) -> pd.DataFrame:
    """Build as-of token-attention context for an arbitrary trade ledger.

    This is used by paper-live reporting to append diagnostic fields to future
    samples.  The returned frame is counterfactual/logging-only; it never marks
    a trade as actionable.
    """
    cfg = cfg or V67Config()
    if trades.empty:
        return pd.DataFrame()
    v66 = cfg.v66
    mapping = _read_mapping(v66.token_mapping_path)
    mapped_symbols = set(mapping["cex_symbol"].astype(str)) if not mapping.empty else set()
    events = _read_token_events(v66.token_events_path, mapped_symbols)
    market_days = _read_market_days(v66.market_attention_days_path)

    local = trades.copy()
    local["symbol"] = local.get("symbol", pd.Series("", index=local.index)).fillna("").astype(str).str.upper()
    local = local[local["symbol"].isin(mapped_symbols)].copy()
    if local.empty:
        return pd.DataFrame()
    local["entry_time"] = pd.to_datetime(local.get("entry_time"), utc=True, errors="coerce")
    local["signal_time"] = pd.to_datetime(local.get("signal_time"), utc=True, errors="coerce")
    local["candidate"] = local.get("candidate", pd.Series("", index=local.index)).fillna("").astype(str)
    if "trade_id" not in local.columns:
        local["trade_id"] = (
            local["symbol"].astype(str)
            + "|"
            + local["candidate"].astype(str)
            + "|"
            + local["entry_time"].astype(str)
            + "|"
            + local.index.astype(str)
        )
    if "signal_id" not in local.columns:
        local["signal_id"] = local["trade_id"].astype(str)
    if "net20" not in local.columns:
        if "net_return_20bp" in local.columns:
            local["net20"] = _num(local, "net_return_20bp")
        elif "net_return" in local.columns:
            local["net20"] = _num(local, "net_return")
        else:
            local["net20"] = np.nan
    local["month"] = local["entry_time"].dt.strftime("%Y-%m")
    candidate_text = local["candidate"].str.upper()
    local["is_cic1"] = candidate_text.str.contains("CIC1", regex=False)
    local["is_cic2"] = candidate_text.str.contains("CIC2", regex=False)
    if "burst_count_so_far" not in local.columns:
        local["burst_count_so_far"] = np.nan
    local["is_o6"] = _num(local, "burst_count_so_far").ge(9)
    local = local.dropna(subset=["entry_time"]).reset_index(drop=True)
    scored = _attach_prior_flags(local, events, v66.prior_windows, prefix="token")
    return _context_ledger(scored, events, market_days, cfg)


def _control_features(controls: pd.DataFrame) -> pd.DataFrame:
    if controls.empty:
        return pd.DataFrame()
    rows = []
    for (module, window), group in controls.groupby(["module", "lookback_window"], sort=False):
        actual = _num(group, "actual_prior_net20")
        p75 = _num(group, "random_p75_net20")
        p90 = _num(group, "random_p90_net20")
        percentile = _num(group, "actual_percentile")
        rows.append(
            {
                "module": module,
                "lookback_window": window,
                "min_random_percentile": float(percentile.min()) if percentile.notna().any() else np.nan,
                "max_random_percentile": float(percentile.max()) if percentile.notna().any() else np.nan,
                "beats_all_random_p75": bool((actual > p75).all()) if len(group) else False,
                "beats_all_random_p90": bool((actual > p90).all()) if len(group) else False,
            }
        )
    return pd.DataFrame(rows)


def _leave_one_features(leave_one: pd.DataFrame) -> pd.DataFrame:
    if leave_one.empty:
        return pd.DataFrame()
    rows = []
    for module, group in leave_one.groupby("module", sort=False):
        lift = _num(group, "prior_minus_no_prior_net20")
        rows.append(
            {
                "module": module,
                "leave_one_month_min_lift": float(lift.min()) if lift.notna().any() else np.nan,
                "leave_one_month_all_positive": bool(lift.gt(0).all()) if len(group) else False,
            }
        )
    return pd.DataFrame(rows)


def _context_status(row: pd.Series, cfg: V67Config) -> str:
    module = str(row.get("module", ""))
    window = str(row.get("lookback_window", ""))
    prior_trades = int(row.get("prior_trades", 0) or 0)
    min_percentile = float(row.get("min_random_percentile", np.nan))
    month_lift_positive = bool(row.get("leave_one_month_all_positive", False))
    month_cap = float(row.get("prior_month_cap35_net20", np.nan))
    if window != cfg.main_prior_window:
        return "diagnostic_secondary_window"
    if (
        module == "CIC1"
        and prior_trades >= cfg.min_p2_prior_trades
        and min_percentile >= cfg.random_p90_threshold
        and month_lift_positive
        and month_cap > 0
    ):
        return "forward_counterfactual_candidate"
    if module == "P2_all" and prior_trades >= cfg.min_p2_prior_trades and min_percentile >= cfg.random_p75_threshold:
        return "forward_context_diagnostic"
    if module == "O6_late9" and prior_trades >= cfg.min_o6_prior_trades and min_percentile >= cfg.random_p75_threshold:
        return "overflow_context_diagnostic"
    return "diagnostic_only"


def _context_summary(
    scored: pd.DataFrame,
    controls: pd.DataFrame,
    leave_one: pd.DataFrame,
    cfg: V67Config,
) -> pd.DataFrame:
    fusion = _fusion_summary(scored, cfg.v66)
    if fusion.empty:
        return pd.DataFrame()
    summary = fusion.rename(columns={"prior_minus_no_prior_net20": "lift_vs_no_prior"}).copy()
    control_features = _control_features(controls)
    if not control_features.empty:
        summary = summary.merge(control_features, on=["module", "lookback_window"], how="left")
    leave_features = _leave_one_features(leave_one)
    if not leave_features.empty:
        summary = summary.merge(leave_features, on="module", how="left")
    for col in ("min_random_percentile", "max_random_percentile", "leave_one_month_min_lift"):
        if col not in summary.columns:
            summary[col] = np.nan
    for col in ("beats_all_random_p75", "beats_all_random_p90", "leave_one_month_all_positive"):
        if col not in summary.columns:
            summary[col] = False
    summary["context_status"] = summary.apply(_context_status, axis=1, cfg=cfg)
    return summary


def _decision_table(summary: pd.DataFrame, cfg: V67Config) -> pd.DataFrame:
    columns = [
        "context_id",
        "module",
        "lookback_window",
        "decision",
        "live_action_allowed",
        "shadow_portfolio_allowed",
        "reason",
    ]
    if summary.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    main = summary[summary["lookback_window"].astype(str).eq(cfg.main_prior_window)].copy()
    for row in main.to_dict("records"):
        status = str(row.get("context_status", "diagnostic_only"))
        module = str(row.get("module", ""))
        if status == "forward_counterfactual_candidate":
            decision = "ENABLE_FORWARD_COUNTERFACTUAL_LOGGING"
            reason = "passes random controls, leave-one-month, month-cap, and sample floor; no trade action"
        elif status in {"forward_context_diagnostic", "overflow_context_diagnostic"}:
            decision = "KEEP_DIAGNOSTIC_CONTEXT"
            reason = "positive context but random-control cleanliness is not enough for a rule"
        else:
            decision = "DIAGNOSTIC_ONLY"
            reason = "insufficient attribution cleanliness or sample stability"
        rows.append(
            {
                "context_id": f"{module}_token_prior_{cfg.main_prior_window}",
                "module": module,
                "lookback_window": cfg.main_prior_window,
                "decision": decision,
                "live_action_allowed": False,
                "shadow_portfolio_allowed": False,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _live_field_spec(cfg: V67Config) -> pd.DataFrame:
    rows = [
        {
            "field": "token_attention_context_version",
            "type": "string",
            "asof_requirement": "constant config",
            "description": "Use v6.7 context fields for forward diagnostics only.",
        },
        {
            "field": "token_attention_live_action_allowed",
            "type": "bool",
            "asof_requirement": "constant false",
            "description": "Must remain false until a future promotion audit changes permissions.",
        },
    ]
    for window in cfg.v66.prior_windows:
        rows.extend(
            [
                {
                    "field": f"token_prior_{window}",
                    "type": "bool",
                    "asof_requirement": "event_available_time <= entry_decision_time",
                    "description": f"Whether same-symbol token/pool attention event is visible within prior {window}.",
                },
                {
                    "field": f"token_prior_{window}_count",
                    "type": "int",
                    "asof_requirement": "event_available_time <= entry_decision_time",
                    "description": f"Count of visible same-symbol token/pool events within prior {window}.",
                },
                {
                    "field": f"token_event_types_{window}",
                    "type": "string",
                    "asof_requirement": "event_available_time <= entry_decision_time",
                    "description": "Pipe-delimited event types for audit and future stratification.",
                },
                {
                    "field": f"token_event_sources_{window}",
                    "type": "string",
                    "asof_requirement": "event_available_time <= entry_decision_time",
                    "description": "Pipe-delimited source systems for provenance checks.",
                },
                {
                    "field": f"token_event_age_minutes_{window}",
                    "type": "float",
                    "asof_requirement": "event_available_time <= entry_decision_time",
                    "description": "Age of the latest visible token attention event.",
                },
            ]
        )
    return pd.DataFrame(rows)


def _write_notes(
    path: Path,
    coverage: pd.DataFrame,
    summary: pd.DataFrame,
    decisions: pd.DataFrame,
    cfg: V67Config,
) -> None:
    lines = [
        "# v6.7 Token Attention Forward Context",
        "",
        "Status: forward/counterfactual logging only. No selector, gate, sizing rule, shadow portfolio, or real-live permission is changed.",
        "",
    ]
    if not coverage.empty:
        for row in coverage.to_dict("records"):
            lines.append(f"- {row['dataset']}: {row['rows']} ({row['status']}).")
    main = summary[summary["lookback_window"].astype(str).eq(cfg.main_prior_window)] if not summary.empty else pd.DataFrame()
    if not main.empty:
        lines.extend(["", "Main context decisions:"])
        for row in main.sort_values("module").to_dict("records"):
            lines.append(
                f"- {row['module']}: prior={int(row['prior_trades'])} net20={row['prior_net20']:.4%}, "
                f"without={int(row['no_prior_trades'])} net20={row['no_prior_net20']:.4%}, "
                f"lift={row['lift_vs_no_prior']:.4%}, min_random_pct={row['min_random_percentile']:.2f}, "
                f"status={row['context_status']}."
            )
    if not decisions.empty:
        lines.extend(["", "Decision guardrail:"])
        for row in decisions.to_dict("records"):
            lines.append(f"- {row['context_id']}: {row['decision']}; live_action_allowed=false.")
    lines.extend(
        [
            "",
            "Forward use:",
            "- Log token-prior fields beside future P2/CIC/O6 candidates.",
            "- Evaluate CIC1 token-prior as the first forward counterfactual context.",
            "- Do not skip, size, rank, enable O6, disable CP60, or protect CP60 exits from this context yet.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_v67_token_attention_forward_context(cfg: V67Config | None = None) -> dict[str, Path]:
    cfg = cfg or V67Config()
    report_root = ensure_dir(cfg.report_root)
    v66 = cfg.v66
    mapping = _read_mapping(v66.token_mapping_path)
    mapped_symbols = set(mapping["cex_symbol"].astype(str)) if not mapping.empty else set()
    events = _read_token_events(v66.token_events_path, mapped_symbols)
    trades = _prepare_p2_trades(v66, mapped_symbols)
    scored = _attach_prior_flags(trades, events, v66.prior_windows, prefix="token")
    market_days = _read_market_days(v66.market_attention_days_path)
    controls = _read_csv(cfg.v66_report_root / "random_control_summary.csv")
    if controls.empty:
        controls = _control_distribution(scored, events, mapping, v66)
    leave_one = _read_csv(cfg.v66_report_root / "leave_one_month.csv")
    if leave_one.empty:
        leave_one = _leave_one_month(scored, v66)
    ledger = _context_ledger(scored, events, market_days, cfg)
    summary = _context_summary(scored, controls, leave_one, cfg)
    decisions = _decision_table(summary, cfg)
    live_spec = _live_field_spec(cfg)
    coverage = pd.DataFrame(
        [
            {"dataset": "mapped_A_B_symbols", "rows": int(mapping["cex_symbol"].nunique()) if not mapping.empty else 0, "status": "reference"},
            {"dataset": "token_events", "rows": int(len(events)), "status": "ok" if len(events) else "missing"},
            {"dataset": "forward_context_trades", "rows": int(len(ledger)), "status": "ok" if len(ledger) else "missing"},
            {
                "dataset": f"token_prior_{cfg.main_prior_window}_trades",
                "rows": int(_bool(ledger, f"token_prior_{cfg.main_prior_window}").sum()) if not ledger.empty else 0,
                "status": "coverage",
            },
        ]
    )

    outputs = {
        "forward_context_ledger": report_root / "token_attention_forward_context_ledger.csv",
        "context_summary": report_root / "token_attention_context_summary.csv",
        "decision_table": report_root / "token_attention_decision_table.csv",
        "random_control_reference": report_root / "token_attention_random_control_reference.csv",
        "leave_one_month_reference": report_root / "token_attention_leave_one_month_reference.csv",
        "live_field_spec": report_root / "token_attention_live_field_spec.csv",
        "coverage": report_root / "token_attention_forward_coverage.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    ledger.to_csv(outputs["forward_context_ledger"], index=False)
    summary.to_csv(outputs["context_summary"], index=False)
    decisions.to_csv(outputs["decision_table"], index=False)
    controls.to_csv(outputs["random_control_reference"], index=False)
    leave_one.to_csv(outputs["leave_one_month_reference"], index=False)
    live_spec.to_csv(outputs["live_field_spec"], index=False)
    coverage.to_csv(outputs["coverage"], index=False)
    _write_notes(outputs["candidate_notes"], coverage, summary, decisions, cfg)
    return outputs
