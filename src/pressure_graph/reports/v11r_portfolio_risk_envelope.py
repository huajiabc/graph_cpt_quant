from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v09b import _max_contribution, _month_cap_expectancy, _prepare_trade_features
from pressure_graph.reports.v09d import _period_hours
from pressure_graph.reports.v10a_cic_basket_portfolio import V10AConfig, _load_or_build_trades
from pressure_graph.reports.v10b_slot_turnover_attribution import _focus_pool
from pressure_graph.reports.v10c_burst_phase_allocation import _add_asof_burst_phase


REPORT_ROOT = Path("reports/v1_1r_portfolio_risk_envelope")
CORE_MAX_POSITIONS = 8
O6_MIN_BURST_COUNT = 9
O6_CIC1_SIZE = 0.50
O6_CIC2_SIZE = 0.25
TOTAL_EXPOSURE_CAPS = (8.0, 9.0, 10.0, 12.0)
OVERFLOW_SLOT_CAPS = (1, 2, 4)
DAILY_NEW_EXPOSURE_CAPS = (4.0, 6.0, 8.0, 10.0)
ROLLING_4H_NEW_EXPOSURE_CAPS = (4.0, 6.0, 8.0, 10.0)


@dataclass(frozen=True)
class RiskPolicy:
    policy_id: str
    policy_type: str
    overflow_enabled: bool
    overflow_max_slots: int = 0
    total_exposure_cap: float | None = None
    daily_new_exposure_cap: float | None = None
    rolling_4h_new_exposure_cap: float | None = None
    cic1_size: float = O6_CIC1_SIZE
    cic2_size: float = O6_CIC2_SIZE
    notes: str = ""


@dataclass(frozen=True)
class V11RConfig:
    report_root: Path = REPORT_ROOT
    v10a: V10AConfig = V10AConfig()


def _overflow_size(row: pd.Series | dict[str, Any], policy: RiskPolicy) -> float:
    if not policy.overflow_enabled:
        return 0.0
    candidate = str(row.get("candidate", ""))
    if candidate == "CIC1_beta_extreme":
        return float(policy.cic1_size)
    if candidate == "CIC2_beta_broad":
        return float(policy.cic2_size)
    return 0.0


def _overflow_allowed(row: pd.Series | dict[str, Any], policy: RiskPolicy) -> bool:
    if not policy.overflow_enabled or policy.overflow_max_slots <= 0:
        return False
    if int(row.get("burst_count_so_far", 0)) < O6_MIN_BURST_COUNT:
        return False
    return _overflow_size(row, policy) > 0


def _ledger_row(
    row: pd.Series,
    *,
    sleeve: str,
    weight: float,
    selection_status: str,
    skip_reason: str = "",
    active_exposure_at_decision: float = 0.0,
    daily_new_exposure_at_decision: float = 0.0,
    rolling_4h_new_exposure_at_decision: float = 0.0,
) -> dict[str, Any]:
    payload = row.to_dict()
    net = float(pd.to_numeric(row.get("net_return", np.nan), errors="coerce"))
    payload["sleeve"] = sleeve
    payload["exposure_weight"] = float(weight)
    payload["selection_status"] = selection_status
    payload["skip_reason"] = skip_reason
    payload["weighted_return"] = net * float(weight)
    payload["active_exposure_at_decision"] = float(active_exposure_at_decision)
    payload["daily_new_exposure_at_decision"] = float(daily_new_exposure_at_decision)
    payload["rolling_4h_new_exposure_at_decision"] = float(rolling_4h_new_exposure_at_decision)
    return payload


def _current_exposure(active: list[dict[str, Any]]) -> float:
    return float(sum(float(item["weight"]) for item in active))


def _rolling_new_exposure(entries: list[tuple[pd.Timestamp, float]], entry: pd.Timestamp, hours: int = 4) -> float:
    start = entry - pd.Timedelta(hours=hours)
    return float(sum(weight for ts, weight in entries if start < ts <= entry))


def _daily_new_exposure(entries: list[tuple[pd.Timestamp, float]], entry: pd.Timestamp) -> float:
    day = entry.date()
    return float(sum(weight for ts, weight in entries if ts.date() == day))


def _cap_allows(
    *,
    weight: float,
    active: list[dict[str, Any]],
    entries: list[tuple[pd.Timestamp, float]],
    entry: pd.Timestamp,
    policy: RiskPolicy,
) -> tuple[bool, str, float, float, float]:
    active_exposure = _current_exposure(active)
    daily_exposure = _daily_new_exposure(entries, entry)
    rolling_exposure = _rolling_new_exposure(entries, entry, 4)
    eps = 1e-12
    if policy.total_exposure_cap is not None and active_exposure + weight > float(policy.total_exposure_cap) + eps:
        return False, "total_exposure_cap", active_exposure, daily_exposure, rolling_exposure
    if policy.daily_new_exposure_cap is not None and daily_exposure + weight > float(policy.daily_new_exposure_cap) + eps:
        return False, "daily_new_exposure_cap", active_exposure, daily_exposure, rolling_exposure
    if policy.rolling_4h_new_exposure_cap is not None and rolling_exposure + weight > float(policy.rolling_4h_new_exposure_cap) + eps:
        return False, "rolling_4h_new_exposure_cap", active_exposure, daily_exposure, rolling_exposure
    return True, "", active_exposure, daily_exposure, rolling_exposure


def simulate_risk_policy(pool: pd.DataFrame, policy: RiskPolicy) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_core: list[dict[str, Any]] = []
    active_overflow: list[dict[str, Any]] = []
    selected_entries: list[tuple[pd.Timestamp, float]] = []
    ledger_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for _, row in pool.sort_values(["entry_time", "symbol"]).iterrows():
        entry = pd.Timestamp(row["entry_time"])
        active_core = [item for item in active_core if pd.Timestamp(item["exit_time"]) > entry]
        active_overflow = [item for item in active_overflow if pd.Timestamp(item["exit_time"]) > entry]
        active = [*active_core, *active_overflow]
        active_symbols = {str(item["symbol"]) for item in active}
        if str(row["symbol"]) in active_symbols:
            skipped_rows.append(_ledger_row(row, sleeve="skipped", weight=0.0, selection_status="skipped", skip_reason="symbol_already_active"))
            continue
        if len(active_core) < CORE_MAX_POSITIONS:
            allowed, reason, active_exp, daily_exp, rolling_exp = _cap_allows(
                weight=1.0,
                active=active,
                entries=selected_entries,
                entry=entry,
                policy=policy,
            )
            if not allowed:
                skipped_rows.append(
                    _ledger_row(
                        row,
                        sleeve="skipped",
                        weight=0.0,
                        selection_status="skipped",
                        skip_reason=reason,
                        active_exposure_at_decision=active_exp,
                        daily_new_exposure_at_decision=daily_exp,
                        rolling_4h_new_exposure_at_decision=rolling_exp,
                    )
                )
                continue
            ledger_rows.append(
                _ledger_row(
                    row,
                    sleeve="core",
                    weight=1.0,
                    selection_status="selected",
                    active_exposure_at_decision=active_exp,
                    daily_new_exposure_at_decision=daily_exp,
                    rolling_4h_new_exposure_at_decision=rolling_exp,
                )
            )
            active_core.append({"symbol": str(row["symbol"]), "exit_time": row["exit_time"], "weight": 1.0})
            selected_entries.append((entry, 1.0))
            continue
        if _overflow_allowed(row, policy) and len(active_overflow) < policy.overflow_max_slots:
            size = _overflow_size(row, policy)
            allowed, reason, active_exp, daily_exp, rolling_exp = _cap_allows(
                weight=size,
                active=active,
                entries=selected_entries,
                entry=entry,
                policy=policy,
            )
            if not allowed:
                skipped_rows.append(
                    _ledger_row(
                        row,
                        sleeve="skipped",
                        weight=0.0,
                        selection_status="skipped",
                        skip_reason=reason,
                        active_exposure_at_decision=active_exp,
                        daily_new_exposure_at_decision=daily_exp,
                        rolling_4h_new_exposure_at_decision=rolling_exp,
                    )
                )
                continue
            ledger_rows.append(
                _ledger_row(
                    row,
                    sleeve="overflow",
                    weight=size,
                    selection_status="selected",
                    active_exposure_at_decision=active_exp,
                    daily_new_exposure_at_decision=daily_exp,
                    rolling_4h_new_exposure_at_decision=rolling_exp,
                )
            )
            active_overflow.append({"symbol": str(row["symbol"]), "exit_time": row["exit_time"], "weight": size})
            selected_entries.append((entry, size))
            continue
        reason = "overflow_full" if _overflow_allowed(row, policy) else "portfolio_full_not_overflow_eligible"
        skipped_rows.append(_ledger_row(row, sleeve="skipped", weight=0.0, selection_status="skipped", skip_reason=reason))
    return pd.DataFrame(ledger_rows), pd.DataFrame(skipped_rows)


def _exposure_stats(ledger: pd.DataFrame) -> dict[str, float]:
    if ledger.empty:
        return {
            "avg_exposure_units": np.nan,
            "max_total_exposure": 0.0,
            "max_concurrent_positions": 0.0,
            "max_core_positions": 0.0,
            "max_overflow_exposure": 0.0,
            "period_hours": np.nan,
        }
    events: list[tuple[pd.Timestamp, int, float, str]] = []
    holding = 0.0
    for row in ledger.itertuples(index=False):
        entry = pd.Timestamp(getattr(row, "entry_time"))
        exit_time = pd.Timestamp(getattr(row, "exit_time"))
        weight = float(getattr(row, "exposure_weight"))
        sleeve = str(getattr(row, "sleeve"))
        events.append((entry, 1, weight, sleeve))
        events.append((exit_time, -1, -weight, sleeve))
        holding += max(float((exit_time - entry).total_seconds() / 3600.0), 0.0) * weight
    total_exposure = 0.0
    concurrent = 0
    core_positions = 0
    overflow_exposure = 0.0
    max_total_exposure = 0.0
    max_concurrent = 0
    max_core = 0
    max_overflow = 0.0
    for _, count_delta, weight_delta, sleeve in sorted(events, key=lambda item: (item[0], item[1])):
        total_exposure += weight_delta
        concurrent += count_delta
        if sleeve == "core":
            core_positions += count_delta
        if sleeve == "overflow":
            overflow_exposure += weight_delta
        max_total_exposure = max(max_total_exposure, total_exposure)
        max_concurrent = max(max_concurrent, concurrent)
        max_core = max(max_core, core_positions)
        max_overflow = max(max_overflow, overflow_exposure)
    period_hours = _period_hours(ledger)
    return {
        "avg_exposure_units": float(holding / period_hours) if period_hours else np.nan,
        "max_total_exposure": float(max_total_exposure),
        "max_concurrent_positions": float(max_concurrent),
        "max_core_positions": float(max_core),
        "max_overflow_exposure": float(max_overflow),
        "period_hours": float(period_hours) if period_hours else np.nan,
    }


def _period_return(ledger: pd.DataFrame, period: str) -> pd.Series:
    if ledger.empty:
        return pd.Series(dtype=float)
    data = ledger.copy()
    data["entry_time"] = pd.to_datetime(data["entry_time"], utc=True, errors="coerce")
    if period == "day":
        key = data["entry_time"].dt.strftime("%Y-%m-%d")
    elif period == "week":
        key = data["entry_time"].dt.tz_localize(None).dt.to_period("W").astype(str)
    elif period == "month":
        key = data["entry_time"].dt.strftime("%Y-%m")
    elif period == "burst":
        key = data.get("burst_id", pd.Series("unknown", index=data.index)).astype(str)
    else:
        raise ValueError(f"unsupported period: {period}")
    weighted = pd.to_numeric(data.get("weighted_return", pd.Series(dtype=float)), errors="coerce")
    return weighted.groupby(key, sort=False, dropna=False).sum() / CORE_MAX_POSITIONS


def _drawdown(contribution: pd.Series) -> float:
    if contribution.empty:
        return np.nan
    equity = contribution.cumsum()
    drawdown = equity - equity.cummax()
    return float(drawdown.min()) if len(drawdown) else np.nan


def _risk_summary_row(policy: RiskPolicy, ledger: pd.DataFrame, skipped: pd.DataFrame) -> dict[str, Any]:
    core = ledger[ledger.get("sleeve", pd.Series(dtype=str)).astype(str).eq("core")] if not ledger.empty else ledger
    overflow = ledger[ledger.get("sleeve", pd.Series(dtype=str)).astype(str).eq("overflow")] if not ledger.empty else ledger
    weighted = pd.to_numeric(ledger.get("weighted_return", pd.Series(dtype=float)), errors="coerce")
    core_weighted = pd.to_numeric(core.get("weighted_return", pd.Series(dtype=float)), errors="coerce")
    overflow_weighted = pd.to_numeric(overflow.get("weighted_return", pd.Series(dtype=float)), errors="coerce")
    overflow_exposure = pd.to_numeric(overflow.get("exposure_weight", pd.Series(dtype=float)), errors="coerce").sum()
    skipped_net = pd.to_numeric(skipped.get("net_return", pd.Series(dtype=float)), errors="coerce")
    contribution = weighted / CORE_MAX_POSITIONS
    overflow_contribution = overflow_weighted / CORE_MAX_POSITIONS
    row = {
        "policy_id": policy.policy_id,
        "policy_type": policy.policy_type,
        "overflow_enabled": policy.overflow_enabled,
        "overflow_max_slots": policy.overflow_max_slots,
        "total_exposure_cap": policy.total_exposure_cap,
        "daily_new_exposure_cap": policy.daily_new_exposure_cap,
        "rolling_4h_new_exposure_cap": policy.rolling_4h_new_exposure_cap,
        "selected_trades": int(len(ledger)),
        "core_trades": int(len(core)),
        "overflow_trades": int(len(overflow)),
        "skipped_trades": int(len(skipped)),
        "core_net20": float(core_weighted.sum() / CORE_MAX_POSITIONS) if len(core_weighted) else 0.0,
        "overflow_net20": float(overflow_weighted.sum() / CORE_MAX_POSITIONS) if len(overflow_weighted) else 0.0,
        "combined_net20": float(weighted.sum() / CORE_MAX_POSITIONS) if len(weighted) else 0.0,
        "skipped_net20": float(skipped_net.mean()) if len(skipped_net) else np.nan,
        "extra_exposure": float(overflow_exposure),
        "return_per_extra_exposure": float(overflow_weighted.sum() / overflow_exposure) if overflow_exposure else np.nan,
        "max_drawdown_proxy": _drawdown(contribution),
        "overflow_drawdown_contribution": _drawdown(overflow_contribution),
        "worst_burst_net20": float(_period_return(ledger, "burst").min()) if not ledger.empty else np.nan,
        "worst_day_net20": float(_period_return(ledger, "day").min()) if not ledger.empty else np.nan,
        "worst_week_net20": float(_period_return(ledger, "week").min()) if not ledger.empty else np.nan,
        "worst_month_net20": float(_period_return(ledger, "month").min()) if not ledger.empty else np.nan,
        "month_cap35_net20": _month_cap_expectancy(ledger.assign(net_return=ledger.get("weighted_return", np.nan))),
        "max_month_contribution": _max_contribution(ledger.assign(net_return=ledger.get("weighted_return", np.nan)), "month"),
        "max_symbol_contribution": _max_contribution(ledger.assign(net_return=ledger.get("weighted_return", np.nan)), "symbol"),
        "notes": policy.notes,
    }
    return {**row, **_exposure_stats(ledger)}


def _policies() -> list[RiskPolicy]:
    policies = [
        RiskPolicy(
            "core_p2_max8",
            "core_only",
            False,
            notes="P2 CIC1+CIC2 combined max8, no overflow.",
        ),
        RiskPolicy(
            "core_p2_max8_plus_o6",
            "core_plus_o6",
            True,
            overflow_max_slots=4,
            notes="Baseline P2 max8 plus O6 late-burst additive overflow.",
        ),
    ]
    for cap in TOTAL_EXPOSURE_CAPS:
        policies.append(
            RiskPolicy(
                f"o6_total_exposure_cap_{cap:g}",
                "total_exposure_cap",
                True,
                overflow_max_slots=4,
                total_exposure_cap=cap,
            )
        )
    for slots in OVERFLOW_SLOT_CAPS:
        policies.append(
            RiskPolicy(
                f"o6_overflow_slots_{slots}",
                "overflow_slot_cap",
                True,
                overflow_max_slots=slots,
            )
        )
    for cap in DAILY_NEW_EXPOSURE_CAPS:
        policies.append(
            RiskPolicy(
                f"o6_daily_new_exposure_cap_{cap:g}",
                "daily_new_exposure_cap",
                True,
                overflow_max_slots=4,
                daily_new_exposure_cap=cap,
            )
        )
    for cap in ROLLING_4H_NEW_EXPOSURE_CAPS:
        policies.append(
            RiskPolicy(
                f"o6_rolling_4h_new_exposure_cap_{cap:g}",
                "rolling_4h_new_exposure_cap",
                True,
                overflow_max_slots=4,
                rolling_4h_new_exposure_cap=cap,
            )
        )
    return policies


def _risk_reports(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    ledger_frames = []
    skipped_frames = []
    period_rows = []
    burst_rows = []
    for policy in _policies():
        ledger, skipped = simulate_risk_policy(pool, policy)
        summary_rows.append(_risk_summary_row(policy, ledger, skipped))
        if not ledger.empty:
            ledger_frames.append(ledger.assign(policy_id=policy.policy_id, policy_type=policy.policy_type))
            local = ledger.copy()
            local["entry_time"] = pd.to_datetime(local["entry_time"], utc=True, errors="coerce")
            for period in ["day", "week", "month"]:
                returns = _period_return(local, period)
                for key, value in returns.items():
                    period_rows.append({"policy_id": policy.policy_id, "period_type": period, "period": key, "net20": value})
            for burst_id, group in local.groupby("burst_id", sort=False, dropna=False):
                core = group[group["sleeve"].astype(str).eq("core")]
                overflow = group[group["sleeve"].astype(str).eq("overflow")]
                burst_rows.append(
                    {
                        "policy_id": policy.policy_id,
                        "burst_id": burst_id,
                        "burst_start": pd.to_datetime(group["entry_time"], utc=True, errors="coerce").min(),
                        "trades": int(len(group)),
                        "core_trades": int(len(core)),
                        "overflow_trades": int(len(overflow)),
                        "core_exposure": float(pd.to_numeric(core.get("exposure_weight", pd.Series(dtype=float)), errors="coerce").sum()),
                        "overflow_exposure": float(pd.to_numeric(overflow.get("exposure_weight", pd.Series(dtype=float)), errors="coerce").sum()),
                        "net20": float(pd.to_numeric(group["weighted_return"], errors="coerce").sum() / CORE_MAX_POSITIONS),
                        "overflow_net20": float(pd.to_numeric(overflow.get("weighted_return", pd.Series(dtype=float)), errors="coerce").sum() / CORE_MAX_POSITIONS),
                    }
                )
        if not skipped.empty:
            skipped_frames.append(skipped.assign(policy_id=policy.policy_id, policy_type=policy.policy_type))
    summary = pd.DataFrame(summary_rows)
    ledger_all = pd.concat(ledger_frames, ignore_index=True) if ledger_frames else pd.DataFrame()
    skipped_all = pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame()
    period = pd.DataFrame(period_rows)
    burst = pd.DataFrame(burst_rows)
    return summary, ledger_all, skipped_all, period, burst


def _overflow_risk_contribution(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    base = summary[summary["policy_id"].eq("core_p2_max8")]
    o6 = summary[summary["policy_id"].eq("core_p2_max8_plus_o6")]
    if base.empty or o6.empty:
        return pd.DataFrame()
    base_row = base.iloc[0]
    o6_row = o6.iloc[0]
    fields = [
        "combined_net20",
        "worst_burst_net20",
        "worst_day_net20",
        "worst_week_net20",
        "worst_month_net20",
        "max_drawdown_proxy",
        "avg_exposure_units",
        "max_total_exposure",
        "max_concurrent_positions",
    ]
    rows = []
    for field in fields:
        rows.append(
            {
                "metric": field,
                "core_only": base_row.get(field, np.nan),
                "core_plus_o6": o6_row.get(field, np.nan),
                "delta": o6_row.get(field, np.nan) - base_row.get(field, np.nan),
            }
        )
    rows.append(
        {
            "metric": "return_per_extra_exposure",
            "core_only": np.nan,
            "core_plus_o6": o6_row.get("return_per_extra_exposure", np.nan),
            "delta": np.nan,
        }
    )
    return pd.DataFrame(rows)


def _write_notes(report_root: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# v1.1R Portfolio Risk Envelope",
        "",
        "Purpose: define the risk envelope for P2 max8 core basket plus O6 late-burst additive overflow.",
        "This report does not change paper-live or real-live permissions.",
        "",
    ]
    if not summary.empty:
        base = summary[summary["policy_id"].eq("core_p2_max8")]
        o6 = summary[summary["policy_id"].eq("core_p2_max8_plus_o6")]
        if not base.empty and not o6.empty:
            base_row = base.iloc[0]
            o6_row = o6.iloc[0]
            lines.extend(
                [
                    "## Core vs O6",
                    f"- Core combined_net20={base_row.combined_net20:.4%}, worst_burst={base_row.worst_burst_net20:.4%}, "
                    f"worst_month={base_row.worst_month_net20:.4%}.",
                    f"- Core+O6 combined_net20={o6_row.combined_net20:.4%}, overflow_net20={o6_row.overflow_net20:.4%}, "
                    f"return_per_extra_exposure={o6_row.return_per_extra_exposure:.4%}.",
                    f"- Max total exposure moves from {base_row.max_total_exposure:.2f} to {o6_row.max_total_exposure:.2f}.",
                    "",
                ]
            )
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v11r_portfolio_risk_envelope(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V11RConfig = V11RConfig(),
) -> dict[str, Path]:
    report_root = ensure_dir(cfg.report_root)
    trades = _prepare_trade_features(_load_or_build_trades(feature_path, instruments, config, report_root, cfg.v10a))
    pool = _add_asof_burst_phase(_focus_pool(trades), "1h")
    if pool.empty:
        raise ValueError("No P2 CIC trades available for v1.1R risk envelope.")
    summary, ledger, skipped, period, burst = _risk_reports(pool)
    overflow_contribution = _overflow_risk_contribution(summary)
    outputs = {
        "risk_envelope_summary": report_root / "risk_envelope_summary.csv",
        "risk_trade_ledger": report_root / "risk_trade_ledger.csv",
        "risk_skipped_candidates": report_root / "risk_skipped_candidates.csv",
        "period_risk_summary": report_root / "period_risk_summary.csv",
        "burst_risk_summary": report_root / "burst_risk_summary.csv",
        "overflow_risk_contribution": report_root / "overflow_risk_contribution.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    summary.to_csv(outputs["risk_envelope_summary"], index=False)
    ledger.to_csv(outputs["risk_trade_ledger"], index=False)
    skipped.to_csv(outputs["risk_skipped_candidates"], index=False)
    period.to_csv(outputs["period_risk_summary"], index=False)
    burst.to_csv(outputs["burst_risk_summary"], index=False)
    overflow_contribution.to_csv(outputs["overflow_risk_contribution"], index=False)
    _write_notes(report_root, summary)
    return outputs


__all__ = [
    "REPORT_ROOT",
    "RiskPolicy",
    "V11RConfig",
    "simulate_risk_policy",
    "write_v11r_portfolio_risk_envelope",
]
