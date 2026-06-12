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


REPORT_ROOT = Path("reports/v1_0d_late_burst_overflow")
BASELINE_MAX_POSITIONS = 8


@dataclass(frozen=True)
class OverflowPolicy:
    policy_id: str
    min_burst_count: int
    overflow_max_slots: int
    cic1_size: float
    cic2_size: float
    cic1_only: bool = False


POLICIES = (
    OverflowPolicy("B0_baseline_max8_no_overflow", 999_999, 0, 0.0, 0.0),
    OverflowPolicy("O1_late9_slots2_size025", 9, 2, 0.25, 0.25),
    OverflowPolicy("O2_late9_slots2_size050", 9, 2, 0.50, 0.50),
    OverflowPolicy("O3_late9_slots4_size025", 9, 4, 0.25, 0.25),
    OverflowPolicy("O4_late15_slots2_size050", 15, 2, 0.50, 0.50),
    OverflowPolicy("O5_late9_cic1_slots2_size050", 9, 2, 0.50, 0.0, True),
    OverflowPolicy("O6_late9_slots4_cic1_050_cic2_025", 9, 4, 0.50, 0.25),
)
STRESS_COSTS = (20, 30, 50)
STRESS_SIZES = ((0.25, 0.125), (0.50, 0.25), (0.75, 0.375))
STRESS_OVERFLOW_SLOTS = (2, 4, 6)


@dataclass(frozen=True)
class V10DConfig:
    report_root: Path = REPORT_ROOT
    v10a: V10AConfig = V10AConfig()


def _overflow_size(row: pd.Series | dict[str, Any], policy: OverflowPolicy) -> float:
    candidate = str(row.get("candidate", ""))
    if policy.cic1_only and candidate != "CIC1_beta_extreme":
        return 0.0
    if candidate == "CIC1_beta_extreme":
        return policy.cic1_size
    if candidate == "CIC2_beta_broad":
        return policy.cic2_size
    return 0.0


def _overflow_allowed(row: pd.Series, policy: OverflowPolicy) -> bool:
    if policy.overflow_max_slots <= 0:
        return False
    if int(row.get("burst_count_so_far", 0)) < policy.min_burst_count:
        return False
    return _overflow_size(row, policy) > 0


def _ledger_row(row: pd.Series, *, sleeve: str, weight: float, selection_status: str, skip_reason: str = "") -> dict[str, Any]:
    payload = row.to_dict()
    payload["sleeve"] = sleeve
    payload["exposure_weight"] = float(weight)
    payload["selection_status"] = selection_status
    payload["skip_reason"] = skip_reason
    payload["weighted_return"] = float(pd.to_numeric(row.get("net_return", np.nan), errors="coerce")) * float(weight)
    return payload


def _simulate_overflow_policy(pool: pd.DataFrame, policy: OverflowPolicy) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_base: list[dict[str, Any]] = []
    active_overflow: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for _, row in pool.sort_values(["entry_time", "symbol"]).iterrows():
        entry = pd.Timestamp(row["entry_time"])
        active_base = [item for item in active_base if pd.Timestamp(item["exit_time"]) > entry]
        active_overflow = [item for item in active_overflow if pd.Timestamp(item["exit_time"]) > entry]
        active_symbols = {str(item["symbol"]) for item in [*active_base, *active_overflow]}
        if str(row["symbol"]) in active_symbols:
            skipped_rows.append(_ledger_row(row, sleeve="skipped", weight=0.0, selection_status="skipped", skip_reason="symbol_already_active"))
            continue
        if len(active_base) < BASELINE_MAX_POSITIONS:
            ledger_rows.append(_ledger_row(row, sleeve="baseline", weight=1.0, selection_status="selected"))
            active_base.append({"symbol": str(row["symbol"]), "exit_time": row["exit_time"]})
            continue
        if _overflow_allowed(row, policy) and len(active_overflow) < policy.overflow_max_slots:
            size = _overflow_size(row, policy)
            ledger_rows.append(_ledger_row(row, sleeve="overflow", weight=size, selection_status="selected"))
            active_overflow.append({"symbol": str(row["symbol"]), "exit_time": row["exit_time"], "weight": size})
            continue
        reason = "overflow_full" if _overflow_allowed(row, policy) else "portfolio_full_not_overflow_eligible"
        skipped_rows.append(_ledger_row(row, sleeve="skipped", weight=0.0, selection_status="skipped", skip_reason=reason))
    return pd.DataFrame(ledger_rows), pd.DataFrame(skipped_rows)


def _weighted_month_cap(ledger: pd.DataFrame) -> float:
    if ledger.empty:
        return np.nan
    sample = ledger.copy()
    sample["net_return"] = pd.to_numeric(sample["weighted_return"], errors="coerce")
    return _month_cap_expectancy(sample)


def _exposure_stats(ledger: pd.DataFrame) -> dict[str, float]:
    if ledger.empty:
        return {"avg_exposure_units": np.nan, "max_exposure_units": 0.0, "period_hours": np.nan}
    events: list[tuple[pd.Timestamp, float]] = []
    holding = 0.0
    for row in ledger.itertuples(index=False):
        entry = pd.Timestamp(getattr(row, "entry_time"))
        exit_time = pd.Timestamp(getattr(row, "exit_time"))
        weight = float(getattr(row, "exposure_weight"))
        events.append((entry, weight))
        events.append((exit_time, -weight))
        holding += max(float((exit_time - entry).total_seconds() / 3600.0), 0.0) * weight
    active = 0.0
    max_active = 0.0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        max_active = max(max_active, active)
    period_hours = _period_hours(ledger)
    return {
        "avg_exposure_units": float(holding / period_hours) if period_hours else np.nan,
        "max_exposure_units": float(max_active),
        "period_hours": float(period_hours) if period_hours else np.nan,
    }


def _policy_summary_row(
    ledger: pd.DataFrame,
    skipped: pd.DataFrame,
    policy: OverflowPolicy,
    baseline_portfolio_net20: float,
) -> dict[str, Any]:
    net = pd.to_numeric(ledger.get("weighted_return", pd.Series(dtype=float)), errors="coerce")
    baseline = ledger[ledger.get("sleeve", pd.Series(dtype=str)).astype(str).eq("baseline")]
    overflow = ledger[ledger.get("sleeve", pd.Series(dtype=str)).astype(str).eq("overflow")]
    overflow_weight_sum = pd.to_numeric(overflow.get("exposure_weight", pd.Series(dtype=float)), errors="coerce").sum()
    overflow_weighted_return = pd.to_numeric(overflow.get("weighted_return", pd.Series(dtype=float)), errors="coerce").sum()
    skipped_net = pd.to_numeric(skipped.get("net_return", pd.Series(dtype=float)), errors="coerce")
    portfolio_net20 = float(net.sum() / BASELINE_MAX_POSITIONS) if len(net) else 0.0
    extra_capacity = float(policy.overflow_max_slots * max(policy.cic1_size, policy.cic2_size))
    scale = BASELINE_MAX_POSITIONS / (BASELINE_MAX_POSITIONS + extra_capacity) if extra_capacity > 0 else 1.0
    exposure = _exposure_stats(ledger)
    return {
        "policy_id": policy.policy_id,
        "min_burst_count": policy.min_burst_count if policy.overflow_max_slots > 0 else np.nan,
        "overflow_max_slots": policy.overflow_max_slots,
        "cic1_size": policy.cic1_size,
        "cic2_size": policy.cic2_size,
        "selected_trades": int(len(ledger)),
        "baseline_trades": int(len(baseline)),
        "overflow_trades": int(len(overflow)),
        "skipped_trades": int(len(skipped)),
        "overflow_net20": float(pd.to_numeric(overflow.get("net_return", pd.Series(dtype=float)), errors="coerce").mean())
        if len(overflow)
        else np.nan,
        "skipped_net20": float(skipped_net.mean()) if len(skipped_net) else np.nan,
        "overflow_minus_skipped_net20": float(
            pd.to_numeric(overflow.get("net_return", pd.Series(dtype=float)), errors="coerce").mean() - skipped_net.mean()
        )
        if len(overflow) and len(skipped_net)
        else np.nan,
        "portfolio_net20": portfolio_net20,
        "incremental_net20_vs_baseline": portfolio_net20 - baseline_portfolio_net20,
        "overflow_weighted_return_sum": float(overflow_weighted_return),
        "overflow_exposure_weight_sum": float(overflow_weight_sum),
        "incremental_return_per_extra_exposure": float(overflow_weighted_return / overflow_weight_sum)
        if overflow_weight_sum
        else np.nan,
        "capital_neutral_scale": scale,
        "capital_neutral_net20": portfolio_net20 * scale,
        "capital_neutral_delta_vs_baseline": portfolio_net20 * scale - baseline_portfolio_net20,
        "month_cap35_net20": _weighted_month_cap(ledger),
        "max_month_contribution": _max_contribution(ledger.assign(net_return=ledger.get("weighted_return", np.nan)), "month"),
        "max_symbol_contribution": _max_contribution(ledger.assign(net_return=ledger.get("weighted_return", np.nan)), "symbol"),
        **exposure,
    }


def _overflow_reports(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_ledger, baseline_skipped = _simulate_overflow_policy(pool, POLICIES[0])
    baseline_net = float(pd.to_numeric(baseline_ledger["weighted_return"], errors="coerce").sum() / BASELINE_MAX_POSITIONS)
    rows = []
    ledgers = []
    skipped_frames = []
    for policy in POLICIES:
        ledger, skipped = _simulate_overflow_policy(pool, policy)
        rows.append(_policy_summary_row(ledger, skipped, policy, baseline_net))
        ledgers.append(ledger.assign(policy_id=policy.policy_id))
        skipped_frames.append(skipped.assign(policy_id=policy.policy_id))
    summary = pd.DataFrame(rows)
    ledger_all = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()
    skipped_all = pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame()
    neutral = summary[
        [
            "policy_id",
            "portfolio_net20",
            "capital_neutral_scale",
            "capital_neutral_net20",
            "capital_neutral_delta_vs_baseline",
            "avg_exposure_units",
            "max_exposure_units",
        ]
    ].copy()
    return summary, ledger_all, skipped_all, neutral


def _stress_adjusted_pool(pool: pd.DataFrame, cost: int) -> pd.DataFrame:
    out = pool.copy()
    cost_delta = (float(cost) - 20.0) * 2.0 / 10_000.0
    out["net_return"] = pd.to_numeric(out["net_return"], errors="coerce") - cost_delta
    out["cost_single_side_bps"] = float(cost)
    return out


def _overflow_stress_summary(pool: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cost in STRESS_COSTS:
        local = _stress_adjusted_pool(pool, cost)
        baseline_ledger, _ = _simulate_overflow_policy(local, POLICIES[0])
        baseline_net = float(pd.to_numeric(baseline_ledger["weighted_return"], errors="coerce").sum() / BASELINE_MAX_POSITIONS)
        for slots in STRESS_OVERFLOW_SLOTS:
            for cic1_size, cic2_size in STRESS_SIZES:
                policy = OverflowPolicy(
                    f"stress_late9_slots{slots}_cic1_{cic1_size:.3f}_cic2_{cic2_size:.3f}",
                    9,
                    slots,
                    cic1_size,
                    cic2_size,
                )
                ledger, skipped = _simulate_overflow_policy(local, policy)
                row = _policy_summary_row(ledger, skipped, policy, baseline_net)
                row["stress_cost_single_side_bps"] = cost
                rows.append(row)
    return pd.DataFrame(rows)


def _incremental_pnl_summary(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "policy_id",
        "overflow_trades",
        "overflow_net20",
        "incremental_net20_vs_baseline",
        "incremental_return_per_extra_exposure",
        "overflow_exposure_weight_sum",
        "overflow_minus_skipped_net20",
    ]
    return summary[[col for col in cols if col in summary.columns]].copy()


def _overflow_selected_vs_skipped(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "policy_id",
        "overflow_trades",
        "skipped_trades",
        "overflow_net20",
        "skipped_net20",
        "overflow_minus_skipped_net20",
    ]
    return summary[[col for col in cols if col in summary.columns]].copy()


def _burst_phase_attribution(ledger: pd.DataFrame, skipped: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if not ledger.empty:
        frames.append(ledger.assign(result="selected"))
    if not skipped.empty:
        frames.append(skipped.assign(result="skipped"))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    rows = []
    for keys, group in combined.groupby(["policy_id", "burst_phase_bucket", "sleeve", "result"], sort=False, dropna=False):
        net = pd.to_numeric(group.get("net_return", pd.Series(dtype=float)), errors="coerce")
        weight = pd.to_numeric(group.get("exposure_weight", pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "policy_id": keys[0],
                "burst_phase_bucket": keys[1],
                "sleeve": keys[2],
                "result": keys[3],
                "trades": int(len(group)),
                "avg_net20": float(net.mean()) if len(net) else np.nan,
                "weighted_return_sum": float(pd.to_numeric(group.get("weighted_return", pd.Series(dtype=float)), errors="coerce").sum()),
                "exposure_weight_sum": float(weight.sum()) if len(weight) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _exposure_and_risk_summary(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "policy_id",
        "avg_exposure_units",
        "max_exposure_units",
        "portfolio_net20",
        "capital_neutral_net20",
        "month_cap35_net20",
        "max_month_contribution",
        "max_symbol_contribution",
    ]
    return summary[[col for col in cols if col in summary.columns]].copy()


def _write_notes(report_root: Path, summary: pd.DataFrame, baseline_net: float) -> None:
    lines = [
        "# v1.0D Late-Burst Overflow Sleeve",
        "",
        "Purpose: test small late-burst overflow positions on top of the P2 max8 basket.",
        "This report is attribution only; it does not change paper-live or real-live permissions.",
        "",
    ]
    non_base = summary[~summary["policy_id"].eq("B0_baseline_max8_no_overflow")].copy()
    if not non_base.empty:
        best_add = non_base.sort_values("portfolio_net20", ascending=False).head(1).iloc[0]
        best_neutral = non_base.sort_values("capital_neutral_net20", ascending=False).head(1).iloc[0]
        lines.extend(
            [
                "## Additive Overflow",
                f"- Baseline P2 max8 portfolio_net20={baseline_net:.4%}.",
                f"- Best additive policy: {best_add.policy_id}, portfolio_net20={best_add.portfolio_net20:.4%}, "
                f"incremental={best_add.incremental_net20_vs_baseline:.4%}.",
                f"- Overflow trades={int(best_add.overflow_trades)}, overflow_net20={best_add.overflow_net20:.4%}, "
                f"return_per_extra_exposure={best_add.incremental_return_per_extra_exposure:.4%}.",
                "",
                "## Capital Neutral",
                f"- Best capital-neutral policy: {best_neutral.policy_id}, "
                f"capital_neutral_net20={best_neutral.capital_neutral_net20:.4%}, "
                f"delta_vs_baseline={best_neutral.capital_neutral_delta_vs_baseline:.4%}.",
                "",
            ]
        )
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v10d_late_burst_overflow(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V10DConfig = V10DConfig(),
) -> dict[str, Path]:
    report_root = ensure_dir(cfg.report_root)
    trades = _prepare_trade_features(_load_or_build_trades(feature_path, instruments, config, report_root, cfg.v10a))
    pool = _add_asof_burst_phase(_focus_pool(trades), "1h")
    if pool.empty:
        raise ValueError("No P2 CIC trades available for v1.0D late-burst overflow.")
    summary, ledger, skipped, neutral = _overflow_reports(pool)
    stress = _overflow_stress_summary(pool)
    baseline = summary[summary["policy_id"].eq("B0_baseline_max8_no_overflow")].iloc[0]
    baseline_net = float(baseline["portfolio_net20"])
    incremental = _incremental_pnl_summary(summary)
    selected_vs_skipped = _overflow_selected_vs_skipped(summary)
    phase = _burst_phase_attribution(ledger, skipped)
    risk = _exposure_and_risk_summary(summary)
    outputs = {
        "overflow_policy_summary": report_root / "overflow_policy_summary.csv",
        "overflow_trade_ledger": report_root / "overflow_trade_ledger.csv",
        "capital_neutral_summary": report_root / "capital_neutral_summary.csv",
        "incremental_pnl_summary": report_root / "incremental_pnl_summary.csv",
        "overflow_selected_vs_skipped": report_root / "overflow_selected_vs_skipped.csv",
        "burst_phase_overflow_attribution": report_root / "burst_phase_overflow_attribution.csv",
        "exposure_and_risk_summary": report_root / "exposure_and_risk_summary.csv",
        "overflow_stress_summary": report_root / "overflow_stress_summary.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    summary.to_csv(outputs["overflow_policy_summary"], index=False)
    ledger.to_csv(outputs["overflow_trade_ledger"], index=False)
    neutral.to_csv(outputs["capital_neutral_summary"], index=False)
    incremental.to_csv(outputs["incremental_pnl_summary"], index=False)
    selected_vs_skipped.to_csv(outputs["overflow_selected_vs_skipped"], index=False)
    phase.to_csv(outputs["burst_phase_overflow_attribution"], index=False)
    risk.to_csv(outputs["exposure_and_risk_summary"], index=False)
    stress.to_csv(outputs["overflow_stress_summary"], index=False)
    _write_notes(report_root, summary, baseline_net)
    return outputs
