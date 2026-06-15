"""v1.3E CP60 beta-protection stability audit.

This report stress-tests the beta-high CP60 protection found in v1.3D. It is
offline-only and does not create a live shadow rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v10a_cic_basket_portfolio import V10AConfig
from pressure_graph.reports.v13d_cp60_context_protection import (
    PROTECTION_COST_BPS,
    _add_context_features,
    _add_high_flags,
    _apply_protected_checkpoint,
    _num,
    _portfolio_summary,
    _protection_masks,
    _simulate_core_max8,
    _slot_impact_of_protection,
    _thresholds_from_exits,
)
from pressure_graph.reports.v13a_checkpoint_robustness import (
    CORE_MAX_POSITIONS,
    O6_CIC1_SIZE,
    O6_CIC2_SIZE,
    O6_MAX_SLOTS,
    O6_MIN_BURST_COUNT,
    CheckpointSpec,
    _load_price_frame,
    _pool_base20,
    _prepare_checkpoint_sample,
)
from pressure_graph.reports.v10a_cic_basket_portfolio import _load_or_build_trades
from pressure_graph.reports.v10c_burst_phase_allocation import _add_asof_burst_phase


REPORT_ROOT = Path("reports/v1_3e_cp60_beta_protection_stability")
STRESS_COSTS = (20.0, 30.0, 50.0)


@dataclass(frozen=True)
class V13EConfig:
    report_root: Path = REPORT_ROOT
    v10a: V10AConfig = V10AConfig()


def _prepare_sample_at_cost(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    root: Path,
    v10a: V10AConfig,
    cost_bps: float,
) -> pd.DataFrame:
    trades = _load_or_build_trades(feature_path, instruments, config, root, v10a)
    base = _add_asof_burst_phase(_pool_base20(trades), "1h")
    if base.empty:
        raise ValueError("No P2 CIC trades available for v1.3E beta protection stability.")
    prices = _load_price_frame(feature_path, base)
    sample = _prepare_checkpoint_sample(
        base,
        prices,
        CheckpointSpec(f"60m_net_lte_0_cost_{cost_bps:g}", 60, 0.0, "net_lte_threshold", cost_bps),
    )
    sample = _add_context_features(sample)
    return _add_high_flags(sample, _thresholds_from_exits(sample))


def _protect_a_sample(sample: pd.DataFrame) -> pd.DataFrame:
    return _apply_protected_checkpoint(sample, _protection_masks(sample)["Protect_A_beta_high"], "Protect_A_beta_high")


def _cp60_sample(sample: pd.DataFrame) -> pd.DataFrame:
    return _apply_protected_checkpoint(sample, _protection_masks(sample)["CP60_all"], "CP60_all")


def _protect_a_ledger(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cp_sample = _cp60_sample(sample)
    pa_sample = _protect_a_sample(sample)
    cp_ledger, cp_skipped = _simulate_core_max8(cp_sample, "CP60_all")
    pa_ledger, pa_skipped = _simulate_core_max8(pa_sample, "Protect_A_beta_high")
    return cp_ledger, cp_skipped, pa_ledger, pa_skipped


def _selected_protected_exits(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame()
    mask = ledger.get("kept_due_to_protection", pd.Series(False, index=ledger.index)).fillna(False).astype(bool)
    out = ledger[mask].copy()
    if out.empty:
        return out
    out["delta_vs_cp60"] = _num(out, "net_return_at_cost") - _num(out, "checkpoint_net_at_cost")
    out["slot_blocked_minutes"] = (
        pd.to_datetime(out["exit_time"], utc=True, errors="coerce")
        - pd.to_datetime(out["checkpoint_time"], utc=True, errors="coerce")
    ).dt.total_seconds() / 60.0
    return out.sort_values(["entry_time", "symbol"]).reset_index(drop=True)


def _leave_one_protected_exit_out(sample: pd.DataFrame) -> pd.DataFrame:
    cp_ledger, cp_skipped, pa_ledger, _ = _protect_a_ledger(sample)
    base_net = float(_portfolio_summary("CP60_all", cp_ledger, cp_skipped)["portfolio_net20"])
    protected = _selected_protected_exits(pa_ledger)
    rows: list[dict[str, object]] = []
    for row in protected.itertuples(index=False):
        trade_key = str(getattr(row, "trade_key", ""))
        mask = _protection_masks(sample)["Protect_A_beta_high"].copy()
        mask = mask & ~sample["trade_key"].astype(str).eq(trade_key)
        local = _apply_protected_checkpoint(sample, mask, "Protect_A_minus_one")
        ledger, skipped = _simulate_core_max8(local, "Protect_A_minus_one")
        net = float(_portfolio_summary("Protect_A_minus_one", ledger, skipped)["portfolio_net20"])
        rows.append(
            {
                "removed_trade_id": getattr(row, "signal_id", trade_key),
                "removed_symbol": getattr(row, "symbol", ""),
                "removed_time": getattr(row, "entry_time", pd.NaT),
                "removed_delta_vs_cp60": getattr(row, "delta_vs_cp60", np.nan),
                "net20_after_removal": net,
                "delta_vs_CP60_after_removal": net - base_net,
                "still_above_CP60": net > base_net,
            }
        )
    return pd.DataFrame(rows)


def _protected_exit_ledger(sample: pd.DataFrame) -> pd.DataFrame:
    _, _, pa_ledger, _ = _protect_a_ledger(sample)
    protected = _selected_protected_exits(pa_ledger)
    if protected.empty:
        return protected
    preferred = [
        "signal_id",
        "trade_key",
        "symbol",
        "candidate",
        "entry_time",
        "checkpoint_time",
        "checkpoint_price",
        "entry_price",
        "net_return_at_cost",
        "checkpoint_net_at_cost",
        "delta_vs_cp60",
        "cic_type",
        "beta_extreme_strength",
        "cluster_density",
        "market_impulse_density",
        "local_volume_shock_strength",
        "slot_blocked_minutes",
        "burst_id",
        "month",
    ]
    cols = [col for col in preferred if col in protected.columns]
    rest = [col for col in protected.columns if col not in cols]
    return protected[cols + rest]


def _protected_distribution(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    data = ledger.copy()
    if "month" not in data.columns:
        data["month"] = pd.to_datetime(data["entry_time"], utc=True, errors="coerce").dt.strftime("%Y-%m")
    for group_col in ("month", "symbol", "burst_id"):
        if group_col not in data.columns:
            continue
        for value, group in data.groupby(group_col, sort=False, dropna=False):
            rows.append(
                {
                    "group_col": group_col,
                    "group_value": value,
                    "protected_exits": int(len(group)),
                    "delta_vs_cp60_sum": float(_num(group, "delta_vs_cp60").sum()),
                    "delta_vs_cp60_avg": float(_num(group, "delta_vs_cp60").mean()),
                    "symbols": ",".join(sorted(set(group["symbol"].astype(str)))) if "symbol" in group.columns else "",
                }
            )
    return pd.DataFrame(rows)


def _overflow_size(row: pd.Series) -> float:
    candidate = str(row.get("candidate", ""))
    if candidate == "CIC1_beta_extreme":
        return O6_CIC1_SIZE
    if candidate == "CIC2_beta_broad":
        return O6_CIC2_SIZE
    return 0.0


def _simulate_max8_o6(sample: pd.DataFrame, rule_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = sample.sort_values(["entry_time", "symbol", "candidate"]).copy()
    active_core: list[dict[str, object]] = []
    active_overflow: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for _, row in data.iterrows():
        entry = pd.Timestamp(row["entry_time"])
        active_core = [item for item in active_core if pd.Timestamp(item["exit_time"]) > entry]
        active_overflow = [item for item in active_overflow if pd.Timestamp(item["exit_time"]) > entry]
        active = [*active_core, *active_overflow]
        if str(row["symbol"]) in {str(item["symbol"]) for item in active}:
            payload = row.to_dict()
            payload.update({"portfolio_id": rule_name, "selection_status": "skipped", "skip_reason": "symbol_already_active"})
            skipped.append(payload)
            continue
        payload = row.to_dict()
        if len(active_core) < CORE_MAX_POSITIONS:
            payload.update({"portfolio_id": rule_name, "selection_status": "selected", "sleeve": "core", "exposure_weight": 1.0})
            payload["weighted_return"] = float(pd.to_numeric(row.get("effective_net_return", np.nan), errors="coerce"))
            selected.append(payload)
            active_core.append({"symbol": str(row["symbol"]), "exit_time": row["effective_exit_time"]})
            continue
        overflow_allowed = int(row.get("burst_count_so_far", 0)) >= O6_MIN_BURST_COUNT and _overflow_size(row) > 0
        if overflow_allowed and len(active_overflow) < O6_MAX_SLOTS:
            size = _overflow_size(row)
            payload.update({"portfolio_id": rule_name, "selection_status": "selected", "sleeve": "overflow", "exposure_weight": size})
            payload["weighted_return"] = float(pd.to_numeric(row.get("effective_net_return", np.nan), errors="coerce")) * size
            selected.append(payload)
            active_overflow.append({"symbol": str(row["symbol"]), "exit_time": row["effective_exit_time"]})
            continue
        payload.update(
            {
                "portfolio_id": rule_name,
                "selection_status": "skipped",
                "skip_reason": "overflow_full" if overflow_allowed else "portfolio_full_not_overflow_eligible",
            }
        )
        skipped.append(payload)
    return pd.DataFrame(selected), pd.DataFrame(skipped)


def _portfolio_net(ledger: pd.DataFrame) -> float:
    return float(_num(ledger, "weighted_return").sum() / CORE_MAX_POSITIONS) if not ledger.empty else 0.0


def _slot_impact_between(base: pd.DataFrame, protected: pd.DataFrame) -> float:
    if base.empty or protected.empty:
        return 0.0
    base_keys = set(base["trade_key"].astype(str))
    protected_keys = set(protected["trade_key"].astype(str))
    base_selected = base[["trade_key", "entry_time", "effective_net_return"]].copy()
    total = 0.0
    protected_rows = protected[
        protected.get("kept_due_to_protection", pd.Series(False, index=protected.index)).fillna(False).astype(bool)
    ].copy()
    for row in protected_rows.itertuples(index=False):
        checkpoint_time = pd.Timestamp(getattr(row, "checkpoint_time"))
        original_exit = pd.Timestamp(getattr(row, "exit_time"))
        missed = base_selected[
            base_selected["trade_key"].astype(str).isin(base_keys - protected_keys)
            & pd.to_datetime(base_selected["entry_time"], utc=True, errors="coerce").ge(checkpoint_time)
            & pd.to_datetime(base_selected["entry_time"], utc=True, errors="coerce").lt(original_exit)
        ]
        keep_gain = float(getattr(row, "net_return_at_cost", np.nan) - getattr(row, "checkpoint_net_at_cost", np.nan))
        total += keep_gain - float(_num(missed, "effective_net_return").sum())
    return total


def _protection_o6_integration(sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cases = {
        "S3_CP60_O6": _cp60_sample(sample),
        "S3_Protect_A_O6": _protect_a_sample(sample),
    }
    baseline_net = np.nan
    simulated: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for name, local in cases.items():
        ledger, skipped = _simulate_max8_o6(local, name)
        simulated[name] = (ledger, skipped)
        if name == "S3_CP60_O6":
            baseline_net = _portfolio_net(ledger)
    slot_impact_total = _slot_impact_between(
        simulated.get("S3_CP60_O6", (pd.DataFrame(), pd.DataFrame()))[0],
        simulated.get("S3_Protect_A_O6", (pd.DataFrame(), pd.DataFrame()))[0],
    )
    for name, (ledger, skipped) in simulated.items():
        protected = ledger.get("kept_due_to_protection", pd.Series(False, index=ledger.index)).fillna(False).astype(bool) if not ledger.empty else pd.Series(dtype=bool)
        overflow = ledger.get("sleeve", pd.Series(dtype=str)).astype(str).eq("overflow") if not ledger.empty else pd.Series(dtype=bool)
        rows.append(
            {
                "structure": name,
                "net20": _portfolio_net(ledger),
                "delta_vs_s3": _portfolio_net(ledger) - baseline_net if pd.notna(baseline_net) else np.nan,
                "selected_trades": int(len(ledger)),
                "skipped_trades": int(len(skipped)),
                "overflow_trades": int(overflow.sum()) if len(overflow) else 0,
                "protected_exits": int(protected.sum()) if len(protected) else 0,
                "slot_impact": slot_impact_total if name == "S3_Protect_A_O6" else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _cost_stress(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    root: Path,
    cfg: V13EConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cost in STRESS_COSTS:
        sample = _prepare_sample_at_cost(feature_path, instruments, config, root, cfg.v10a, cost)
        for rule_name, local in {"CP60_all": _cp60_sample(sample), "Protect_A_beta_high": _protect_a_sample(sample)}.items():
            ledger, skipped = _simulate_core_max8(local, rule_name)
            summary = _portfolio_summary(rule_name, ledger, skipped)
            rows.append(
                {
                    "rule": rule_name,
                    "cost_single_side_bps": cost,
                    "net20_equivalent": summary["portfolio_net20"],
                    "selected_trades": summary["selected_trades"],
                    "protected_cp60_exits": summary["protected_cp60_exits"],
                    "cp60_exits_executed": summary["cp60_exits_executed"],
                    "worst_month": summary["worst_month"],
                    "month_cap35_net20": summary["month_cap35_net20"],
                }
            )
    return pd.DataFrame(rows)


def _write_notes(root: Path, loo: pd.DataFrame, o6: pd.DataFrame, cost: pd.DataFrame) -> None:
    lines = [
        "# v1.3E CP60 Beta-Protection Stability Audit",
        "",
        "Purpose: test whether Protect_A beta_high is stable enough to remain a CP60_v2 research candidate.",
        "Status: offline diagnostic only. No live shadow or production rule changes.",
        "",
        "## Leave-One-Protected-Exit-Out",
    ]
    if loo.empty:
        lines.append("- No protected exits.")
    else:
        lines.append(f"- protected_exits={len(loo)}")
        lines.append(f"- still_above_CP60={int(loo['still_above_CP60'].sum())}/{len(loo)}")
        lines.append(f"- min_delta_vs_CP60_after_removal={loo['delta_vs_CP60_after_removal'].min():.4%}")
    lines.append("")
    lines.append("## O6 Integration")
    if o6.empty:
        lines.append("- No O6 integration rows.")
    else:
        for row in o6.itertuples(index=False):
            lines.append(f"- {row.structure}: net20={row.net20:.4%}, delta_vs_s3={row.delta_vs_s3:.4%}.")
    lines.append("")
    lines.append("## Cost Stress")
    if not cost.empty:
        for row in cost.itertuples(index=False):
            lines.append(f"- {row.rule} cost={row.cost_single_side_bps:g}bp net={row.net20_equivalent:.4%}.")
    lines.extend(["", "Discipline: keep CP60_all live shadow unchanged until future protected-exit sample is materially larger."])
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v13e_cp60_beta_protection_stability(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V13EConfig = V13EConfig(),
) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    sample = _prepare_sample_at_cost(feature_path, instruments, config, root, cfg.v10a, PROTECTION_COST_BPS)
    protected_ledger = _protected_exit_ledger(sample)
    loo = _leave_one_protected_exit_out(sample)
    distribution = _protected_distribution(protected_ledger)
    o6 = _protection_o6_integration(sample)
    cost = _cost_stress(feature_path, instruments, config, root, cfg)
    cp_ledger, _, protect_ledger, _ = _protect_a_ledger(sample)
    slot = _slot_impact_of_protection(pd.concat([cp_ledger, protect_ledger], ignore_index=True))
    outputs = {
        "leave_one_protected_exit_out": root / "leave_one_protected_exit_out.csv",
        "protected_exit_ledger": root / "protected_exit_ledger.csv",
        "protected_exit_distribution": root / "protected_exit_distribution.csv",
        "protection_o6_integration": root / "protection_o6_integration.csv",
        "protection_cost_stress": root / "protection_cost_stress.csv",
        "slot_impact_of_protection": root / "slot_impact_of_protection.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    loo.to_csv(outputs["leave_one_protected_exit_out"], index=False)
    protected_ledger.to_csv(outputs["protected_exit_ledger"], index=False)
    distribution.to_csv(outputs["protected_exit_distribution"], index=False)
    o6.to_csv(outputs["protection_o6_integration"], index=False)
    cost.to_csv(outputs["protection_cost_stress"], index=False)
    slot.to_csv(outputs["slot_impact_of_protection"], index=False)
    _write_notes(root, loo, o6, cost)
    return outputs


__all__ = [
    "REPORT_ROOT",
    "V13EConfig",
    "write_v13e_cp60_beta_protection_stability",
]
