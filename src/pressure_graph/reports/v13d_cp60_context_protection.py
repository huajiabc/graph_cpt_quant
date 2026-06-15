"""v1.3D CP60 context-protection audit.

This is an offline diagnostic. It tests whether CP60 false exits cluster in a
small number of continuation contexts and whether protecting those exits would
improve the P2 max8 checkpoint portfolio after slot opportunity costs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v09b import _month_cap_expectancy
from pressure_graph.reports.v09d import _period_hours
from pressure_graph.reports.v10a_cic_basket_portfolio import V10AConfig, _load_or_build_trades
from pressure_graph.reports.v10c_burst_phase_allocation import _add_asof_burst_phase
from pressure_graph.reports.v13a_checkpoint_robustness import (
    CORE_MAX_POSITIONS,
    CheckpointSpec,
    _load_price_frame,
    _pool_base20,
    _prepare_checkpoint_sample,
)
from pressure_graph.reports.v13c_cp60_false_exit_attribution import (
    NEUTRAL_DELTA,
    _classify_cp60_exits,
)


REPORT_ROOT = Path("reports/v1_3d_cp60_context_protection")
PROTECTION_COST_BPS = 20.0


@dataclass(frozen=True)
class V13DConfig:
    report_root: Path = REPORT_ROOT
    v10a: V10AConfig = V10AConfig()
    neutral_delta: float = NEUTRAL_DELTA


def _num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def _first_numeric(frame: pd.DataFrame, cols: tuple[str, ...], default: float = np.nan) -> pd.Series:
    out = pd.Series(default, index=frame.index, dtype="float64")
    for col in cols:
        if col not in frame.columns:
            continue
        candidate = pd.to_numeric(frame[col], errors="coerce")
        out = out.where(out.notna(), candidate)
    return out


def _cic_type(candidate: object) -> str:
    text = str(candidate)
    if text.startswith("CIC1"):
        return "CIC1"
    if text.startswith("CIC2"):
        return "CIC2"
    return text or "unknown"


def _trade_key(frame: pd.DataFrame) -> pd.Series:
    if "signal_id" in frame.columns:
        return frame["signal_id"].astype(str)
    return (
        frame["exchange"].astype(str)
        + "|"
        + frame["symbol"].astype(str)
        + "|"
        + pd.to_datetime(frame["entry_time"], utc=True, errors="coerce").astype(str)
    )


def _add_context_features(sample: pd.DataFrame) -> pd.DataFrame:
    out = sample.copy()
    out["trade_key"] = _trade_key(out)
    out["cic_type"] = out["candidate"].map(_cic_type)
    out["market_impulse_density"] = _first_numeric(out, ("volume_impulse_density", "rank_market_impulse_density"))
    out["beta_extreme_strength"] = _first_numeric(
        out,
        ("c2_beta_extension_score", "rank_beta_extreme_strength", "ret_4h_percentile"),
    )
    out["local_volume_shock_strength"] = _first_numeric(
        out,
        ("volume_z_1h", "volume_z_4h", "rank_local_volume_shock_strength"),
    )
    out["cluster_density"] = _first_numeric(out, ("cluster_impulse_density", "rank_cluster_impulse_density"))
    return out


def _cp60_exit_mask(sample: pd.DataFrame) -> pd.Series:
    checkpoint_before_exit = pd.to_datetime(sample["checkpoint_time"], utc=True, errors="coerce") < pd.to_datetime(
        sample["exit_time"], utc=True, errors="coerce"
    )
    return (
        sample.get("checkpoint_price_covered", pd.Series(False, index=sample.index)).fillna(False).astype(bool)
        & checkpoint_before_exit
        & _num(sample, "checkpoint_net_at_cost").le(0.0)
    )


def _thresholds_from_exits(sample: pd.DataFrame) -> dict[str, float]:
    exits = sample[_cp60_exit_mask(sample)].copy()
    thresholds: dict[str, float] = {}
    for feature in ("beta_extreme_strength", "cluster_density", "market_impulse_density", "local_volume_shock_strength"):
        values = pd.to_numeric(exits.get(feature), errors="coerce").dropna()
        thresholds[feature] = float(values.quantile(0.8)) if not values.empty else np.nan
    return thresholds


def _add_high_flags(sample: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    out = sample.copy()
    mapping = {
        "beta_extreme_strength": "beta_extreme_strength_high",
        "cluster_density": "cluster_density_high",
        "market_impulse_density": "market_impulse_density_high",
        "local_volume_shock_strength": "local_volume_shock_strength_high",
    }
    for feature, flag in mapping.items():
        threshold = thresholds.get(feature, np.nan)
        values = pd.to_numeric(out.get(feature), errors="coerce")
        out[flag] = values.ge(threshold) if pd.notna(threshold) else False
        out[f"{feature}_high_threshold"] = threshold
    return out


def _prepare_sample(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    root: Path,
    v10a: V10AConfig,
) -> pd.DataFrame:
    trades = _load_or_build_trades(feature_path, instruments, config, root, v10a)
    base = _add_asof_burst_phase(_pool_base20(trades), "1h")
    if base.empty:
        raise ValueError("No P2 CIC trades available for v1.3D CP60 context protection.")
    prices = _load_price_frame(feature_path, base)
    sample = _prepare_checkpoint_sample(
        base,
        prices,
        CheckpointSpec("60m_net_lte_0", 60, 0.0, "net_lte_threshold", PROTECTION_COST_BPS),
    )
    sample = _add_context_features(sample)
    return _add_high_flags(sample, _thresholds_from_exits(sample))


def _exit_class_for_sample(sample: pd.DataFrame, neutral_delta: float) -> pd.DataFrame:
    cp = sample.copy()
    cp["checkpoint_early_exit"] = _cp60_exit_mask(cp)
    cp["effective_exit_time"] = cp["exit_time"]
    cp["effective_net_return"] = cp["net_return_at_cost"]
    cp.loc[cp["checkpoint_early_exit"], "effective_exit_time"] = cp.loc[cp["checkpoint_early_exit"], "checkpoint_time"]
    cp.loc[cp["checkpoint_early_exit"], "effective_net_return"] = cp.loc[cp["checkpoint_early_exit"], "checkpoint_net_at_cost"]
    return _classify_cp60_exits(cp, neutral_delta)


def _context_cross_table(sample: pd.DataFrame, neutral_delta: float) -> pd.DataFrame:
    exits = _exit_class_for_sample(sample, neutral_delta)
    if exits.empty:
        return pd.DataFrame()
    flag_map = sample.set_index("trade_key")[
        [
            "beta_extreme_strength_high",
            "cluster_density_high",
            "market_impulse_density_high",
            "local_volume_shock_strength_high",
        ]
    ]
    exits = exits.join(flag_map, on="signal_id", how="left", rsuffix="_flag")
    combos = {
        "beta_high_and_cluster_high": ("beta_extreme_strength_high", "cluster_density_high"),
        "beta_high_and_market_high": ("beta_extreme_strength_high", "market_impulse_density_high"),
        "cluster_high_and_market_high": ("cluster_density_high", "market_impulse_density_high"),
        "beta_high_and_local_shock_high": ("beta_extreme_strength_high", "local_volume_shock_strength_high"),
    }
    rows: list[dict[str, object]] = []
    for name, (left, right) in combos.items():
        if left not in exits.columns or right not in exits.columns:
            group = pd.DataFrame()
            available = False
        else:
            mask = exits[left].fillna(False).astype(bool) & exits[right].fillna(False).astype(bool)
            group = exits[mask].copy()
            available = bool(mask.notna().any())
        classes = group.get("exit_class", pd.Series(dtype=str)).astype(str)
        true_group = group[classes.eq("true_good_exit")] if not group.empty else group
        false_group = group[classes.eq("false_exit")] if not group.empty else group
        rows.append(
            {
                "context": name,
                "available": available,
                "cp_exit_count": int(len(group)),
                "true_good_exit_rate": float(classes.eq("true_good_exit").mean()) if len(group) else np.nan,
                "false_exit_rate": float(classes.eq("false_exit").mean()) if len(group) else np.nan,
                "neutral_exit_rate": float(classes.eq("neutral_exit").mean()) if len(group) else np.nan,
                "avg_delta_vs_keep": float(_num(group, "delta_exit_vs_keep").mean()) if len(group) else np.nan,
                "sum_delta_vs_keep": float(_num(group, "delta_exit_vs_keep").sum()) if len(group) else 0.0,
                "avg_loss_avoided": float(_num(true_group, "delta_exit_vs_keep").mean()) if len(true_group) else np.nan,
                "avg_false_exit_cost": float(_num(false_group, "delta_exit_vs_keep").mean()) if len(false_group) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _apply_protected_checkpoint(sample: pd.DataFrame, protect_mask: pd.Series, rule_name: str) -> pd.DataFrame:
    out = sample.copy()
    early_raw = _cp60_exit_mask(out)
    protect = protect_mask.reindex(out.index).fillna(False).astype(bool)
    early = early_raw & ~protect
    out["protection_rule"] = rule_name
    out["would_have_exited_at_cp60"] = early_raw
    out["kept_due_to_protection"] = early_raw & protect
    out["checkpoint_early_exit"] = early
    out["effective_exit_time"] = out["exit_time"]
    out["effective_net_return"] = out["net_return_at_cost"]
    out.loc[early, "effective_exit_time"] = out.loc[early, "checkpoint_time"]
    out.loc[early, "effective_net_return"] = out.loc[early, "checkpoint_net_at_cost"]
    out["effective_holding_minutes"] = (
        pd.to_datetime(out["effective_exit_time"], utc=True, errors="coerce")
        - pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    ).dt.total_seconds() / 60.0
    return out


def _ledger_row(row: pd.Series, *, status: str, reason: str = "") -> dict[str, object]:
    payload = row.to_dict()
    payload["selection_status"] = status
    payload["skip_reason"] = reason
    payload["sleeve"] = "core"
    payload["exposure_weight"] = 1.0 if status == "selected" else 0.0
    payload["weighted_return"] = float(pd.to_numeric(row.get("effective_net_return", np.nan), errors="coerce")) if status == "selected" else 0.0
    return payload


def _simulate_core_max8(sample: pd.DataFrame, rule_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = sample.sort_values(["entry_time", "symbol", "candidate"]).copy()
    active: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for _, row in data.iterrows():
        entry = pd.Timestamp(row["entry_time"])
        active = [item for item in active if pd.Timestamp(item["exit_time"]) > entry]
        if str(row["symbol"]) in {str(item["symbol"]) for item in active}:
            skipped.append(_ledger_row(row, status="skipped", reason="symbol_already_active"))
            continue
        if len(active) >= CORE_MAX_POSITIONS:
            skipped.append(_ledger_row(row, status="skipped", reason="portfolio_full"))
            continue
        selected.append(_ledger_row(row, status="selected"))
        active.append({"symbol": str(row["symbol"]), "exit_time": row["effective_exit_time"]})
    ledger = pd.DataFrame(selected)
    skipped_frame = pd.DataFrame(skipped)
    if not ledger.empty:
        ledger["portfolio_id"] = rule_name
    if not skipped_frame.empty:
        skipped_frame["portfolio_id"] = rule_name
    return ledger, skipped_frame


def _drawdown(contribution: pd.Series) -> float:
    if contribution.empty:
        return np.nan
    equity = contribution.cumsum()
    return float((equity - equity.cummax()).min())


def _period_return(ledger: pd.DataFrame, period: str) -> pd.Series:
    if ledger.empty:
        return pd.Series(dtype=float)
    data = ledger.copy()
    data["entry_time"] = pd.to_datetime(data["entry_time"], utc=True, errors="coerce")
    if period == "month":
        key = data["entry_time"].dt.strftime("%Y-%m")
    elif period == "burst":
        key = data.get("burst_id", pd.Series("unknown", index=data.index)).astype(str)
    else:
        raise KeyError(period)
    return _num(data, "weighted_return").groupby(key, sort=False, dropna=False).sum() / CORE_MAX_POSITIONS


def _portfolio_summary(rule_name: str, ledger: pd.DataFrame, skipped: pd.DataFrame) -> dict[str, object]:
    ledger = ledger.copy()
    if not ledger.empty and "month" not in ledger.columns:
        ledger["month"] = pd.to_datetime(ledger["entry_time"], utc=True, errors="coerce").dt.strftime("%Y-%m")
    weighted = _num(ledger, "weighted_return")
    contribution = weighted / CORE_MAX_POSITIONS
    period_hours = _period_hours(ledger.assign(exit_time=ledger.get("effective_exit_time"))) if not ledger.empty else np.nan
    holding_hours = _num(ledger, "effective_holding_minutes").sum() / 60.0 if not ledger.empty else 0.0
    skipped_net = _num(skipped, "net_return_at_cost")
    protected = ledger.get("kept_due_to_protection", pd.Series(False, index=ledger.index)).fillna(False).astype(bool) if not ledger.empty else pd.Series(dtype=bool)
    raw_cp = ledger.get("would_have_exited_at_cp60", pd.Series(False, index=ledger.index)).fillna(False).astype(bool) if not ledger.empty else pd.Series(dtype=bool)
    return {
        "rule": rule_name,
        "selected_trades": int(len(ledger)),
        "skipped_trades": int(len(skipped)),
        "portfolio_net20": float(contribution.sum()) if len(contribution) else 0.0,
        "selected_effective_net20": float(_num(ledger, "effective_net_return").mean()) if len(ledger) else np.nan,
        "skipped_counterfactual_net20": float(skipped_net.mean()) if len(skipped_net) else np.nan,
        "selected_minus_skipped": float(_num(ledger, "effective_net_return").mean() - skipped_net.mean()) if len(ledger) and len(skipped_net) else np.nan,
        "month_cap35_net20": _month_cap_expectancy(ledger.assign(net_return=_num(ledger, "weighted_return"))) if not ledger.empty else np.nan,
        "worst_month": float(_period_return(ledger, "month").min()) if not ledger.empty else np.nan,
        "worst_burst": float(_period_return(ledger, "burst").min()) if not ledger.empty else np.nan,
        "max_drawdown_proxy": _drawdown(contribution),
        "capital_utilization": float(holding_hours / (period_hours * CORE_MAX_POSITIONS)) if period_hours else np.nan,
        "cp60_exits_executed": int((raw_cp & ~protected).sum()) if len(raw_cp) else 0,
        "protected_cp60_exits": int(protected.sum()) if len(protected) else 0,
    }


def _protection_masks(sample: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "CP60_all": pd.Series(False, index=sample.index),
        "Protect_A_beta_high": sample["beta_extreme_strength_high"].fillna(False).astype(bool),
        "Protect_B_cluster_high": sample["cluster_density_high"].fillna(False).astype(bool),
        "Protect_C_beta_cluster_high": sample["beta_extreme_strength_high"].fillna(False).astype(bool)
        & sample["cluster_density_high"].fillna(False).astype(bool),
        "Protect_D_beta_market_high": sample["beta_extreme_strength_high"].fillna(False).astype(bool)
        & sample["market_impulse_density_high"].fillna(False).astype(bool),
    }


def _protection_counterfactual(sample: pd.DataFrame, neutral_delta: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ledgers: list[pd.DataFrame] = []
    skipped_frames: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    masks = _protection_masks(sample)
    baseline_sample = _apply_protected_checkpoint(sample, masks["CP60_all"], "CP60_all")
    baseline_ledger, baseline_skipped = _simulate_core_max8(baseline_sample, "CP60_all")
    baseline_summary = _portfolio_summary("CP60_all", baseline_ledger, baseline_skipped)
    baseline_net = float(baseline_summary["portfolio_net20"])
    base_classes = _exit_class_for_protection_sample(baseline_sample, neutral_delta).set_index("trade_key")
    for rule_name, mask in masks.items():
        protected_sample = _apply_protected_checkpoint(sample, mask, rule_name)
        ledger, skipped = _simulate_core_max8(protected_sample, rule_name)
        ledgers.append(ledger)
        skipped_frames.append(skipped)
        summary = _portfolio_summary(rule_name, ledger, skipped)
        protected_keys = set(
            ledger.get("trade_key", pd.Series(dtype=str))[
                ledger.get("kept_due_to_protection", pd.Series(False, index=ledger.index)).fillna(False).astype(bool)
            ].astype(str)
        )
        protected_classes = base_classes.loc[base_classes.index.intersection(protected_keys)] if protected_keys else pd.DataFrame()
        true_lost = int(protected_classes["exit_class"].astype(str).eq("true_good_exit").sum()) if not protected_classes.empty else 0
        false_saved = int(protected_classes["exit_class"].astype(str).eq("false_exit").sum()) if not protected_classes.empty else 0
        neutral_saved = int(protected_classes["exit_class"].astype(str).eq("neutral_exit").sum()) if not protected_classes.empty else 0
        summary.update(
            {
                "baseline_CP60_net20": baseline_net,
                "protected_CP60_net20": summary["portfolio_net20"],
                "delta_vs_CP60": float(summary["portfolio_net20"] - baseline_net),
                "false_exits_saved": false_saved,
                "true_good_exits_lost": true_lost,
                "neutral_exits_saved": neutral_saved,
            }
        )
        rows.append(summary)
    ledger_all = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()
    skipped_all = pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame()
    return pd.DataFrame(rows), ledger_all, skipped_all


def _exit_class_for_protection_sample(sample: pd.DataFrame, neutral_delta: float) -> pd.DataFrame:
    cp = sample[sample["would_have_exited_at_cp60"].fillna(False).astype(bool)].copy()
    if cp.empty:
        return pd.DataFrame()
    cp["net_if_kept"] = _num(cp, "net_return_at_cost")
    cp["net_if_exited"] = _num(cp, "checkpoint_net_at_cost")
    cp["delta_exit_vs_keep"] = cp["net_if_exited"] - cp["net_if_kept"]
    cp["exit_class"] = np.select(
        [cp["delta_exit_vs_keep"].gt(neutral_delta), cp["delta_exit_vs_keep"].lt(-neutral_delta)],
        ["true_good_exit", "false_exit"],
        default="neutral_exit",
    )
    return cp


def _slot_impact_of_protection(ledger_all: pd.DataFrame) -> pd.DataFrame:
    if ledger_all.empty:
        return pd.DataFrame()
    base = ledger_all[ledger_all["portfolio_id"].eq("CP60_all")].copy()
    base_keys = set(base["trade_key"].astype(str)) if not base.empty else set()
    base_selected = base[["trade_key", "entry_time", "effective_net_return"]].copy() if not base.empty else pd.DataFrame()
    rows: list[dict[str, object]] = []
    protected = ledger_all[
        ledger_all.get("kept_due_to_protection", pd.Series(False, index=ledger_all.index)).fillna(False).astype(bool)
    ].copy()
    for row in protected.itertuples(index=False):
        rule = str(getattr(row, "portfolio_id", ""))
        rule_keys = set(ledger_all[ledger_all["portfolio_id"].astype(str).eq(rule)]["trade_key"].astype(str))
        checkpoint_time = pd.Timestamp(getattr(row, "checkpoint_time"))
        original_exit = pd.Timestamp(getattr(row, "exit_time"))
        blocked = (original_exit - checkpoint_time).total_seconds() / 60.0
        missed = base_selected[
            base_selected["trade_key"].astype(str).isin(base_keys - rule_keys)
            & pd.to_datetime(base_selected["entry_time"], utc=True, errors="coerce").ge(checkpoint_time)
            & pd.to_datetime(base_selected["entry_time"], utc=True, errors="coerce").lt(original_exit)
        ].copy()
        if missed.empty:
            rows.append(
                {
                    "protection_rule": rule,
                    "protected_trade_id": getattr(row, "signal_id", getattr(row, "trade_key", "")),
                    "symbol": getattr(row, "symbol", ""),
                    "candidate": getattr(row, "candidate", ""),
                    "would_have_exited_at_cp60": True,
                    "kept_due_to_protection": True,
                    "net_if_kept": getattr(row, "net_return_at_cost", np.nan),
                    "net_if_cp60_exited": getattr(row, "checkpoint_net_at_cost", np.nan),
                    "slot_blocked_minutes": blocked,
                    "new_trade_missed_due_to_slot": False,
                    "missed_trade_id": "",
                    "missed_trade_net20": np.nan,
                    "total_effect": getattr(row, "net_return_at_cost", np.nan) - getattr(row, "checkpoint_net_at_cost", np.nan),
                }
            )
            continue
        for missed_row in missed.itertuples(index=False):
            rows.append(
                {
                    "protection_rule": rule,
                    "protected_trade_id": getattr(row, "signal_id", getattr(row, "trade_key", "")),
                    "symbol": getattr(row, "symbol", ""),
                    "candidate": getattr(row, "candidate", ""),
                    "would_have_exited_at_cp60": True,
                    "kept_due_to_protection": True,
                    "net_if_kept": getattr(row, "net_return_at_cost", np.nan),
                    "net_if_cp60_exited": getattr(row, "checkpoint_net_at_cost", np.nan),
                    "slot_blocked_minutes": blocked,
                    "new_trade_missed_due_to_slot": True,
                    "missed_trade_id": getattr(missed_row, "trade_key", ""),
                    "missed_trade_net20": getattr(missed_row, "effective_net_return", np.nan),
                    "total_effect": (
                        getattr(row, "net_return_at_cost", np.nan)
                        - getattr(row, "checkpoint_net_at_cost", np.nan)
                        - getattr(missed_row, "effective_net_return", 0.0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _write_notes(root: Path, protection: pd.DataFrame, context: pd.DataFrame) -> None:
    lines = [
        "# v1.3D CP60 Context Protection Audit",
        "",
        "Purpose: diagnose whether CP60 false exits concentrate in protected continuation contexts.",
        "Status: offline diagnostic only. No live shadow or primary rule changes.",
        "",
        "## Protection Counterfactuals",
    ]
    if protection.empty:
        lines.append("- No protection rows.")
    else:
        for row in protection.sort_values("delta_vs_CP60", ascending=False).itertuples(index=False):
            lines.append(
                f"- {row.rule}: net20={row.protected_CP60_net20:.4%}, "
                f"delta_vs_CP60={row.delta_vs_CP60:.4%}, false_saved={row.false_exits_saved}, "
                f"true_lost={row.true_good_exits_lost}."
            )
        lines.append("")
        lines.append(
            "Field definitions: false_saved / true_lost count selected trades whose CP60 exit was protected; "
            "protected_cp60_exits is the same selected-trade protected count before exit-class filtering."
        )
    lines.append("")
    lines.append("## Context Cross Table")
    if context.empty:
        lines.append("- No context rows.")
    else:
        for row in context.itertuples(index=False):
            lines.append(
                f"- {row.context}: exits={row.cp_exit_count}, false={row.false_exit_rate:.2%} "
                f"sum_delta={row.sum_delta_vs_keep:.4%}."
            )
    lines.extend(
        [
            "",
            "## Discipline",
            "- Do not upgrade CP60_v2 unless protection improves portfolio net20, month-cap, and slot impact together.",
            "- A high false-exit rate alone is insufficient if true weak exits still dominate sum_delta.",
        ]
    )
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v13d_cp60_context_protection(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V13DConfig = V13DConfig(),
) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    sample = _prepare_sample(feature_path, instruments, config, root, cfg.v10a)
    context = _context_cross_table(sample, cfg.neutral_delta)
    protection, ledger, skipped = _protection_counterfactual(sample, cfg.neutral_delta)
    slot = _slot_impact_of_protection(ledger)
    outputs = {
        "context_cross_table": root / "context_cross_table.csv",
        "protection_counterfactual": root / "protection_counterfactual.csv",
        "slot_impact_of_protection": root / "slot_impact_of_protection.csv",
        "protected_trade_ledger": root / "protected_trade_ledger.csv",
        "protected_skipped_candidates": root / "protected_skipped_candidates.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    context.to_csv(outputs["context_cross_table"], index=False)
    protection.to_csv(outputs["protection_counterfactual"], index=False)
    slot.to_csv(outputs["slot_impact_of_protection"], index=False)
    ledger.to_csv(outputs["protected_trade_ledger"], index=False)
    skipped.to_csv(outputs["protected_skipped_candidates"], index=False)
    _write_notes(root, protection, context)
    return outputs


__all__ = [
    "REPORT_ROOT",
    "V13DConfig",
    "write_v13d_cp60_context_protection",
]
