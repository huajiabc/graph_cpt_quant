"""v1.3A checkpoint robustness and integration.

This report validates whether the post-entry no-follow-through checkpoint found
in v1.3 is stable enough to consider as a future shadow slot-management rule.
It deliberately does not search for new entry signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v09b import FOCAL_COST, _max_contribution, _month_cap_expectancy, _pool_trades
from pressure_graph.reports.v09d import _period_hours
from pressure_graph.reports.v10a_cic_basket_portfolio import V10AConfig, _load_or_build_trades
from pressure_graph.reports.v10c_burst_phase_allocation import _add_asof_burst_phase


REPORT_ROOT = Path("reports/v1_3a_checkpoint_robustness")
POOL_NAME = "P2_CIC1_CIC2_COMBINED"
CORE_MAX_POSITIONS = 8
O6_MIN_BURST_COUNT = 9
O6_MAX_SLOTS = 4
O6_CIC1_SIZE = 0.50
O6_CIC2_SIZE = 0.25
TIME_GRID_MINUTES = (30, 60, 90, 120)
THRESHOLD_GRID = (-0.005, 0.0, 0.005, 0.01)
STRESS_COSTS = (10.0, 20.0, 30.0, 50.0)
PRICE_COLUMNS = ("exchange", "symbol", "feature_time", "high", "low", "close")


@dataclass(frozen=True)
class V13AConfig:
    report_root: Path = REPORT_ROOT
    v10a: V10AConfig = V10AConfig()


@dataclass(frozen=True)
class CheckpointSpec:
    spec_id: str
    checkpoint_minutes: int = 60
    threshold: float = 0.0
    variant: str = "net_lte_threshold"
    cost_bps: float = FOCAL_COST


@dataclass(frozen=True)
class PortfolioSpec:
    portfolio_id: str
    checkpoint_enabled: bool
    overflow_enabled: bool
    checkpoint: CheckpointSpec = CheckpointSpec("hold_baseline")


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce")


def _pool_base20(trades: pd.DataFrame) -> pd.DataFrame:
    pool = _pool_trades(trades, POOL_NAME)
    if pool.empty:
        return pool.copy()
    pool = pool[pd.to_numeric(pool["cost_single_side_bps"], errors="coerce").eq(FOCAL_COST)].copy()
    for col in ("signal_time", "entry_time", "exit_time"):
        pool[col] = pd.to_datetime(pool[col], utc=True, errors="coerce")
    for col in ("entry_price", "gross_return", "funding_cost", "net_return"):
        pool[col] = pd.to_numeric(pool[col], errors="coerce")
    pool["month"] = pool["entry_time"].dt.strftime("%Y-%m")
    return (
        pool.dropna(subset=["signal_time", "entry_time", "exit_time", "entry_price", "net_return"])
        .sort_values(["entry_time", "symbol", "candidate"])
        .reset_index(drop=True)
    )


def _load_price_frame(feature_path: Path, trades: pd.DataFrame) -> pd.DataFrame:
    symbols = sorted(set(trades["symbol"].dropna().astype(str)))
    start = pd.to_datetime(trades["entry_time"], utc=True).min() - pd.Timedelta("30min")
    end = pd.to_datetime(trades["entry_time"], utc=True).max() + pd.Timedelta(minutes=max(TIME_GRID_MINUTES) + 30)
    prices = pd.read_parquet(feature_path, columns=list(PRICE_COLUMNS))
    prices["feature_time"] = pd.to_datetime(prices["feature_time"], utc=True, errors="coerce")
    prices = prices[
        prices["symbol"].astype(str).isin(symbols)
        & prices["feature_time"].ge(start)
        & prices["feature_time"].le(end)
    ].copy()
    for col in ("high", "low", "close"):
        prices[col] = pd.to_numeric(prices[col], errors="coerce")
    return prices.dropna(subset=["feature_time", "symbol", "close"]).sort_values(["symbol", "feature_time"])


def _attach_checkpoint_path(pool: pd.DataFrame, prices: pd.DataFrame, minutes: int) -> pd.DataFrame:
    out = pool.copy()
    out["checkpoint_minutes"] = int(minutes)
    out["checkpoint_time"] = out["entry_time"] + pd.Timedelta(minutes=int(minutes))
    frames: list[pd.DataFrame] = []
    for symbol, group in out.groupby("symbol", sort=False):
        symbol_prices = prices[prices["symbol"].astype(str).eq(str(symbol))].sort_values("feature_time")
        local_rows = []
        for row in group.sort_values("checkpoint_time").itertuples(index=False):
            payload = row._asdict()
            entry = pd.Timestamp(payload["entry_time"])
            checkpoint = pd.Timestamp(payload["checkpoint_time"])
            window = symbol_prices[
                symbol_prices["feature_time"].gt(entry)
                & symbol_prices["feature_time"].le(checkpoint)
            ]
            if window.empty:
                payload["checkpoint_price_time"] = pd.NaT
                payload["checkpoint_price"] = np.nan
                payload["checkpoint_mfe"] = np.nan
                payload["checkpoint_mae"] = np.nan
            else:
                last = window.iloc[-1]
                entry_price = float(payload["entry_price"])
                payload["checkpoint_price_time"] = last["feature_time"]
                payload["checkpoint_price"] = float(last["close"])
                payload["checkpoint_mfe"] = float(window["high"].max() / entry_price - 1.0)
                payload["checkpoint_mae"] = float(window["low"].min() / entry_price - 1.0)
            local_rows.append(payload)
        frames.append(pd.DataFrame(local_rows))
    out = pd.concat(frames, ignore_index=True) if frames else out
    out["checkpoint_gross_return"] = out["checkpoint_price"] / out["entry_price"] - 1.0
    out["checkpoint_price_covered"] = out["checkpoint_price"].notna()
    return out.sort_values(["entry_time", "symbol", "candidate"]).reset_index(drop=True)


def _net_at_cost(frame: pd.DataFrame, cost_bps: float) -> pd.Series:
    if "gross_return" in frame.columns:
        return _num(frame, "gross_return") - 2.0 * float(cost_bps) / 10_000.0
    return _num(frame, "net_return") - 2.0 * (float(cost_bps) - FOCAL_COST) / 10_000.0


def _prepare_checkpoint_sample(base: pd.DataFrame, prices: pd.DataFrame, spec: CheckpointSpec) -> pd.DataFrame:
    out = _attach_checkpoint_path(base, prices, spec.checkpoint_minutes)
    out["cost_single_side_bps"] = float(spec.cost_bps)
    out["net_return_at_cost"] = _net_at_cost(out, spec.cost_bps)
    out["checkpoint_net_at_cost"] = _num(out, "checkpoint_gross_return") - 2.0 * float(spec.cost_bps) / 10_000.0
    return out


def _checkpoint_exit_mask(sample: pd.DataFrame, spec: CheckpointSpec) -> pd.Series:
    checkpoint_before_exit = pd.to_datetime(sample["checkpoint_time"], utc=True, errors="coerce") < pd.to_datetime(
        sample["exit_time"], utc=True, errors="coerce"
    )
    base = sample["checkpoint_price_covered"].fillna(False).astype(bool) & checkpoint_before_exit
    net = _num(sample, "checkpoint_net_at_cost")
    mfe = _num(sample, "checkpoint_mfe")
    mae = _num(sample, "checkpoint_mae")
    if spec.variant == "net_lte_threshold":
        return base & net.le(float(spec.threshold))
    if spec.variant == "net_lte_0_and_mfe_lt_1p5":
        return base & net.le(0.0) & mfe.lt(0.015)
    if spec.variant == "net_lte_0_and_mae_lte_minus_1":
        return base & net.le(0.0) & mae.le(-0.01)
    if spec.variant == "net_lte_0_and_no_plus_1p5":
        return base & net.le(0.0) & mfe.lt(0.015)
    raise KeyError(spec.variant)


def _apply_checkpoint(sample: pd.DataFrame, spec: CheckpointSpec, enabled: bool) -> pd.DataFrame:
    out = sample.copy()
    out["checkpoint_spec_id"] = spec.spec_id
    out["checkpoint_threshold"] = spec.threshold
    out["checkpoint_variant"] = spec.variant
    out["checkpoint_early_exit"] = False
    out["effective_exit_time"] = out["exit_time"]
    out["effective_net_return"] = out["net_return_at_cost"]
    if enabled:
        early = _checkpoint_exit_mask(out, spec)
        out["checkpoint_early_exit"] = early
        out.loc[early, "effective_exit_time"] = out.loc[early, "checkpoint_time"]
        out.loc[early, "effective_net_return"] = out.loc[early, "checkpoint_net_at_cost"]
    out["effective_holding_minutes"] = (
        pd.to_datetime(out["effective_exit_time"], utc=True, errors="coerce")
        - pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    ).dt.total_seconds() / 60.0
    return out


def _overflow_size(row: pd.Series | dict[str, Any]) -> float:
    candidate = str(row.get("candidate", ""))
    if candidate == "CIC1_beta_extreme":
        return O6_CIC1_SIZE
    if candidate == "CIC2_beta_broad":
        return O6_CIC2_SIZE
    return 0.0


def _overflow_allowed(row: pd.Series | dict[str, Any], enabled: bool) -> bool:
    return bool(enabled and int(row.get("burst_count_so_far", 0)) >= O6_MIN_BURST_COUNT and _overflow_size(row) > 0)


def _ledger_row(
    row: pd.Series,
    *,
    sleeve: str,
    weight: float,
    status: str,
    reason: str = "",
) -> dict[str, Any]:
    payload = row.to_dict()
    net = float(pd.to_numeric(row.get("effective_net_return", np.nan), errors="coerce"))
    payload["sleeve"] = sleeve
    payload["exposure_weight"] = float(weight)
    payload["selection_status"] = status
    payload["skip_reason"] = reason
    payload["weighted_return"] = net * float(weight)
    return payload


def _simulate_portfolio(sample: pd.DataFrame, spec: PortfolioSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = _apply_checkpoint(sample, spec.checkpoint, spec.checkpoint_enabled)
    active_core: list[dict[str, Any]] = []
    active_overflow: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for _, row in pool.sort_values(["entry_time", "symbol"]).iterrows():
        entry = pd.Timestamp(row["entry_time"])
        active_core = [item for item in active_core if pd.Timestamp(item["exit_time"]) > entry]
        active_overflow = [item for item in active_overflow if pd.Timestamp(item["exit_time"]) > entry]
        active = [*active_core, *active_overflow]
        if str(row["symbol"]) in {str(item["symbol"]) for item in active}:
            skipped_rows.append(_ledger_row(row, sleeve="skipped", weight=0.0, status="skipped", reason="symbol_already_active"))
            continue
        if len(active_core) < CORE_MAX_POSITIONS:
            ledger_rows.append(_ledger_row(row, sleeve="core", weight=1.0, status="selected"))
            active_core.append({"symbol": str(row["symbol"]), "exit_time": row["effective_exit_time"]})
            continue
        if _overflow_allowed(row, spec.overflow_enabled) and len(active_overflow) < O6_MAX_SLOTS:
            size = _overflow_size(row)
            ledger_rows.append(_ledger_row(row, sleeve="overflow", weight=size, status="selected"))
            active_overflow.append({"symbol": str(row["symbol"]), "exit_time": row["effective_exit_time"], "weight": size})
            continue
        reason = "overflow_full" if _overflow_allowed(row, spec.overflow_enabled) else "portfolio_full_not_overflow_eligible"
        skipped_rows.append(_ledger_row(row, sleeve="skipped", weight=0.0, status="skipped", reason=reason))
    ledger = pd.DataFrame(ledger_rows)
    skipped = pd.DataFrame(skipped_rows)
    if not ledger.empty:
        ledger["portfolio_id"] = spec.portfolio_id
    if not skipped.empty:
        skipped["portfolio_id"] = spec.portfolio_id
    return ledger, skipped


def _period_return(ledger: pd.DataFrame, period: str) -> pd.Series:
    if ledger.empty:
        return pd.Series(dtype=float)
    data = ledger.copy()
    data["entry_time"] = pd.to_datetime(data["entry_time"], utc=True, errors="coerce")
    if period == "month":
        key = data["entry_time"].dt.strftime("%Y-%m")
    elif period == "day":
        key = data["entry_time"].dt.strftime("%Y-%m-%d")
    elif period == "burst":
        key = data.get("burst_id", pd.Series("unknown", index=data.index)).astype(str)
    else:
        raise KeyError(period)
    return _num(data, "weighted_return").groupby(key, sort=False, dropna=False).sum() / CORE_MAX_POSITIONS


def _drawdown(contribution: pd.Series) -> float:
    if contribution.empty:
        return np.nan
    equity = contribution.cumsum()
    return float((equity - equity.cummax()).min())


def _summary_row(spec: PortfolioSpec, ledger: pd.DataFrame, skipped: pd.DataFrame) -> dict[str, Any]:
    weighted = _num(ledger, "weighted_return")
    core = ledger[ledger.get("sleeve", pd.Series(dtype=str)).astype(str).eq("core")] if not ledger.empty else ledger
    overflow = ledger[ledger.get("sleeve", pd.Series(dtype=str)).astype(str).eq("overflow")] if not ledger.empty else ledger
    skipped_net = _num(skipped, "net_return_at_cost")
    contribution = weighted / CORE_MAX_POSITIONS
    effective = ledger.copy()
    if not effective.empty:
        effective["exit_time"] = effective["effective_exit_time"]
        effective["holding_minutes"] = effective["effective_holding_minutes"]
    period_hours = _period_hours(effective)
    holding_hours = _num(effective, "effective_holding_minutes").mul(_num(effective, "exposure_weight")).sum() / 60.0
    overflow_exposure = _num(overflow, "exposure_weight").sum()
    return {
        "portfolio_id": spec.portfolio_id,
        "checkpoint_enabled": spec.checkpoint_enabled,
        "overflow_enabled": spec.overflow_enabled,
        "checkpoint_minutes": spec.checkpoint.checkpoint_minutes,
        "checkpoint_threshold": spec.checkpoint.threshold,
        "checkpoint_variant": spec.checkpoint.variant,
        "cost_single_side_bps": spec.checkpoint.cost_bps,
        "selected_trades": int(len(ledger)),
        "core_trades": int(len(core)),
        "overflow_trades": int(len(overflow)),
        "skipped_trades": int(len(skipped)),
        "early_exit_trades": int(ledger.get("checkpoint_early_exit", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "portfolio_net20": float(contribution.sum()) if len(contribution) else 0.0,
        "core_net20": float(_num(core, "weighted_return").sum() / CORE_MAX_POSITIONS) if len(core) else 0.0,
        "overflow_net20": float(_num(overflow, "weighted_return").sum() / CORE_MAX_POSITIONS) if len(overflow) else 0.0,
        "selected_effective_net_avg": float(_num(ledger, "effective_net_return").mean()) if len(ledger) else np.nan,
        "skipped_counterfactual_net_avg": float(skipped_net.mean()) if len(skipped_net) else np.nan,
        "selected_minus_skipped": float(_num(ledger, "effective_net_return").mean() - skipped_net.mean()) if len(ledger) and len(skipped_net) else np.nan,
        "return_per_capital_day": float(contribution.sum() / period_hours * 24.0) if period_hours else np.nan,
        "capital_utilization": float(holding_hours / (period_hours * CORE_MAX_POSITIONS)) if period_hours else np.nan,
        "max_drawdown_proxy": _drawdown(contribution),
        "worst_burst_net20": float(_period_return(ledger, "burst").min()) if not ledger.empty else np.nan,
        "worst_day_net20": float(_period_return(ledger, "day").min()) if not ledger.empty else np.nan,
        "worst_month_net20": float(_period_return(ledger, "month").min()) if not ledger.empty else np.nan,
        "month_cap35_net20": _month_cap_expectancy(ledger.assign(net_return=_num(ledger, "weighted_return"))),
        "max_symbol_contribution": _max_contribution(ledger.assign(net_return=_num(ledger, "weighted_return")), "symbol"),
        "extra_exposure": float(overflow_exposure),
        "return_per_extra_exposure": float(_num(overflow, "weighted_return").sum() / overflow_exposure) if overflow_exposure else np.nan,
    }


def _base_specs() -> dict[str, PortfolioSpec]:
    checkpoint = CheckpointSpec("60m_net_lte_0", 60, 0.0, "net_lte_threshold", FOCAL_COST)
    hold = CheckpointSpec("hold", 60, 0.0, "net_lte_threshold", FOCAL_COST)
    return {
        "S0_core": PortfolioSpec("S0_P2_MAX8_BASELINE", False, False, hold),
        "S1_checkpoint": PortfolioSpec("S1_P2_MAX8_CHECKPOINT_60M", True, False, checkpoint),
        "S2_o6": PortfolioSpec("S2_P2_MAX8_PLUS_O6", False, True, hold),
        "S3_checkpoint_o6": PortfolioSpec("S3_P2_MAX8_CHECKPOINT_60M_PLUS_O6", True, True, checkpoint),
    }


def _run_spec(sample: pd.DataFrame, spec: PortfolioSpec) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    ledger, skipped = _simulate_portfolio(sample, spec)
    return _summary_row(spec, ledger, skipped), ledger, skipped


def _checkpoint_time_sensitivity(base: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for minutes in TIME_GRID_MINUTES:
        spec = PortfolioSpec(
            f"time_{minutes}m",
            True,
            False,
            CheckpointSpec(f"{minutes}m_net_lte_0", minutes, 0.0, "net_lte_threshold", FOCAL_COST),
        )
        sample = _prepare_checkpoint_sample(base, prices, spec.checkpoint)
        row, _, _ = _run_spec(sample, spec)
        rows.append(row)
    return pd.DataFrame(rows)


def _checkpoint_threshold_sensitivity(base: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLD_GRID:
        spec = PortfolioSpec(
            f"threshold_{threshold:+.3f}",
            True,
            False,
            CheckpointSpec(f"60m_net_lte_{threshold:+.3f}", 60, threshold, "net_lte_threshold", FOCAL_COST),
        )
        sample = _prepare_checkpoint_sample(base, prices, spec.checkpoint)
        row, _, _ = _run_spec(sample, spec)
        rows.append(row)
    return pd.DataFrame(rows)


def _checkpoint_mfe_mae_variants(base: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variants = (
        ("net_lte_0", "net_lte_threshold"),
        ("net_lte_0_and_mfe_lt_1p5", "net_lte_0_and_mfe_lt_1p5"),
        ("net_lte_0_and_mae_lte_minus_1", "net_lte_0_and_mae_lte_minus_1"),
        ("net_lte_0_and_no_plus_1p5", "net_lte_0_and_no_plus_1p5"),
    )
    for label, variant in variants:
        spec = PortfolioSpec(
            f"variant_{label}",
            True,
            False,
            CheckpointSpec(f"60m_{label}", 60, 0.0, variant, FOCAL_COST),
        )
        sample = _prepare_checkpoint_sample(base, prices, spec.checkpoint)
        row, _, _ = _run_spec(sample, spec)
        rows.append(row)
    return pd.DataFrame(rows)


def _checkpoint_o6_integration(base: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    ledgers = []
    skipped_frames = []
    for spec in _base_specs().values():
        sample = _prepare_checkpoint_sample(base, prices, spec.checkpoint)
        row, ledger, skipped = _run_spec(sample, spec)
        rows.append(row)
        ledgers.append(ledger)
        skipped_frames.append(skipped)
    return (
        pd.DataFrame(rows),
        pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame(),
        pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame(),
    )


def _checkpoint_cost_stress(base: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cost in STRESS_COSTS:
        checkpoint = CheckpointSpec(f"60m_net_lte_0_cost_{cost:g}", 60, 0.0, "net_lte_threshold", cost)
        for label, overflow in (("checkpoint", False), ("checkpoint_o6", True)):
            spec = PortfolioSpec(f"cost_{cost:g}_{label}", True, overflow, checkpoint)
            sample = _prepare_checkpoint_sample(base, prices, checkpoint)
            row, _, _ = _run_spec(sample, spec)
            rows.append(row)
    return pd.DataFrame(rows)


def _month_symbol_attribution(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame()
    rows = []
    for group_col in ("month", "symbol"):
        for value, group in ledger.groupby(group_col, sort=False, dropna=False):
            rows.append(
                {
                    "group_col": group_col,
                    "group_value": value,
                    "trades": int(len(group)),
                    "weighted_net20": float(_num(group, "weighted_return").sum() / CORE_MAX_POSITIONS),
                    "avg_effective_net": float(_num(group, "effective_net_return").mean()),
                    "early_exit_trades": int(group.get("checkpoint_early_exit", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                }
            )
    return pd.DataFrame(rows)


def _slot_release_attribution(baseline_ledger: pd.DataFrame, checkpoint_ledger: pd.DataFrame) -> pd.DataFrame:
    base_ids = set(baseline_ledger.get("signal_id", pd.Series(dtype=str)).astype(str))
    rows = []
    for row in checkpoint_ledger.itertuples(index=False):
        signal_id = str(getattr(row, "signal_id", ""))
        early = bool(getattr(row, "checkpoint_early_exit", False))
        new_trade = signal_id not in base_ids
        rows.append(
            {
                "signal_id": signal_id,
                "symbol": getattr(row, "symbol", ""),
                "candidate": getattr(row, "candidate", ""),
                "entry_time": getattr(row, "entry_time", pd.NaT),
                "checkpoint_time": getattr(row, "checkpoint_time", pd.NaT),
                "checkpoint_net": getattr(row, "checkpoint_net_at_cost", np.nan),
                "exit_by_checkpoint": early,
                "net_if_kept": getattr(row, "net_return_at_cost", np.nan),
                "net_if_exited": getattr(row, "effective_net_return", np.nan),
                "slot_released": early,
                "new_trade_entered_due_to_release": new_trade,
                "new_trade_net20": getattr(row, "effective_net_return", np.nan) if new_trade else np.nan,
                "avoidance_pnl": (getattr(row, "effective_net_return", np.nan) - getattr(row, "net_return_at_cost", np.nan))
                if early
                else 0.0,
                "opportunity_capture": getattr(row, "effective_net_return", np.nan) if new_trade else 0.0,
                "sleeve": getattr(row, "sleeve", ""),
                "exposure_weight": getattr(row, "exposure_weight", np.nan),
            }
        )
    return pd.DataFrame(rows)


def _candidate_family(candidate: object) -> str:
    text = str(candidate)
    if text.startswith("CIC1"):
        return "CIC1"
    if text.startswith("CIC2"):
        return "CIC2"
    return text or "unknown"


def _checkpoint_by_cic_type(baseline_ledger: pd.DataFrame, checkpoint_ledger: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base = baseline_ledger.copy()
    cp = checkpoint_ledger.copy()
    if not base.empty:
        base["cic_type"] = base["candidate"].map(_candidate_family)
    if not cp.empty:
        cp["cic_type"] = cp["candidate"].map(_candidate_family)
        cp["avoidance_pnl"] = _num(cp, "effective_net_return") - _num(cp, "net_return_at_cost")
        cp["false_exit"] = cp.get("checkpoint_early_exit", pd.Series(False, index=cp.index)).fillna(False).astype(bool) & cp[
            "avoidance_pnl"
        ].lt(0)
    cic_types = sorted(set(base.get("cic_type", pd.Series(dtype=str)).dropna()) | set(cp.get("cic_type", pd.Series(dtype=str)).dropna()))
    for cic_type in cic_types:
        b = base[base["cic_type"].eq(cic_type)] if not base.empty else pd.DataFrame()
        c = cp[cp["cic_type"].eq(cic_type)] if not cp.empty else pd.DataFrame()
        exits = c.get("checkpoint_early_exit", pd.Series(False, index=c.index)).fillna(False).astype(bool) if not c.empty else pd.Series(dtype=bool)
        exit_rows = c[exits].copy() if not c.empty else pd.DataFrame()
        false_exits = exit_rows.get("false_exit", pd.Series(False, index=exit_rows.index)).fillna(False).astype(bool) if not exit_rows.empty else pd.Series(dtype=bool)
        rows.append(
            {
                "cic_type": cic_type,
                "baseline_trades": int(len(b)),
                "checkpoint_trades": int(len(c)),
                "baseline_net20_avg": float(_num(b, "net_return_at_cost").mean()) if len(b) else np.nan,
                "checkpoint_effective_net20_avg": float(_num(c, "effective_net_return").mean()) if len(c) else np.nan,
                "checkpoint_kept_counterfactual_net20_avg": float(_num(c, "net_return_at_cost").mean()) if len(c) else np.nan,
                "checkpoint_vs_kept_delta_avg": float((_num(c, "effective_net_return") - _num(c, "net_return_at_cost")).mean())
                if len(c)
                else np.nan,
                "checkpoint_exits": int(exits.sum()) if len(exits) else 0,
                "checkpoint_exit_rate": float(exits.mean()) if len(exits) else np.nan,
                "avoided_loss_sum": float(_num(exit_rows, "avoidance_pnl").sum()) if len(exit_rows) else 0.0,
                "avoided_loss_avg": float(_num(exit_rows, "avoidance_pnl").mean()) if len(exit_rows) else np.nan,
                "false_exit_count": int(false_exits.sum()) if len(false_exits) else 0,
                "false_exit_rate": float(false_exits.mean()) if len(false_exits) else np.nan,
                "false_exit_cost_sum": float(_num(exit_rows[false_exits], "avoidance_pnl").sum()) if len(false_exits) else 0.0,
                "true_prune_count": int((exits.sum() - false_exits.sum())) if len(exits) else 0,
            }
        )
    return pd.DataFrame(rows)


def _write_notes(
    root: Path,
    integration: pd.DataFrame,
    time_sensitivity: pd.DataFrame,
    threshold: pd.DataFrame,
) -> None:
    lines = [
        "# v1.3A Checkpoint Robustness & Integration",
        "",
        "Purpose: validate whether the post-entry no-follow-through checkpoint is robust and compatible with O6.",
        "Status: research-only. No paper-live, shadow selector, or real-live permission change.",
        "",
        "## O6 Integration",
    ]
    if integration.empty:
        lines.append("- No integration rows.")
    else:
        for row in integration.sort_values("portfolio_net20", ascending=False).itertuples(index=False):
            lines.append(
                f"- {row.portfolio_id}: net20={row.portfolio_net20:.4%}, "
                f"early_exits={row.early_exit_trades}, overflow={row.overflow_trades}, "
                f"worst_month={row.worst_month_net20:.4%}."
            )
    lines.append("")
    lines.append("## Time Sensitivity")
    for row in time_sensitivity.sort_values("checkpoint_minutes").itertuples(index=False):
        lines.append(f"- {row.checkpoint_minutes}m: net20={row.portfolio_net20:.4%}, early_exits={row.early_exit_trades}.")
    lines.append("")
    lines.append("## Threshold Sensitivity")
    for row in threshold.sort_values("checkpoint_threshold").itertuples(index=False):
        lines.append(
            f"- threshold {row.checkpoint_threshold:.2%}: net20={row.portfolio_net20:.4%}, early_exits={row.early_exit_trades}."
        )
    lines.extend(
        [
            "",
            "## Discipline",
            "- Checkpoint decisions use only as-of checkpoint price path.",
            "- O6 integration keeps the existing late-burst overflow definition.",
            "- Promotion requires robustness across adjacent checkpoint times and acceptable risk envelope.",
        ]
    )
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v13a_checkpoint_robustness(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V13AConfig = V13AConfig(),
) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    trades = _load_or_build_trades(feature_path, instruments, config, root, cfg.v10a)
    base = _add_asof_burst_phase(_pool_base20(trades), "1h")
    if base.empty:
        raise ValueError("No P2 CIC trades available for v1.3A checkpoint robustness.")
    prices = _load_price_frame(feature_path, base)
    time_sensitivity = _checkpoint_time_sensitivity(base, prices)
    threshold = _checkpoint_threshold_sensitivity(base, prices)
    variants = _checkpoint_mfe_mae_variants(base, prices)
    integration, integration_ledger, integration_skipped = _checkpoint_o6_integration(base, prices)
    cost_stress = _checkpoint_cost_stress(base, prices)

    s1_ledger = integration_ledger[integration_ledger["portfolio_id"].eq("S1_P2_MAX8_CHECKPOINT_60M")].copy()
    s0_ledger = integration_ledger[integration_ledger["portfolio_id"].eq("S0_P2_MAX8_BASELINE")].copy()
    attribution = _slot_release_attribution(s0_ledger, s1_ledger)
    month_symbol = _month_symbol_attribution(s1_ledger)
    checkpoint_by_cic = _checkpoint_by_cic_type(s0_ledger, s1_ledger)

    outputs = {
        "checkpoint_time_sensitivity": root / "checkpoint_time_sensitivity.csv",
        "checkpoint_threshold_sensitivity": root / "checkpoint_threshold_sensitivity.csv",
        "checkpoint_mfe_mae_variants": root / "checkpoint_mfe_mae_variants.csv",
        "checkpoint_o6_integration": root / "checkpoint_o6_integration.csv",
        "checkpoint_cost_stress": root / "checkpoint_cost_stress.csv",
        "checkpoint_month_symbol_attribution": root / "checkpoint_month_symbol_attribution.csv",
        "checkpoint_by_cic_type": root / "checkpoint_by_cic_type.csv",
        "slot_release_attribution": root / "slot_release_attribution.csv",
        "integration_trade_ledger": root / "integration_trade_ledger.csv",
        "integration_skipped_candidates": root / "integration_skipped_candidates.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    time_sensitivity.to_csv(outputs["checkpoint_time_sensitivity"], index=False)
    threshold.to_csv(outputs["checkpoint_threshold_sensitivity"], index=False)
    variants.to_csv(outputs["checkpoint_mfe_mae_variants"], index=False)
    integration.to_csv(outputs["checkpoint_o6_integration"], index=False)
    cost_stress.to_csv(outputs["checkpoint_cost_stress"], index=False)
    month_symbol.to_csv(outputs["checkpoint_month_symbol_attribution"], index=False)
    checkpoint_by_cic.to_csv(outputs["checkpoint_by_cic_type"], index=False)
    attribution.to_csv(outputs["slot_release_attribution"], index=False)
    integration_ledger.to_csv(outputs["integration_trade_ledger"], index=False)
    integration_skipped.to_csv(outputs["integration_skipped_candidates"], index=False)
    _write_notes(root, integration, time_sensitivity, threshold)
    return outputs


__all__ = [
    "REPORT_ROOT",
    "V13AConfig",
    "write_v13a_checkpoint_robustness",
]
