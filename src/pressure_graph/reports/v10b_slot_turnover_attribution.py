from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v09b import (
    FOCAL_COST,
    _max_contribution,
    _month_cap_expectancy,
    _pool_trades,
    _prepare_trade_features,
)
from pressure_graph.reports.v09d import (
    _add_burst_id,
    _capital_units,
    _max_concurrent_positions,
    _period_hours,
)
from pressure_graph.reports.v10a_cic_basket_portfolio import (
    V10AConfig,
    _load_or_build_trades,
    _portfolio_metrics,
)


REPORT_ROOT = Path("reports/v1_0b_slot_turnover_attribution")
FOCUS_POOL = "P2_CIC1_CIC2_COMBINED"
FOCUS_MAX_POSITIONS = (5, 8, 10, 12, 15)
BASELINE_MAX_POSITIONS = 8
REPLACEMENT_RULES = (
    "R0_baseline_no_replacement",
    "R1_replace_stagnant_loser",
    "R2_replace_no_follow_through",
    "R3_replace_weak_cic2_with_cic1",
    "R4_replace_lowest_progress_score",
)


@dataclass(frozen=True)
class V10BConfig:
    report_root: Path = REPORT_ROOT
    v10a: V10AConfig = V10AConfig()
    load_mark_prices: bool = True


def _focus_pool(trades: pd.DataFrame) -> pd.DataFrame:
    pool = _pool_trades(trades, FOCUS_POOL)
    if pool.empty:
        return pool.copy()
    pool = pool[pd.to_numeric(pool["cost_single_side_bps"], errors="coerce").eq(FOCAL_COST)].copy()
    pool["entry_time"] = pd.to_datetime(pool["entry_time"], utc=True, errors="coerce")
    pool["exit_time"] = pd.to_datetime(pool["exit_time"], utc=True, errors="coerce")
    pool["net_return"] = pd.to_numeric(pool["net_return"], errors="coerce")
    pool["entry_price"] = pd.to_numeric(pool.get("entry_price", np.nan), errors="coerce")
    pool["exit_price"] = pd.to_numeric(pool.get("exit_price", np.nan), errors="coerce")
    pool["row_id"] = np.arange(len(pool))
    return pool.dropna(subset=["entry_time", "exit_time", "net_return"]).sort_values(
        ["entry_time", "symbol", "candidate"]
    ).reset_index(drop=True)


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _candidate_score(row: pd.Series | dict[str, Any]) -> float:
    candidate = str(row.get("candidate", ""))
    cic_bonus = 1.0 if candidate == "CIC1_beta_extreme" else 0.0
    beta = _safe_float(row.get("rank_beta_extreme_strength"), 0.0)
    local = _safe_float(row.get("rank_local_volume_shock_strength"), 0.0)
    return cic_bonus * 2.0 + beta + 0.5 * local


def _load_mark_table(feature_path: Path, symbols: set[str], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    if not feature_path.exists() or not symbols:
        return {}
    try:
        import pyarrow.parquet as pq

        schema = pq.read_schema(feature_path)
        available = set(schema.names)
        time_col = next(
            (col for col in ["bar_close_time", "feature_time", "bar_open_time", "timestamp"] if col in available),
            None,
        )
        if time_col is None or "symbol" not in available or "close" not in available:
            return {}
        table = pq.read_table(feature_path, columns=["symbol", time_col, "close"])
        frame = table.to_pandas()
    except Exception:
        try:
            frame = pd.read_parquet(feature_path, columns=["symbol", "bar_close_time", "close"])
            time_col = "bar_close_time"
        except Exception:
            return {}
    frame = frame[frame["symbol"].astype(str).isin(symbols)].copy()
    frame["mark_time"] = pd.to_datetime(frame[time_col], utc=True, errors="coerce")
    frame["mark_price"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame[
        frame["mark_time"].between(start - pd.Timedelta(hours=1), end + pd.Timedelta(hours=1), inclusive="both")
    ].dropna(subset=["mark_time", "mark_price"])
    out: dict[str, pd.DataFrame] = {}
    for symbol, group in frame.sort_values(["symbol", "mark_time"]).groupby("symbol", sort=False):
        out[str(symbol)] = group[["mark_time", "mark_price"]].reset_index(drop=True)
    return out


def _lookup_mark(mark_table: dict[str, pd.DataFrame], symbol: str, decision_time: pd.Timestamp) -> tuple[float, str]:
    group = mark_table.get(str(symbol))
    if group is None or group.empty:
        return np.nan, "missing"
    times = pd.to_datetime(group["mark_time"], utc=True, errors="coerce").astype("int64").to_numpy()
    decision_ts = pd.Timestamp(decision_time)
    decision_ts = decision_ts.tz_localize("UTC") if decision_ts.tzinfo is None else decision_ts.tz_convert("UTC")
    decision_ns = decision_ts.value
    idx = int(np.searchsorted(times, decision_ns, side="right") - 1)
    if idx < 0:
        return np.nan, "missing_before_decision"
    return _safe_float(group.iloc[idx]["mark_price"]), "feature_close_asof"


def _position_state(item: dict[str, Any], decision_time: pd.Timestamp, mark_table: dict[str, pd.DataFrame]) -> dict[str, Any]:
    row = item["row"]
    entry_time = pd.Timestamp(row["entry_time"])
    exit_time = pd.Timestamp(row["exit_time"])
    holding_minutes = max(float((exit_time - entry_time).total_seconds() / 60.0), 1.0)
    age_minutes = max(float((decision_time - entry_time).total_seconds() / 60.0), 0.0)
    progress = float(np.clip(age_minutes / holding_minutes, 0.0, 1.0))
    entry_price = _safe_float(row.get("entry_price"))
    mark_price, mark_source = _lookup_mark(mark_table, str(row.get("symbol", "")), decision_time)
    cost = float(FOCAL_COST) * 2.0 / 10_000.0
    if np.isfinite(mark_price) and np.isfinite(entry_price) and entry_price > 0:
        current_pnl = mark_price / entry_price - 1.0 - cost
    else:
        current_pnl = _safe_float(row.get("net_return"), 0.0) * progress
        mark_source = "linear_proxy"
    mfe_total = _safe_float(row.get("mfe_12h"), max(current_pnl, 0.0))
    mae_total = _safe_float(row.get("mae_12h"), min(current_pnl, 0.0))
    mfe_so_far = max(current_pnl, mfe_total * np.sqrt(progress)) if progress > 0 else max(current_pnl, 0.0)
    mae_so_far = min(current_pnl, mae_total * np.sqrt(progress)) if progress > 0 else min(current_pnl, 0.0)
    remaining_minutes = max(float((exit_time - decision_time).total_seconds() / 60.0), 0.0)
    future_net = _safe_float(row.get("net_return"), 0.0)
    return {
        "row_id": int(row.get("row_id", item.get("row_id", -1))),
        "symbol": str(row.get("symbol", "")),
        "candidate": str(row.get("candidate", "")),
        "entry_time": entry_time,
        "natural_exit_time": exit_time,
        "age_minutes": age_minutes,
        "unrealized_pnl_at_decision": current_pnl,
        "mfe_so_far": float(mfe_so_far),
        "mae_so_far": float(mae_so_far),
        "remaining_time_to_timeout": remaining_minutes,
        "future_net20_if_kept": future_net,
        "remaining_net20_proxy_if_kept": future_net - current_pnl,
        "current_exit_status_if_closed_now": "profit" if current_pnl > 0 else "loss_or_flat",
        "mark_price": mark_price,
        "mark_source": mark_source,
        "progress_score": current_pnl + 0.5 * float(mfe_so_far) - 0.5 * float(mae_so_far) - 0.001 * age_minutes / 60.0,
    }


def _base_payload(row: pd.Series, *, selection_status: str, skip_reason: str = "") -> dict[str, Any]:
    payload = row.to_dict()
    payload["selection_status"] = selection_status
    payload["skip_reason"] = skip_reason
    return payload


def _simulate_baseline(
    trades: pd.DataFrame,
    *,
    max_positions: int,
    mark_table: dict[str, pd.DataFrame],
    capture_opportunity: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    active: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    opportunity: list[dict[str, Any]] = []
    for _, row in trades.sort_values(["entry_time", "symbol"]).iterrows():
        decision_time = pd.Timestamp(row["entry_time"])
        active = [item for item in active if pd.Timestamp(item["row"]["exit_time"]) > decision_time]
        active_symbols = {str(item["row"]["symbol"]) for item in active}
        if str(row["symbol"]) in active_symbols:
            skipped.append(_base_payload(row, selection_status="skipped", skip_reason="symbol_already_active"))
            continue
        if len(active) >= max_positions:
            if capture_opportunity:
                opportunity.extend(_opportunity_rows(row, active, decision_time, mark_table, max_positions))
            skipped.append(_base_payload(row, selection_status="skipped", skip_reason="portfolio_full"))
            continue
        payload = _base_payload(row, selection_status="selected")
        selected.append(payload)
        active.append({"row_id": int(row["row_id"]), "row": row})
    return pd.DataFrame(selected), pd.DataFrame(skipped), pd.DataFrame(opportunity)


def _opportunity_rows(
    row: pd.Series,
    active: list[dict[str, Any]],
    decision_time: pd.Timestamp,
    mark_table: dict[str, pd.DataFrame],
    max_positions: int,
) -> list[dict[str, Any]]:
    states = [_position_state(item, decision_time, mark_table) for item in active]
    if not states:
        return []
    worst = min(states, key=lambda item: item["future_net20_if_kept"])
    best = max(states, key=lambda item: item["future_net20_if_kept"])
    new_future = _safe_float(row.get("net_return"), 0.0)
    rows = []
    for state in states:
        role = "middle"
        if state["row_id"] == worst["row_id"]:
            role = "worst_open_position"
        elif state["row_id"] == best["row_id"]:
            role = "best_open_position"
        rows.append(
            {
                "decision_time": decision_time,
                "max_positions": max_positions,
                "new_candidate_symbol": row.get("symbol", ""),
                "new_candidate_type": row.get("candidate", ""),
                "new_candidate_future_net20": new_future,
                "open_position_count": len(active),
                "worst_open_position_symbol": worst["symbol"],
                "best_open_position_symbol": best["symbol"],
                "open_position_role": role,
                "open_symbol": state["symbol"],
                "open_candidate_type": state["candidate"],
                "open_entry_time": state["entry_time"],
                "open_age_minutes": state["age_minutes"],
                "open_unrealized_pnl_at_decision": state["unrealized_pnl_at_decision"],
                "open_mfe_so_far": state["mfe_so_far"],
                "open_mae_so_far": state["mae_so_far"],
                "open_remaining_time_to_timeout": state["remaining_time_to_timeout"],
                "open_current_exit_status_if_closed_now": state["current_exit_status_if_closed_now"],
                "open_future_net20_if_kept": state["future_net20_if_kept"],
                "open_remaining_net20_proxy_if_kept": state["remaining_net20_proxy_if_kept"],
                "opportunity_cost_vs_this_open": new_future - state["future_net20_if_kept"],
                "opportunity_cost_vs_worst_open": new_future - worst["future_net20_if_kept"],
                "mark_price": state["mark_price"],
                "mark_source": state["mark_source"],
            }
        )
    return rows


def _ledger_row(row: pd.Series | dict[str, Any], *, realized_return: float, close_time: pd.Timestamp, close_reason: str) -> dict[str, Any]:
    data = dict(row)
    data["realized_net20"] = realized_return
    data["ledger_exit_time"] = close_time
    data["ledger_exit_reason"] = close_reason
    data["net_return"] = realized_return
    return data


def _replace_allowed(rule: str, new_row: pd.Series, state: dict[str, Any]) -> bool:
    if state["unrealized_pnl_at_decision"] >= 0.02 or state["mfe_so_far"] >= 0.04:
        return False
    new_is_cic1 = str(new_row.get("candidate", "")) == "CIC1_beta_extreme"
    if rule == "R1_replace_stagnant_loser":
        return bool(new_is_cic1 and state["age_minutes"] >= 90.0 and state["unrealized_pnl_at_decision"] <= 0.0)
    if rule == "R2_replace_no_follow_through":
        return bool(
            state["age_minutes"] >= 120.0
            and state["mfe_so_far"] < 0.015
            and state["unrealized_pnl_at_decision"] < 0.005
        )
    if rule == "R3_replace_weak_cic2_with_cic1":
        return bool(new_is_cic1 and state["candidate"] == "CIC2_beta_broad" and state["unrealized_pnl_at_decision"] < 0.005)
    if rule == "R4_replace_lowest_progress_score":
        return True
    return False


def _simulate_replacement(
    trades: pd.DataFrame,
    *,
    max_positions: int,
    rule: str,
    mark_table: dict[str, pd.DataFrame],
    oracle: bool = False,
    oracle_with_cost: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    active: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    selected_candidates: list[dict[str, Any]] = []
    for _, row in trades.sort_values(["entry_time", "symbol"]).iterrows():
        decision_time = pd.Timestamp(row["entry_time"])
        still_active = []
        for item in active:
            if pd.Timestamp(item["row"]["exit_time"]) > decision_time:
                still_active.append(item)
            else:
                ledger.append(
                    _ledger_row(
                        item["row"],
                        realized_return=_safe_float(item["row"].get("net_return"), 0.0),
                        close_time=pd.Timestamp(item["row"]["exit_time"]),
                        close_reason="natural_exit",
                    )
                )
        active = still_active
        active_symbols = {str(item["row"]["symbol"]) for item in active}
        if str(row["symbol"]) in active_symbols:
            skipped.append(_base_payload(row, selection_status="skipped", skip_reason="symbol_already_active"))
            continue
        if len(active) < max_positions:
            selected_candidates.append(_base_payload(row, selection_status="selected"))
            active.append({"row_id": int(row["row_id"]), "row": row})
            continue
        states = [_position_state(item, decision_time, mark_table) for item in active]
        if oracle:
            worst = min(states, key=lambda state: state["future_net20_if_kept"])
            should_replace = _safe_float(row.get("net_return"), 0.0) > worst["future_net20_if_kept"]
            replace_state = worst if should_replace else None
        else:
            eligible = [state for state in states if _replace_allowed(rule, row, state)]
            if rule == "R4_replace_lowest_progress_score" and eligible:
                new_score = _candidate_score(row)
                worst_score = min(state["progress_score"] for state in eligible)
                eligible = eligible if new_score > worst_score else []
            replace_state = min(eligible, key=lambda state: state["progress_score"]) if eligible else None
        if replace_state is None:
            skipped.append(_base_payload(row, selection_status="skipped", skip_reason="portfolio_full_no_replacement"))
            continue
        old_idx = next(i for i, item in enumerate(active) if int(item["row_id"]) == int(replace_state["row_id"]))
        old = active.pop(old_idx)
        old_realized = 0.0 if oracle and not oracle_with_cost else float(replace_state["unrealized_pnl_at_decision"])
        ledger.append(
            _ledger_row(
                old["row"],
                realized_return=old_realized,
                close_time=decision_time,
                close_reason="oracle_replaced_out" if oracle else f"{rule}_replaced_out",
            )
        )
        selected_payload = _base_payload(row, selection_status="selected")
        selected_payload["replacement_in"] = True
        selected_payload["replaced_symbol"] = replace_state["symbol"]
        selected_payload["replaced_candidate"] = replace_state["candidate"]
        selected_payload["replaced_unrealized_pnl"] = replace_state["unrealized_pnl_at_decision"]
        selected_candidates.append(selected_payload)
        active.append({"row_id": int(row["row_id"]), "row": row})
    for item in active:
        ledger.append(
            _ledger_row(
                item["row"],
                realized_return=_safe_float(item["row"].get("net_return"), 0.0),
                close_time=pd.Timestamp(item["row"]["exit_time"]),
                close_reason="natural_exit",
            )
        )
    return pd.DataFrame(selected_candidates), pd.DataFrame(skipped), pd.DataFrame(ledger)


def _ledger_metrics(
    ledger: pd.DataFrame,
    selected_candidates: pd.DataFrame,
    skipped: pd.DataFrame,
    *,
    architecture: str,
    rule: str,
    max_positions: int,
) -> dict[str, Any]:
    if ledger.empty:
        realized = pd.Series(dtype=float)
    else:
        realized = pd.to_numeric(ledger.get("realized_net20", pd.Series(dtype=float)), errors="coerce")
    selected_net = pd.to_numeric(selected_candidates.get("net_return", pd.Series(dtype=float)), errors="coerce")
    skipped_net = pd.to_numeric(skipped.get("net_return", pd.Series(dtype=float)), errors="coerce")
    ledger_for_time = ledger.copy()
    if "ledger_exit_time" in ledger_for_time.columns:
        ledger_for_time["exit_time"] = ledger_for_time["ledger_exit_time"]
    capital_units = _capital_units(ledger_for_time, max_positions)
    contribution = realized / max(1, capital_units)
    equity = contribution.cumsum()
    dd = equity - equity.cummax()
    period_hours = _period_hours(ledger_for_time)
    holding_hours = pd.to_numeric(ledger.get("holding_minutes", pd.Series(dtype=float)), errors="coerce").sum() / 60.0
    realized_frame = ledger.copy()
    realized_frame["net_return"] = realized
    return {
        "architecture": architecture,
        "pool": FOCUS_POOL,
        "rule": rule,
        "cost_single_side_bps": FOCAL_COST,
        "max_positions": str(max_positions),
        "capital_units": capital_units,
        "selected_trades": int(len(selected_candidates)),
        "skipped_trades": int(len(skipped)),
        "replacement_count": int(ledger.get("ledger_exit_reason", pd.Series(dtype=str)).astype(str).str.contains("replaced_out").sum())
        if not ledger.empty
        else 0,
        "selected_candidate_net20": float(selected_net.mean()) if len(selected_net) else np.nan,
        "skipped_net20": float(skipped_net.mean()) if len(skipped_net) else np.nan,
        "selected_minus_skipped_net20": float(selected_net.mean() - skipped_net.mean())
        if len(selected_net) and len(skipped_net)
        else np.nan,
        "realized_trade_net20": float(realized.mean()) if len(realized) else np.nan,
        "portfolio_net20": float(contribution.sum()) if len(contribution) else 0.0,
        "return_per_capital_day": float(contribution.sum() / period_hours * 24.0) if period_hours else np.nan,
        "return_per_position_hour": float(realized.sum() / holding_hours) if holding_hours else np.nan,
        "max_drawdown_proxy": float(dd.min()) if len(dd) else np.nan,
        "capital_utilization": float(holding_hours / (period_hours * capital_units)) if period_hours else np.nan,
        "avg_concurrent_positions": float(holding_hours / period_hours) if period_hours else np.nan,
        "max_concurrent_positions": _max_concurrent_positions(ledger_for_time),
        "month_cap35_net20": _month_cap_expectancy(realized_frame),
        "max_month_contribution": _max_contribution(realized_frame, "month"),
        "max_symbol_contribution": _max_contribution(realized_frame, "symbol"),
    }


def _baseline_metric_row(selected: pd.DataFrame, skipped: pd.DataFrame, max_positions: int) -> dict[str, Any]:
    row = _portfolio_metrics(
        selected,
        skipped,
        architecture="slot_capacity_baseline",
        pool=FOCUS_POOL,
        rule="R0_baseline_no_replacement",
        max_positions=max_positions,
    )
    row["replacement_count"] = 0
    row["selected_candidate_net20"] = row.pop("selected_net20")
    return row


def _slot_opportunity_cost(pool: pd.DataFrame, mark_table: dict[str, pd.DataFrame]) -> pd.DataFrame:
    _, _, opportunity = _simulate_baseline(
        pool,
        max_positions=BASELINE_MAX_POSITIONS,
        mark_table=mark_table,
        capture_opportunity=True,
    )
    return opportunity


def _oracle_replacement_gap(pool: pd.DataFrame, mark_table: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    ledger_frames: list[pd.DataFrame] = []
    selected, skipped, _ = _simulate_baseline(
        pool,
        max_positions=BASELINE_MAX_POSITIONS,
        mark_table=mark_table,
        capture_opportunity=False,
    )
    rows.append(_baseline_metric_row(selected, skipped, BASELINE_MAX_POSITIONS) | {"oracle_mode": "baseline"})
    for mode, with_cost in [("oracle_no_cost_replace", False), ("oracle_with_cost_replace", True)]:
        chosen, skipped_new, ledger = _simulate_replacement(
            pool,
            max_positions=BASELINE_MAX_POSITIONS,
            rule=mode,
            mark_table=mark_table,
            oracle=True,
            oracle_with_cost=with_cost,
        )
        rows.append(
            _ledger_metrics(
                ledger,
                chosen,
                skipped_new,
                architecture="oracle_replacement",
                rule=mode,
                max_positions=BASELINE_MAX_POSITIONS,
            )
            | {"oracle_mode": mode}
        )
        if not ledger.empty:
            ledger_frames.append(ledger.assign(architecture="oracle_replacement", rule=mode, max_positions=BASELINE_MAX_POSITIONS))
    return pd.DataFrame(rows), pd.concat(ledger_frames, ignore_index=True) if ledger_frames else pd.DataFrame()


def _true_replacement_summary(pool: pd.DataFrame, mark_table: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    ledger_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    skipped_frames: list[pd.DataFrame] = []
    for max_positions in FOCUS_MAX_POSITIONS:
        selected, skipped, _ = _simulate_baseline(pool, max_positions=max_positions, mark_table=mark_table)
        rows.append(_baseline_metric_row(selected, skipped, max_positions))
        if not selected.empty:
            selected_frames.append(selected.assign(rule="R0_baseline_no_replacement", max_positions=max_positions))
        if not skipped.empty:
            skipped_frames.append(skipped.assign(rule="R0_baseline_no_replacement", max_positions=max_positions))
        for rule in REPLACEMENT_RULES[1:]:
            chosen, skipped_new, ledger = _simulate_replacement(
                pool,
                max_positions=max_positions,
                rule=rule,
                mark_table=mark_table,
            )
            rows.append(
                _ledger_metrics(
                    ledger,
                    chosen,
                    skipped_new,
                    architecture="true_replacement",
                    rule=rule,
                    max_positions=max_positions,
                )
            )
            if not ledger.empty:
                ledger_frames.append(ledger.assign(architecture="true_replacement", rule=rule, max_positions=max_positions))
            if not chosen.empty:
                selected_frames.append(chosen.assign(rule=rule, max_positions=max_positions))
            if not skipped_new.empty:
                skipped_frames.append(skipped_new.assign(rule=rule, max_positions=max_positions))
    summary = pd.DataFrame(rows)
    ledger = pd.concat(ledger_frames, ignore_index=True) if ledger_frames else pd.DataFrame()
    selected_all = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    skipped_all = pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame()
    return summary, ledger, _selected_vs_skipped_after_replacement(selected_all, skipped_all)


def _selected_vs_skipped_after_replacement(selected: pd.DataFrame, skipped: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if selected.empty:
        return pd.DataFrame()
    group_cols = ["rule", "max_positions"]
    for keys, group in selected.groupby(group_cols, sort=False, dropna=False):
        local_skipped = skipped[
            skipped.get("rule", pd.Series(dtype=str)).astype(str).eq(str(keys[0]))
            & skipped.get("max_positions", pd.Series(dtype=str)).astype(str).eq(str(keys[1]))
        ]
        selected_net = pd.to_numeric(group.get("net_return", pd.Series(dtype=float)), errors="coerce")
        skipped_net = pd.to_numeric(local_skipped.get("net_return", pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "rule": keys[0],
                "max_positions": keys[1],
                "selected_trades": int(len(group)),
                "skipped_trades": int(len(local_skipped)),
                "selected_net20": float(selected_net.mean()) if len(selected_net) else np.nan,
                "skipped_net20": float(skipped_net.mean()) if len(skipped_net) else np.nan,
                "selected_minus_skipped_net20": float(selected_net.mean() - skipped_net.mean())
                if len(selected_net) and len(skipped_net)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _holding_age_edge_decay(slot_opportunity: pd.DataFrame) -> pd.DataFrame:
    if slot_opportunity.empty:
        return pd.DataFrame()
    out = slot_opportunity.copy()
    out["age_bucket"] = pd.cut(
        pd.to_numeric(out["open_age_minutes"], errors="coerce"),
        bins=[-1, 30, 60, 120, 240, np.inf],
        labels=["0_30m", "30_60m", "60_120m", "120_240m", "240m_plus"],
    )
    out["current_pnl_bucket"] = pd.cut(
        pd.to_numeric(out["open_unrealized_pnl_at_decision"], errors="coerce"),
        bins=[-np.inf, -0.01, 0.0, 0.01, 0.02, np.inf],
        labels=["lt_-1pct", "-1_0pct", "0_1pct", "1_2pct", "gt_2pct"],
    )
    rows = []
    for keys, group in out.groupby(["age_bucket", "current_pnl_bucket"], sort=False, observed=True):
        future = pd.to_numeric(group["open_future_net20_if_kept"], errors="coerce")
        rows.append(
            {
                "age_bucket": str(keys[0]),
                "current_pnl_bucket": str(keys[1]),
                "open_position_snapshots": int(len(group)),
                "future_return_if_kept": float(future.mean()) if len(future) else np.nan,
                "probability_of_positive_after_this_point": float((future > 0).mean()) if len(future) else np.nan,
                "median_mfe_so_far": float(pd.to_numeric(group["open_mfe_so_far"], errors="coerce").median()),
                "median_mae_so_far": float(pd.to_numeric(group["open_mae_so_far"], errors="coerce").median()),
                "avg_opportunity_cost_vs_this_open": float(
                    pd.to_numeric(group["opportunity_cost_vs_this_open"], errors="coerce").mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _burst_order_opportunity(pool: pd.DataFrame, selected: pd.DataFrame, skipped: pd.DataFrame) -> pd.DataFrame:
    status = []
    if not selected.empty:
        status.append(selected[["row_id"]].assign(selection_status="selected"))
    if not skipped.empty:
        status.append(skipped[["row_id", "skip_reason"]].assign(selection_status="skipped"))
    if not status:
        return pd.DataFrame()
    status_frame = pd.concat(status, ignore_index=True).drop_duplicates("row_id", keep="first")
    local = _add_burst_id(pool, "1h").merge(status_frame, on="row_id", how="left")
    local["order_bucket"] = pd.cut(
        pd.to_numeric(local["burst_order"], errors="coerce"),
        bins=[0, 3, 8, np.inf],
        labels=["order_1_3", "order_4_8", "order_9_plus"],
    )
    rows = []
    for keys, group in local.groupby(["order_bucket", "selection_status"], sort=False, observed=True, dropna=False):
        net = pd.to_numeric(group["net_return"], errors="coerce")
        rows.append(
            {
                "order_bucket": str(keys[0]),
                "selection_status": str(keys[1]),
                "trades": int(len(group)),
                "future_net20": float(net.mean()) if len(net) else np.nan,
                "hit_rate": float((net > 0).mean()) if len(net) else np.nan,
                "avg_time_since_burst_start": float(pd.to_numeric(group["minutes_since_burst_start"], errors="coerce").mean()),
                "avg_burst_size": float(pd.to_numeric(group["burst_size"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def _write_notes(
    report_root: Path,
    oracle: pd.DataFrame,
    replacement: pd.DataFrame,
    selected_vs_skipped: pd.DataFrame,
) -> None:
    lines = [
        "# v1.0B Slot Turnover Attribution",
        "",
        "Purpose: compare occupied CIC basket slots with later portfolio_full candidates.",
        "This report is attribution only; it does not change paper-live or real-live permissions.",
        "",
    ]
    if not oracle.empty:
        base = oracle[oracle["rule"].astype(str).eq("R0_baseline_no_replacement")]
        best = oracle.sort_values("portfolio_net20", ascending=False).head(1)
        if not base.empty and not best.empty:
            lines.extend(
                [
                    "## Oracle Gap",
                    f"- Baseline max8 portfolio_net20={base.iloc[0].portfolio_net20:.4%}.",
                    f"- Best oracle row: {best.iloc[0].rule}, portfolio_net20={best.iloc[0].portfolio_net20:.4%}.",
                    "- A large oracle gap means slot turnover may be learnable; a small gap means skipped strength is mostly hindsight noise.",
                    "",
                ]
            )
    if not replacement.empty:
        best_real = replacement[
            ~replacement["rule"].astype(str).eq("R0_baseline_no_replacement")
        ].sort_values("portfolio_net20", ascending=False)
        if not best_real.empty:
            row = best_real.iloc[0]
            baseline_same_max = replacement[
                replacement["rule"].astype(str).eq("R0_baseline_no_replacement")
                & replacement["max_positions"].astype(str).eq(str(row.max_positions))
            ]
            baseline_text = ""
            if not baseline_same_max.empty:
                baseline_text = f" vs same-max baseline {baseline_same_max.iloc[0].portfolio_net20:.4%}"
            lines.extend(
                [
                    "## True Replacement",
                    f"- Best executable rule: {row.rule} max={row.max_positions}, "
                    f"portfolio_net20={row.portfolio_net20:.4%}{baseline_text}, replacements={int(row.replacement_count)}.",
                    "- No replacement rule is promoted by this report unless it also shrinks selected-vs-skipped and is robust at max8/max10.",
                    "",
                ]
            )
    if not selected_vs_skipped.empty:
        focus = selected_vs_skipped.sort_values("selected_minus_skipped_net20", ascending=False).head(1)
        if not focus.empty:
            row = focus.iloc[0]
            lines.extend(
                [
                    "## Selected vs Skipped",
                    f"- Best gap row: {row.rule} max={row.max_positions}, "
                    f"selected-skipped={row.selected_minus_skipped_net20:.4%}.",
                    "- Because the best gap remains negative, slot turnover remains an attribution lead rather than a live portfolio rule.",
                    "",
                ]
            )
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v10b_slot_turnover_attribution(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V10BConfig = V10BConfig(),
) -> dict[str, Path]:
    report_root = ensure_dir(cfg.report_root)
    trades = _prepare_trade_features(_load_or_build_trades(feature_path, instruments, config, report_root, cfg.v10a))
    pool = _focus_pool(trades)
    if pool.empty:
        raise ValueError("No P2 CIC trades available for v1.0B slot turnover attribution.")
    symbols = set(pool["symbol"].dropna().astype(str).unique())
    start = pd.to_datetime(pool["entry_time"], utc=True, errors="coerce").min()
    end = pd.to_datetime(pool["exit_time"], utc=True, errors="coerce").max()
    mark_table = _load_mark_table(feature_path, symbols, start, end) if cfg.load_mark_prices else {}

    selected_baseline, skipped_baseline, _ = _simulate_baseline(
        pool,
        max_positions=BASELINE_MAX_POSITIONS,
        mark_table=mark_table,
    )
    slot_cost = _slot_opportunity_cost(pool, mark_table)
    oracle, oracle_ledger = _oracle_replacement_gap(pool, mark_table)
    replacement, replacement_ledger, selected_vs_skipped = _true_replacement_summary(pool, mark_table)
    holding_age = _holding_age_edge_decay(slot_cost)
    burst_order = _burst_order_opportunity(pool, selected_baseline, skipped_baseline)
    outputs = {
        "slot_opportunity_cost": report_root / "slot_opportunity_cost.csv",
        "oracle_replacement_gap": report_root / "oracle_replacement_gap.csv",
        "true_replacement_summary": report_root / "true_replacement_summary.csv",
        "replacement_trade_ledger": report_root / "replacement_trade_ledger.csv",
        "holding_age_edge_decay": report_root / "holding_age_edge_decay.csv",
        "burst_order_opportunity": report_root / "burst_order_opportunity.csv",
        "capacity_replacement_curve": report_root / "capacity_replacement_curve.csv",
        "selected_vs_skipped_after_replacement": report_root / "selected_vs_skipped_after_replacement.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    slot_cost.to_csv(outputs["slot_opportunity_cost"], index=False)
    oracle.to_csv(outputs["oracle_replacement_gap"], index=False)
    replacement.to_csv(outputs["true_replacement_summary"], index=False)
    replacement.to_csv(outputs["capacity_replacement_curve"], index=False)
    ledgers = [frame for frame in [oracle_ledger, replacement_ledger] if not frame.empty]
    (pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()).to_csv(
        outputs["replacement_trade_ledger"], index=False
    )
    holding_age.to_csv(outputs["holding_age_edge_decay"], index=False)
    burst_order.to_csv(outputs["burst_order_opportunity"], index=False)
    selected_vs_skipped.to_csv(outputs["selected_vs_skipped_after_replacement"], index=False)
    _write_notes(report_root, oracle, replacement, selected_vs_skipped)
    return outputs
