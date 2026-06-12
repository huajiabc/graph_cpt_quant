from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v09b import _month_cap_expectancy, _prepare_trade_features
from pressure_graph.reports.v09d import _add_burst_id
from pressure_graph.reports.v10a_cic_basket_portfolio import V10AConfig, _load_or_build_trades, _portfolio_metrics
from pressure_graph.reports.v10b_slot_turnover_attribution import (
    _focus_pool,
    _ledger_metrics,
    _load_mark_table,
    _position_state,
    _safe_float,
)


REPORT_ROOT = Path("reports/v1_0c_burst_phase_allocation")
BASELINE_MAX_POSITIONS = 8
PHASE_BUCKETS = ("order_1_3", "order_4_8", "order_9_14", "order_15_plus")


@dataclass(frozen=True)
class V10CConfig:
    report_root: Path = REPORT_ROOT
    v10a: V10AConfig = V10AConfig()
    load_mark_prices: bool = True


def _phase_bucket(count: int) -> str:
    if count <= 3:
        return "order_1_3"
    if count <= 8:
        return "order_4_8"
    if count <= 14:
        return "order_9_14"
    return "order_15_plus"


def _add_asof_burst_phase(trades: pd.DataFrame, window: str = "1h") -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    out = _add_burst_id(trades, window).sort_values(["entry_time", "symbol"]).copy()
    out["burst_count_so_far"] = out.groupby("burst_id", sort=False).cumcount() + 1
    starts = out.groupby("burst_id", sort=False)["entry_time"].transform("min")
    out["time_since_burst_start_so_far"] = (out["entry_time"] - starts).dt.total_seconds() / 60.0
    out["final_burst_size"] = out.groupby("burst_id", sort=False)["symbol"].transform("size")
    out["same_timestamp_peer_count"] = out.groupby("entry_time", sort=False)["symbol"].transform("size")
    out["burst_phase_bucket"] = out["burst_count_so_far"].map(lambda value: _phase_bucket(int(value)))
    out["asof_phase_passed"] = True
    out["uses_final_burst_size_for_decision"] = False
    return out.reset_index(drop=True)


def _burst_phase_asof_audit(pool: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "row_id",
        "symbol",
        "candidate",
        "entry_time",
        "burst_id",
        "burst_count_so_far",
        "time_since_burst_start_so_far",
        "final_burst_size",
        "same_timestamp_peer_count",
        "burst_phase_bucket",
        "asof_phase_passed",
        "uses_final_burst_size_for_decision",
    ]
    return pool[[col for col in cols if col in pool.columns]].copy()


def _cap_from_schedule(row: pd.Series | dict[str, Any], schedule: tuple[int, int, int]) -> int:
    count = int(row.get("burst_count_so_far", 1))
    if count <= 3:
        return schedule[0]
    if count <= 8:
        return schedule[1]
    return schedule[2]


def _simulate_capacity_rule(
    pool: pd.DataFrame,
    *,
    rule: str,
    cap_fn: Callable[[pd.Series], int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for _, row in pool.sort_values(["entry_time", "symbol"]).iterrows():
        entry = pd.Timestamp(row["entry_time"])
        active = [item for item in active if pd.Timestamp(item["exit_time"]) > entry]
        active_symbols = {str(item["symbol"]) for item in active}
        allowed = int(cap_fn(row))
        payload = row.to_dict()
        payload["rule"] = rule
        payload["allowed_positions_at_decision"] = allowed
        payload["active_positions_at_decision"] = len(active)
        if str(row["symbol"]) in active_symbols:
            payload["selection_status"] = "skipped"
            payload["skip_reason"] = "symbol_already_active"
            skipped_rows.append(payload)
            continue
        if allowed <= 0 or len(active) >= allowed:
            payload["selection_status"] = "skipped"
            payload["skip_reason"] = "portfolio_full"
            skipped_rows.append(payload)
            continue
        payload["selection_status"] = "selected"
        payload["skip_reason"] = ""
        selected_rows.append(payload)
        active.append({"symbol": str(row["symbol"]), "exit_time": row["exit_time"], "row_id": row["row_id"]})
    return pd.DataFrame(selected_rows), pd.DataFrame(skipped_rows)


def _summarize_selected(
    selected: pd.DataFrame,
    skipped: pd.DataFrame,
    *,
    architecture: str,
    rule: str,
    max_positions: int,
) -> dict[str, object]:
    row = _portfolio_metrics(
        selected,
        skipped,
        architecture=architecture,
        pool="P2_CIC1_CIC2_COMBINED",
        rule=rule,
        max_positions=max_positions,
    )
    row["late_phase_selected_trades"] = int(
        selected.get("burst_count_so_far", pd.Series(dtype=float)).ge(9).sum()
    ) if not selected.empty else 0
    row["late_phase_skipped_trades"] = int(
        skipped.get("burst_count_so_far", pd.Series(dtype=float)).ge(9).sum()
    ) if not skipped.empty else 0
    return row


def _dynamic_slot_ramp_summary(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    schedules = {
        "R0_fixed_max8": (8, 8, 8),
        "Ramp_A_3_5_8": (3, 5, 8),
        "Ramp_B_2_5_10": (2, 5, 10),
        "Ramp_C_4_6_10": (4, 6, 10),
        "Ramp_D_0_5_8": (0, 5, 8),
    }
    rows = []
    selected_frames = []
    skipped_frames = []
    for rule, schedule in schedules.items():
        selected, skipped = _simulate_capacity_rule(
            pool,
            rule=rule,
            cap_fn=lambda row, schedule=schedule: _cap_from_schedule(row, schedule),
        )
        rows.append(
            _summarize_selected(
                selected,
                skipped,
                architecture="dynamic_slot_ramp",
                rule=rule,
                max_positions=max(schedule),
            )
        )
        selected_frames.append(selected.assign(architecture="dynamic_slot_ramp", rule=rule))
        skipped_frames.append(skipped.assign(architecture="dynamic_slot_ramp", rule=rule))
    return (
        pd.DataFrame(rows),
        pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame(),
        pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame(),
    )


def _late_phase_expansion_summary(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rules = {
        "LateExpand_base5_late9_to8": lambda row: 8 if int(row["burst_count_so_far"]) >= 9 else 5,
        "LateExpand_base5_late9_to10": lambda row: 10 if int(row["burst_count_so_far"]) >= 9 else 5,
        "LateExpand_base5_late15_to10": lambda row: 10 if int(row["burst_count_so_far"]) >= 15 else 5,
        "LateExpand_base3_late9_to8": lambda row: 8 if int(row["burst_count_so_far"]) >= 9 else 3,
    }
    rows = []
    selected_frames = []
    skipped_frames = []
    for rule, cap_fn in rules.items():
        selected, skipped = _simulate_capacity_rule(pool, rule=rule, cap_fn=cap_fn)
        max_positions = 10 if "to10" in rule else 8
        rows.append(
            _summarize_selected(
                selected,
                skipped,
                architecture="late_phase_expansion",
                rule=rule,
                max_positions=max_positions,
            )
        )
        selected_frames.append(selected.assign(architecture="late_phase_expansion", rule=rule))
        skipped_frames.append(skipped.assign(architecture="late_phase_expansion", rule=rule))
    return (
        pd.DataFrame(rows),
        pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame(),
        pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame(),
    )


def _close_ledger_row(row: pd.Series | dict[str, Any], *, realized_return: float, close_time: pd.Timestamp, close_reason: str) -> dict[str, Any]:
    data = dict(row)
    entry = pd.Timestamp(data.get("entry_time"))
    data["realized_net20"] = realized_return
    data["ledger_exit_time"] = close_time
    data["ledger_exit_reason"] = close_reason
    data["net_return"] = realized_return
    data["holding_minutes"] = max(float((close_time - entry).total_seconds() / 60.0), 0.0)
    return data


def _beta_strength_pct(pool: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(pool.get("rank_beta_extreme_strength", pd.Series(dtype=float)), errors="coerce").rank(pct=True)


def _density_pct(pool: pd.DataFrame) -> pd.Series:
    market = pd.to_numeric(pool.get("rank_market_impulse_density", pd.Series(dtype=float)), errors="coerce").rank(pct=True)
    cluster = pd.to_numeric(pool.get("rank_cluster_impulse_density", pd.Series(dtype=float)), errors="coerce").rank(pct=True)
    return pd.concat([market, cluster], axis=1).max(axis=1)


def _late_replacement_allowed(rule: str, new_row: pd.Series, state: dict[str, Any]) -> bool:
    if int(new_row.get("burst_count_so_far", 0)) < 9:
        return False
    if int(state.get("entry_burst_count_so_far", 999)) > 3:
        return False
    if state["age_minutes"] < 60.0 or state["unrealized_pnl_at_decision"] > 0.0 or state["mfe_so_far"] >= 0.015:
        return False
    if rule == "LateReplace_1_late9_replace_early_weak":
        return True
    if rule == "LateReplace_2_late9_cic1_replace_early_weak":
        return str(new_row.get("candidate", "")) == "CIC1_beta_extreme"
    if rule == "LateReplace_3_late9_beta_high_replace_early_weak":
        return _safe_float(new_row.get("beta_strength_pct"), 0.0) >= 0.75
    if rule == "LateReplace_4_late9_density_high_replace_early_weak":
        return _safe_float(new_row.get("density_strength_pct"), 0.0) >= 0.75
    return False


def _simulate_late_replacement(
    pool: pd.DataFrame,
    *,
    rule: str,
    max_positions: int,
    mark_table: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    active: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    for _, row in pool.sort_values(["entry_time", "symbol"]).iterrows():
        entry = pd.Timestamp(row["entry_time"])
        still_active = []
        for item in active:
            if pd.Timestamp(item["row"]["exit_time"]) > entry:
                still_active.append(item)
            else:
                ledger_rows.append(
                    _close_ledger_row(
                        item["row"],
                        realized_return=_safe_float(item["row"].get("net_return"), 0.0),
                        close_time=pd.Timestamp(item["row"]["exit_time"]),
                        close_reason="natural_exit",
                    )
                )
        active = still_active
        active_symbols = {str(item["row"]["symbol"]) for item in active}
        payload = row.to_dict()
        payload["rule"] = rule
        payload["active_positions_at_decision"] = len(active)
        if str(row["symbol"]) in active_symbols:
            payload["selection_status"] = "skipped"
            payload["skip_reason"] = "symbol_already_active"
            skipped_rows.append(payload)
            continue
        if len(active) < max_positions:
            payload["selection_status"] = "selected"
            payload["skip_reason"] = ""
            selected_rows.append(payload)
            active.append({"row_id": int(row["row_id"]), "row": row})
            continue
        states = []
        for item in active:
            state = _position_state(item, entry, mark_table)
            state["entry_burst_count_so_far"] = int(item["row"].get("burst_count_so_far", 999))
            states.append(state)
        eligible = [state for state in states if _late_replacement_allowed(rule, row, state)]
        if not eligible:
            payload["selection_status"] = "skipped"
            payload["skip_reason"] = "portfolio_full_no_late_replacement"
            skipped_rows.append(payload)
            continue
        replace_state = min(eligible, key=lambda state: state["unrealized_pnl_at_decision"] + 0.5 * state["mfe_so_far"])
        old_idx = next(i for i, item in enumerate(active) if int(item["row_id"]) == int(replace_state["row_id"]))
        old = active.pop(old_idx)
        ledger_rows.append(
            _close_ledger_row(
                old["row"],
                realized_return=float(replace_state["unrealized_pnl_at_decision"]),
                close_time=entry,
                close_reason=f"{rule}_replaced_out",
            )
        )
        payload["selection_status"] = "selected"
        payload["skip_reason"] = ""
        payload["replacement_in"] = True
        payload["replaced_symbol"] = replace_state["symbol"]
        payload["replaced_unrealized_pnl"] = replace_state["unrealized_pnl_at_decision"]
        selected_rows.append(payload)
        active.append({"row_id": int(row["row_id"]), "row": row})
    for item in active:
        ledger_rows.append(
            _close_ledger_row(
                item["row"],
                realized_return=_safe_float(item["row"].get("net_return"), 0.0),
                close_time=pd.Timestamp(item["row"]["exit_time"]),
                close_reason="natural_exit",
            )
        )
    return pd.DataFrame(selected_rows), pd.DataFrame(skipped_rows), pd.DataFrame(ledger_rows)


def _late_signal_replacement_summary(pool: pd.DataFrame, mark_table: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    ledgers = []
    for max_positions in [8, 10]:
        selected, skipped = _simulate_capacity_rule(
            pool,
            rule="R0_baseline_no_replacement",
            cap_fn=lambda _row, max_positions=max_positions: max_positions,
        )
        rows.append(
            _summarize_selected(
                selected,
                skipped,
                architecture="late_signal_replacement",
                rule="R0_baseline_no_replacement",
                max_positions=max_positions,
            )
        )
        for rule in [
            "LateReplace_1_late9_replace_early_weak",
            "LateReplace_2_late9_cic1_replace_early_weak",
            "LateReplace_3_late9_beta_high_replace_early_weak",
            "LateReplace_4_late9_density_high_replace_early_weak",
        ]:
            chosen, skipped_new, ledger = _simulate_late_replacement(
                pool,
                rule=rule,
                max_positions=max_positions,
                mark_table=mark_table,
            )
            metric = _ledger_metrics(
                ledger,
                chosen,
                skipped_new,
                architecture="late_signal_replacement",
                rule=rule,
                max_positions=max_positions,
            )
            metric["late_phase_selected_trades"] = int(chosen.get("burst_count_so_far", pd.Series(dtype=float)).ge(9).sum())
            metric["late_phase_skipped_trades"] = int(skipped_new.get("burst_count_so_far", pd.Series(dtype=float)).ge(9).sum())
            rows.append(metric)
            if not ledger.empty:
                ledgers.append(ledger.assign(rule=rule, max_positions=max_positions))
    return pd.DataFrame(rows), pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()


def _burst_order_bucket_summary(pool: pd.DataFrame, selected: pd.DataFrame, skipped: pd.DataFrame) -> pd.DataFrame:
    status_frames = []
    if not selected.empty:
        status_frames.append(selected[["row_id"]].assign(selection_status="selected"))
    if not skipped.empty:
        status_frames.append(skipped[["row_id", "skip_reason"]].assign(selection_status="skipped"))
    status = pd.concat(status_frames, ignore_index=True) if status_frames else pd.DataFrame(columns=["row_id", "selection_status"])
    local = pool.merge(status.drop_duplicates("row_id"), on="row_id", how="left")
    rows = []
    for keys, group in local.groupby(["burst_phase_bucket", "selection_status"], sort=False, dropna=False):
        net = pd.to_numeric(group["net_return"], errors="coerce")
        rows.append(
            {
                "burst_phase_bucket": str(keys[0]),
                "selection_status": str(keys[1]),
                "trades": int(len(group)),
                "net10": float(pd.to_numeric(group.get("net_return_10bp", net), errors="coerce").mean()),
                "net20": float(net.mean()) if len(net) else np.nan,
                "tp_rate_proxy": float((net > 0).mean()) if len(net) else np.nan,
                "month_cap35_net20": _month_cap_expectancy(group),
                "avg_time_since_burst_start": float(
                    pd.to_numeric(group["time_since_burst_start_so_far"], errors="coerce").mean()
                ),
            }
        )
    all_rows = []
    for phase, group in local.groupby("burst_phase_bucket", sort=False, dropna=False):
        net = pd.to_numeric(group["net_return"], errors="coerce")
        all_rows.append(
            {
                "burst_phase_bucket": str(phase),
                "selection_status": "all",
                "trades": int(len(group)),
                "net10": float(pd.to_numeric(group.get("net_return_10bp", net), errors="coerce").mean()),
                "net20": float(net.mean()) if len(net) else np.nan,
                "tp_rate_proxy": float((net > 0).mean()) if len(net) else np.nan,
                "month_cap35_net20": _month_cap_expectancy(group),
                "avg_time_since_burst_start": float(
                    pd.to_numeric(group["time_since_burst_start_so_far"], errors="coerce").mean()
                ),
            }
        )
    return pd.DataFrame([*rows, *all_rows])


def _selected_vs_skipped_by_phase(selected: pd.DataFrame, skipped: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if not selected.empty:
        frames.append(selected.assign(selection_status="selected"))
    if not skipped.empty:
        frames.append(skipped.assign(selection_status="skipped"))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    rows = []
    for keys, group in combined.groupby(["architecture", "rule", "burst_phase_bucket", "selection_status"], sort=False):
        net = pd.to_numeric(group["net_return"], errors="coerce")
        rows.append(
            {
                "architecture": keys[0],
                "rule": keys[1],
                "burst_phase_bucket": keys[2],
                "selection_status": keys[3],
                "trades": int(len(group)),
                "net20": float(net.mean()) if len(net) else np.nan,
                "hit_rate": float((net > 0).mean()) if len(net) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _phase_weight_for_scheme(phase: str, scheme: str) -> float:
    weights = {
        "Tranche_30_30_40": {"order_1_3": 0.30, "order_4_8": 0.30, "order_9_14": 0.20, "order_15_plus": 0.20},
        "Tranche_20_30_50": {"order_1_3": 0.20, "order_4_8": 0.30, "order_9_14": 0.25, "order_15_plus": 0.25},
        "Tranche_10_30_60": {"order_1_3": 0.10, "order_4_8": 0.30, "order_9_14": 0.30, "order_15_plus": 0.30},
        "Tranche_equal_phase": {phase: 0.25 for phase in PHASE_BUCKETS},
    }
    return weights[scheme].get(phase, 0.0)


def _burst_tranche_allocation_summary(pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scheme in ["Tranche_30_30_40", "Tranche_20_30_50", "Tranche_10_30_60", "Tranche_equal_phase"]:
        burst_returns = []
        details = []
        for burst_id, group in pool.groupby("burst_id", sort=False, dropna=False):
            contribution = 0.0
            for phase, phase_group in group.groupby("burst_phase_bucket", sort=False, dropna=False):
                phase_budget = _phase_weight_for_scheme(str(phase), scheme)
                if phase_budget <= 0 or phase_group.empty:
                    continue
                contribution += phase_budget * float(pd.to_numeric(phase_group["net_return"], errors="coerce").mean())
            burst_returns.append(contribution)
            details.append({"burst_id": burst_id, "burst_return_net20": contribution, "month": group["month"].iloc[0]})
        returns = pd.Series(burst_returns, dtype="float64")
        detail = pd.DataFrame(details)
        rows.append(
            {
                "scheme": scheme,
                "bursts": int(len(returns)),
                "portfolio_burst_net20": float(returns.sum()) if len(returns) else np.nan,
                "avg_burst_net20": float(returns.mean()) if len(returns) else np.nan,
                "median_burst_net20": float(returns.median()) if len(returns) else np.nan,
                "burst_hit_rate": float((returns > 0).mean()) if len(returns) else np.nan,
                "worst_burst_net20": float(returns.min()) if len(returns) else np.nan,
                "month_cap35_burst_net20": _month_cap_expectancy(detail.rename(columns={"burst_return_net20": "net_return"}))
                if not detail.empty
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _write_notes(
    report_root: Path,
    baseline_net: float,
    ramp: pd.DataFrame,
    expansion: pd.DataFrame,
    replacement: pd.DataFrame,
    tranche: pd.DataFrame,
) -> None:
    lines = [
        "# v1.0C Burst-Phase Allocation",
        "",
        "Purpose: test whether as-of burst phase can carry CIC capacity better than fixed max8.",
        "This report is attribution only; it does not change paper-live or real-live permissions.",
        "",
    ]
    for title, frame in [
        ("Dynamic Slot Ramp", ramp),
        ("Late Phase Expansion", expansion),
        ("Late Signal Replacement", replacement),
    ]:
        if frame.empty:
            continue
        best = frame.sort_values("portfolio_net20", ascending=False).head(1).iloc[0]
        lines.extend(
            [
                f"## {title}",
                f"- Best row: {best.rule}, portfolio_net20={best.portfolio_net20:.4%} "
                f"vs fixed max8 baseline={baseline_net:.4%}.",
                f"- selected-skipped={best.selected_minus_skipped_net20:.4%} "
                f"late_selected={int(getattr(best, 'late_phase_selected_trades', 0))} "
                f"late_skipped={int(getattr(best, 'late_phase_skipped_trades', 0))}.",
                "",
            ]
        )
    if not tranche.empty:
        best = tranche.sort_values("portfolio_burst_net20", ascending=False).head(1).iloc[0]
        lines.extend(
            [
                "## Burst Tranche Allocation",
                f"- Best tranche scheme: {best.scheme}, portfolio_burst_net20={best.portfolio_burst_net20:.4%}.",
                "- Tranche allocation is a burst-budget diagnostic, not a slot-capacity replacement.",
                "",
            ]
        )
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v10c_burst_phase_allocation(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V10CConfig = V10CConfig(),
) -> dict[str, Path]:
    report_root = ensure_dir(cfg.report_root)
    trades = _prepare_trade_features(_load_or_build_trades(feature_path, instruments, config, report_root, cfg.v10a))
    pool = _add_asof_burst_phase(_focus_pool(trades), "1h")
    pool["beta_strength_pct"] = _beta_strength_pct(pool)
    pool["density_strength_pct"] = _density_pct(pool)
    if pool.empty:
        raise ValueError("No P2 CIC trades available for v1.0C burst-phase allocation.")

    symbols = set(pool["symbol"].dropna().astype(str).unique())
    start = pd.to_datetime(pool["entry_time"], utc=True, errors="coerce").min()
    end = pd.to_datetime(pool["exit_time"], utc=True, errors="coerce").max()
    mark_table = _load_mark_table(feature_path, symbols, start, end) if cfg.load_mark_prices else {}

    baseline_selected, baseline_skipped = _simulate_capacity_rule(
        pool,
        rule="R0_fixed_max8",
        cap_fn=lambda _row: BASELINE_MAX_POSITIONS,
    )
    baseline_metric = _summarize_selected(
        baseline_selected,
        baseline_skipped,
        architecture="baseline",
        rule="R0_fixed_max8",
        max_positions=BASELINE_MAX_POSITIONS,
    )
    baseline_net = float(baseline_metric["portfolio_net20"])

    ramp, ramp_selected, ramp_skipped = _dynamic_slot_ramp_summary(pool)
    expansion, expansion_selected, expansion_skipped = _late_phase_expansion_summary(pool)
    replacement, replacement_ledger = _late_signal_replacement_summary(pool, mark_table)
    bucket = _burst_order_bucket_summary(pool, baseline_selected, baseline_skipped)
    selected_vs_skipped = _selected_vs_skipped_by_phase(
        pd.concat([ramp_selected, expansion_selected], ignore_index=True),
        pd.concat([ramp_skipped, expansion_skipped], ignore_index=True),
    )
    tranche = _burst_tranche_allocation_summary(pool)
    timeline = pd.concat([ramp_selected, expansion_selected], ignore_index=True)

    outputs = {
        "burst_phase_asof_audit": report_root / "burst_phase_asof_audit.csv",
        "burst_order_bucket_summary": report_root / "burst_order_bucket_summary.csv",
        "dynamic_slot_ramp_summary": report_root / "dynamic_slot_ramp_summary.csv",
        "late_phase_expansion_summary": report_root / "late_phase_expansion_summary.csv",
        "late_signal_replacement_summary": report_root / "late_signal_replacement_summary.csv",
        "burst_tranche_allocation_summary": report_root / "burst_tranche_allocation_summary.csv",
        "selected_vs_skipped_by_phase": report_root / "selected_vs_skipped_by_phase.csv",
        "portfolio_timeline": report_root / "portfolio_timeline.csv",
        "late_replacement_trade_ledger": report_root / "late_replacement_trade_ledger.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    _burst_phase_asof_audit(pool).to_csv(outputs["burst_phase_asof_audit"], index=False)
    bucket.to_csv(outputs["burst_order_bucket_summary"], index=False)
    ramp.to_csv(outputs["dynamic_slot_ramp_summary"], index=False)
    expansion.to_csv(outputs["late_phase_expansion_summary"], index=False)
    replacement.to_csv(outputs["late_signal_replacement_summary"], index=False)
    tranche.to_csv(outputs["burst_tranche_allocation_summary"], index=False)
    selected_vs_skipped.to_csv(outputs["selected_vs_skipped_by_phase"], index=False)
    timeline.to_csv(outputs["portfolio_timeline"], index=False)
    replacement_ledger.to_csv(outputs["late_replacement_trade_ledger"], index=False)
    _write_notes(report_root, baseline_net, ramp, expansion, replacement, tranche)
    return outputs
