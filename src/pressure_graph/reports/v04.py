from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v03 import (
    C2_SIGNAL_COL,
    COST_BPS,
    V03_REQUIRED_COLUMNS,
    V03_TURNOVER_COLUMNS,
    _add_v03_report_columns,
    _attach_trade_context,
    _baseline_signal_count,
    _concat_or_empty,
    _downcast_frame,
    _entry_only_rows,
    _expand_costs,
    _matched_random_rows,
    _month_start,
    _read_existing_columns,
    _simulate_base_trades,
    _summary_row,
    build_dynamic_rank_table,
)


REPORT_ROOT = Path("reports/v0_4_regime_liquidity_gate")
RANK30_COL = "turnover_rank_30d"
RANK90_COL = "turnover_rank_90d"


GATES = {
    "G0_C2_raw_dynamic_top30": "gate_g0_raw",
    "G1_C2_BTC_up": "gate_g1_btc_up",
    "G2_C2_core_liquidity": "gate_g2_core_liquidity",
    "G3_C2_BTC_up_core_liquidity": "gate_g3_btc_up_core",
    "G4_C2_BTC_chop": "gate_g4_btc_chop",
    "G4b_C2_BTC_down": "gate_g4b_btc_down",
    "G5_C2_transient_hot": "gate_g5_transient_hot",
}


def _rank_table_with_lookback(
    turnover: pd.DataFrame,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    lookback_days: int,
    rank_col: str,
    turnover_col: str,
) -> pd.DataFrame:
    cfg = config.model_copy(deep=True)
    cfg.universe.dynamic_lookback_days = lookback_days
    ranks = build_dynamic_rank_table(turnover, instruments, cfg)
    if ranks.empty:
        return pd.DataFrame(columns=["month_start", "symbol", rank_col, turnover_col])
    return ranks.rename(
        columns={
            "dynamic_all_rank": rank_col,
            "dynamic_all_trailing_turnover": turnover_col,
        }
    )[["month_start", "symbol", rank_col, turnover_col]]


def _v04_required_columns() -> list[str]:
    return list(dict.fromkeys([*V03_REQUIRED_COLUMNS, "ret_15m", "volume_z_4h"]))


def _read_symbol_v04(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    symbol: str,
    config: ExperimentConfig,
) -> pd.DataFrame:
    data = _read_existing_columns(feature_path, _v04_required_columns(), filters=[("symbol", "==", symbol)])
    if data.empty:
        return data
    data = _downcast_frame(data)
    data["month_start"] = _month_start(data["bar_open_time"])
    data = data.merge(rank30, on=["month_start", "symbol"], how="left")
    data = data.merge(rank90, on=["month_start", "symbol"], how="left")
    data = data[pd.to_numeric(data[RANK30_COL], errors="coerce") <= 30].copy()
    if data.empty:
        return data
    data["dynamic_all_rank"] = pd.to_numeric(data[RANK30_COL], errors="coerce", downcast="integer")
    data["dynamic_all_trailing_turnover"] = pd.to_numeric(
        data["trailing_30d_turnover"], errors="coerce", downcast="float"
    )
    data = _add_v03_report_columns(data, config)
    data[RANK30_COL] = pd.to_numeric(data[RANK30_COL], errors="coerce", downcast="integer")
    data[RANK90_COL] = pd.to_numeric(data[RANK90_COL], errors="coerce", downcast="integer")
    data["core_liquidity"] = (data[RANK30_COL] <= 30) & (data[RANK90_COL] <= 50)
    data["transient_hot"] = (data[RANK30_COL] <= 30) & (data[RANK90_COL] > 50)
    data["non_core_liquidity"] = (data[RANK30_COL] <= 30) & ~data["core_liquidity"]
    state = data["btc_market_state"].astype("string")
    raw = data["universe_dynamic_all_top30"].fillna(False)
    data["gate_g0_raw"] = raw
    data["gate_g1_btc_up"] = raw & state.eq("BTC_up")
    data["gate_g2_core_liquidity"] = raw & data["core_liquidity"]
    data["gate_g3_btc_up_core"] = raw & state.eq("BTC_up") & data["core_liquidity"]
    data["gate_g4_btc_chop"] = raw & state.eq("BTC_chop")
    data["gate_g4b_btc_down"] = raw & state.eq("BTC_down")
    data["gate_g5_transient_hot"] = raw & data["transient_hot"]
    data["liquidity_quality"] = np.select(
        [
            data["core_liquidity"],
            data["transient_hot"],
            data[RANK90_COL].isna(),
        ],
        ["core_liquidity", "transient_hot", "rank90_missing"],
        default="non_core_liquidity",
    )
    return data


def _summarize_trades(
    trades_by_gate: dict[str, pd.DataFrame],
    signal_counts: dict[str, int],
    matched_by_gate: dict[str, pd.DataFrame] | None = None,
    entry_by_gate: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    rows = []
    matched_by_gate = matched_by_gate or {}
    entry_by_gate = entry_by_gate or {}
    for gate, trades in trades_by_gate.items():
        if trades.empty:
            continue
        for cost, group in trades.groupby("cost_single_side_bps", sort=False):
            row = _summary_row(group, signal_counts.get(gate, 0), gate_name=gate, cost_single_side_bps=cost)
            matched = matched_by_gate.get(gate, pd.DataFrame())
            entry = entry_by_gate.get(gate, pd.DataFrame())
            if not matched.empty:
                m = matched[matched["cost_single_side_bps"].eq(cost)]
                row["matched_random_net"] = pd.to_numeric(m["net_return_ex_fee_slippage"], errors="coerce").mean()
            if not entry.empty:
                e = entry[entry["cost_single_side_bps"].eq(cost)]
                row["entry_only_net"] = pd.to_numeric(e["net_return_ex_fee_slippage"], errors="coerce").mean()
            row["matched_random_lift_within_gate"] = row.get("net_expectancy", np.nan) - row.get(
                "matched_random_net", np.nan
            )
            row["entry_only_lift_within_gate"] = row.get("net_expectancy", np.nan) - row.get(
                "entry_only_net", np.nan
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _partition_summary(trades_by_gate: dict[str, pd.DataFrame], signal_context: pd.DataFrame) -> pd.DataFrame:
    rows = []
    partitions = {
        "full_12m": lambda month: pd.Series(True, index=month.index),
        "ex_2026_05": lambda month: ~month.eq("2026-05"),
        "pre_2026_05": lambda month: month.lt("2026-05"),
        "only_2026_05": lambda month: month.eq("2026-05"),
    }
    for gate, trades in trades_by_gate.items():
        if trades.empty:
            continue
        gate_col = GATES[gate]
        context_month = signal_context["month"].astype(str)
        trade_month = trades["month"].astype(str)
        for partition, fn in partitions.items():
            trade_mask = fn(trade_month)
            signal_n = int((signal_context[gate_col].fillna(False) & fn(context_month)).sum())
            for cost, group in trades[trade_mask].groupby("cost_single_side_bps", sort=False):
                rows.append(
                    _summary_row(
                        group,
                        signal_n,
                        gate_name=gate,
                        partition=partition,
                        cost_single_side_bps=cost,
                    )
                )
    return pd.DataFrame(rows)


def _leave_one_month_out(trades_by_gate: dict[str, pd.DataFrame], signal_context: pd.DataFrame) -> pd.DataFrame:
    rows = []
    months = sorted(signal_context["month"].dropna().unique()) if "month" in signal_context else []
    for gate, trades in trades_by_gate.items():
        if trades.empty:
            continue
        gate_col = GATES[gate]
        for excluded in months:
            data = trades[~trades["month"].astype(str).eq(str(excluded))]
            signal_n = int(
                (
                    signal_context[gate_col].fillna(False)
                    & ~signal_context["month"].astype(str).eq(str(excluded))
                ).sum()
            )
            for cost, group in data.groupby("cost_single_side_bps", sort=False):
                rows.append(
                    _summary_row(
                        group,
                        signal_n,
                        gate_name=gate,
                        excluded_month=excluded,
                        cost_single_side_bps=cost,
                    )
                )
    return pd.DataFrame(rows)


def _group_summary(
    trades_by_gate: dict[str, pd.DataFrame],
    signal_context: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    rows = []
    for gate, trades in trades_by_gate.items():
        if trades.empty or group_col not in trades.columns:
            continue
        gate_col = GATES[gate]
        counts = (
            signal_context[signal_context[gate_col].fillna(False)]
            .groupby(group_col, dropna=False, observed=True)
            .size()
            .to_dict()
            if group_col in signal_context.columns
            else {}
        )
        for (value, cost), group in trades.groupby(
            [group_col, "cost_single_side_bps"], dropna=False, sort=False, observed=True
        ):
            rows.append(
                _summary_row(
                    group,
                    int(counts.get(value, 0)),
                    gate_name=gate,
                    **{group_col: value},
                    cost_single_side_bps=cost,
                )
            )
    return pd.DataFrame(rows)


def _contribution_summary(trades_by_gate: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for gate, trades in trades_by_gate.items():
        if trades.empty:
            continue
        for cost, cost_group in trades.groupby("cost_single_side_bps", sort=False):
            total_net = pd.to_numeric(cost_group["net_return_ex_fee_slippage"], errors="coerce").sum()
            for dimension in ["month", "symbol"]:
                for value, group in cost_group.groupby(dimension, dropna=False, sort=False):
                    net_sum = pd.to_numeric(group["net_return_ex_fee_slippage"], errors="coerce").sum()
                    rows.append(
                        {
                            "gate_name": gate,
                            "cost_single_side_bps": cost,
                            "dimension": dimension,
                            "value": value,
                            "trades": len(group),
                            "trade_share": len(group) / len(cost_group) if len(cost_group) else np.nan,
                            "net_sum": net_sum,
                            "net_contribution": net_sum / total_net if total_net else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def _wide_gate_matrix(
    summary: pd.DataFrame,
    ex_may: pd.DataFrame,
    contribution: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for gate, group in summary.groupby("gate_name", sort=False):
        base = {"gate_name": gate}
        for cost in COST_BPS:
            cost_row = group[group["cost_single_side_bps"].eq(cost)]
            if cost_row.empty:
                continue
            row = cost_row.iloc[0]
            base[f"net_{int(cost)}bp"] = row["net_expectancy"]
            base[f"gross_{int(cost)}bp"] = row["gross_expectancy"]
            base[f"trades_{int(cost)}bp"] = row["trades"]
            if cost == 5:
                base["signals"] = row["signals"]
                base["fill_rate"] = row["fill_rate"]
                base["tp_rate"] = row["tp_rate"]
                base["sl_rate"] = row["sl_rate"]
                base["timeout_rate"] = row["timeout_rate"]
                base["matched_random_lift_5bp"] = row.get("matched_random_lift_within_gate")
                base["entry_only_lift_5bp"] = row.get("entry_only_lift_within_gate")
        ex = ex_may[
            ex_may["gate_name"].eq(gate)
            & ex_may["partition"].eq("ex_2026_05")
            & ex_may["cost_single_side_bps"].eq(5)
        ]
        if not ex.empty:
            base["ex_may_5bp_net"] = ex.iloc[0]["net_expectancy"]
            base["ex_may_trades"] = ex.iloc[0]["trades"]
        for dimension in ["month", "symbol"]:
            contrib = contribution[
                contribution["gate_name"].eq(gate)
                & contribution["cost_single_side_bps"].eq(5)
                & contribution["dimension"].eq(dimension)
            ].copy()
            if contrib.empty:
                continue
            contrib["abs_contribution"] = pd.to_numeric(contrib["net_contribution"], errors="coerce").abs()
            top = contrib.sort_values("abs_contribution", ascending=False).iloc[0]
            base[f"max_{dimension}_contribution_5bp"] = top["net_contribution"]
            base[f"max_{dimension}_contribution_value"] = top["value"]
        rows.append(base)
    return pd.DataFrame(rows)


def _write_candidate_reclassification(report_root: Path, gate_matrix: pd.DataFrame) -> None:
    lines = ["# Crypto Pressure Graph v0.4 Candidate Reclassification", ""]
    lines.append("C2 is frozen. v0.4 tests fixed BTC regime and liquidity-quality gates only.")
    combo = gate_matrix[gate_matrix["gate_name"].eq("G3_C2_BTC_up_core_liquidity")]
    if not combo.empty:
        row = combo.iloc[0]
        lines.append("")
        lines.append("## G3 BTC_up + core_liquidity")
        lines.append(
            f"- trades={int(row.get('trades_5bp', 0))}, net5={row.get('net_5bp', np.nan):.4%}, "
            f"net10={row.get('net_10bp', np.nan):.4%}, net20={row.get('net_20bp', np.nan):.4%}"
        )
        lines.append(
            f"- ex-May net5={row.get('ex_may_5bp_net', np.nan):.4%}, "
            f"matched lift={row.get('matched_random_lift_5bp', np.nan):.4%}, "
            f"entry lift={row.get('entry_only_lift_5bp', np.nan):.4%}"
        )
        passed = (
            row.get("net_5bp", -1) >= 0.002
            and row.get("net_10bp", -1) >= 0.0005
            and row.get("ex_may_5bp_net", -1) > 0
            and row.get("matched_random_lift_5bp", -1) > 0
            and row.get("entry_only_lift_5bp", -1) > 0
            and row.get("trades_5bp", 0) >= 100
        )
        lines.append("")
        if passed:
            lines.append("Classification: B- conditional paper-live overlay candidate.")
        else:
            lines.append("Classification: keep C+ regime-specific pressure overlay; do not promote yet.")
    (report_root / "candidate_reclassification.md").write_text("\n".join(lines), encoding="utf-8")


def write_v04_regime_liquidity_gate(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path = REPORT_ROOT,
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    turnover = _read_existing_columns(feature_path, V03_TURNOVER_COLUMNS)
    rank30 = _rank_table_with_lookback(turnover, instruments, config, 30, RANK30_COL, "trailing_30d_turnover")
    rank90 = _rank_table_with_lookback(turnover, instruments, config, 90, RANK90_COL, "trailing_90d_turnover")
    del turnover
    symbols = sorted(rank30[rank30[RANK30_COL] <= 30]["symbol"].dropna().astype(str).unique())

    trade_frames: dict[str, list[pd.DataFrame]] = defaultdict(list)
    matched_frames: dict[str, list[pd.DataFrame]] = defaultdict(list)
    entry_frames: dict[str, list[pd.DataFrame]] = defaultdict(list)
    signal_counts: dict[str, int] = defaultdict(int)
    matched_counts: dict[str, int] = defaultdict(int)
    entry_counts: dict[str, int] = defaultdict(int)
    signal_context_frames: list[pd.DataFrame] = []

    context_cols = list(
        dict.fromkeys(
            [
                "exchange",
                "symbol",
                "feature_time",
                "month",
                "btc_market_state",
                "liquidity_bucket",
                "liquidity_quality",
                RANK30_COL,
                RANK90_COL,
                "core_liquidity",
                "transient_hot",
                "dynamic_all_trailing_turnover",
                "symbol_volatility_percentile",
                *GATES.values(),
                C2_SIGNAL_COL,
            ]
        )
    )

    for idx, symbol in enumerate(symbols, start=1):
        data = _read_symbol_v04(feature_path, rank30, rank90, symbol, config)
        if data.empty:
            continue
        if idx % 25 == 0:
            print(f"v0.4 processed {idx}/{len(symbols)} symbols", flush=True)

        signal_rows = data[data[C2_SIGNAL_COL].fillna(False)]
        if not signal_rows.empty:
            signal_context_frames.append(signal_rows[[col for col in context_cols if col in signal_rows.columns]])

        for gate, gate_col in GATES.items():
            gate_events = data[gate_col].fillna(False) & data[C2_SIGNAL_COL].fillna(False)
            signal_n = int(gate_events.sum())
            if signal_n <= 0:
                continue
            signal_counts[gate] += signal_n

            signal_col = f"{gate}_signal_event"
            simulation_data = data.copy()
            simulation_data[signal_col] = gate_events
            trades = _attach_trade_context(
                _expand_costs(_simulate_base_trades(simulation_data, signal_col)),
                simulation_data,
            )
            if not trades.empty:
                trade_frames[gate].append(trades)

            gate_pool = data[data[gate_col].fillna(False)].copy()
            matched_rows = _matched_random_rows(gate_pool)
            matched_counts[gate] += _baseline_signal_count(matched_rows)
            matched_simulation = data.copy()
            matched_times = (
                set(pd.to_datetime(matched_rows["feature_time"], utc=True, errors="coerce"))
                if not matched_rows.empty and "feature_time" in matched_rows.columns
                else set()
            )
            matched_simulation["baseline_event"] = pd.to_datetime(
                matched_simulation["feature_time"], utc=True, errors="coerce"
            ).isin(matched_times)
            matched = _attach_trade_context(
                _expand_costs(_simulate_base_trades(matched_simulation, "baseline_event")),
                matched_simulation,
            )
            if not matched.empty:
                matched_frames[gate].append(matched)

            entry_rows = _entry_only_rows(gate_pool)
            entry_counts[gate] += _baseline_signal_count(entry_rows)
            entry_simulation = data.copy()
            entry_times = (
                set(pd.to_datetime(entry_rows["feature_time"], utc=True, errors="coerce"))
                if not entry_rows.empty and "feature_time" in entry_rows.columns
                else set()
            )
            entry_simulation["baseline_event"] = pd.to_datetime(
                entry_simulation["feature_time"], utc=True, errors="coerce"
            ).isin(entry_times)
            entry = _attach_trade_context(
                _expand_costs(_simulate_base_trades(entry_simulation, "baseline_event")),
                entry_simulation,
            )
            if not entry.empty:
                entry_frames[gate].append(entry)

    trades_by_gate = {gate: _concat_or_empty(frames) for gate, frames in trade_frames.items()}
    matched_by_gate = {gate: _concat_or_empty(frames) for gate, frames in matched_frames.items()}
    entry_by_gate = {gate: _concat_or_empty(frames) for gate, frames in entry_frames.items()}
    signal_context = _concat_or_empty(signal_context_frames)

    summary = _summarize_trades(trades_by_gate, signal_counts, matched_by_gate, entry_by_gate)
    ex_may = _partition_summary(trades_by_gate, signal_context)
    leave_one = _leave_one_month_out(trades_by_gate, signal_context)
    liquidity = _group_summary(trades_by_gate, signal_context, "liquidity_quality")
    regime = _group_summary(trades_by_gate, signal_context, "btc_market_state")
    contribution = _contribution_summary(trades_by_gate)
    gate_matrix = _wide_gate_matrix(summary, ex_may, contribution)

    outputs = {
        "gate_matrix_summary": report_root / "gate_matrix_summary.csv",
        "within_gate_baselines": report_root / "within_gate_baselines.csv",
        "monthly_leave_one_out": report_root / "monthly_leave_one_out.csv",
        "ex_may_summary": report_root / "ex_may_summary.csv",
        "liquidity_quality_split": report_root / "liquidity_quality_split.csv",
        "regime_negative_controls": report_root / "regime_negative_controls.csv",
        "symbol_month_contribution": report_root / "symbol_month_contribution.csv",
        "candidate_reclassification": report_root / "candidate_reclassification.md",
    }
    gate_matrix.to_csv(outputs["gate_matrix_summary"], index=False)
    summary.to_csv(outputs["within_gate_baselines"], index=False)
    leave_one.to_csv(outputs["monthly_leave_one_out"], index=False)
    ex_may.to_csv(outputs["ex_may_summary"], index=False)
    liquidity.to_csv(outputs["liquidity_quality_split"], index=False)
    regime.to_csv(outputs["regime_negative_controls"], index=False)
    contribution.to_csv(outputs["symbol_month_contribution"], index=False)
    _write_candidate_reclassification(report_root, gate_matrix)
    return outputs
