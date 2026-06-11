from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.backtest.minute_execution import simulate_1m_execution
from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir, raw_path, read_parquet, write_parquet
from pressure_graph.reports.v03 import V03_TURNOVER_COLUMNS, _concat_or_empty, _read_existing_columns, build_dynamic_rank_table
from pressure_graph.reports.v04 import _rank_table_with_lookback
from pressure_graph.reports.v06a1 import (
    MAX_BOUNDARY_TOP_N,
    _expand_costs,
    _policy,
    _read_symbol_features,
    _rule,
    _signal_mask,
    _simulate_rows,
)
from pressure_graph.reports.v06a2 import (
    V06A2Config,
    _entry_policy,
    _load_1m_bars,
    _signal_col,
    build_v06a2_signal_rows,
)


REPORT_ROOT = Path("reports/v0_6a3_1_entry_gate_revalidation")


def _required_state(candidate_gate: str) -> str:
    return f"BTC_{candidate_gate.removeprefix('BTC_')}" if not candidate_gate.startswith("BTC_") else candidate_gate


def _entry_state_asof(trades: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        out = trades.copy()
        out["btc_state_at_entry"] = pd.Series(dtype=object)
        return out
    if context.empty:
        out = trades.copy()
        out["btc_state_at_entry"] = "unknown"
        return out
    left = trades.copy()
    left["entry_time"] = pd.to_datetime(left["entry_time"], utc=True, errors="coerce")
    ctx = context.copy()
    ctx["feature_time"] = pd.to_datetime(ctx["feature_time"], utc=True, errors="coerce")
    ctx = ctx.dropna(subset=["feature_time"]).sort_values(["exchange", "symbol", "feature_time"])
    frames = []
    for key, group in left.groupby(["exchange", "symbol"], sort=False, dropna=False):
        exchange, symbol = key
        local_ctx = ctx[
            ctx["exchange"].astype(str).eq(str(exchange))
            & ctx["symbol"].astype(str).eq(str(symbol))
        ][["feature_time", "btc_market_state"]].sort_values("feature_time")
        data = group.sort_values("entry_time")
        if local_ctx.empty:
            data["btc_state_at_entry"] = "unknown"
            frames.append(data)
            continue
        merged = pd.merge_asof(
            data,
            local_ctx,
            left_on="entry_time",
            right_on="feature_time",
            direction="backward",
        ).drop(columns=["feature_time"], errors="ignore")
        merged = merged.rename(columns={"btc_market_state": "btc_state_at_entry"})
        frames.append(merged)
    return pd.concat(frames, ignore_index=True) if frames else left


def _add_signal_state_1m(trades: pd.DataFrame, signal_rows: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        out = trades.copy()
        out["btc_state_at_signal"] = pd.Series(dtype=object)
        return out
    context = signal_rows[["exchange", "symbol", "feature_time", "btc_market_state", "month"]].copy()
    context["feature_time"] = pd.to_datetime(context["feature_time"], utc=True, errors="coerce")
    out = trades.copy()
    out["signal_time"] = pd.to_datetime(out["signal_time"], utc=True, errors="coerce")
    out = out.merge(
        context.rename(
            columns={
                "feature_time": "signal_time",
                "btc_market_state": "btc_state_at_signal",
            }
        ),
        on=["exchange", "symbol", "signal_time"],
        how="left",
    )
    out["month"] = out["month"].fillna(pd.to_datetime(out["signal_time"], utc=True).dt.strftime("%Y-%m"))
    return out


def _normalize_trades(trades: pd.DataFrame, execution_granularity: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    out = trades.copy()
    out["execution_granularity"] = execution_granularity
    if "net_return" not in out.columns and "net_expectancy" in out.columns:
        out["net_return"] = pd.to_numeric(out["net_expectancy"], errors="coerce")
    if "gross_return" not in out.columns:
        out["gross_return"] = np.nan
    if "month" not in out.columns:
        out["month"] = pd.to_datetime(out["signal_time"], utc=True, errors="coerce").dt.strftime("%Y-%m")
    out["cost_single_side_bps"] = pd.to_numeric(out["cost_single_side_bps"], errors="coerce")
    out["net_return"] = pd.to_numeric(out["net_return"], errors="coerce")
    out["gross_return"] = pd.to_numeric(out["gross_return"], errors="coerce")
    return out


def _entry_gate_pass(data: pd.DataFrame, required: str) -> pd.Series:
    return (
        data["btc_state_at_signal"].astype(str).eq(required)
        & data["btc_state_at_entry"].astype(str).eq(required)
    )


def _summary_metrics(trades: pd.DataFrame) -> dict[str, object]:
    if trades.empty:
        return {
            "trades": 0,
            "gross_expectancy": np.nan,
            "net_expectancy": np.nan,
            "net_sum": 0.0,
            "tp_rate": np.nan,
            "sl_rate": np.nan,
            "timeout_rate": np.nan,
            "max_loss": np.nan,
        }
    exits = trades.get("exit_reason", pd.Series(dtype=str)).astype(str)
    net = pd.to_numeric(trades["net_return"], errors="coerce")
    return {
        "trades": int(len(trades)),
        "gross_expectancy": float(pd.to_numeric(trades["gross_return"], errors="coerce").mean()),
        "net_expectancy": float(net.mean()),
        "net_sum": float(net.sum()),
        "tp_rate": float(exits.str.startswith("tp").mean()),
        "sl_rate": float(exits.str.startswith("sl").mean()),
        "timeout_rate": float(exits.isin(["max_hold", "open"]).mean()),
        "max_loss": float(net.min()),
    }


def _candidate_metrics_after_gate(new_trades: pd.DataFrame) -> pd.DataFrame:
    if new_trades.empty:
        return pd.DataFrame()
    rows = []
    for key, group in new_trades.groupby(
        ["execution_granularity", "candidate", "cost_single_side_bps"],
        sort=False,
        dropna=False,
    ):
        execution, candidate, cost = key
        rows.append(
            {
                "execution_granularity": execution,
                "candidate": candidate,
                "cost_single_side_bps": cost,
                **_summary_metrics(group),
            }
        )
    return pd.DataFrame(rows)


def _month_capped_expectancy(trades: pd.DataFrame, cost: float, cap: float) -> float:
    sample = trades[pd.to_numeric(trades["cost_single_side_bps"], errors="coerce").eq(float(cost))]
    if sample.empty:
        return np.nan
    total_net = pd.to_numeric(sample["net_return"], errors="coerce").sum()
    total_trades = len(sample)
    cap_value = total_net * cap if total_net > 0 else 0.0
    capped = []
    for _, group in sample.groupby("month", dropna=False):
        month_sum = pd.to_numeric(group["net_return"], errors="coerce").sum()
        capped.append(min(month_sum, cap_value) if month_sum > 0 and cap_value > 0 else month_sum)
    return float(np.sum(capped) / total_trades) if total_trades else np.nan


def _net_expectancy(trades: pd.DataFrame, cost: float, exclude_month: str | None = None) -> float:
    sample = trades[pd.to_numeric(trades["cost_single_side_bps"], errors="coerce").eq(float(cost))]
    if exclude_month is not None:
        sample = sample[~sample["month"].astype(str).eq(str(exclude_month))]
    if sample.empty:
        return np.nan
    return float(pd.to_numeric(sample["net_return"], errors="coerce").mean())


def _entry_gate_impact(old_trades: pd.DataFrame, new_trades: pd.DataFrame, focal_month: str) -> pd.DataFrame:
    rows = []
    keys = sorted(
        set(zip(old_trades.get("execution_granularity", []), old_trades.get("candidate", []), strict=False))
        | set(zip(new_trades.get("execution_granularity", []), new_trades.get("candidate", []), strict=False))
    )
    for execution, candidate in keys:
        old = old_trades[
            old_trades["execution_granularity"].astype(str).eq(str(execution))
            & old_trades["candidate"].astype(str).eq(str(candidate))
        ]
        new = new_trades[
            new_trades["execution_granularity"].astype(str).eq(str(execution))
            & new_trades["candidate"].astype(str).eq(str(candidate))
        ]
        old10 = old[pd.to_numeric(old["cost_single_side_bps"], errors="coerce").eq(10.0)]
        new10 = new[pd.to_numeric(new["cost_single_side_bps"], errors="coerce").eq(10.0)]
        old_trade_n = int(len(old10))
        new_trade_n = int(len(new10))
        old_net10 = _net_expectancy(old, 10)
        new_net10 = _net_expectancy(new, 10)
        old_net20 = _net_expectancy(old, 20)
        new_net20 = _net_expectancy(new, 20)
        rows.append(
            {
                "execution_granularity": execution,
                "candidate": candidate,
                "old_trades": old_trade_n,
                "new_trades": new_trade_n,
                "invalidated_trades": old_trade_n - new_trade_n,
                "invalidated_pct": float((old_trade_n - new_trade_n) / old_trade_n)
                if old_trade_n
                else np.nan,
                "old_net10": old_net10,
                "new_net10": new_net10,
                "delta_net10": new_net10 - old_net10 if pd.notna(old_net10) and pd.notna(new_net10) else np.nan,
                "old_net20": old_net20,
                "new_net20": new_net20,
                "delta_net20": new_net20 - old_net20 if pd.notna(old_net20) and pd.notna(new_net20) else np.nan,
                f"old_ex_{focal_month}_net10": _net_expectancy(old, 10, focal_month),
                f"new_ex_{focal_month}_net10": _net_expectancy(new, 10, focal_month),
                "old_month_cap_35_net20": _month_capped_expectancy(old, 20, 0.35),
                "new_month_cap_35_net20": _month_capped_expectancy(new, 20, 0.35),
            }
        )
    return pd.DataFrame(rows)


def _invalidated_summary(old_trades: pd.DataFrame) -> pd.DataFrame:
    if old_trades.empty:
        return pd.DataFrame()
    sample = old_trades[
        pd.to_numeric(old_trades["cost_single_side_bps"], errors="coerce").eq(10.0)
        & ~old_trades["entry_gate_pass"].fillna(False)
    ].copy()
    if sample.empty:
        return pd.DataFrame()
    sample["old_trade_return_10bp"] = sample["net_return"]
    sample["invalid_reason"] = "entry_btc_not_up:" + sample["btc_state_at_entry"].astype(str)
    cols = [
        "execution_granularity",
        "candidate",
        "symbol",
        "signal_time",
        "entry_time",
        "btc_state_at_signal",
        "btc_state_at_entry",
        "old_trade_return_10bp",
        "invalid_reason",
    ]
    return sample[[col for col in cols if col in sample.columns]].sort_values(
        ["execution_granularity", "candidate", "entry_time", "symbol"]
    )


def _monthly_impact(old_trades: pd.DataFrame, new_trades: pd.DataFrame) -> pd.DataFrame:
    if old_trades.empty:
        return pd.DataFrame()
    rows = []
    keys = sorted(
        set(
            zip(
                old_trades["execution_granularity"].astype(str),
                old_trades["candidate"].astype(str),
                old_trades["month"].astype(str),
                strict=False,
            )
        )
    )
    for execution, candidate, month in keys:
        old = old_trades[
            old_trades["execution_granularity"].astype(str).eq(execution)
            & old_trades["candidate"].astype(str).eq(candidate)
            & old_trades["month"].astype(str).eq(month)
        ]
        new = new_trades[
            new_trades["execution_granularity"].astype(str).eq(execution)
            & new_trades["candidate"].astype(str).eq(candidate)
            & new_trades["month"].astype(str).eq(month)
        ]
        invalidated = old[
            pd.to_numeric(old["cost_single_side_bps"], errors="coerce").eq(10.0)
            & ~old["entry_gate_pass"].fillna(False)
        ]
        rows.append(
            {
                "execution_granularity": execution,
                "candidate": candidate,
                "month": month,
                "old_trades_10bp": int(
                    pd.to_numeric(old["cost_single_side_bps"], errors="coerce").eq(10.0).sum()
                ),
                "new_trades_10bp": int(
                    pd.to_numeric(new["cost_single_side_bps"], errors="coerce").eq(10.0).sum()
                ),
                "invalidated_trades": int(len(invalidated)),
                "old_net10": _net_expectancy(old, 10),
                "new_net10": _net_expectancy(new, 10),
                "old_net20": _net_expectancy(old, 20),
                "new_net20": _net_expectancy(new, 20),
            }
        )
    return pd.DataFrame(rows)


def _load_context_and_15m_trades(
    feature_path: Path,
    instruments: pd.DataFrame,
    base_config: ExperimentConfig,
    v06a2_config: V06A2Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    turnover = _read_existing_columns(feature_path, V03_TURNOVER_COLUMNS)
    rank30 = build_dynamic_rank_table(turnover, instruments, base_config)
    rank90 = _rank_table_with_lookback(
        turnover,
        instruments,
        base_config,
        90,
        "turnover_rank_90d",
        "trailing_90d_turnover",
    )
    symbols = sorted(
        rank30[pd.to_numeric(rank30["dynamic_all_rank"], errors="coerce") <= MAX_BOUNDARY_TOP_N][
            "symbol"
        ]
        .dropna()
        .astype(str)
        .unique()
    )
    del turnover
    trade_frames: list[pd.DataFrame] = []
    invalid_frames: list[pd.DataFrame] = []
    context_frames: list[pd.DataFrame] = []
    for idx, symbol in enumerate(symbols, start=1):
        data = _read_symbol_features(feature_path, rank30, rank90, symbol, base_config)
        if data.empty:
            continue
        print(f"v0.6A.3.1 processing {idx}/{len(symbols)} {symbol}", flush=True)
        context_frames.append(
            data[["exchange", "symbol", "feature_time", "btc_market_state"]].drop_duplicates().copy()
        )
        for candidate in v06a2_config.candidates:
            signal = _signal_mask(data, candidate)
            policy = _policy(candidate.entry_policy)
            trades = _simulate_rows(
                data,
                signal,
                candidate.candidate,
                candidate.anchor,
                policy,
                candidate.execution_rule,
                base_config,
            )
            if trades.empty:
                continue
            trades = trades.rename(columns={"btc_market_state": "btc_state_at_signal"})
            trades = _entry_state_asof(trades, context_frames[-1])
            trades["entry_gate_pass"] = _entry_gate_pass(trades, _required_state(candidate.gate))
            expanded = _expand_costs(trades, v06a2_config.cost_bps)
            trade_frames.append(_normalize_trades(expanded, "15m_bar"))
            invalid_frames.append(_invalidated_summary(_normalize_trades(expanded, "15m_bar")))
    return _concat_or_empty(trade_frames), _concat_or_empty(context_frames), _concat_or_empty(invalid_frames)


def _one_min_revalidation(
    feature_path: Path,
    instruments: pd.DataFrame,
    base_config: ExperimentConfig,
    v06a2_config: V06A2Config,
    context: pd.DataFrame,
) -> pd.DataFrame:
    signal_rows = build_v06a2_signal_rows(feature_path, instruments, base_config, v06a2_config)
    minute_bars = _load_1m_bars(base_config, v06a2_config)
    if signal_rows.empty or minute_bars.empty:
        return pd.DataFrame()
    frames = []
    for candidate in v06a2_config.candidates:
        rule, resolver = _rule(candidate.execution_rule, base_config)
        for cost in v06a2_config.cost_bps:
            trades = simulate_1m_execution(
                signal_rows,
                minute_bars,
                _signal_col(candidate),
                candidate.candidate,
                _entry_policy(candidate),
                candidate.execution_rule,
                rule,
                float(cost),
                resolver,
            )
            if trades.empty:
                continue
            trades["candidate"] = candidate.candidate
            trades = _add_signal_state_1m(trades, signal_rows)
            trades = _entry_state_asof(trades, context)
            trades["entry_gate_pass"] = _entry_gate_pass(trades, _required_state(candidate.gate))
            frames.append(_normalize_trades(trades, "1m_bar"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _write_status(report_root: Path, impact: pd.DataFrame, invalidated: pd.DataFrame) -> None:
    lines = [
        "# v0.6A.3.1 Entry-Gate Revalidation",
        "",
        "Frozen candidates only: IR1, IR2, IR3. No parameter search or threshold changes.",
        "",
        "New rule: requires BTC_up at both signal time and entry time. Entry state is as-of `feature_time <= entry_time` to avoid future leakage.",
        "",
    ]
    if impact.empty:
        lines.append("- Revalidation produced no impact rows.")
    else:
        focus = impact[
            impact["execution_granularity"].eq("1m_bar")
            & impact["candidate"].eq("IR2")
        ]
        if focus.empty:
            focus = impact[
                impact["execution_granularity"].eq("15m_bar")
                & impact["candidate"].eq("IR2")
            ]
        if not focus.empty:
            row10 = focus.iloc[0]
            lines.append(
                f"- IR2 {row10.execution_granularity}: old_trades={int(row10.old_trades)}, "
                f"new_trades={int(row10.new_trades)}, invalidated={int(row10.invalidated_trades)}, "
                f"new_net10={row10.new_net10:.4%}, new_net20={row10.new_net20:.4%}"
            )
            if row10.new_trades >= 100 and row10.new_net10 > 0.003 and row10.new_net20 > 0.001:
                decision = "B- primary paper-live candidate maintained after entry-gate revalidation."
            elif row10.new_net10 > 0:
                decision = "B- maintained with execution gate sensitivity; keep paper-live, no real-live."
            else:
                decision = "Downgrade pressure: pause candidate evaluation until attribution is reviewed."
            lines.append(f"- Decision: {decision}")
    lines.extend(
        [
            "",
            f"- invalidated_trade_rows: {len(invalidated)}",
            "- real_live_allowed remains False.",
            "- tick validation remains pending unless a trade-sequence cache is available.",
        ]
    )
    (report_root / "candidate_status.md").write_text("\n".join(lines), encoding="utf-8")


def write_v06a31_entry_gate_revalidation(
    feature_path: Path,
    instruments: pd.DataFrame,
    base_config: ExperimentConfig,
    v06a2_config: V06A2Config,
    report_root: Path = REPORT_ROOT,
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    old_15m, context, invalid_15m = _load_context_and_15m_trades(
        feature_path,
        instruments,
        base_config,
        v06a2_config,
    )
    old_1m = _one_min_revalidation(feature_path, instruments, base_config, v06a2_config, context)
    old_trades = _concat_or_empty([old_15m, old_1m])
    new_trades = old_trades[old_trades["entry_gate_pass"].fillna(False)].copy() if not old_trades.empty else old_trades
    invalidated = _concat_or_empty([invalid_15m, _invalidated_summary(old_1m)])
    focal_month = v06a2_config.execution_validation.focal_month
    impact = _entry_gate_impact(old_trades, new_trades, focal_month)
    metrics = _candidate_metrics_after_gate(new_trades)
    monthly = _monthly_impact(old_trades, new_trades)

    outputs = {
        "entry_gate_impact": report_root / "entry_gate_impact.csv",
        "candidate_metrics_after_entry_gate": report_root / "candidate_metrics_after_entry_gate.csv",
        "invalidated_trade_summary": report_root / "invalidated_trade_summary.csv",
        "monthly_impact": report_root / "monthly_impact.csv",
        "candidate_status": report_root / "candidate_status.md",
        "debug_old_trades": report_root / "debug_old_trades.parquet",
        "debug_new_trades": report_root / "debug_new_trades.parquet",
    }
    impact.to_csv(outputs["entry_gate_impact"], index=False)
    metrics.to_csv(outputs["candidate_metrics_after_entry_gate"], index=False)
    invalidated.to_csv(outputs["invalidated_trade_summary"], index=False)
    monthly.to_csv(outputs["monthly_impact"], index=False)
    write_parquet(old_trades, outputs["debug_old_trades"])
    write_parquet(new_trades, outputs["debug_new_trades"])
    _write_status(report_root, impact, invalidated)
    return outputs


def run_v06a31_entry_gate_revalidation_from_features(
    config: ExperimentConfig,
    v06a2_config: V06A2Config,
) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = read_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v06a31_entry_gate_revalidation(features_path, instruments, config, v06a2_config)
