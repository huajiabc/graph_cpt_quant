from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from pressure_graph.backtest import EntryPolicy, simulate_entry_policy_trades
from pressure_graph.config import ExperimentConfig
from pressure_graph.config.models import ExecutionRule
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v03 import (
    C2_SIGNAL_COL,
    C2_STATE_COL,
    V03_REQUIRED_COLUMNS,
    V03_TURNOVER_COLUMNS,
    _add_v03_report_columns,
    _concat_or_empty,
    _downcast_frame,
    _event_flags,
    _month_start,
    _read_existing_columns,
    build_dynamic_rank_table,
)
from pressure_graph.reports.v04 import _rank_table_with_lookback


REPORT_ROOT = Path("reports/v0_6a_reclaim_alpha")
TOP_NS = [30, 50]
COST_BPS = [5, 10, 20]
ANCHORS = {
    "entry_only_all_bars": "entry_only_anchor_event",
    "momentum_breakout": "momentum_breakout_anchor_event",
    "short_squeeze_C2": C2_SIGNAL_COL,
    "bullish_volume_shock": "bullish_volume_shock_anchor_event",
}
GATES = {
    "all": "gate_all",
    "BTC_up": "gate_btc_up",
    "BTC_chop": "gate_btc_chop",
    "BTC_down": "gate_btc_down",
    "non_transient_hot": "gate_non_transient_hot",
}
ENTRY_POLICIES = [
    EntryPolicy("P2_pullback_0.5pct_valid_4_bars", "pullback", valid_bars=4, pullback_pct=0.005),
    EntryPolicy("P1_pullback_1.0pct_valid_8_bars", "pullback", valid_bars=8, pullback_pct=0.010),
    EntryPolicy(
        "R2_pullback_0.5pct_reclaim_signal_close_valid_8_bars",
        "pullback_reclaim",
        valid_bars=8,
        pullback_pct=0.005,
    ),
    EntryPolicy(
        "R1_pullback_1.0pct_reclaim_signal_close_valid_8_bars",
        "pullback_reclaim",
        valid_bars=8,
        pullback_pct=0.010,
    ),
]


def _v06a_required_columns() -> list[str]:
    return list(
        dict.fromkeys(
            [
                *V03_REQUIRED_COLUMNS,
                "ret_15m",
                "btc_ret_1h",
                "btc_volatility_4h",
            ]
        )
    )


def _top_symbols(ranks: pd.DataFrame) -> list[str]:
    return sorted(
        ranks[pd.to_numeric(ranks["dynamic_all_rank"], errors="coerce") <= max(TOP_NS)]["symbol"]
        .dropna()
        .astype(str)
        .unique()
    )


def _read_symbol_features(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    symbol: str,
    config: ExperimentConfig,
) -> pd.DataFrame:
    data = _read_existing_columns(
        feature_path,
        _v06a_required_columns(),
        filters=[("symbol", "==", symbol)],
    )
    if data.empty:
        return data
    data = _downcast_frame(data)
    data["month_start"] = _month_start(data["bar_open_time"])
    data = data.merge(rank30, on=["month_start", "symbol"], how="left")
    data = data.merge(rank90, on=["month_start", "symbol"], how="left")
    data = data[pd.to_numeric(data["dynamic_all_rank"], errors="coerce") <= max(TOP_NS)].copy()
    if data.empty:
        return data
    data["dynamic_all_rank"] = pd.to_numeric(data["dynamic_all_rank"], errors="coerce")
    data["turnover_rank_90d"] = pd.to_numeric(data["turnover_rank_90d"], errors="coerce")
    data = _add_v03_report_columns(data, config)
    return _add_reclaim_anchor_columns(data, config)


def _add_reclaim_anchor_columns(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    out = df.sort_values(["exchange", "symbol", "bar_open_time"]).copy()
    warmup = out.get("warmup_complete", True)
    if not isinstance(warmup, pd.Series):
        warmup = pd.Series(bool(warmup), index=out.index)

    ret_4h_pct = pd.to_numeric(out.get("ret_4h_percentile"), errors="coerce")
    volume_z_4h = pd.to_numeric(out.get("volume_z_4h"), errors="coerce")
    oi_4h_pct = pd.to_numeric(out.get("oi_value_delta_4h_percentile"), errors="coerce")
    funding_pct = pd.to_numeric(out.get("funding_percentile"), errors="coerce")
    btc_ret_4h = pd.to_numeric(out.get("btc_ret_4h"), errors="coerce")
    ret_4h = pd.to_numeric(out.get("ret_4h"), errors="coerce")

    momentum_state = (
        warmup
        & (btc_ret_4h > config.path_rules.momentum_ignition.btc_ret_4h_min)
        & (ret_4h_pct > 70)
        & (volume_z_4h > 1.0)
        & (oi_4h_pct > 60)
        & (oi_4h_pct < 95)
        & (funding_pct < 80)
    )
    volume_state = warmup & (volume_z_4h > 2.0) & (ret_4h > 0)
    entry_only_state = warmup

    states = {
        "entry_only_anchor_state": entry_only_state,
        "momentum_breakout_anchor_state": momentum_state,
        "bullish_volume_shock_anchor_state": volume_state,
    }
    for state_col, state in states.items():
        event_col = state_col.replace("_state", "_event")
        out[state_col] = state.fillna(False).astype(bool)
        out[event_col] = (
            out.groupby(["exchange", "symbol"], group_keys=False, sort=False, observed=True)[state_col]
            .apply(lambda series: _event_flags(series, config.events.cooldown_bars_4h))
            .reindex(out.index)
        )

    state = out["btc_market_state"].astype("string")
    out["gate_all"] = True
    out["gate_btc_up"] = state.eq("BTC_up")
    out["gate_btc_chop"] = state.eq("BTC_chop")
    out["gate_btc_down"] = state.eq("BTC_down")
    out["core_liquidity"] = (out["dynamic_all_rank"] <= 30) & (out["turnover_rank_90d"] <= 50)
    out["transient_hot"] = (out["dynamic_all_rank"] <= 30) & (out["turnover_rank_90d"] > 50)
    out["gate_non_transient_hot"] = ~out["transient_hot"].fillna(False)
    out["liquidity_quality"] = np.select(
        [
            out["core_liquidity"],
            out["transient_hot"],
            out["turnover_rank_90d"].isna(),
        ],
        ["core_liquidity", "transient_hot", "rank90_missing"],
        default="non_core_liquidity",
    )
    return out


def _vol_regime_rule(row: pd.Series) -> ExecutionRule:
    vol_pct = pd.to_numeric(row.get("symbol_volatility_percentile"), errors="coerce")
    if pd.isna(vol_pct) or vol_pct < 40:
        return ExecutionRule(tp=0.03, sl=0.02, max_hold_bars=16)
    if vol_pct < 80:
        return ExecutionRule(tp=0.04, sl=0.025, max_hold_bars=16)
    return ExecutionRule(tp=0.05, sl=0.03, max_hold_bars=16)


def _execution_rules(config: ExperimentConfig) -> list[tuple[str, ExecutionRule, Callable[[pd.Series], ExecutionRule] | None]]:
    return [
        ("swing", ExecutionRule(tp=0.05, sl=0.03, max_hold_bars=48), None),
        ("vol_regime_fast", config.execution.rules["fast"], _vol_regime_rule),
    ]


def _expand_costs(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    frames = []
    for cost in COST_BPS:
        data = trades.copy()
        data["cost_single_side_bps"] = float(cost)
        data["net_return_ex_fee_slippage"] = pd.to_numeric(data["gross_return"], errors="coerce") - (
            2.0 * float(cost) / 10_000.0
        )
        if "funding_cost" in data.columns:
            data["net_return_ex_fee_slippage_funding"] = (
                data["net_return_ex_fee_slippage"]
                - pd.to_numeric(data["funding_cost"], errors="coerce").fillna(0)
            )
        frames.append(data)
    return pd.concat(frames, ignore_index=True)


def _attach_context(trades: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    context_cols = [
        "exchange",
        "symbol",
        "feature_time",
        "month",
        "dynamic_all_rank",
        "liquidity_bucket",
        "dynamic_all_trailing_turnover",
        "btc_market_state",
        "symbol_volatility_percentile",
        "liquidity_quality",
        "core_liquidity",
        "transient_hot",
        "turnover_rank_90d",
    ]
    context = df[[col for col in context_cols if col in df.columns]].drop_duplicates(
        ["exchange", "symbol", "feature_time"]
    )
    out = trades.merge(
        context,
        left_on=["exchange", "symbol", "signal_time"],
        right_on=["exchange", "symbol", "feature_time"],
        how="left",
    )
    return out.drop(columns=["feature_time"], errors="ignore")


def _trade_summary(trades: pd.DataFrame, signal_n: int, **keys: object) -> dict[str, object]:
    net = pd.to_numeric(trades.get("net_return_ex_fee_slippage"), errors="coerce")
    gross = pd.to_numeric(trades.get("gross_return"), errors="coerce")
    exit_reason = trades.get("exit_reason", pd.Series(dtype=str)).astype(str)
    return {
        **keys,
        "signals": int(signal_n),
        "trades": int(len(trades)),
        "fill_rate": float(len(trades) / signal_n) if signal_n else np.nan,
        "gross_expectancy": float(gross.mean()) if len(trades) else np.nan,
        "net_expectancy": float(net.mean()) if len(trades) else np.nan,
        "win_rate": float((net > 0).mean()) if len(trades) else np.nan,
        "tp_rate": float(exit_reason.str.startswith("tp").mean()) if len(trades) else np.nan,
        "sl_rate": float(exit_reason.str.startswith("sl").mean()) if len(trades) else np.nan,
        "timeout_rate": float(exit_reason.eq("max_hold").mean()) if len(trades) else np.nan,
        "median_holding_bars": float(pd.to_numeric(trades.get("holding_bars"), errors="coerce").median())
        if len(trades)
        else np.nan,
        "p25_net": float(net.quantile(0.25)) if len(trades) else np.nan,
        "p75_net": float(net.quantile(0.75)) if len(trades) else np.nan,
        "max_loss": float(net.min()) if len(trades) else np.nan,
    }


def _simulate_anchor_set(
    data: pd.DataFrame,
    anchor_name: str,
    base_signal_col: str,
    policy: EntryPolicy,
    rule_name: str,
    rule: ExecutionRule,
    resolver: Callable[[pd.Series], ExecutionRule] | None,
    config: ExperimentConfig,
) -> pd.DataFrame:
    universe_col = f"universe_dynamic_all_top{max(TOP_NS)}"
    if universe_col not in data.columns:
        return pd.DataFrame()
    signal_col = "__v06a_signal_event"
    sim = data.copy()
    sim[signal_col] = sim[universe_col].fillna(False) & sim[base_signal_col].fillna(False)
    if int(sim[signal_col].sum()) <= 0:
        return pd.DataFrame()
    trades = simulate_entry_policy_trades(
        sim,
        signal_col,
        anchor_name,
        policy,
        rule,
        0,
        "sl_first",
        True,
        C2_STATE_COL if base_signal_col == C2_SIGNAL_COL else signal_col,
        resolver,
    )
    trades = _attach_context(_expand_costs(trades), sim)
    if not trades.empty:
        trades["anchor_name"] = anchor_name
        trades["entry_policy"] = policy.name
        trades["execution_rule"] = rule_name
    return trades


def _compare_entry_only(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    keys = ["universe", "gate_name", "entry_policy", "execution_rule", "cost_single_side_bps"]
    baseline = summary[summary["anchor_name"].eq("entry_only_all_bars")][
        keys + ["net_expectancy", "gross_expectancy", "fill_rate", "sl_rate"]
    ].rename(
        columns={
            "net_expectancy": "entry_only_net",
            "gross_expectancy": "entry_only_gross",
            "fill_rate": "entry_only_fill_rate",
            "sl_rate": "entry_only_sl_rate",
        }
    )
    out = summary.merge(baseline, on=keys, how="left")
    out["net_lift_vs_entry_only"] = out["net_expectancy"] - out["entry_only_net"]
    out["gross_lift_vs_entry_only"] = out["gross_expectancy"] - out["entry_only_gross"]
    out["fill_rate_delta_vs_entry_only"] = out["fill_rate"] - out["entry_only_fill_rate"]
    out["sl_rate_delta_vs_entry_only"] = out["sl_rate"] - out["entry_only_sl_rate"]
    return out


def _best_reclaim(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    data = summary[
        summary["entry_policy"].str.startswith("R")
        & summary["cost_single_side_bps"].eq(10)
        & summary["anchor_name"].isin(
            [
                "entry_only_all_bars",
                "momentum_breakout",
                "short_squeeze_C2",
                "bullish_volume_shock",
            ]
        )
    ].copy()
    return data.sort_values(["net_expectancy", "net_lift_vs_entry_only"], ascending=False).head(30)


def _write_candidate_notes(report_root: Path, summary: pd.DataFrame, best: pd.DataFrame) -> None:
    lines = ["# v0.6A Reclaim Alpha Notes", ""]
    lines.append("This is a fixed research sprint. It does not tune C2/G3 and does not affect paper-live.")
    lines.append(
        "The first pass simulates Top50 anchors once, then slices trades by universe and regime context."
    )
    lines.append("")
    lines.append("## Readout")
    if best.empty:
        lines.append("- No reclaim rows were produced.")
    else:
        for row in best.head(12).itertuples(index=False):
            lines.append(
                f"- {row.universe} {row.gate_name} {row.anchor_name} {row.entry_policy} "
                f"/ {row.execution_rule}: trades={row.trades}, net10={row.net_expectancy:.4%}, "
                f"lift_vs_entry_only={row.net_lift_vs_entry_only:.4%}, sl={row.sl_rate:.2%}"
            )
    lines.append("")
    lines.append("## Interpretation Rule")
    lines.append("- If entry_only reclaim is strongest, the alpha is likely the reclaim structure itself.")
    lines.append("- If momentum/C2 improves entry_only reclaim, pressure/context is useful as a quality filter.")
    lines.append("- If BTC_chop remains weak, reclaim should be regime-gated before any paper-live candidate.")
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v06a_reclaim_alpha(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path = REPORT_ROOT,
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    turnover = _read_existing_columns(feature_path, V03_TURNOVER_COLUMNS)
    rank30 = build_dynamic_rank_table(turnover, instruments, config)
    rank90 = _rank_table_with_lookback(
        turnover,
        instruments,
        config,
        90,
        "turnover_rank_90d",
        "trailing_90d_turnover",
    )
    symbols = _top_symbols(rank30)
    del turnover

    trade_frames: list[pd.DataFrame] = []
    signal_counts: list[dict[str, object]] = []
    rules = _execution_rules(config)

    for idx, symbol in enumerate(symbols, start=1):
        data = _read_symbol_features(feature_path, rank30, rank90, symbol, config)
        if data.empty:
            continue
        print(f"v0.6A processing {idx}/{len(symbols)} {symbol}", flush=True)
        for top_n in TOP_NS:
            for anchor_name, signal_col in ANCHORS.items():
                for gate_name, gate_col in GATES.items():
                    signal_counts.append(
                        {
                            "universe": f"dynamic_all_top{top_n}",
                            "anchor_name": anchor_name,
                            "gate_name": gate_name,
                            "signals": int(
                                (
                                    data[f"universe_dynamic_all_top{top_n}"].fillna(False)
                                    & data[signal_col].fillna(False)
                                    & data[gate_col].fillna(False)
                                ).sum()
                            ),
                        }
                    )
        for anchor_name, signal_col in ANCHORS.items():
            for policy in ENTRY_POLICIES:
                for rule_name, rule, resolver in rules:
                    trades = _simulate_anchor_set(
                        data,
                        anchor_name,
                        signal_col,
                        policy,
                        rule_name,
                        rule,
                        resolver,
                        config,
                    )
                    if not trades.empty:
                        trade_frames.append(trades)

    trades = _concat_or_empty(trade_frames)
    counts = pd.DataFrame(signal_counts)
    summary_rows = []
    if not trades.empty:
        counts_lookup = (
            counts.groupby(["universe", "anchor_name", "gate_name"], dropna=False, sort=False)["signals"]
            .sum()
            .to_dict()
        )
        for top_n in TOP_NS:
            universe = f"dynamic_all_top{top_n}"
            universe_mask = pd.to_numeric(trades["dynamic_all_rank"], errors="coerce") <= top_n
            for gate_name in GATES:
                if gate_name == "all":
                    gate_mask = pd.Series(True, index=trades.index)
                elif gate_name == "BTC_up":
                    gate_mask = trades["btc_market_state"].astype(str).eq("BTC_up")
                elif gate_name == "BTC_chop":
                    gate_mask = trades["btc_market_state"].astype(str).eq("BTC_chop")
                elif gate_name == "BTC_down":
                    gate_mask = trades["btc_market_state"].astype(str).eq("BTC_down")
                elif gate_name == "non_transient_hot":
                    gate_mask = ~trades["transient_hot"].fillna(False).astype(bool)
                else:
                    gate_mask = pd.Series(True, index=trades.index)
                sample = trades[universe_mask & gate_mask].copy()
                if sample.empty:
                    continue
                for key, group in sample.groupby(
                    ["anchor_name", "entry_policy", "execution_rule", "cost_single_side_bps"],
                    dropna=False,
                    sort=False,
                ):
                    anchor_name, entry_policy, execution_rule, cost = key
                    signal_n = int(counts_lookup.get((universe, anchor_name, gate_name), 0))
                    summary_rows.append(
                        _trade_summary(
                            group,
                            signal_n,
                            universe=universe,
                            anchor_name=anchor_name,
                            gate_name=gate_name,
                            entry_policy=entry_policy,
                            execution_rule=execution_rule,
                            cost_single_side_bps=cost,
                        )
                    )
    summary = _compare_entry_only(pd.DataFrame(summary_rows))
    best = _best_reclaim(summary)

    regime = (
        summary[summary["cost_single_side_bps"].eq(10)]
        .sort_values(["universe", "anchor_name", "entry_policy", "execution_rule", "gate_name"])
        .copy()
    )
    topn = (
        summary[summary["cost_single_side_bps"].eq(10)]
        .groupby(["universe", "anchor_name", "entry_policy", "execution_rule"], as_index=False)
        .agg(
            trades=("trades", "sum"),
            net_expectancy=("net_expectancy", "mean"),
            net_lift_vs_entry_only=("net_lift_vs_entry_only", "mean"),
        )
    )

    outputs = {
        "reclaim_entry_summary": report_root / "reclaim_entry_summary.csv",
        "baseline_comparison": report_root / "baseline_comparison.csv",
        "regime_split": report_root / "regime_split.csv",
        "universe_topn_compare": report_root / "universe_topn_compare.csv",
        "best_reclaim_candidates": report_root / "best_reclaim_candidates.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    summary.to_csv(outputs["reclaim_entry_summary"], index=False)
    summary.to_csv(outputs["baseline_comparison"], index=False)
    regime.to_csv(outputs["regime_split"], index=False)
    topn.to_csv(outputs["universe_topn_compare"], index=False)
    best.to_csv(outputs["best_reclaim_candidates"], index=False)
    _write_candidate_notes(report_root, summary, best)
    return outputs
