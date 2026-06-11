from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.backtest import EntryPolicy, simulate_entry_policy_trades
from pressure_graph.config import ExperimentConfig
from pressure_graph.config.models import ExecutionRule
from pressure_graph.io import ensure_dir, raw_path, read_parquet
from pressure_graph.reports.v03 import _event_flags
from pressure_graph.reports.v06a1 import _read_symbol_features
from pressure_graph.reports.v06a31 import _entry_state_asof
from pressure_graph.reports.v06c import _build_regime_streaming, _rank_inputs


REPORT_ROOT = Path("reports/v0_7a_motif_atlas")
ATLAS_TOP_N = 50
FOCAL_MONTH = "2025-08"
COST_BPS = [10, 20, 30, 50]
SIM_WINDOW_BARS = 40

NEXT_OPEN = EntryPolicy("next_open", "next_open")
RECLAIM_1 = EntryPolicy(
    "pullback_1.0pct_reclaim_signal_close_valid_8_bars",
    "pullback_reclaim",
    valid_bars=8,
    pullback_pct=0.010,
)


@dataclass(frozen=True)
class MarketGate:
    name: str
    col: str
    entry_state: str | None = None
    negative_control: bool = False


@dataclass(frozen=True)
class LocalState:
    name: str
    event_col: str
    family: str
    negative_control: bool = False


@dataclass(frozen=True)
class ExecutionSpec:
    name: str
    policy: EntryPolicy


@dataclass(frozen=True)
class ExitSpec:
    name: str
    base_rule: str


MARKET_GATES = [
    MarketGate("all_market", "market_all"),
    MarketGate("BTC_up", "market_btc_up", "BTC_up"),
    MarketGate("volume_impulse_density_high", "market_volume_impulse_density_high"),
    MarketGate("BTC_chop_negative_control", "market_btc_chop", "BTC_chop", True),
]

LOCAL_STATES = [
    LocalState("bullish_volume_shock", "bullish_volume_shock_event", "impulse"),
    LocalState("momentum_breakout", "momentum_breakout_event", "momentum"),
    LocalState("squeeze_pressure", "squeeze_pressure_event", "pressure"),
    LocalState("volatility_compression_break", "volatility_compression_break_event", "compression"),
    LocalState("oi_flush_control", "oi_flush_control_event", "flush", True),
]

EXECUTIONS = [
    ExecutionSpec("next_open", NEXT_OPEN),
    ExecutionSpec("reclaim_1pct", RECLAIM_1),
]

EXITS = [
    ExitSpec("fast", "fast"),
    ExitSpec("vol_regime_fast", "vol_regime_fast"),
]


def _vol_regime_rule(row: pd.Series) -> ExecutionRule:
    vol_pct = pd.to_numeric(row.get("symbol_volatility_percentile"), errors="coerce")
    if pd.isna(vol_pct) or vol_pct < 40:
        return ExecutionRule(tp=0.03, sl=0.02, max_hold_bars=16)
    if vol_pct < 80:
        return ExecutionRule(tp=0.04, sl=0.025, max_hold_bars=16)
    return ExecutionRule(tp=0.05, sl=0.03, max_hold_bars=16)


def _rule(name: str, config: ExperimentConfig) -> tuple[ExecutionRule, Callable[[pd.Series], ExecutionRule] | None]:
    if name == "fast":
        return config.execution.rules["fast"], None
    if name == "swing":
        return ExecutionRule(tp=0.05, sl=0.03, max_hold_bars=48), None
    if name == "vol_regime_fast":
        return config.execution.rules["fast"], _vol_regime_rule
    raise KeyError(name)


def _future_extreme(series: pd.Series, bars: int, op: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").shift(-1).iloc[::-1]
    rolled = values.rolling(bars, min_periods=1)
    result = rolled.max() if op == "max" else rolled.min()
    return result.iloc[::-1].reindex(series.index)


def _add_right_tail_labels(data: pd.DataFrame) -> pd.DataFrame:
    out = data.sort_values(["exchange", "symbol", "bar_open_time"]).copy()
    close = pd.to_numeric(out["close"], errors="coerce").replace(0, np.nan)
    groups = out.groupby(["exchange", "symbol"], group_keys=False, sort=False, observed=True)
    for bars, suffix in [(48, "12h"), (96, "24h")]:
        future_high = groups["high"].apply(lambda s: _future_extreme(s, bars, "max")).reindex(out.index)
        future_low = groups["low"].apply(lambda s: _future_extreme(s, bars, "min")).reindex(out.index)
        out[f"mfe_{suffix}"] = (pd.to_numeric(future_high, errors="coerce") / close - 1.0).astype("float32")
        out[f"mae_{suffix}"] = (pd.to_numeric(future_low, errors="coerce") / close - 1.0).astype("float32")
    out["hit_10pct_12h"] = pd.to_numeric(out["mfe_12h"], errors="coerce") >= 0.10
    out["hit_20pct_24h"] = pd.to_numeric(out["mfe_24h"], errors="coerce") >= 0.20
    return out


def _add_motif_columns(data: pd.DataFrame, regime: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    out = data.merge(regime, on="feature_time", how="left", suffixes=("", "_market"))
    out["market_all"] = True
    state = out["btc_market_state"].astype(str)
    out["market_btc_up"] = state.eq("BTC_up")
    out["market_btc_chop"] = state.eq("BTC_chop")
    out["market_alt_breadth_high"] = out["alt_breadth_high"].fillna(False)
    out["market_volume_impulse_density_high"] = out["volume_impulse_density_high"].fillna(False)
    out["market_low_volume_impulse_density"] = out.get("low_volume_impulse_density", False)
    if not isinstance(out["market_low_volume_impulse_density"], pd.Series):
        out["market_low_volume_impulse_density"] = False
    out["market_low_volume_impulse_density"] = out["market_low_volume_impulse_density"].fillna(False)
    out["market_btc_up_volume_density"] = out["market_btc_up"] & out["market_volume_impulse_density_high"]

    warmup = out.get("warmup_complete", True)
    if not isinstance(warmup, pd.Series):
        warmup = pd.Series(bool(warmup), index=out.index)
    ret_1h = pd.to_numeric(out.get("ret_1h"), errors="coerce")
    ret_4h = pd.to_numeric(out.get("ret_4h"), errors="coerce")
    ret_4h_pct = pd.to_numeric(out.get("ret_4h_percentile"), errors="coerce")
    volume_z_1h = pd.to_numeric(out.get("volume_z_1h"), errors="coerce")
    volume_z_4h = pd.to_numeric(out.get("volume_z_4h"), errors="coerce")
    funding_pct = pd.to_numeric(out.get("funding_percentile"), errors="coerce")
    funding_z = pd.to_numeric(out.get("funding_z"), errors="coerce")
    oi_1h = pd.to_numeric(out.get("oi_value_delta_1h_percentile"), errors="coerce")
    oi_4h = pd.to_numeric(out.get("oi_value_delta_4h_percentile"), errors="coerce")
    vol_pct = pd.to_numeric(out.get("symbol_volatility_percentile"), errors="coerce")

    out["momentum_breakout_state"] = warmup & (ret_4h_pct > 70) & (volume_z_4h > 1.0) & (ret_4h > 0)
    out["squeeze_pressure_state"] = (
        warmup
        & ((oi_1h > 75) | (oi_4h > 75))
        & ((funding_pct < 60) | (funding_z < 0.5))
        & ((ret_1h > 0) | (ret_4h > 0))
        & (volume_z_1h > 1.0)
    )
    out["oi_expansion_funding_safe_state"] = warmup & (oi_4h > 75) & (funding_pct < 80) & (ret_4h > 0)
    out["volatility_compression_break_state"] = warmup & (vol_pct < 30) & (ret_4h > 0) & (volume_z_4h > 1.0)
    out["oi_flush_control_state"] = warmup & (oi_4h < 25) & (ret_4h < 0) & (volume_z_4h > 1.0)

    for state_col in [
        "momentum_breakout_state",
        "squeeze_pressure_state",
        "oi_expansion_funding_safe_state",
        "volatility_compression_break_state",
        "oi_flush_control_state",
    ]:
        event_col = state_col.replace("_state", "_event")
        out[event_col] = (
            out.groupby(["exchange", "symbol"], group_keys=False, sort=False, observed=True)[state_col]
            .apply(lambda series: _event_flags(series, config.events.cooldown_bars_4h))
            .reindex(out.index)
        )

    random_pool = warmup & ~out[[local.event_col for local in LOCAL_STATES if local.event_col in out.columns]].any(
        axis=1
    )
    hashed = pd.util.hash_pandas_object(out[["symbol", "bar_open_time"]], index=False)
    out["matched_random_event"] = random_pool & ((hashed % 53) == 0)
    return _add_right_tail_labels(out)


def _motif_name(market: MarketGate, local: LocalState, execution: ExecutionSpec, exit_spec: ExitSpec) -> str:
    return f"{market.name}__{local.name}__{execution.name}__{exit_spec.name}"


def _attach_context(trades: pd.DataFrame, data: pd.DataFrame, market: MarketGate) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    context_cols = [
        "exchange",
        "symbol",
        "feature_time",
        "month",
        "dynamic_all_rank",
        "dynamic_all_trailing_turnover",
        "liquidity_bucket",
        "btc_market_state",
        "symbol_volatility_percentile",
        "mfe_12h",
        "mae_12h",
        "mfe_24h",
        "mae_24h",
        "hit_10pct_12h",
        "hit_20pct_24h",
        "volume_impulse_density",
        "alt_ret_4h_positive_ratio",
    ]
    context = data[[col for col in context_cols if col in data.columns]].drop_duplicates(
        ["exchange", "symbol", "feature_time"]
    )
    out = trades.merge(
        context,
        left_on=["exchange", "symbol", "signal_time"],
        right_on=["exchange", "symbol", "feature_time"],
        how="left",
    ).drop(columns=["feature_time"], errors="ignore")
    out = out.rename(columns={"btc_market_state": "btc_state_at_signal"})
    if market.entry_state is not None:
        entry_context = data[["exchange", "symbol", "feature_time", "btc_market_state"]].copy()
        out = _entry_state_asof(out, entry_context)
        out = out[out["btc_state_at_entry"].astype(str).eq(market.entry_state)].copy()
    else:
        out["btc_state_at_entry"] = pd.Series(dtype=object)
    return out


def _expand_costs(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    frames = []
    for cost in COST_BPS:
        sample = trades.copy()
        sample["cost_single_side_bps"] = float(cost)
        sample["net_return"] = pd.to_numeric(sample["gross_return"], errors="coerce") - 2.0 * float(cost) / 10_000.0
        frames.append(sample)
    return pd.concat(frames, ignore_index=True)


def _simulate_motif(
    data: pd.DataFrame,
    market: MarketGate,
    local: LocalState,
    execution: ExecutionSpec,
    exit_spec: ExitSpec,
    baseline: str,
    signal_col: str,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, int]:
    mask = (
        (pd.to_numeric(data["dynamic_all_rank"], errors="coerce") <= ATLAS_TOP_N)
        & data[market.col].fillna(False).astype(bool)
        & data[signal_col].fillna(False).astype(bool)
    )
    signal_n = int(mask.sum())
    if signal_n <= 0:
        return pd.DataFrame(), 0
    sim = _signal_window(data, mask, SIM_WINDOW_BARS)
    sim["__v07a_signal"] = mask.loc[sim.index].fillna(False).astype(bool)
    rule, resolver = _rule(exit_spec.base_rule, config)
    trades = simulate_entry_policy_trades(
        sim,
        "__v07a_signal",
        "motif_atlas",
        execution.policy,
        rule,
        0,
        "sl_first",
        True,
        "__v07a_signal",
        resolver,
    )
    if trades.empty:
        return trades, signal_n
    trades = _attach_context(trades, data, market)
    if trades.empty:
        return trades, signal_n
    trades["motif_name"] = _motif_name(market, local, execution, exit_spec)
    trades["market_state"] = market.name
    trades["local_state"] = local.name
    trades["local_family"] = local.family
    trades["execution_type"] = execution.name
    trades["exit_type"] = exit_spec.name
    trades["baseline"] = baseline
    trades["negative_control"] = market.negative_control or local.negative_control
    return _expand_costs(trades), signal_n


def _signal_window(data: pd.DataFrame, signal_mask: pd.Series, bars: int) -> pd.DataFrame:
    mask = signal_mask.fillna(False).to_numpy(dtype=bool)
    positions = np.flatnonzero(mask)
    if len(positions) == 0:
        return data.iloc[0:0].copy()
    take = np.zeros(len(data), dtype=bool)
    for pos in positions:
        take[pos : min(len(data), pos + bars)] = True
    return data.iloc[take].copy(deep=False)


def _summary_row(trades: pd.DataFrame, signal_n: int, **keys: object) -> dict[str, object]:
    net = pd.to_numeric(trades.get("net_return"), errors="coerce")
    gross = pd.to_numeric(trades.get("gross_return"), errors="coerce")
    exits = trades.get("exit_reason", pd.Series(dtype=str)).astype(str)
    mfe12 = pd.to_numeric(trades.get("mfe_12h"), errors="coerce")
    mfe24 = pd.to_numeric(trades.get("mfe_24h"), errors="coerce")
    mae12 = pd.to_numeric(trades.get("mae_12h"), errors="coerce")
    p05 = net.quantile(0.05) if len(trades) else np.nan
    p95 = net.quantile(0.95) if len(trades) else np.nan
    return {
        **keys,
        "signals": int(signal_n),
        "trades": int(len(trades)),
        "fill_rate": float(len(trades) / signal_n) if signal_n else np.nan,
        "gross_expectancy": float(gross.mean()) if len(trades) else np.nan,
        "net_expectancy": float(net.mean()) if len(trades) else np.nan,
        "net_sum": float(net.sum()) if len(trades) else 0.0,
        "tp_rate": float(exits.str.startswith("tp").mean()) if len(trades) else np.nan,
        "sl_rate": float(exits.str.startswith("sl").mean()) if len(trades) else np.nan,
        "timeout_rate": float(exits.eq("max_hold").mean()) if len(trades) else np.nan,
        "hit_10pct_12h": float(trades.get("hit_10pct_12h", pd.Series(dtype=bool)).fillna(False).mean())
        if len(trades)
        else np.nan,
        "hit_20pct_24h": float(trades.get("hit_20pct_24h", pd.Series(dtype=bool)).fillna(False).mean())
        if len(trades)
        else np.nan,
        "median_mfe_12h": float(mfe12.median()) if len(trades) else np.nan,
        "p90_mfe_12h": float(mfe12.quantile(0.90)) if len(trades) else np.nan,
        "p95_mfe_24h": float(mfe24.quantile(0.95)) if len(trades) else np.nan,
        "median_mae_12h": float(mae12.median()) if len(trades) else np.nan,
        "payoff_skew": float(p95 / abs(p05)) if pd.notna(p95) and pd.notna(p05) and p05 < 0 else np.nan,
    }


def _aggregate(trades: pd.DataFrame, signal_counts: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    count_lookup = signal_counts.set_index(["motif_name", "baseline"])["signals"].to_dict()
    rows = []
    group_cols = [
        "motif_name",
        "market_state",
        "local_state",
        "local_family",
        "execution_type",
        "exit_type",
        "baseline",
        "cost_single_side_bps",
    ]
    for key, group in trades.groupby(group_cols, sort=False, dropna=False):
        payload = dict(zip(group_cols, key, strict=False))
        rows.append(_summary_row(group, int(count_lookup.get((payload["motif_name"], payload["baseline"]), 0)), **payload))
    return pd.DataFrame(rows)


def _motif_summary(baselines: pd.DataFrame) -> pd.DataFrame:
    if baselines.empty:
        return pd.DataFrame()
    primary = baselines[
        baselines["baseline"].eq("candidate") & pd.to_numeric(baselines["cost_single_side_bps"], errors="coerce").eq(10)
    ].copy()
    rows = []
    for row in primary.to_dict("records"):
        motif = row["motif_name"]
        cost_rows = baselines[baselines["motif_name"].eq(motif)]
        candidate_costs = cost_rows[cost_rows["baseline"].eq("candidate")]
        row["net20"] = _lookup_metric(candidate_costs, 20, "net_expectancy")
        row["net30"] = _lookup_metric(candidate_costs, 30, "net_expectancy")
        row["net50"] = _lookup_metric(candidate_costs, 50, "net_expectancy")
        for baseline, col in [("entry_only_neutral", "entry_only_lift"), ("matched_random", "matched_random_lift")]:
            base = cost_rows[
                cost_rows["baseline"].eq(baseline)
                & pd.to_numeric(cost_rows["cost_single_side_bps"], errors="coerce").eq(10)
            ]
            row[col] = row["net_expectancy"] - float(base["net_expectancy"].iloc[0]) if not base.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows).rename(columns={"net_expectancy": "net10"})


def _lookup_metric(rows: pd.DataFrame, cost: float, metric: str) -> float:
    sample = rows[pd.to_numeric(rows["cost_single_side_bps"], errors="coerce").eq(float(cost))]
    return float(sample[metric].iloc[0]) if not sample.empty else np.nan


def _month_cap(trades: pd.DataFrame, cap: float = 0.35) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    sample = trades[trades["baseline"].eq("candidate")]
    for (motif, cost), group in sample.groupby(["motif_name", "cost_single_side_bps"], sort=False, dropna=False):
        total = pd.to_numeric(group["net_return"], errors="coerce").sum()
        cap_value = total * cap if total > 0 else 0.0
        capped = []
        month_rows = []
        for month, month_group in group.groupby("month", sort=False, dropna=False):
            value = pd.to_numeric(month_group["net_return"], errors="coerce").sum()
            adjusted = min(value, cap_value) if value > 0 and cap_value > 0 else value
            capped.append(adjusted)
            month_rows.append((month, value))
        max_contrib = np.nan
        if total:
            max_contrib = max(abs(value / total) for _, value in month_rows)
        rows.append(
            {
                "motif_name": motif,
                "cost_single_side_bps": cost,
                "month_cap35_expectancy": float(np.sum(capped) / len(group)) if len(group) else np.nan,
                "max_month_contribution": float(max_contrib) if pd.notna(max_contrib) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _regime_split(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    sample = trades[trades["baseline"].eq("candidate")]
    rows = []
    for key, group in sample.groupby(["motif_name", "cost_single_side_bps", "btc_state_at_signal"], sort=False, dropna=False):
        motif, cost, btc_state = key
        rows.append(_summary_row(group, len(group), motif_name=motif, cost_single_side_bps=cost, btc_state_at_signal=btc_state))
    return pd.DataFrame(rows)


def _top_candidates(summary: pd.DataFrame, caps: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    cap10 = caps[pd.to_numeric(caps["cost_single_side_bps"], errors="coerce").eq(20.0)][
        ["motif_name", "month_cap35_expectancy", "max_month_contribution"]
    ]
    out = summary.merge(cap10, on="motif_name", how="left")
    out["atlas_score"] = (
        pd.to_numeric(out["net10"], errors="coerce").fillna(0)
        + 0.7 * pd.to_numeric(out["net20"], errors="coerce").fillna(0)
        + 0.5 * pd.to_numeric(out["hit_10pct_12h"], errors="coerce").fillna(0)
        + 0.3 * pd.to_numeric(out["hit_20pct_24h"], errors="coerce").fillna(0)
        + 0.5 * pd.to_numeric(out["matched_random_lift"], errors="coerce").fillna(0)
        - 0.5 * pd.to_numeric(out["sl_rate"], errors="coerce").fillna(0)
    )
    eligible = out[
        (pd.to_numeric(out["trades"], errors="coerce") >= 80)
        & (pd.to_numeric(out["net20"], errors="coerce") > 0)
    ].copy()
    return eligible.sort_values("atlas_score", ascending=False).head(50)


def _write_notes(report_root: Path, top: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = ["# v0.7A Motif Atlas", ""]
    lines.append("Discovery-only 15m motif map. No paper-live promotion, no tick validation, and no new data source.")
    lines.append("IR2 remains the running paper-live candidate; v0.7A is for finding broader motif families.")
    lines.append("First pass uses a controlled Top50 grid and matched-random baseline; top motifs should get deeper baselines in v0.7A.1.")
    lines.append("")
    if top.empty:
        lines.append("- No motif passed the minimal atlas screen of trades >= 80 and net20 > 0.")
    else:
        lines.append("## Top Atlas Rows")
        for row in top.head(12).to_dict("records"):
            lines.append(
                f"- {row['motif_name']}: trades={row['trades']}, net10={row['net10']:.4%}, "
                f"net20={row['net20']:.4%}, hit10_12h={row['hit_10pct_12h']:.2%}, "
                f"matched_lift={row['matched_random_lift']:.4%}, cap20={row['month_cap35_expectancy']:.4%}"
            )
    if not summary.empty:
        family = (
            summary.groupby("local_family", as_index=False)["net10"]
            .median()
            .sort_values("net10", ascending=False)
        )
        lines.append("")
        lines.append("## Family Median Net10")
        for row in family.to_dict("records"):
            lines.append(f"- {row['local_family']}: median_net10={row['net10']:.4%}")
    (report_root / "motif_notes.md").write_text("\n".join(lines), encoding="utf-8")


def _simulate_streaming(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    symbols: list[str],
    regime: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trade_path = report_root / "_v07a_trades_tmp.csv"
    if trade_path.exists():
        trade_path.unlink()
    wrote_header = False
    signal_rows: list[dict[str, object]] = []
    baselines = [
        ("candidate", None),
        ("matched_random", "matched_random_event"),
    ]
    for idx, symbol in enumerate(symbols, start=1):
        data = _read_symbol_features(feature_path, rank30, rank90, symbol, config)
        if data.empty:
            continue
        data = data[pd.to_numeric(data["dynamic_all_rank"], errors="coerce") <= ATLAS_TOP_N].copy()
        if data.empty:
            continue
        data = _add_motif_columns(data, regime, config)
        for market in MARKET_GATES:
            for local in LOCAL_STATES:
                for execution in EXECUTIONS:
                    for exit_spec in EXITS:
                        motif = _motif_name(market, local, execution, exit_spec)
                        for baseline, override_col in baselines:
                            signal_col = override_col or local.event_col
                            trades, signal_n = _simulate_motif(
                                data,
                                market,
                                local,
                                execution,
                                exit_spec,
                                baseline,
                                signal_col,
                                config,
                            )
                            signal_rows.append({"motif_name": motif, "baseline": baseline, "signals": signal_n})
                            if not trades.empty:
                                trades.to_csv(trade_path, mode="a", header=not wrote_header, index=False)
                                wrote_header = True
                                del trades
        print(f"v0.7A motif pass {idx}/{len(symbols)} {symbol}", flush=True)
        del data
        gc.collect()
    trades = pd.read_csv(trade_path, low_memory=False) if wrote_header else pd.DataFrame()
    if trade_path.exists():
        trade_path.unlink()
    signals = pd.DataFrame(signal_rows)
    if not signals.empty:
        signals = signals.groupby(["motif_name", "baseline"], as_index=False, sort=False, dropna=False)["signals"].sum()
    return trades, signals


def _write_outputs(report_root: Path, trades: pd.DataFrame, signal_counts: pd.DataFrame) -> dict[str, Path]:
    baselines = _aggregate(trades, signal_counts)
    summary = _motif_summary(baselines)
    caps = _month_cap(trades)
    regime_split = _regime_split(trades)
    top = _top_candidates(summary, caps)
    outputs = {
        "motif_summary": report_root / "motif_summary.csv",
        "motif_baselines": report_root / "motif_baselines.csv",
        "motif_regime_split": report_root / "motif_regime_split.csv",
        "motif_month_cap": report_root / "motif_month_cap.csv",
        "motif_top_candidates": report_root / "motif_top_candidates.csv",
        "motif_notes": report_root / "motif_notes.md",
    }
    summary.to_csv(outputs["motif_summary"], index=False)
    baselines.to_csv(outputs["motif_baselines"], index=False)
    regime_split.to_csv(outputs["motif_regime_split"], index=False)
    caps.to_csv(outputs["motif_month_cap"], index=False)
    top.to_csv(outputs["motif_top_candidates"], index=False)
    _write_notes(report_root, top, summary)
    return outputs


def _assemble_batches(report_root: Path) -> dict[str, Path]:
    batch_root = report_root / "batches"
    trade_files = sorted(batch_root.glob("trades_*.csv"))
    signal_files = sorted(batch_root.glob("signals_*.csv"))
    trades = pd.concat((pd.read_csv(path, low_memory=False) for path in trade_files), ignore_index=True) if trade_files else pd.DataFrame()
    signals = pd.concat((pd.read_csv(path) for path in signal_files), ignore_index=True) if signal_files else pd.DataFrame()
    if not signals.empty:
        signals = signals.groupby(["motif_name", "baseline"], as_index=False, sort=False, dropna=False)["signals"].sum()
    return _write_outputs(report_root, trades, signals)


def write_v07a_motif_atlas(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path = REPORT_ROOT,
    symbol_offset: int = 0,
    symbol_limit: int | None = None,
    batch_only: bool = False,
    assemble_only: bool = False,
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    if assemble_only:
        return _assemble_batches(report_root)
    rank30, rank90, symbols = _rank_inputs(feature_path, instruments, config)
    selected_symbols = symbols[symbol_offset : symbol_offset + symbol_limit if symbol_limit is not None else None]
    regime = _build_regime_streaming(feature_path, rank30, rank90, symbols, config)
    trades, signal_counts = _simulate_streaming(
        feature_path, rank30, rank90, selected_symbols, regime, config, report_root
    )
    if batch_only:
        batch_root = ensure_dir(report_root / "batches")
        suffix = f"{symbol_offset}_{symbol_limit if symbol_limit is not None else 'all'}"
        trade_path = batch_root / f"trades_{suffix}.csv"
        signal_path = batch_root / f"signals_{suffix}.csv"
        trades.to_csv(trade_path, index=False)
        signal_counts.to_csv(signal_path, index=False)
        return {"batch_trades": trade_path, "batch_signals": signal_path}
    return _write_outputs(report_root, trades, signal_counts)


def run_v07a_motif_atlas_from_features(
    config: ExperimentConfig,
    symbol_offset: int = 0,
    symbol_limit: int | None = None,
    batch_only: bool = False,
    assemble_only: bool = False,
) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = read_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v07a_motif_atlas(
        features_path,
        instruments,
        config,
        symbol_offset=symbol_offset,
        symbol_limit=symbol_limit,
        batch_only=batch_only,
        assemble_only=assemble_only,
    )
