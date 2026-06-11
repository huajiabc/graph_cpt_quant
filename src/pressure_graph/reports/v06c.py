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
from pressure_graph.reports.v03 import V03_TURNOVER_COLUMNS, _concat_or_empty, _read_existing_columns, build_dynamic_rank_table
from pressure_graph.reports.v04 import _rank_table_with_lookback
from pressure_graph.reports.v06a1 import MAX_BOUNDARY_TOP_N, _read_symbol_features
from pressure_graph.reports.v06a31 import _entry_state_asof


REPORT_ROOT = Path("reports/v0_6c_attack_regime_detector")
COST_BPS = [5, 10, 20, 30, 50]
TOP_N = 50
FOCAL_MONTH = "2025-08"
R1_RECLAIM = EntryPolicy(
    "R1_pullback_1.0pct_reclaim_signal_close_valid_8_bars",
    "pullback_reclaim",
    valid_bars=8,
    pullback_pct=0.010,
)
NEXT_OPEN = EntryPolicy("B0_volume_shock_next_open", "next_open")
P1_PULLBACK = EntryPolicy("B1_pullback_1.0pct_only_valid_8_bars", "pullback", valid_bars=8, pullback_pct=0.010)


@dataclass(frozen=True)
class RegimeGate:
    gate_name: str
    gate_col: str
    signal_state: str
    entry_state: str
    negative_control: bool = False


GATES = [
    RegimeGate("G0_IR2_raw_strict", "gate_ir2_raw", "BTC_up", "BTC_up"),
    RegimeGate("G1_BTC_up_strong", "gate_btc_up_strong", "BTC_up", "BTC_up"),
    RegimeGate("G2_alt_breadth_high", "gate_alt_breadth_high", "BTC_up", "BTC_up"),
    RegimeGate("G3_volume_impulse_density_high", "gate_volume_impulse_density_high", "BTC_up", "BTC_up"),
    RegimeGate("G4_BTC_up_strong_and_volume_density", "gate_btc_up_strong_and_volume_density", "BTC_up", "BTC_up"),
    RegimeGate("G5_alt_breadth_and_volume_density", "gate_alt_breadth_and_volume_density", "BTC_up", "BTC_up"),
    RegimeGate("G6_risk_off_veto", "gate_risk_off_veto", "BTC_up", "BTC_up"),
    RegimeGate("NEG_BTC_chop", "gate_btc_chop", "BTC_chop", "BTC_chop", True),
    RegimeGate("NEG_low_breadth", "gate_low_breadth", "BTC_up", "BTC_up", True),
    RegimeGate("NEG_low_volume_impulse_density", "gate_low_volume_impulse_density", "BTC_up", "BTC_up", True),
]


def _vol_regime_rule(row: pd.Series) -> ExecutionRule:
    vol_pct = pd.to_numeric(row.get("symbol_volatility_percentile"), errors="coerce")
    if pd.isna(vol_pct) or vol_pct < 40:
        return ExecutionRule(tp=0.03, sl=0.02, max_hold_bars=16)
    if vol_pct < 80:
        return ExecutionRule(tp=0.04, sl=0.025, max_hold_bars=16)
    return ExecutionRule(tp=0.05, sl=0.03, max_hold_bars=16)


def _rule(config: ExperimentConfig) -> tuple[ExecutionRule, Callable[[pd.Series], ExecutionRule]]:
    return config.execution.rules["fast"], _vol_regime_rule


def _top_symbols(ranks: pd.DataFrame) -> list[str]:
    return sorted(
        ranks[pd.to_numeric(ranks["dynamic_all_rank"], errors="coerce") <= MAX_BOUNDARY_TOP_N]["symbol"]
        .dropna()
        .astype(str)
        .unique()
    )


def _load_impulse_rows(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    turnover = _read_existing_columns(feature_path, V03_TURNOVER_COLUMNS)
    rank30 = build_dynamic_rank_table(turnover, instruments, config)
    rank90 = _rank_table_with_lookback(turnover, instruments, config, 90, "turnover_rank_90d", "trailing_90d_turnover")
    symbols = _top_symbols(rank30)
    del turnover
    frames = []
    for idx, symbol in enumerate(symbols, start=1):
        data = _read_symbol_features(feature_path, rank30, rank90, symbol, config)
        if data.empty:
            continue
        print(f"v0.6C loading {idx}/{len(symbols)} {symbol}", flush=True)
        frames.append(data[pd.to_numeric(data["dynamic_all_rank"], errors="coerce") <= TOP_N].copy())
    return _concat_or_empty(frames)


def _market_regime_features(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    sample = data[pd.to_numeric(data["dynamic_all_rank"], errors="coerce") <= TOP_N].copy()
    sample["ret_1h_positive"] = pd.to_numeric(sample["ret_1h"], errors="coerce") > 0
    sample["ret_4h_positive"] = pd.to_numeric(sample["ret_4h"], errors="coerce") > 0
    sample["alt_down_4h"] = pd.to_numeric(sample["ret_4h"], errors="coerce") < 0
    sample["alt_volume_expansion"] = pd.to_numeric(sample["volume_z_4h"], errors="coerce") > 1.0
    sample["market_funding_extreme"] = pd.to_numeric(sample["funding_percentile"], errors="coerce") > 90
    sample["market_oi_explosion"] = pd.to_numeric(sample["oi_value_delta_4h_percentile"], errors="coerce") > 90
    sample["volume_impulse"] = sample["bullish_volume_shock_state"].fillna(False).astype(bool)
    grouped = sample.groupby("feature_time", sort=True, observed=True)
    regime = grouped.agg(
        top50_symbols=("symbol", "nunique"),
        alt_ret_1h_positive_ratio=("ret_1h_positive", "mean"),
        alt_ret_4h_positive_ratio=("ret_4h_positive", "mean"),
        alt_down_breadth=("alt_down_4h", "mean"),
        alt_volume_expansion_ratio=("alt_volume_expansion", "mean"),
        volume_impulse_density=("volume_impulse", "mean"),
        market_funding_extreme_ratio=("market_funding_extreme", "mean"),
        market_oi_explosion_ratio=("market_oi_explosion", "mean"),
    ).reset_index()
    btc = sample[sample["symbol"].astype(str).eq("BTCUSDT")][
        ["feature_time", "btc_market_state", "btc_ret_1h", "btc_ret_4h", "btc_volatility_4h"]
    ].drop_duplicates("feature_time")
    regime = regime.merge(btc, on="feature_time", how="left")
    regime["btc_up_strong"] = regime["btc_market_state"].astype(str).eq("BTC_up") & (
        pd.to_numeric(regime["btc_ret_4h"], errors="coerce") >= 0.01
    )
    regime["alt_breadth_high"] = pd.to_numeric(regime["alt_ret_4h_positive_ratio"], errors="coerce") >= 0.60
    regime["volume_impulse_density_high"] = pd.to_numeric(regime["volume_impulse_density"], errors="coerce") >= 0.10
    regime["low_breadth"] = pd.to_numeric(regime["alt_ret_4h_positive_ratio"], errors="coerce") <= 0.40
    regime["low_volume_impulse_density"] = pd.to_numeric(regime["volume_impulse_density"], errors="coerce") <= 0.02
    regime["risk_off"] = (
        regime["btc_market_state"].astype(str).eq("BTC_down")
        | (pd.to_numeric(regime["alt_down_breadth"], errors="coerce") >= 0.65)
        | (pd.to_numeric(regime["market_funding_extreme_ratio"], errors="coerce") >= 0.25)
        | (pd.to_numeric(regime["market_oi_explosion_ratio"], errors="coerce") >= 0.25)
    )
    return regime


def _market_regime_partial(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    sample = data[pd.to_numeric(data["dynamic_all_rank"], errors="coerce") <= TOP_N].copy()
    if sample.empty:
        return pd.DataFrame(), pd.DataFrame()
    sample["ret_1h_positive"] = (pd.to_numeric(sample["ret_1h"], errors="coerce") > 0).astype("int8")
    sample["ret_4h_positive"] = (pd.to_numeric(sample["ret_4h"], errors="coerce") > 0).astype("int8")
    sample["alt_down_4h"] = (pd.to_numeric(sample["ret_4h"], errors="coerce") < 0).astype("int8")
    sample["alt_volume_expansion"] = (pd.to_numeric(sample["volume_z_4h"], errors="coerce") > 1.0).astype("int8")
    sample["market_funding_extreme"] = (pd.to_numeric(sample["funding_percentile"], errors="coerce") > 90).astype(
        "int8"
    )
    sample["market_oi_explosion"] = (
        pd.to_numeric(sample["oi_value_delta_4h_percentile"], errors="coerce") > 90
    ).astype("int8")
    sample["volume_impulse"] = sample["bullish_volume_shock_state"].fillna(False).astype("int8")
    grouped = sample.groupby("feature_time", sort=True, observed=True)
    partial = grouped.agg(
        top50_symbols=("symbol", "nunique"),
        ret_1h_positive_sum=("ret_1h_positive", "sum"),
        ret_4h_positive_sum=("ret_4h_positive", "sum"),
        alt_down_4h_sum=("alt_down_4h", "sum"),
        alt_volume_expansion_sum=("alt_volume_expansion", "sum"),
        volume_impulse_sum=("volume_impulse", "sum"),
        market_funding_extreme_sum=("market_funding_extreme", "sum"),
        market_oi_explosion_sum=("market_oi_explosion", "sum"),
    ).reset_index()
    btc = sample[sample["symbol"].astype(str).eq("BTCUSDT")][
        ["feature_time", "btc_market_state", "btc_ret_1h", "btc_ret_4h", "btc_volatility_4h"]
    ].drop_duplicates("feature_time")
    return partial, btc


def _finalize_market_regime(partials: list[pd.DataFrame], btc_frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not partials:
        return pd.DataFrame()
    sums = pd.concat(partials, ignore_index=True).groupby("feature_time", sort=True, observed=True).sum().reset_index()
    denom = pd.to_numeric(sums["top50_symbols"], errors="coerce").replace(0, np.nan)
    regime = pd.DataFrame(
        {
            "feature_time": sums["feature_time"],
            "top50_symbols": sums["top50_symbols"],
            "alt_ret_1h_positive_ratio": sums["ret_1h_positive_sum"] / denom,
            "alt_ret_4h_positive_ratio": sums["ret_4h_positive_sum"] / denom,
            "alt_down_breadth": sums["alt_down_4h_sum"] / denom,
            "alt_volume_expansion_ratio": sums["alt_volume_expansion_sum"] / denom,
            "volume_impulse_density": sums["volume_impulse_sum"] / denom,
            "market_funding_extreme_ratio": sums["market_funding_extreme_sum"] / denom,
            "market_oi_explosion_ratio": sums["market_oi_explosion_sum"] / denom,
        }
    )
    if btc_frames:
        btc = (
            pd.concat(btc_frames, ignore_index=True)
            .sort_values("feature_time")
            .drop_duplicates("feature_time", keep="last")
        )
        regime = regime.merge(btc, on="feature_time", how="left")
    else:
        regime["btc_market_state"] = pd.Series(dtype=object)
        regime["btc_ret_1h"] = np.nan
        regime["btc_ret_4h"] = np.nan
        regime["btc_volatility_4h"] = np.nan
    regime["btc_up_strong"] = regime["btc_market_state"].astype(str).eq("BTC_up") & (
        pd.to_numeric(regime["btc_ret_4h"], errors="coerce") >= 0.01
    )
    regime["alt_breadth_high"] = pd.to_numeric(regime["alt_ret_4h_positive_ratio"], errors="coerce") >= 0.60
    regime["volume_impulse_density_high"] = pd.to_numeric(regime["volume_impulse_density"], errors="coerce") >= 0.10
    regime["low_breadth"] = pd.to_numeric(regime["alt_ret_4h_positive_ratio"], errors="coerce") <= 0.40
    regime["low_volume_impulse_density"] = pd.to_numeric(regime["volume_impulse_density"], errors="coerce") <= 0.02
    regime["risk_off"] = (
        regime["btc_market_state"].astype(str).eq("BTC_down")
        | (pd.to_numeric(regime["alt_down_breadth"], errors="coerce") >= 0.65)
        | (pd.to_numeric(regime["market_funding_extreme_ratio"], errors="coerce") >= 0.25)
        | (pd.to_numeric(regime["market_oi_explosion_ratio"], errors="coerce") >= 0.25)
    )
    return regime


def _add_gate_columns(data: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    out = data.merge(regime, on="feature_time", how="left", suffixes=("", "_market"))
    if "matched_random_event" not in out.columns:
        random_state = (
            out.get("warmup_complete", True)
            & ~out["bullish_volume_shock_state"].fillna(False).astype(bool)
        )
        out["matched_random_state"] = random_state
        hashed = pd.util.hash_pandas_object(out[["symbol", "bar_open_time"]], index=False)
        out["matched_random_event"] = random_state & ((hashed % 97) == 0)
    state = out["btc_market_state"].astype(str)
    out["gate_ir2_raw"] = state.eq("BTC_up")
    out["gate_btc_up_strong"] = out["gate_ir2_raw"] & out["btc_up_strong"].fillna(False)
    out["gate_alt_breadth_high"] = out["gate_ir2_raw"] & out["alt_breadth_high"].fillna(False)
    out["gate_volume_impulse_density_high"] = out["gate_ir2_raw"] & out["volume_impulse_density_high"].fillna(False)
    out["gate_btc_up_strong_and_volume_density"] = out["gate_btc_up_strong"] & out["gate_volume_impulse_density_high"]
    out["gate_alt_breadth_and_volume_density"] = out["gate_alt_breadth_high"] & out["gate_volume_impulse_density_high"]
    out["gate_risk_off_veto"] = out["gate_ir2_raw"] & ~out["risk_off"].fillna(True)
    out["gate_btc_chop"] = state.eq("BTC_chop")
    out["gate_low_breadth"] = out["gate_ir2_raw"] & out["low_breadth"].fillna(False)
    out["gate_low_volume_impulse_density"] = out["gate_ir2_raw"] & out["low_volume_impulse_density"].fillna(False)
    return out


def _rank_inputs(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    turnover = _read_existing_columns(feature_path, V03_TURNOVER_COLUMNS)
    rank30 = build_dynamic_rank_table(turnover, instruments, config)
    rank90 = _rank_table_with_lookback(turnover, instruments, config, 90, "turnover_rank_90d", "trailing_90d_turnover")
    symbols = _top_symbols(rank30)
    del turnover
    return rank30, rank90, symbols


def _build_regime_streaming(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    symbols: list[str],
    config: ExperimentConfig,
) -> pd.DataFrame:
    partials: list[pd.DataFrame] = []
    btc_frames: list[pd.DataFrame] = []
    for idx, symbol in enumerate(symbols, start=1):
        data = _read_symbol_features(feature_path, rank30, rank90, symbol, config)
        if data.empty:
            continue
        data = data[pd.to_numeric(data["dynamic_all_rank"], errors="coerce") <= TOP_N].copy()
        partial, btc = _market_regime_partial(data)
        if not partial.empty:
            partials.append(partial)
        if not btc.empty:
            btc_frames.append(btc)
        print(f"v0.6C regime pass {idx}/{len(symbols)} {symbol}", flush=True)
    return _finalize_market_regime(partials, btc_frames)


def _simulate_streaming(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    symbols: list[str],
    regime: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path,
) -> tuple[Path | None, pd.DataFrame]:
    count_rows: list[dict[str, object]] = []
    trade_path = report_root / "_v06c_trades_tmp.csv"
    if trade_path.exists():
        trade_path.unlink()
    wrote_header = False
    variants = [
        ("candidate_reclaim", "bullish_volume_shock_event", R1_RECLAIM),
        ("entry_only_reclaim", "neutral_volume_event", R1_RECLAIM),
        ("volume_shock_next_open", "bullish_volume_shock_event", NEXT_OPEN),
        ("volume_shock_pullback_only", "bullish_volume_shock_event", P1_PULLBACK),
        ("matched_random_reclaim", "matched_random_event", R1_RECLAIM),
    ]
    for idx, symbol in enumerate(symbols, start=1):
        data = _read_symbol_features(feature_path, rank30, rank90, symbol, config)
        if data.empty:
            continue
        data = data[pd.to_numeric(data["dynamic_all_rank"], errors="coerce") <= TOP_N].copy()
        if data.empty:
            continue
        data = _add_gate_columns(data, regime)
        for gate in GATES:
            for baseline, signal_col, policy in variants:
                trades, signal_n = _simulate(data, gate, baseline, signal_col, policy, config)
                count_rows.append({"gate_name": gate.gate_name, "baseline": baseline, "signals": signal_n})
                if not trades.empty:
                    trades.to_csv(trade_path, mode="a", header=not wrote_header, index=False)
                    wrote_header = True
                    del trades
        print(f"v0.6C simulation pass {idx}/{len(symbols)} {symbol}", flush=True)
        del data
        gc.collect()
    signal_counts = pd.DataFrame(count_rows)
    if not signal_counts.empty:
        signal_counts = (
            signal_counts.groupby(["gate_name", "baseline"], as_index=False, sort=False, dropna=False)["signals"]
            .sum()
        )
    return (trade_path if wrote_header else None), signal_counts


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


def _attach_context(trades: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    context_cols = [
        "exchange",
        "symbol",
        "feature_time",
        "month",
        "dynamic_all_rank",
        "btc_market_state",
        "volume_impulse_density",
        "alt_ret_4h_positive_ratio",
        "alt_ret_1h_positive_ratio",
        "alt_volume_expansion_ratio",
        "alt_down_breadth",
        "market_funding_extreme_ratio",
        "market_oi_explosion_ratio",
        "btc_up_strong",
        "alt_breadth_high",
        "volume_impulse_density_high",
        "risk_off",
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
    entry_context = data[["exchange", "symbol", "feature_time", "btc_market_state"]].copy()
    return _entry_state_asof(out, entry_context)


def _simulate(
    data: pd.DataFrame,
    gate: RegimeGate,
    baseline: str,
    signal_col: str,
    policy: EntryPolicy,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, int]:
    mask = data[gate.gate_col].fillna(False).astype(bool) & data[signal_col].fillna(False).astype(bool)
    signal_n = int(mask.sum())
    if signal_n <= 0:
        return pd.DataFrame(), 0
    sim = data.copy(deep=False)
    sim["__v06c_signal"] = mask
    rule, resolver = _rule(config)
    trades = simulate_entry_policy_trades(
        sim,
        "__v06c_signal",
        "IR2_regime",
        policy,
        rule,
        0,
        "sl_first",
        True,
        "__v06c_signal",
        resolver,
    )
    if trades.empty:
        return trades, signal_n
    trades = _attach_context(trades, data)
    trades = trades[
        trades["btc_state_at_signal"].astype(str).eq(gate.signal_state)
        & trades["btc_state_at_entry"].astype(str).eq(gate.entry_state)
    ].copy()
    if trades.empty:
        return trades, signal_n
    trades["gate_name"] = gate.gate_name
    trades["baseline"] = baseline
    trades["negative_control"] = gate.negative_control
    trades["entry_policy"] = policy.name
    return _expand_costs(trades), signal_n


def _net(sample: pd.DataFrame, cost: float, exclude_month: str | None = None) -> float:
    data = sample[pd.to_numeric(sample["cost_single_side_bps"], errors="coerce").eq(float(cost))]
    if exclude_month is not None:
        data = data[~data["month"].astype(str).eq(exclude_month)]
    if data.empty:
        return np.nan
    return float(pd.to_numeric(data["net_return"], errors="coerce").mean())


def _month_cap(sample: pd.DataFrame, cost: float, cap: float = 0.35) -> float:
    data = sample[pd.to_numeric(sample["cost_single_side_bps"], errors="coerce").eq(float(cost))]
    if data.empty:
        return np.nan
    total = pd.to_numeric(data["net_return"], errors="coerce").sum()
    cap_value = total * cap if total > 0 else 0.0
    capped = []
    for _, group in data.groupby("month", dropna=False):
        value = pd.to_numeric(group["net_return"], errors="coerce").sum()
        capped.append(min(value, cap_value) if value > 0 and cap_value > 0 else value)
    return float(np.sum(capped) / len(data)) if len(data) else np.nan


def _max_month_contribution(sample: pd.DataFrame, cost: float = 10.0) -> float:
    data = sample[pd.to_numeric(sample["cost_single_side_bps"], errors="coerce").eq(float(cost))]
    if data.empty:
        return np.nan
    by_month = data.groupby("month", dropna=False)["net_return"].sum()
    total = by_month.sum()
    return float((by_month / total).abs().max()) if total else np.nan


def _summary(trades: pd.DataFrame, signal_counts: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    count_lookup = signal_counts.set_index(["gate_name", "baseline"])["signals"].to_dict()
    rows = []
    for gate_name, group in trades.groupby("gate_name", sort=False):
        candidate = group[group["baseline"].eq("candidate_reclaim")]
        if candidate.empty:
            continue
        row = {
            "gate_name": gate_name,
            "negative_control": bool(candidate["negative_control"].iloc[0]),
            "signals": int(count_lookup.get((gate_name, "candidate_reclaim"), 0)),
            "trades": int(pd.to_numeric(candidate["cost_single_side_bps"], errors="coerce").eq(10.0).sum()),
            "net10": _net(candidate, 10),
            "net20": _net(candidate, 20),
            "net30": _net(candidate, 30),
            f"ex_{FOCAL_MONTH}_net10": _net(candidate, 10, FOCAL_MONTH),
            "month_cap35_net20": _month_cap(candidate, 20, 0.35),
            "max_month_contribution": _max_month_contribution(candidate, 10),
            "tp_rate": float(
                candidate[pd.to_numeric(candidate["cost_single_side_bps"], errors="coerce").eq(10.0)][
                    "exit_reason"
                ].astype(str).str.startswith("tp").mean()
            ),
            "sl_rate": float(
                candidate[pd.to_numeric(candidate["cost_single_side_bps"], errors="coerce").eq(10.0)][
                    "exit_reason"
                ].astype(str).str.startswith("sl").mean()
            ),
        }
        for baseline, out_col in [
            ("entry_only_reclaim", "entry_only_lift"),
            ("volume_shock_next_open", "volume_shock_only_lift"),
            ("volume_shock_pullback_only", "volume_shock_pullback_lift"),
            ("matched_random_reclaim", "matched_lift"),
        ]:
            base = group[group["baseline"].eq(baseline)]
            row[out_col] = row["net10"] - _net(base, 10) if not base.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_attribution(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    sample = trades[
        trades["baseline"].eq("candidate_reclaim")
        & pd.to_numeric(trades["cost_single_side_bps"], errors="coerce").eq(10.0)
    ].copy()
    numeric = [
        "volume_impulse_density",
        "alt_ret_4h_positive_ratio",
        "alt_ret_1h_positive_ratio",
        "alt_volume_expansion_ratio",
        "alt_down_breadth",
        "market_funding_extreme_ratio",
        "market_oi_explosion_ratio",
    ]
    rows = []
    for gate_name, group in sample.groupby("gate_name", sort=False):
        row = {"gate_name": gate_name, "trades": len(group)}
        for col in numeric:
            row[f"avg_{col}"] = float(pd.to_numeric(group[col], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _monthly(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    sample = trades[
        trades["baseline"].eq("candidate_reclaim")
        & pd.to_numeric(trades["cost_single_side_bps"], errors="coerce").isin([10.0, 20.0])
    ]
    rows = []
    for key, group in sample.groupby(["gate_name", "cost_single_side_bps", "month"], sort=False, dropna=False):
        gate_name, cost, month = key
        rows.append(
            {
                "gate_name": gate_name,
                "cost_single_side_bps": cost,
                "month": month,
                "trades": len(group),
                "net_expectancy": float(pd.to_numeric(group["net_return"], errors="coerce").mean()),
                "net_sum": float(pd.to_numeric(group["net_return"], errors="coerce").sum()),
            }
        )
    return pd.DataFrame(rows)


def _write_notes(report_root: Path, summary: pd.DataFrame) -> None:
    lines = ["# v0.6C Attack / Impulse Regime Detector", ""]
    lines.append("Fixed-gate attribution only. No IR2 parameter tuning and no new data source.")
    lines.append("IR2 primary paper-live should continue unchanged; any passing gate is shadow-only until future samples confirm it.")
    lines.append("")
    if summary.empty:
        lines.append("- No regime rows produced.")
    else:
        pass_mask = (
            ~summary["negative_control"].fillna(False)
            & (pd.to_numeric(summary["trades"], errors="coerce") >= 100)
            & (pd.to_numeric(summary["net10"], errors="coerce") >= 0.005)
            & (pd.to_numeric(summary["net20"], errors="coerce") >= 0.0025)
            & (pd.to_numeric(summary[f"ex_{FOCAL_MONTH}_net10"], errors="coerce") >= 0.0015)
            & (pd.to_numeric(summary["month_cap35_net20"], errors="coerce") >= 0.0015)
        )
        if pass_mask.any():
            passed = ", ".join(summary.loc[pass_mask, "gate_name"].astype(str).tolist())
            lines.append(f"- Promotion screen passed by: {passed}. Keep as shadow-only until future paper-live samples confirm.")
        else:
            lines.append(
                "- Promotion screen passed by no gate. Strong absolute gates remain too focal-month dependent; "
                "keep IR2 primary unchanged and treat regime gates as diagnostics only."
            )
        for row in summary.sort_values("net10", ascending=False).to_dict("records"):
            lines.append(
                f"- {row['gate_name']}: trades={row['trades']}, net10={row['net10']:.4%}, "
                f"net20={row['net20']:.4%}, ex-{FOCAL_MONTH}={row[f'ex_{FOCAL_MONTH}_net10']:.4%}, "
                f"cap35_net20={row['month_cap35_net20']:.4%}, matched_lift={row['matched_lift']:.4%}"
            )
    lines.append("")
    lines.append("Promotion rule: a gate must improve net10/net20, remain positive ex-2025-08 and under month-cap, and beat negative controls before becoming a paper-live shadow gate.")
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v06c_attack_regime_detector(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path = REPORT_ROOT,
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    rank30, rank90, symbols = _rank_inputs(feature_path, instruments, config)
    regime = _build_regime_streaming(feature_path, rank30, rank90, symbols, config)
    trade_path, signal_counts = _simulate_streaming(feature_path, rank30, rank90, symbols, regime, config, report_root)
    trades = pd.read_csv(trade_path, low_memory=False) if trade_path is not None and trade_path.exists() else pd.DataFrame()
    gate_summary = _summary(trades, signal_counts)
    feature_attr = _feature_attribution(trades)
    monthly = _monthly(trades)
    negative = gate_summary[gate_summary["negative_control"].fillna(False)].copy() if not gate_summary.empty else gate_summary
    ex_focal = gate_summary[["gate_name", f"ex_{FOCAL_MONTH}_net10", "month_cap35_net20"]].copy() if not gate_summary.empty else gate_summary
    outputs = {
        "regime_feature_attribution": report_root / "regime_feature_attribution.csv",
        "ir2_regime_gate_summary": report_root / "ir2_regime_gate_summary.csv",
        "ex_2025_08_regime_summary": report_root / "ex_2025_08_regime_summary.csv",
        "month_cap_regime_summary": report_root / "month_cap_regime_summary.csv",
        "negative_controls": report_root / "negative_controls.csv",
        "regime_monthly_distribution": report_root / "regime_monthly_distribution.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    feature_attr.to_csv(outputs["regime_feature_attribution"], index=False)
    gate_summary.to_csv(outputs["ir2_regime_gate_summary"], index=False)
    ex_focal.to_csv(outputs["ex_2025_08_regime_summary"], index=False)
    gate_summary[["gate_name", "month_cap35_net20", "max_month_contribution"]].to_csv(
        outputs["month_cap_regime_summary"], index=False
    )
    negative.to_csv(outputs["negative_controls"], index=False)
    monthly.to_csv(outputs["regime_monthly_distribution"], index=False)
    _write_notes(report_root, gate_summary)
    if trade_path is not None and trade_path.exists():
        trade_path.unlink()
    return outputs


def run_v06c_attack_regime_detector_from_features(config: ExperimentConfig) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = read_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v06c_attack_regime_detector(features_path, instruments, config)
