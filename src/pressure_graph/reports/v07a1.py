from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.backtest import EntryPolicy, simulate_entry_policy_trades
from pressure_graph.backtest.minute_execution import simulate_1m_execution
from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir, raw_path, read_parquet
from pressure_graph.reports.v06a1 import _read_symbol_features
from pressure_graph.reports.v06c import _build_regime_streaming, _rank_inputs
from pressure_graph.reports.v07a import (
    COST_BPS,
    FOCAL_MONTH,
    NEXT_OPEN,
    RECLAIM_1,
    SIM_WINDOW_BARS,
    _add_motif_columns,
    _rule,
    _signal_window,
)


REPORT_ROOT = Path("reports/v0_7a1_mir1_validation")
PULLBACK_1 = EntryPolicy(
    "pullback_1.0pct_only_valid_8_bars",
    "pullback",
    valid_bars=8,
    pullback_pct=0.010,
)


@dataclass(frozen=True)
class FrozenAtlasCandidate:
    candidate: str
    universe_top_n: int
    market_gate: str
    event_col: str
    entry_policy: EntryPolicy
    exit_rule: str
    role: str
    negative_control: bool = False


@dataclass(frozen=True)
class BaselineSpec:
    baseline: str
    event_col: str
    entry_policy: EntryPolicy


FROZEN_CANDIDATES = [
    FrozenAtlasCandidate(
        "MIR1",
        50,
        "market_volume_impulse_density_high",
        "bullish_volume_shock_event",
        RECLAIM_1,
        "vol_regime_fast",
        "primary_validation",
    ),
    FrozenAtlasCandidate(
        "IR2_ref",
        50,
        "market_btc_up",
        "bullish_volume_shock_event",
        RECLAIM_1,
        "vol_regime_fast",
        "reference_current_primary",
    ),
    FrozenAtlasCandidate(
        "CHOP_tail_shadow",
        50,
        "market_btc_chop",
        "bullish_volume_shock_event",
        RECLAIM_1,
        "vol_regime_fast",
        "right_tail_shadow",
        True,
    ),
    FrozenAtlasCandidate(
        "LOW_VOLUME_DENSITY_NEG",
        50,
        "market_low_volume_impulse_density",
        "bullish_volume_shock_event",
        RECLAIM_1,
        "vol_regime_fast",
        "negative_control",
        True,
    ),
]

BASELINES = [
    BaselineSpec("candidate_reclaim", "bullish_volume_shock_event", RECLAIM_1),
    BaselineSpec("entry_only_reclaim", "neutral_volume_event", RECLAIM_1),
    BaselineSpec("volume_shock_next_open", "bullish_volume_shock_event", NEXT_OPEN),
    BaselineSpec("volume_shock_pullback_only", "bullish_volume_shock_event", PULLBACK_1),
    BaselineSpec("matched_random_reclaim", "matched_random_event", RECLAIM_1),
]
GATE_MODES = ["signal_gate_only", "signal_and_entry_gate"]
ONE_MIN_CANDIDATES = {"MIR1", "IR2_ref"}
ONE_MIN_GATE_MODES = {"signal_and_entry_gate"}
ONE_MIN_COST_BPS = [10, 20]
RECONCILIATION_COST_BPS = 10.0


def _add_signal_col(data: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    out = _signal_window(data, mask, SIM_WINDOW_BARS)
    out["__v07a1_signal"] = mask.loc[out.index].fillna(False).astype(bool)
    return out


def _context_cols(data: pd.DataFrame, gate_col: str) -> list[str]:
    cols = [
        "exchange",
        "symbol",
        "feature_time",
        "month",
        "dynamic_all_rank",
        "dynamic_all_trailing_turnover",
        "turnover_rank_90d",
        "liquidity_bucket",
        "btc_market_state",
        "symbol_volatility_percentile",
        "volume_z_4h",
        "ret_4h",
        "mfe_12h",
        "mae_12h",
        "mfe_24h",
        "mae_24h",
        "hit_10pct_12h",
        "hit_20pct_24h",
        "volume_impulse_density",
        "alt_ret_4h_positive_ratio",
        "alt_ret_1h_positive_ratio",
        "alt_volume_expansion_ratio",
        "market_btc_up",
        "market_btc_chop",
        "market_volume_impulse_density_high",
        "market_low_volume_impulse_density",
        gate_col,
    ]
    return list(dict.fromkeys([col for col in cols if col in data.columns]))


def _entry_context_asof(
    trades: pd.DataFrame,
    context: pd.DataFrame,
    gate_col: str,
) -> pd.DataFrame:
    if trades.empty:
        out = trades.copy()
        out["btc_state_at_entry"] = pd.Series(dtype=object)
        out["market_gate_at_entry"] = pd.Series(dtype=bool)
        return out
    if context.empty:
        out = trades.copy()
        out["btc_state_at_entry"] = "unknown"
        out["market_gate_at_entry"] = False
        return out
    left = trades.copy()
    left["entry_time"] = pd.to_datetime(left["entry_time"], utc=True, errors="coerce")
    ctx_cols = ["exchange", "symbol", "feature_time", "btc_market_state", gate_col]
    ctx = context[[col for col in ctx_cols if col in context.columns]].copy()
    ctx["feature_time"] = pd.to_datetime(ctx["feature_time"], utc=True, errors="coerce")
    ctx = ctx.dropna(subset=["feature_time"]).sort_values(["exchange", "symbol", "feature_time"])
    frames = []
    for key, group in left.groupby(["exchange", "symbol"], sort=False, dropna=False):
        exchange, symbol = key
        local = ctx[
            ctx["exchange"].astype(str).eq(str(exchange))
            & ctx["symbol"].astype(str).eq(str(symbol))
        ].sort_values("feature_time")
        data = group.sort_values("entry_time")
        if local.empty:
            data["btc_state_at_entry"] = "unknown"
            data["market_gate_at_entry"] = False
            frames.append(data)
            continue
        merged = pd.merge_asof(
            data,
            local[["feature_time", "btc_market_state", gate_col]],
            left_on="entry_time",
            right_on="feature_time",
            direction="backward",
        ).drop(columns=["feature_time"], errors="ignore")
        merged = merged.rename(columns={"btc_market_state": "btc_state_at_entry", gate_col: "market_gate_at_entry"})
        frames.append(merged)
    return pd.concat(frames, ignore_index=True) if frames else left


def _attach_context(
    trades: pd.DataFrame,
    data: pd.DataFrame,
    candidate: FrozenAtlasCandidate,
) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    context = data[_context_cols(data, candidate.market_gate)].drop_duplicates(
        ["exchange", "symbol", "feature_time"]
    )
    out = trades.merge(
        context,
        left_on=["exchange", "symbol", "signal_time"],
        right_on=["exchange", "symbol", "feature_time"],
        how="left",
    ).drop(columns=["feature_time"], errors="ignore")
    out = out.rename(
        columns={
            "btc_market_state": "btc_state_at_signal",
            candidate.market_gate: "market_gate_at_signal",
        }
    )
    return _entry_context_asof(out, context, candidate.market_gate)


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


def _simulate_variant(
    data: pd.DataFrame,
    candidate: FrozenAtlasCandidate,
    gate_mode: str,
    baseline: BaselineSpec,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, int, int]:
    rank = pd.to_numeric(data["dynamic_all_rank"], errors="coerce")
    mask = (
        (rank <= candidate.universe_top_n)
        & data[candidate.market_gate].fillna(False).astype(bool)
        & data[baseline.event_col].fillna(False).astype(bool)
    )
    signal_n = int(mask.sum())
    if signal_n <= 0:
        return pd.DataFrame(), 0, 0
    sim = _add_signal_col(data, mask)
    rule, resolver = _rule(candidate.exit_rule, config)
    trades = simulate_entry_policy_trades(
        sim,
        "__v07a1_signal",
        candidate.candidate,
        baseline.entry_policy,
        rule,
        0,
        "sl_first",
        True,
        "__v07a1_signal",
        resolver,
    )
    if trades.empty:
        return trades, signal_n, 0
    trades = _attach_context(trades, data, candidate)
    pre_entry_gate_n = len(trades)
    if gate_mode == "signal_and_entry_gate":
        trades = trades[trades["market_gate_at_entry"].fillna(False).astype(bool)].copy()
    if trades.empty:
        return trades, signal_n, pre_entry_gate_n
    trades["candidate"] = candidate.candidate
    trades["candidate_role"] = candidate.role
    trades["gate_mode"] = gate_mode
    trades["market_gate"] = candidate.market_gate
    trades["baseline"] = baseline.baseline
    trades["entry_policy"] = baseline.entry_policy.name
    trades["execution_rule"] = candidate.exit_rule
    trades["negative_control"] = candidate.negative_control
    trades["entry_gate_invalidated"] = int(pre_entry_gate_n - len(trades))
    return _expand_costs(trades), signal_n, pre_entry_gate_n


def _simulate_streaming(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    symbols: list[str],
    regime: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trade_path = report_root / "_v07a1_trades_tmp.csv"
    if trade_path.exists():
        trade_path.unlink()
    wrote_header = False
    signal_rows: list[dict[str, object]] = []
    for idx, symbol in enumerate(symbols, start=1):
        data = _read_symbol_features(feature_path, rank30, rank90, symbol, config)
        if data.empty:
            continue
        data = data[pd.to_numeric(data["dynamic_all_rank"], errors="coerce") <= 50].copy()
        if data.empty:
            continue
        data = _add_motif_columns(data, regime, config)
        for candidate in FROZEN_CANDIDATES:
            for gate_mode in GATE_MODES:
                for baseline in BASELINES:
                    trades, signal_n, pre_entry_gate_n = _simulate_variant(
                        data,
                        candidate,
                        gate_mode,
                        baseline,
                        config,
                    )
                    signal_rows.append(
                        {
                            "candidate": candidate.candidate,
                            "gate_mode": gate_mode,
                            "baseline": baseline.baseline,
                            "signals": signal_n,
                            "pre_entry_gate_trades": pre_entry_gate_n,
                        }
                    )
                    if not trades.empty:
                        trades.to_csv(trade_path, mode="a", header=not wrote_header, index=False)
                        wrote_header = True
                        del trades
        print(f"v0.7A.1 MIR1 validation pass {idx}/{len(symbols)} {symbol}", flush=True)
        del data
        gc.collect()
    trades = pd.read_csv(trade_path, low_memory=False) if wrote_header else pd.DataFrame()
    if trade_path.exists():
        trade_path.unlink()
    signals = pd.DataFrame(signal_rows)
    if not signals.empty:
        signals = signals.groupby(
            ["candidate", "gate_mode", "baseline"],
            as_index=False,
            sort=False,
            dropna=False,
        ).agg(signals=("signals", "sum"), pre_entry_gate_trades=("pre_entry_gate_trades", "sum"))
    return trades, signals


def _metric(sample: pd.DataFrame, cost: float, metric: str = "net_return") -> float:
    data = sample[pd.to_numeric(sample["cost_single_side_bps"], errors="coerce").eq(float(cost))]
    if data.empty:
        return np.nan
    values = pd.to_numeric(data[metric], errors="coerce")
    return float(values.mean())


def _net_sum(sample: pd.DataFrame, cost: float) -> float:
    data = sample[pd.to_numeric(sample["cost_single_side_bps"], errors="coerce").eq(float(cost))]
    if data.empty:
        return 0.0
    return float(pd.to_numeric(data["net_return"], errors="coerce").sum())


def _month_cap_expectancy(sample: pd.DataFrame, cost: float, cap: float = 0.35) -> float:
    data = sample[pd.to_numeric(sample["cost_single_side_bps"], errors="coerce").eq(float(cost))]
    if data.empty:
        return np.nan
    total = pd.to_numeric(data["net_return"], errors="coerce").sum()
    cap_value = total * cap if total > 0 else 0.0
    capped = []
    for _, group in data.groupby("month", sort=False, dropna=False):
        value = pd.to_numeric(group["net_return"], errors="coerce").sum()
        capped.append(min(value, cap_value) if value > 0 and cap_value > 0 else value)
    return float(np.sum(capped) / len(data)) if len(data) else np.nan


def _max_contribution(sample: pd.DataFrame, group_col: str, cost: float = 10.0) -> float:
    data = sample[pd.to_numeric(sample["cost_single_side_bps"], errors="coerce").eq(float(cost))]
    if data.empty or group_col not in data.columns:
        return np.nan
    grouped = data.groupby(group_col, sort=False, dropna=False)["net_return"].sum()
    total = grouped.sum()
    return float((grouped / total).abs().max()) if total else np.nan


def _ex_month_net(sample: pd.DataFrame, month: str, cost: float = 10.0) -> float:
    data = sample[
        pd.to_numeric(sample["cost_single_side_bps"], errors="coerce").eq(float(cost))
        & ~sample["month"].astype(str).eq(month)
    ]
    if data.empty:
        return np.nan
    return float(pd.to_numeric(data["net_return"], errors="coerce").mean())


def _ex_top_month_net(sample: pd.DataFrame, cost: float = 10.0) -> float:
    data = sample[pd.to_numeric(sample["cost_single_side_bps"], errors="coerce").eq(float(cost))]
    if data.empty:
        return np.nan
    top_month = data.groupby("month", sort=False, dropna=False)["net_return"].sum().abs().idxmax()
    return _ex_month_net(sample, str(top_month), cost)


def _summary_long(trades: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    lookup = signals.set_index(["candidate", "gate_mode", "baseline"]).to_dict("index") if not signals.empty else {}
    rows = []
    group_cols = ["candidate", "gate_mode", "market_gate", "baseline", "cost_single_side_bps"]
    for key, group in trades.groupby(group_cols, sort=False, dropna=False):
        candidate, gate_mode, market_gate, baseline, cost = key
        signal_info = lookup.get((candidate, gate_mode, baseline), {})
        net = pd.to_numeric(group["net_return"], errors="coerce")
        gross = pd.to_numeric(group["gross_return"], errors="coerce")
        exits = group["exit_reason"].astype(str)
        rows.append(
            {
                "candidate": candidate,
                "gate_mode": gate_mode,
                "market_gate": market_gate,
                "baseline": baseline,
                "cost_single_side_bps": cost,
                "signals": int(signal_info.get("signals", 0)),
                "pre_entry_gate_trades": int(signal_info.get("pre_entry_gate_trades", 0)),
                "trades": int(len(group)),
                "fill_rate": float(len(group) / signal_info.get("signals", 0))
                if signal_info.get("signals", 0)
                else np.nan,
                "gross_expectancy": float(gross.mean()) if len(group) else np.nan,
                "net_expectancy": float(net.mean()) if len(group) else np.nan,
                "net_sum": float(net.sum()) if len(group) else 0.0,
                "tp_rate": float(exits.str.startswith("tp").mean()) if len(group) else np.nan,
                "sl_rate": float(exits.str.startswith("sl").mean()) if len(group) else np.nan,
                "timeout_rate": float(exits.eq("max_hold").mean()) if len(group) else np.nan,
                "hit_10pct_12h": float(group.get("hit_10pct_12h", pd.Series(dtype=bool)).fillna(False).mean()),
                "hit_20pct_24h": float(group.get("hit_20pct_24h", pd.Series(dtype=bool)).fillna(False).mean()),
                "median_mfe_12h": float(pd.to_numeric(group.get("mfe_12h"), errors="coerce").median()),
                "p95_mfe_24h": float(pd.to_numeric(group.get("mfe_24h"), errors="coerce").quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def _lookup(rows: pd.DataFrame, candidate: str, gate_mode: str, baseline: str, cost: float, col: str) -> float:
    sample = rows[
        rows["candidate"].astype(str).eq(candidate)
        & rows["gate_mode"].astype(str).eq(gate_mode)
        & rows["baseline"].astype(str).eq(baseline)
        & pd.to_numeric(rows["cost_single_side_bps"], errors="coerce").eq(float(cost))
    ]
    return float(sample[col].iloc[0]) if not sample.empty else np.nan


def _lookup_exact_net(exact: pd.DataFrame, candidate: str, gate_mode: str, cost: float) -> float:
    if exact.empty:
        return np.nan
    col = f"net{int(cost)}"
    if col not in exact.columns:
        return np.nan
    sample = exact[
        exact["candidate"].astype(str).eq(candidate)
        & exact["gate_mode"].astype(str).eq(gate_mode)
    ]
    return float(sample[col].iloc[0]) if not sample.empty else np.nan


def _exact_gate_replay(trades: pd.DataFrame, long_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate in [item.candidate for item in FROZEN_CANDIDATES]:
        for gate_mode in GATE_MODES:
            cand = trades[
                trades["candidate"].astype(str).eq(candidate)
                & trades["gate_mode"].astype(str).eq(gate_mode)
                & trades["baseline"].astype(str).eq("candidate_reclaim")
            ]
            cand10 = cand[pd.to_numeric(cand["cost_single_side_bps"], errors="coerce").eq(10.0)]
            if cand10.empty:
                continue
            net10 = _metric(cand, 10)
            row = {
                "candidate": candidate,
                "gate_mode": gate_mode,
                "market_gate": str(cand10["market_gate"].iloc[0]),
                "role": str(cand10["candidate_role"].iloc[0]),
                "negative_control": bool(cand10["negative_control"].iloc[0]),
                "signals": int(
                    _lookup(long_summary, candidate, gate_mode, "candidate_reclaim", 10, "signals")
                ),
                "trades": int(len(cand10)),
                "net10": net10,
                "net20": _metric(cand, 20),
                "net30": _metric(cand, 30),
                "net50": _metric(cand, 50),
                f"ex_{FOCAL_MONTH}_net10": _ex_month_net(cand, FOCAL_MONTH, 10),
                "ex_top_month_net10": _ex_top_month_net(cand, 10),
                "month_cap35_net20": _month_cap_expectancy(cand, 20, 0.35),
                "max_month_contribution": _max_contribution(cand, "month", 10),
                "max_symbol_contribution": _max_contribution(cand, "symbol", 10),
                "hit10_12h": float(cand10.get("hit_10pct_12h", pd.Series(dtype=bool)).fillna(False).mean()),
                "hit20_24h": float(cand10.get("hit_20pct_24h", pd.Series(dtype=bool)).fillna(False).mean()),
                "median_mfe_12h": float(pd.to_numeric(cand10.get("mfe_12h"), errors="coerce").median()),
                "p95_mfe_24h": float(pd.to_numeric(cand10.get("mfe_24h"), errors="coerce").quantile(0.95)),
            }
            for baseline, col in [
                ("entry_only_reclaim", "entry_only_lift"),
                ("volume_shock_next_open", "volume_shock_next_open_lift"),
                ("volume_shock_pullback_only", "volume_shock_pullback_lift"),
                ("matched_random_reclaim", "matched_random_lift"),
            ]:
                row[col] = net10 - _lookup(long_summary, candidate, gate_mode, baseline, 10, "net_expectancy")
            rows.append(row)
    return pd.DataFrame(rows)


def _month_concentration(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["candidate", "gate_mode", "baseline", "cost_single_side_bps"]
    for key, group in trades.groupby(group_cols, sort=False, dropna=False):
        candidate, gate_mode, baseline, cost = key
        total = pd.to_numeric(group["net_return"], errors="coerce").sum()
        for month, month_group in group.groupby("month", sort=False, dropna=False):
            value = pd.to_numeric(month_group["net_return"], errors="coerce").sum()
            rows.append(
                {
                    "candidate": candidate,
                    "gate_mode": gate_mode,
                    "baseline": baseline,
                    "cost_single_side_bps": cost,
                    "month": month,
                    "trades": int(len(month_group)),
                    "net_expectancy": float(pd.to_numeric(month_group["net_return"], errors="coerce").mean()),
                    "net_sum": float(value),
                    "contribution_pct": float(value / total) if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _leave_one_month_out(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["candidate", "gate_mode", "baseline", "cost_single_side_bps"]
    for key, group in trades.groupby(group_cols, sort=False, dropna=False):
        candidate, gate_mode, baseline, cost = key
        for month in sorted(group["month"].dropna().astype(str).unique()):
            sample = group[~group["month"].astype(str).eq(month)]
            rows.append(
                {
                    "candidate": candidate,
                    "gate_mode": gate_mode,
                    "baseline": baseline,
                    "cost_single_side_bps": cost,
                    "excluded_month": month,
                    "remaining_trades": int(len(sample)),
                    "remaining_net_expectancy": float(pd.to_numeric(sample["net_return"], errors="coerce").mean())
                    if len(sample)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _symbol_contribution(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["candidate", "gate_mode", "baseline", "cost_single_side_bps"]
    for key, group in trades.groupby(group_cols, sort=False, dropna=False):
        candidate, gate_mode, baseline, cost = key
        total = pd.to_numeric(group["net_return"], errors="coerce").sum()
        for symbol, symbol_group in group.groupby("symbol", sort=False, dropna=False):
            value = pd.to_numeric(symbol_group["net_return"], errors="coerce").sum()
            rows.append(
                {
                    "candidate": candidate,
                    "gate_mode": gate_mode,
                    "baseline": baseline,
                    "cost_single_side_bps": cost,
                    "symbol": symbol,
                    "trades": int(len(symbol_group)),
                    "net_expectancy": float(pd.to_numeric(symbol_group["net_return"], errors="coerce").mean()),
                    "net_sum": float(value),
                    "contribution_pct": float(value / total) if total else np.nan,
                    "avg_turnover_rank": float(pd.to_numeric(symbol_group["dynamic_all_rank"], errors="coerce").mean()),
                }
            )
    return pd.DataFrame(rows)


def _load_1m_cache(config: ExperimentConfig, symbols: list[str]) -> pd.DataFrame:
    roots = [
        config.paths.data_root / "raw" / "bybit" / "klines_1m_v06a2",
        config.paths.data_root / "raw" / "bybit" / "klines_1m_v06a2_public",
    ]
    frames = []
    for root in roots:
        if not root.exists():
            continue
        for symbol in symbols:
            path = root / f"{symbol}.parquet"
            if path.exists():
                frames.append(read_parquet(path))
        if frames:
            break
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(["exchange", "symbol", "bar_open_time"])


def _build_signal_rows(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    symbols: list[str],
    regime: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_frames = []
    context_frames = []
    for symbol in symbols:
        data = _read_symbol_features(feature_path, rank30, rank90, symbol, config)
        if data.empty:
            continue
        data = data[pd.to_numeric(data["dynamic_all_rank"], errors="coerce") <= 50].copy()
        if data.empty:
            continue
        data = _add_motif_columns(data, regime, config)
        context_frames.append(
            data[
                [
                    col
                    for col in [
                        "exchange",
                        "symbol",
                        "feature_time",
                        "btc_market_state",
                        "market_volume_impulse_density_high",
                        "market_btc_up",
                        "market_btc_chop",
                        "market_low_volume_impulse_density",
                    ]
                    if col in data.columns
                ]
            ].drop_duplicates(["exchange", "symbol", "feature_time"])
        )
        for candidate in FROZEN_CANDIDATES:
            if candidate.candidate not in ONE_MIN_CANDIDATES:
                continue
            for gate_mode in GATE_MODES:
                if gate_mode not in ONE_MIN_GATE_MODES:
                    continue
                mask = (
                    (pd.to_numeric(data["dynamic_all_rank"], errors="coerce") <= candidate.universe_top_n)
                    & data[candidate.market_gate].fillna(False).astype(bool)
                    & data[candidate.event_col].fillna(False).astype(bool)
                )
                if not mask.any():
                    continue
                rows = data.loc[mask].copy()
                rows["candidate"] = candidate.candidate
                rows["gate_mode"] = gate_mode
                rows["market_gate"] = candidate.market_gate
                rows["__v07a1_1m_signal"] = True
                signal_frames.append(rows)
    signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    context = pd.concat(context_frames, ignore_index=True) if context_frames else pd.DataFrame()
    return signals, context


def _normalize_1m(trades: pd.DataFrame, candidate: FrozenAtlasCandidate, gate_mode: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    out = trades.copy()
    out["candidate"] = candidate.candidate
    out["gate_mode"] = gate_mode
    out["market_gate"] = candidate.market_gate
    out["baseline"] = "candidate_reclaim"
    out["execution_granularity"] = "1m_bar"
    out["net_return"] = pd.to_numeric(out["net_expectancy"], errors="coerce")
    out["month"] = pd.to_datetime(out["signal_time"], utc=True, errors="coerce").dt.strftime("%Y-%m")
    return out


def _signal_id_from_trade(row: pd.Series, candidate: str) -> str:
    signal_time = pd.Timestamp(row["signal_time"])
    if signal_time.tzinfo is None:
        signal_time = signal_time.tz_localize("UTC")
    else:
        signal_time = signal_time.tz_convert("UTC")
    stamp = signal_time.strftime("%Y%m%dT%H%M%SZ")
    return f"{candidate}:candidate_reclaim:{row['exchange']}:{row['symbol']}:{stamp}"


def _normalize_15m_for_reconciliation(trades: pd.DataFrame, candidate: FrozenAtlasCandidate) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "candidate",
                "signal_id",
                "symbol",
                "signal_time",
                "in_15m",
                "entry_time_15m",
                "exit_time_15m",
                "exit_reason_15m",
                "net10_15m",
            ]
        )
    sample = trades[
        pd.to_numeric(trades["cost_single_side_bps"], errors="coerce").eq(RECONCILIATION_COST_BPS)
    ].copy()
    if sample.empty:
        return pd.DataFrame()
    sample["signal_id"] = sample.apply(lambda row: _signal_id_from_trade(row, candidate.candidate), axis=1)
    return pd.DataFrame(
        {
            "candidate": candidate.candidate,
            "signal_id": sample["signal_id"].astype(str),
            "symbol": sample["symbol"].astype(str),
            "signal_time": pd.to_datetime(sample["signal_time"], utc=True, errors="coerce"),
            "in_15m": True,
            "entry_time_15m": pd.to_datetime(sample["entry_time"], utc=True, errors="coerce"),
            "exit_time_15m": pd.to_datetime(sample["exit_time"], utc=True, errors="coerce"),
            "exit_reason_15m": sample["exit_reason"].astype(str),
            "net10_15m": pd.to_numeric(sample["net_return"], errors="coerce"),
        }
    )


def _normalize_1m_for_reconciliation(trades: pd.DataFrame, candidate: FrozenAtlasCandidate) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "candidate",
                "signal_id",
                "symbol",
                "signal_time",
                "in_1m",
                "entry_time_1m",
                "exit_time_1m",
                "exit_reason_1m",
                "net10_1m",
            ]
        )
    sample = trades[
        pd.to_numeric(trades["cost_single_side_bps"], errors="coerce").eq(RECONCILIATION_COST_BPS)
    ].copy()
    if sample.empty:
        return pd.DataFrame()
    sample["signal_id"] = sample.apply(lambda row: _signal_id_from_trade(row, candidate.candidate), axis=1)
    return pd.DataFrame(
        {
            "candidate": candidate.candidate,
            "signal_id": sample["signal_id"].astype(str),
            "symbol": sample["symbol"].astype(str),
            "signal_time": pd.to_datetime(sample["signal_time"], utc=True, errors="coerce"),
            "in_1m": True,
            "entry_time_1m": pd.to_datetime(sample["entry_time"], utc=True, errors="coerce"),
            "exit_time_1m": pd.to_datetime(sample["exit_time"], utc=True, errors="coerce"),
            "exit_reason_1m": sample["exit_reason"].astype(str),
            "net10_1m": pd.to_numeric(sample["net_return"], errors="coerce"),
        }
    )


def _one_min_comparison(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    symbols: list[str],
    regime: pd.DataFrame,
    config: ExperimentConfig,
    exact: pd.DataFrame,
) -> pd.DataFrame:
    signal_rows, context = _build_signal_rows(feature_path, rank30, rank90, symbols, regime, config)
    if signal_rows.empty:
        return pd.DataFrame([{"status": "no_signals"}])
    minute_bars = _load_1m_cache(config, sorted(signal_rows["symbol"].dropna().astype(str).unique()))
    if minute_bars.empty:
        return pd.DataFrame([{"status": "pending_1m_data"}])
    frames = []
    for candidate in FROZEN_CANDIDATES:
        if candidate.candidate not in ONE_MIN_CANDIDATES:
            continue
        rule, resolver = _rule(candidate.exit_rule, config)
        for gate_mode in GATE_MODES:
            if gate_mode not in ONE_MIN_GATE_MODES:
                continue
            rows = signal_rows[
                signal_rows["candidate"].astype(str).eq(candidate.candidate)
                & signal_rows["gate_mode"].astype(str).eq(gate_mode)
            ].copy()
            if rows.empty:
                continue
            for cost in ONE_MIN_COST_BPS:
                trades = simulate_1m_execution(
                    rows,
                    minute_bars,
                    "__v07a1_1m_signal",
                    candidate.candidate,
                    candidate.entry_policy,
                    candidate.exit_rule,
                    rule,
                    float(cost),
                    resolver,
                )
                trades = _normalize_1m(trades, candidate, gate_mode)
                if trades.empty:
                    continue
                trades = _entry_context_asof(trades, context, candidate.market_gate)
                if gate_mode == "signal_and_entry_gate":
                    trades = trades[trades["market_gate_at_entry"].fillna(False).astype(bool)].copy()
                if not trades.empty:
                    frames.append(trades)
    one_min = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if one_min.empty:
        return pd.DataFrame([{"status": "no_1m_fills"}])
    rows = []
    group_cols = ["candidate", "gate_mode", "market_gate", "cost_single_side_bps"]
    for key, group in one_min.groupby(group_cols, sort=False, dropna=False):
        candidate, gate_mode, market_gate, cost = key
        net = pd.to_numeric(group["net_return"], errors="coerce")
        exits = group["exit_reason"].astype(str)
        exact_net = _lookup_exact_net(exact, str(candidate), str(gate_mode), float(cost))
        rows.append(
            {
                "status": "ok",
                "candidate": candidate,
                "gate_mode": gate_mode,
                "market_gate": market_gate,
                "cost_single_side_bps": cost,
                "trades": int(len(group)),
                "net_expectancy_1m": float(net.mean()),
                "gross_expectancy_1m": float(pd.to_numeric(group["gross_return"], errors="coerce").mean()),
                "tp_rate_1m": float(exits.str.startswith("tp").mean()),
                "sl_rate_1m": float(exits.str.startswith("sl").mean()),
                "timeout_rate_1m": float(exits.eq("max_hold").mean()),
                "same_bar_ambiguity_1m": float(group["unresolved_1m_same_bar"].fillna(False).mean()),
                "median_minutes_to_entry": float(
                    pd.to_numeric(group["bars_from_signal_to_entry_1m"], errors="coerce").median()
                ),
                "median_holding_minutes": float(pd.to_numeric(group["holding_minutes"], errors="coerce").median()),
                "expectancy_retention_vs_15m": float(net.mean() / exact_net) if exact_net else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _one_min_trade_rows(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    symbols: list[str],
    regime: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    signal_rows, context = _build_signal_rows(feature_path, rank30, rank90, symbols, regime, config)
    if signal_rows.empty:
        return pd.DataFrame()
    minute_bars = _load_1m_cache(config, sorted(signal_rows["symbol"].dropna().astype(str).unique()))
    if minute_bars.empty:
        return pd.DataFrame()
    frames = []
    for candidate in FROZEN_CANDIDATES:
        if candidate.candidate not in ONE_MIN_CANDIDATES:
            continue
        rule, resolver = _rule(candidate.exit_rule, config)
        rows = signal_rows[
            signal_rows["candidate"].astype(str).eq(candidate.candidate)
            & signal_rows["gate_mode"].astype(str).eq("signal_and_entry_gate")
        ].copy()
        if rows.empty:
            continue
        trades = simulate_1m_execution(
            rows,
            minute_bars,
            "__v07a1_1m_signal",
            candidate.candidate,
            candidate.entry_policy,
            candidate.exit_rule,
            rule,
            RECONCILIATION_COST_BPS,
            resolver,
        )
        trades = _normalize_1m(trades, candidate, "signal_and_entry_gate")
        if trades.empty:
            continue
        trades = _entry_context_asof(trades, context, candidate.market_gate)
        trades = trades[trades["market_gate_at_entry"].fillna(False).astype(bool)].copy()
        if not trades.empty:
            frames.append(trades)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _included_vs_excluded(reconciliation: pd.DataFrame) -> pd.DataFrame:
    if reconciliation.empty:
        return pd.DataFrame()
    data = reconciliation.copy()
    data["bucket"] = np.select(
        [
            data["in_15m"].fillna(False) & data["in_1m"].fillna(False),
            data["in_15m"].fillna(False) & ~data["in_1m"].fillna(False),
            ~data["in_15m"].fillna(False) & data["in_1m"].fillna(False),
        ],
        ["in_both", "15m_only", "1m_only"],
        default="in_neither",
    )
    rows = []
    for (candidate, bucket), group in data.groupby(["candidate", "bucket"], sort=False, dropna=False):
        rows.append(
            {
                "candidate": candidate,
                "bucket": bucket,
                "signals": int(len(group)),
                "net10_15m_avg": float(pd.to_numeric(group["net10_15m"], errors="coerce").mean()),
                "net10_1m_avg": float(pd.to_numeric(group["net10_1m"], errors="coerce").mean()),
                "delta_net10_avg": float(pd.to_numeric(group["delta_net10"], errors="coerce").mean()),
                "tp_rate_15m": float(group["exit_reason_15m"].astype(str).str.startswith("tp").mean()),
                "tp_rate_1m": float(group["exit_reason_1m"].astype(str).str.startswith("tp").mean()),
                "sl_rate_15m": float(group["exit_reason_15m"].astype(str).str.startswith("sl").mean()),
                "sl_rate_1m": float(group["exit_reason_1m"].astype(str).str.startswith("sl").mean()),
            }
        )
    return pd.DataFrame(rows)


def write_v07a1_1m_reconciliation(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path = REPORT_ROOT,
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    rank30, rank90, symbols = _rank_inputs(feature_path, instruments, config)
    regime = _build_regime_streaming(feature_path, rank30, rank90, symbols, config)
    trade_frames = []
    for symbol in symbols:
        data = _read_symbol_features(feature_path, rank30, rank90, symbol, config)
        if data.empty:
            continue
        data = data[pd.to_numeric(data["dynamic_all_rank"], errors="coerce") <= 50].copy()
        if data.empty:
            continue
        data = _add_motif_columns(data, regime, config)
        for candidate in FROZEN_CANDIDATES:
            if candidate.candidate not in ONE_MIN_CANDIDATES:
                continue
            baseline = BASELINES[0]
            trades, _, _ = _simulate_variant(
                data,
                candidate,
                "signal_and_entry_gate",
                baseline,
                config,
            )
            normalized = _normalize_15m_for_reconciliation(trades, candidate)
            if not normalized.empty:
                trade_frames.append(normalized)
        print(f"v0.7A.1 reconciliation 15m pass {symbol}", flush=True)
        del data
        gc.collect()
    fifteen = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    one_frames = []
    one_raw = _one_min_trade_rows(feature_path, rank30, rank90, symbols, regime, config)
    if not one_raw.empty:
        for candidate in FROZEN_CANDIDATES:
            if candidate.candidate not in ONE_MIN_CANDIDATES:
                continue
            sample = one_raw[one_raw["candidate"].astype(str).eq(candidate.candidate)]
            normalized = _normalize_1m_for_reconciliation(sample, candidate)
            if not normalized.empty:
                one_frames.append(normalized)
    one_min = pd.concat(one_frames, ignore_index=True) if one_frames else pd.DataFrame()
    if fifteen.empty and one_min.empty:
        reconciliation = pd.DataFrame(
            [
                {
                    "status": "pending_1m_data_or_no_replay_trades",
                    "candidate": "",
                    "signal_id": "",
                }
            ]
        )
    else:
        reconciliation = fifteen.merge(
            one_min.drop(columns=["symbol", "signal_time"], errors="ignore"),
            on=["candidate", "signal_id"],
            how="outer",
        )
        reconciliation["in_15m"] = reconciliation["in_15m"].fillna(False).astype(bool)
        reconciliation["in_1m"] = reconciliation["in_1m"].fillna(False).astype(bool)
        reconciliation["delta_net10"] = (
            pd.to_numeric(reconciliation.get("net10_1m"), errors="coerce")
            - pd.to_numeric(reconciliation.get("net10_15m"), errors="coerce")
        )
        if "symbol" not in reconciliation.columns and "symbol_x" in reconciliation.columns:
            reconciliation["symbol"] = reconciliation["symbol_x"]
        if "signal_time" not in reconciliation.columns and "signal_time_x" in reconciliation.columns:
            reconciliation["signal_time"] = reconciliation["signal_time_x"]
        preferred = [
            "candidate",
            "signal_id",
            "symbol",
            "signal_time",
            "in_15m",
            "in_1m",
            "entry_time_15m",
            "entry_time_1m",
            "exit_time_15m",
            "exit_time_1m",
            "exit_reason_15m",
            "exit_reason_1m",
            "net10_15m",
            "net10_1m",
            "delta_net10",
        ]
        reconciliation = reconciliation[[col for col in preferred if col in reconciliation.columns]]
    included = _included_vs_excluded(reconciliation) if "in_15m" in reconciliation.columns else pd.DataFrame()
    outputs = {
        "one_min_reconciliation": report_root / "1m_reconciliation.csv",
        "one_min_included_vs_excluded": report_root / "1m_included_vs_excluded.csv",
    }
    reconciliation.to_csv(outputs["one_min_reconciliation"], index=False)
    included.to_csv(outputs["one_min_included_vs_excluded"], index=False)
    return outputs


def _write_notes(report_root: Path, exact: pd.DataFrame, one_min: pd.DataFrame) -> None:
    lines = [
        "# v0.7A.1 MIR1 Atlas Candidate Validation",
        "",
        "Frozen validation only. No new motif search, no parameter tuning.",
        "",
        "Frozen candidates:",
        "- MIR1: Top50 market volume-impulse-density high + bullish volume shock + 1% reclaim + vol-regime-fast.",
        "- IR2_ref: Top50 BTC_up + bullish volume shock + 1% reclaim + vol-regime-fast.",
        "- CHOP_tail_shadow: BTC_chop right-tail shadow, not a paper-live candidate.",
        "",
        "Primary paper-live gate, if promoted, must be `signal_and_entry_gate` using as-of market features.",
        "",
    ]
    focus = exact[
        exact["candidate"].astype(str).eq("MIR1")
        & exact["gate_mode"].astype(str).eq("signal_and_entry_gate")
    ]
    if focus.empty:
        lines.append("- MIR1 strict gate produced no summary row.")
    else:
        row = focus.iloc[0]
        passed = (
            row["net10"] >= 0.003
            and row["net20"] >= 0.001
            and row["month_cap35_net20"] >= 0.001
            and row["ex_top_month_net10"] > 0
            and row["matched_random_lift"] > 0
            and row["entry_only_lift"] > 0
            and row["max_symbol_contribution"] <= 0.35
        )
        lines.append(
            f"- MIR1 strict 15m: trades={int(row['trades'])}, net10={row['net10']:.4%}, "
            f"net20={row['net20']:.4%}, cap20={row['month_cap35_net20']:.4%}, "
            f"matched_lift={row['matched_random_lift']:.4%}, entry_lift={row['entry_only_lift']:.4%}."
        )
        lines.append(
            "- 15m decision: "
            + (
                "passes the frozen validation screen for B- graph-context paper-live review."
                if passed
                else "does not pass all frozen validation thresholds yet."
            )
        )
    if not one_min.empty and "status" in one_min.columns:
        statuses = sorted(one_min["status"].dropna().astype(str).unique())
        lines.append(f"- 1m validation status: {', '.join(statuses)}.")
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v07a1_mir1_validation(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path = REPORT_ROOT,
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    rank30, rank90, symbols = _rank_inputs(feature_path, instruments, config)
    regime = _build_regime_streaming(feature_path, rank30, rank90, symbols, config)
    trades, signals = _simulate_streaming(feature_path, rank30, rank90, symbols, regime, config, report_root)
    baseline = _summary_long(trades, signals)
    exact = _exact_gate_replay(trades, baseline)
    month = _month_concentration(trades)
    leave_one = _leave_one_month_out(trades)
    symbol = _symbol_contribution(trades)
    outputs = {
        "exact_gate_replay": report_root / "exact_gate_replay.csv",
        "baseline_decomposition": report_root / "baseline_decomposition.csv",
        "month_concentration": report_root / "month_concentration.csv",
        "leave_one_month_out": report_root / "leave_one_month_out.csv",
        "symbol_contribution": report_root / "symbol_contribution.csv",
        "execution_1m_comparison": report_root / "execution_1m_comparison.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    exact.to_csv(outputs["exact_gate_replay"], index=False)
    baseline.to_csv(outputs["baseline_decomposition"], index=False)
    month.to_csv(outputs["month_concentration"], index=False)
    leave_one.to_csv(outputs["leave_one_month_out"], index=False)
    symbol.to_csv(outputs["symbol_contribution"], index=False)
    one_min = pd.DataFrame([{"status": "not_started"}])
    one_min.to_csv(outputs["execution_1m_comparison"], index=False)
    _write_notes(report_root, exact, one_min)
    one_min = _one_min_comparison(feature_path, rank30, rank90, symbols, regime, config, exact)
    one_min.to_csv(outputs["execution_1m_comparison"], index=False)
    _write_notes(report_root, exact, one_min)
    return outputs


def run_v07a1_mir1_validation_from_features(config: ExperimentConfig) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = read_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v07a1_mir1_validation(features_path, instruments, config)
