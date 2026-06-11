from __future__ import annotations

import gc
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.backtest.minute_execution import simulate_1m_execution
from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v06c import _build_regime_streaming, _rank_inputs
from pressure_graph.reports.v07a import RECLAIM_1, SIM_WINDOW_BARS, _rule
from pressure_graph.reports.v07a1 import (
    BASELINES,
    FrozenAtlasCandidate,
    _entry_context_asof,
    _ex_top_month_net,
    _load_1m_cache,
    _lookup,
    _max_contribution,
    _metric,
    _month_cap_expectancy,
    _normalize_1m,
    _simulate_variant,
    _summary_long,
)
from pressure_graph.reports.v07b import (
    GRAPH_LOOKBACK_DAYS,
    TOP_N,
    _build_neighbor_graph_edges_for_months,
    _disable_out_of_month_events,
    _ensure_bool,
    _load_month_window,
    _month_symbols,
    add_neighbor_graph_features,
)


REPORT_ROOT = Path("reports/v0_7b1_nir1_validation")
ONE_MIN_COST_BPS = [10, 20, 50]


@dataclass(frozen=True)
class FrozenNeighborGate:
    candidate: str
    role: str
    gate_col: str
    description: str
    negative_control: bool = False


FROZEN_GATES = [
    FrozenNeighborGate(
        "B0_MIR1_raw",
        "raw_reference",
        "market_volume_impulse_density_high",
        "MIR1 raw market impulse density gate.",
    ),
    FrozenNeighborGate(
        "B1_NIR1_neighbor_impulse_high",
        "primary_validation",
        "gate_neighbor_impulse_high",
        "NIR1: MIR1 plus co-occurrence neighbor impulse confirmation.",
    ),
    FrozenNeighborGate(
        "B4_isolated_signal",
        "negative_control",
        "gate_isolated_signal",
        "MIR1 isolated from neighbor impulse and cluster breadth.",
        True,
    ),
    FrozenNeighborGate(
        "B5_low_neighbor_impulse",
        "negative_control",
        "gate_low_neighbor_impulse",
        "MIR1 with low neighbor impulse confirmation.",
        True,
    ),
    FrozenNeighborGate(
        "B6_random_neighbor_graph",
        "graph_negative_control",
        "gate_random_neighbor_impulse_high",
        "Random same-month neighbor graph control.",
        True,
    ),
    FrozenNeighborGate(
        "B7_shuffled_neighbor_graph",
        "graph_negative_control",
        "gate_shuffled_neighbor_impulse_high",
        "Shuffled real-neighbor mapping control.",
        True,
    ),
]

ONE_MIN_GATES = {"B0_MIR1_raw", "B1_NIR1_neighbor_impulse_high", "B5_low_neighbor_impulse"}
ONE_MIN_CONTEXT_COLS = [
    "exchange",
    "symbol",
    "bar_open_time",
    "feature_time",
    "open",
    "high",
    "low",
    "close",
    "dynamic_all_rank",
    "btc_market_state",
    "symbol_volatility_percentile",
    "market_volume_impulse_density_high",
    "gate_neighbor_impulse_high",
    "gate_low_neighbor_impulse",
    "bullish_volume_shock_event",
    "neighbor_impulse_ratio",
    "neighbor_bullish_volume_shock_count",
    "cluster_positive_return_ratio",
    "leader_impulse_recent",
    "leader_return_1h",
    "leader_return_4h",
    "symbol_lag_vs_neighbors",
    "isolated_signal_score",
    "random_neighbor_impulse_ratio",
    "shuffled_neighbor_impulse_ratio",
    "cluster_id",
]


def _candidate_for_gate(gate: FrozenNeighborGate) -> FrozenAtlasCandidate:
    return FrozenAtlasCandidate(
        gate.candidate,
        TOP_N,
        gate.gate_col,
        "bullish_volume_shock_event",
        RECLAIM_1,
        "vol_regime_fast",
        gate.role,
        gate.negative_control,
    )


def _stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def _edge_neighbors(edges: pd.DataFrame, month_start: pd.Timestamp, edge_type: str) -> dict[str, list[str]]:
    if edges.empty:
        return {}
    sample = edges[
        edges["month_start"].eq(month_start)
        & edges["edge_type"].astype(str).eq(edge_type)
    ].sort_values(["source_symbol", "edge_rank"])
    out: dict[str, list[str]] = {}
    for source, group in sample.groupby("source_symbol", sort=False, dropna=False):
        out[str(source)] = group["neighbor_symbol"].dropna().astype(str).tolist()
    return out


def _random_neighbors(month_start: pd.Timestamp, symbols: list[str], k: int = 5) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for symbol in symbols:
        choices = [item for item in symbols if item != symbol]
        if not choices:
            out[symbol] = []
            continue
        rng = np.random.default_rng(_stable_seed("random", month_start.isoformat(), symbol))
        take = min(k, len(choices))
        out[symbol] = list(rng.choice(choices, size=take, replace=False))
    return out


def _shuffled_neighbors(
    month_start: pd.Timestamp,
    symbols: list[str],
    real_neighbors: dict[str, list[str]],
) -> dict[str, list[str]]:
    if not symbols:
        return {}
    rng = np.random.default_rng(_stable_seed("shuffle", month_start.isoformat()))
    shuffled = symbols.copy()
    rng.shuffle(shuffled)
    mapping = dict(zip(symbols, shuffled, strict=False))
    out: dict[str, list[str]] = {}
    for symbol in symbols:
        mapped = [mapping.get(item, item) for item in real_neighbors.get(symbol, [])]
        out[symbol] = [item for item in mapped if item != symbol]
    return out


def _ratio_from_neighbors(
    state: pd.DataFrame,
    feature_time: pd.Series,
    neighbor_map: dict[str, list[str]],
    symbol: str,
) -> np.ndarray:
    neighbors = [item for item in neighbor_map.get(symbol, []) if item in state.columns]
    if not neighbors:
        return np.zeros(len(feature_time), dtype=float)
    count = state[neighbors].sum(axis=1).reindex(feature_time).fillna(0.0).to_numpy(dtype=float)
    return count / float(len(neighbors))


def _top1_cluster_map(edges: pd.DataFrame, month_start: pd.Timestamp, symbols: list[str]) -> dict[str, str]:
    neighbors = _edge_neighbors(edges, month_start, "return_corr_30d")
    out: dict[str, str] = {}
    for symbol in symbols:
        top = neighbors.get(symbol, [])
        if top:
            pair = sorted([symbol, top[0]])
            out[symbol] = f"{month_start:%Y-%m}:corr1:{pair[0]}:{pair[1]}"
        else:
            out[symbol] = f"{month_start:%Y-%m}:solo:{symbol}"
    return out


def _add_graph_controls_and_cluster(data: pd.DataFrame, edges: pd.DataFrame, month_start: pd.Timestamp) -> pd.DataFrame:
    out = data.copy()
    sample = out[pd.to_numeric(out["dynamic_all_rank"], errors="coerce") <= TOP_N].copy()
    if sample.empty:
        out["random_neighbor_impulse_ratio"] = 0.0
        out["shuffled_neighbor_impulse_ratio"] = 0.0
        out["gate_random_neighbor_impulse_high"] = False
        out["gate_shuffled_neighbor_impulse_high"] = False
        out["cluster_id"] = ""
        return out
    symbols = sorted(sample["symbol"].dropna().astype(str).unique())
    state = sample.pivot_table(
        index="feature_time",
        columns="symbol",
        values="bullish_volume_shock_state",
        aggfunc="max",
        observed=True,
    ).fillna(False).infer_objects(copy=False).astype(bool)
    real_neighbors = _edge_neighbors(edges, month_start, "volume_impulse_cooccurrence_30d")
    random_neighbors = _random_neighbors(month_start, symbols)
    shuffled_neighbors = _shuffled_neighbors(month_start, symbols, real_neighbors)
    out["random_neighbor_impulse_ratio"] = 0.0
    out["shuffled_neighbor_impulse_ratio"] = 0.0
    for symbol, group in sample.groupby("symbol", sort=False, observed=True):
        symbol = str(symbol)
        feature_time = pd.to_datetime(group["feature_time"], utc=True, errors="coerce")
        out.loc[group.index, "random_neighbor_impulse_ratio"] = _ratio_from_neighbors(
            state,
            feature_time,
            random_neighbors,
            symbol,
        )
        out.loc[group.index, "shuffled_neighbor_impulse_ratio"] = _ratio_from_neighbors(
            state,
            feature_time,
            shuffled_neighbors,
            symbol,
        )
    market = _ensure_bool(out.get("market_volume_impulse_density_high", pd.Series(False, index=out.index)))
    out["gate_random_neighbor_impulse_high"] = market & (
        pd.to_numeric(out["random_neighbor_impulse_ratio"], errors="coerce").fillna(0.0) >= 0.20
    )
    out["gate_shuffled_neighbor_impulse_high"] = market & (
        pd.to_numeric(out["shuffled_neighbor_impulse_ratio"], errors="coerce").fillna(0.0) >= 0.20
    )
    cluster_map = _top1_cluster_map(edges, month_start, symbols)
    out["cluster_id"] = out["symbol"].astype(str).map(cluster_map).fillna("")
    return out


def _attach_trade_context(trades: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    cols = [
        "exchange",
        "symbol",
        "feature_time",
        "neighbor_impulse_ratio",
        "neighbor_bullish_volume_shock_count",
        "cluster_positive_return_ratio",
        "leader_impulse_recent",
        "leader_return_1h",
        "leader_return_4h",
        "symbol_lag_vs_neighbors",
        "isolated_signal_score",
        "random_neighbor_impulse_ratio",
        "shuffled_neighbor_impulse_ratio",
        "cluster_id",
    ]
    context_cols = [col for col in cols if col in data.columns]
    if len(context_cols) <= 3:
        return trades.copy()
    context = data[context_cols].drop_duplicates(["exchange", "symbol", "feature_time"])
    out = trades.drop(
        columns=[col for col in context_cols if col not in {"exchange", "symbol", "feature_time"}],
        errors="ignore",
    )
    out = out.merge(
        context,
        left_on=["exchange", "symbol", "signal_time"],
        right_on=["exchange", "symbol", "feature_time"],
        how="left",
    ).drop(columns=["feature_time"], errors="ignore")
    return out


def _simulate_gates(data: pd.DataFrame, config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_rows: list[dict[str, object]] = []
    trade_frames = []
    for gate in FROZEN_GATES:
        candidate = _candidate_for_gate(gate)
        for baseline in BASELINES:
            trades, signal_n, pre_entry_gate_n = _simulate_variant(
                data,
                candidate,
                "signal_and_entry_gate",
                baseline,
                config,
            )
            signal_rows.append(
                {
                    "candidate": gate.candidate,
                    "gate_col": gate.gate_col,
                    "gate_mode": "signal_and_entry_gate",
                    "baseline": baseline.baseline,
                    "signals": signal_n,
                    "pre_entry_gate_trades": pre_entry_gate_n,
                }
            )
            if not trades.empty:
                trades["gate_description"] = gate.description
                trade_frames.append(trades)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    signals = pd.DataFrame(signal_rows)
    if not signals.empty:
        signals = signals.groupby(
            ["candidate", "gate_col", "gate_mode", "baseline"],
            as_index=False,
            sort=False,
            dropna=False,
        ).agg(signals=("signals", "sum"), pre_entry_gate_trades=("pre_entry_gate_trades", "sum"))
    return trades, signals


def _leakage_row(
    month_start: pd.Timestamp,
    symbols: list[str],
    window: pd.DataFrame,
    edges: pd.DataFrame,
) -> dict[str, object]:
    hist = window[
        (pd.to_datetime(window["bar_open_time"], utc=True, errors="coerce") < month_start)
        & (pd.to_datetime(window["bar_open_time"], utc=True, errors="coerce") >= month_start - pd.Timedelta(days=GRAPH_LOOKBACK_DAYS))
    ]
    max_hist = pd.to_datetime(hist["bar_open_time"], utc=True, errors="coerce").max() if not hist.empty else pd.NaT
    edge_months = sorted(pd.to_datetime(edges.get("month_start", pd.Series(dtype=object)), utc=True, errors="coerce").dropna().unique())
    return {
        "month_start": month_start,
        "symbols": len(symbols),
        "hist_rows": int(len(hist)),
        "hist_max_bar_open_time": max_hist,
        "graph_uses_prior_month_only": bool(pd.isna(max_hist) or max_hist < month_start),
        "edges": int(len(edges)),
        "edge_months": ",".join(pd.Timestamp(item).strftime("%Y-%m") for item in edge_months),
        "signal_entry_gate_mode": "signal_and_entry_gate",
        "entry_gate_asof_rule": "latest feature_time <= entry_time",
    }


def _stream_15m(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    regime: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trade_path = report_root / "_v07b1_trades_tmp.csv"
    if trade_path.exists():
        trade_path.unlink()
    wrote_header = False
    signal_rows: list[dict[str, object]] = []
    edge_frames: list[pd.DataFrame] = []
    leakage_rows: list[dict[str, object]] = []
    one_min_signal_frames: list[pd.DataFrame] = []
    months = sorted(
        pd.to_datetime(rank30["month_start"], utc=True, errors="coerce")
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    for idx, month_start in enumerate(months, start=1):
        month_start = pd.Timestamp(month_start)
        next_month = pd.Timestamp(months[idx]) if idx < len(months) else month_start + pd.DateOffset(months=1)
        symbols = _month_symbols(rank30, month_start)
        if not symbols:
            continue
        hist_start = month_start - pd.Timedelta(days=GRAPH_LOOKBACK_DAYS)
        window_end = next_month + pd.Timedelta(minutes=15 * SIM_WINDOW_BARS)
        window = _load_month_window(
            feature_path,
            rank30,
            rank90,
            symbols,
            regime,
            config,
            hist_start,
            window_end,
        )
        if window.empty:
            continue
        available_months = {
            pd.Timestamp(item)
            for item in pd.to_datetime(window["month_start"], utc=True, errors="coerce").dropna().unique()
        }
        months_for_edges = [month_start]
        if next_month in available_months:
            months_for_edges.append(next_month)
        edges = _build_neighbor_graph_edges_for_months(window, months_for_edges)
        leakage_rows.append(_leakage_row(month_start, symbols, window, edges[edges["month_start"].eq(month_start)]))
        if not edges.empty:
            edge_frames.append(edges[edges["month_start"].eq(month_start)].copy())
        sim_window = window[
            (pd.to_datetime(window["bar_open_time"], utc=True, errors="coerce") >= month_start)
            & (pd.to_datetime(window["bar_open_time"], utc=True, errors="coerce") < window_end)
        ].copy()
        sim_window = add_neighbor_graph_features(sim_window, edges)
        sim_window = _add_graph_controls_and_cluster(sim_window, edges, month_start)
        month_feature_time = pd.to_datetime(sim_window["feature_time"], utc=True, errors="coerce")
        month_mask = (month_feature_time >= month_start) & (month_feature_time < next_month)
        one_min_cols = [col for col in ONE_MIN_CONTEXT_COLS if col in sim_window.columns]
        one_min_signal_frames.append(sim_window.loc[month_mask, one_min_cols].copy())
        sim_window = _disable_out_of_month_events(sim_window, month_start, next_month)
        trades, signals = _simulate_gates(sim_window, config)
        if not trades.empty:
            trades = _attach_trade_context(trades, sim_window)
            trades.to_csv(trade_path, mode="a", header=not wrote_header, index=False)
            wrote_header = True
            del trades
        if not signals.empty:
            signal_rows.extend(signals.to_dict("records"))
        print(f"v0.7B.1 month {idx}/{len(months)} {month_start:%Y-%m} symbols={len(symbols)}", flush=True)
        del window, sim_window, edges
        gc.collect()
    trades = pd.read_csv(trade_path, low_memory=False) if wrote_header else pd.DataFrame()
    if trade_path.exists():
        trade_path.unlink()
    signals = pd.DataFrame(signal_rows)
    if not signals.empty:
        signals = signals.groupby(
            ["candidate", "gate_col", "gate_mode", "baseline"],
            as_index=False,
            sort=False,
            dropna=False,
        ).agg(signals=("signals", "sum"), pre_entry_gate_trades=("pre_entry_gate_trades", "sum"))
    edges = pd.concat(edge_frames, ignore_index=True).drop_duplicates() if edge_frames else pd.DataFrame()
    leakage = pd.DataFrame(leakage_rows)
    one_min_signals = pd.concat(one_min_signal_frames, ignore_index=True) if one_min_signal_frames else pd.DataFrame()
    return trades, signals, edges, leakage, one_min_signals


def _exact_replay(trades: pd.DataFrame, long_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gate in FROZEN_GATES:
        cand = trades[
            trades["candidate"].astype(str).eq(gate.candidate)
            & trades["baseline"].astype(str).eq("candidate_reclaim")
        ]
        cand10 = cand[pd.to_numeric(cand.get("cost_single_side_bps"), errors="coerce").eq(10.0)]
        if cand10.empty:
            continue
        net10 = _metric(cand, 10)
        row = {
            "candidate": gate.candidate,
            "role": gate.role,
            "negative_control": gate.negative_control,
            "signals": int(_lookup(long_summary, gate.candidate, "signal_and_entry_gate", "candidate_reclaim", 10, "signals")),
            "pre_entry_gate_trades": int(
                _lookup(long_summary, gate.candidate, "signal_and_entry_gate", "candidate_reclaim", 10, "pre_entry_gate_trades")
            ),
            "trades": int(len(cand10)),
            "net10": net10,
            "net20": _metric(cand, 20),
            "net30": _metric(cand, 30),
            "net50": _metric(cand, 50),
            "entry_only_lift": net10
            - _lookup(long_summary, gate.candidate, "signal_and_entry_gate", "entry_only_reclaim", 10, "net_expectancy"),
            "matched_random_lift": net10
            - _lookup(long_summary, gate.candidate, "signal_and_entry_gate", "matched_random_reclaim", 10, "net_expectancy"),
            "volume_shock_next_open_lift": net10
            - _lookup(long_summary, gate.candidate, "signal_and_entry_gate", "volume_shock_next_open", 10, "net_expectancy"),
            "volume_shock_pullback_lift": net10
            - _lookup(long_summary, gate.candidate, "signal_and_entry_gate", "volume_shock_pullback_only", 10, "net_expectancy"),
            "ex_top_month_net10": _ex_top_month_net(cand, 10),
            "month_cap35_net20": _month_cap_expectancy(cand, 20, 0.35),
            "max_month_contribution": _max_contribution(cand, "month", 10),
            "max_symbol_contribution": _max_contribution(cand, "symbol", 10),
            "max_cluster_contribution": _max_contribution(cand, "cluster_id", 10),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _monthly_attribution(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sample = trades[trades["baseline"].astype(str).eq("candidate_reclaim")]
    for (candidate, cost), group in sample.groupby(["candidate", "cost_single_side_bps"], sort=False, dropna=False):
        total = pd.to_numeric(group["net_return"], errors="coerce").sum()
        for month, month_group in group.groupby("month", sort=False, dropna=False):
            net = pd.to_numeric(month_group["net_return"], errors="coerce")
            exits = month_group["exit_reason"].astype(str)
            value = net.sum()
            rows.append(
                {
                    "candidate": candidate,
                    "cost_single_side_bps": cost,
                    "month": month,
                    "trades": int(len(month_group)),
                    "net_expectancy": float(net.mean()),
                    "net_sum": float(value),
                    "contribution_pct": float(value / total) if total else np.nan,
                    "tp_rate": float(exits.str.startswith("tp").mean()),
                    "sl_rate": float(exits.str.startswith("sl").mean()),
                    "timeout_rate": float(exits.eq("max_hold").mean()),
                }
            )
    return pd.DataFrame(rows)


def _leave_one_month_out(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sample = trades[trades["baseline"].astype(str).eq("candidate_reclaim")]
    for (candidate, cost), group in sample.groupby(["candidate", "cost_single_side_bps"], sort=False, dropna=False):
        for month in sorted(group["month"].dropna().astype(str).unique()):
            remain = group[~group["month"].astype(str).eq(month)]
            rows.append(
                {
                    "candidate": candidate,
                    "cost_single_side_bps": cost,
                    "excluded_month": month,
                    "remaining_trades": int(len(remain)),
                    "remaining_net_expectancy": float(pd.to_numeric(remain["net_return"], errors="coerce").mean())
                    if len(remain)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _month_cap_summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sample = trades[trades["baseline"].astype(str).eq("candidate_reclaim")]
    for candidate, group in sample.groupby("candidate", sort=False, dropna=False):
        rows.append(
            {
                "candidate": candidate,
                "net10": _metric(group, 10),
                "net20": _metric(group, 20),
                "month_cap20_net20": _month_cap_expectancy(group, 20, 0.20),
                "month_cap35_net20": _month_cap_expectancy(group, 20, 0.35),
                "month_cap50_net20": _month_cap_expectancy(group, 20, 0.50),
                "ex_top_month_net10": _ex_top_month_net(group, 10),
                "max_month_contribution": _max_contribution(group, "month", 10),
            }
        )
    return pd.DataFrame(rows)


def _contribution(trades: pd.DataFrame, group_col: str, name: str) -> pd.DataFrame:
    rows = []
    sample = trades[
        trades["baseline"].astype(str).eq("candidate_reclaim")
        & pd.to_numeric(trades["cost_single_side_bps"], errors="coerce").eq(10.0)
    ]
    for candidate, group in sample.groupby("candidate", sort=False, dropna=False):
        total = pd.to_numeric(group["net_return"], errors="coerce").sum()
        for key, local in group.groupby(group_col, sort=False, dropna=False):
            value = pd.to_numeric(local["net_return"], errors="coerce").sum()
            rows.append(
                {
                    "candidate": candidate,
                    name: key,
                    "trades": int(len(local)),
                    "net10": float(pd.to_numeric(local["net_return"], errors="coerce").mean()),
                    "gross_contribution_pct": float(value / total) if total else np.nan,
                    "symbols": ",".join(sorted(local["symbol"].dropna().astype(str).unique())[:20]),
                }
            )
    return pd.DataFrame(rows)


def _portfolio_concurrency(trades: pd.DataFrame) -> pd.DataFrame:
    sample = trades[
        trades["baseline"].astype(str).eq("candidate_reclaim")
        & pd.to_numeric(trades["cost_single_side_bps"], errors="coerce").isin([10.0, 20.0])
    ].copy()
    if sample.empty:
        return pd.DataFrame()
    sample["entry_time"] = pd.to_datetime(sample["entry_time"], utc=True, errors="coerce")
    sample["exit_time"] = pd.to_datetime(sample["exit_time"], utc=True, errors="coerce")
    rows = []
    for (candidate, cost), group in sample.groupby(["candidate", "cost_single_side_bps"], sort=False, dropna=False):
        group = group.sort_values(["entry_time", "symbol"]).reset_index(drop=True)
        for max_positions in [1, 3, 5, 10]:
            for cluster_cap in [1, 2, 9999]:
                active: list[tuple[pd.Timestamp, str]] = []
                selected = []
                skipped = 0
                max_concurrent = 0
                for row in group.itertuples(index=False):
                    entry_time = pd.Timestamp(row.entry_time)
                    active = [(exit_time, cluster) for exit_time, cluster in active if exit_time > entry_time]
                    cluster = str(getattr(row, "cluster_id", ""))
                    cluster_active = sum(1 for _, item in active if item == cluster)
                    if len(active) >= max_positions or cluster_active >= cluster_cap:
                        skipped += 1
                        continue
                    selected.append(row)
                    active.append((pd.Timestamp(row.exit_time), cluster))
                    max_concurrent = max(max_concurrent, len(active))
                if selected:
                    net = pd.Series([float(getattr(item, "net_return", np.nan)) for item in selected])
                    equity = net.cumsum()
                    drawdown = equity - equity.cummax()
                    net_expectancy = float(net.mean())
                    total_net = float(net.sum())
                    max_dd = float(drawdown.min())
                else:
                    net_expectancy = np.nan
                    total_net = 0.0
                    max_dd = np.nan
                rows.append(
                    {
                        "candidate": candidate,
                        "cost_single_side_bps": cost,
                        "max_positions": max_positions,
                        "max_positions_per_cluster": "unlimited" if cluster_cap == 9999 else cluster_cap,
                        "selected_trades": len(selected),
                        "skipped_trades": skipped,
                        "net_expectancy": net_expectancy,
                        "total_net": total_net,
                        "max_drawdown_proxy": max_dd,
                        "max_concurrent_positions": max_concurrent,
                    }
                )
    return pd.DataFrame(rows)


def _one_min_validation(
    one_min_signals: pd.DataFrame,
    trades_15m: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if one_min_signals.empty:
        return pd.DataFrame([{"status": "no_signals"}]), pd.DataFrame()
    trade_frames = []
    for gate in FROZEN_GATES:
        if gate.candidate not in ONE_MIN_GATES:
            continue
        candidate = _candidate_for_gate(gate)
        signal_rows = one_min_signals[
            (pd.to_numeric(one_min_signals["dynamic_all_rank"], errors="coerce") <= TOP_N)
            & _ensure_bool(one_min_signals[candidate.market_gate])
            & _ensure_bool(one_min_signals[candidate.event_col])
        ].copy()
        if signal_rows.empty:
            continue
        signal_rows["__v07b1_1m_signal"] = True
        symbols = sorted(signal_rows["symbol"].dropna().astype(str).unique())
        minute_bars = _load_1m_cache(config, symbols)
        if minute_bars.empty:
            continue
        start = pd.to_datetime(signal_rows["feature_time"], utc=True, errors="coerce").min()
        end = pd.to_datetime(signal_rows["feature_time"], utc=True, errors="coerce").max() + pd.Timedelta(hours=14)
        minute_bars["bar_open_time"] = pd.to_datetime(minute_bars["bar_open_time"], utc=True, errors="coerce")
        minute_bars = minute_bars[(minute_bars["bar_open_time"] >= start) & (minute_bars["bar_open_time"] <= end)].copy()
        context_cols = [
            col
            for col in [
                "exchange",
                "symbol",
                "feature_time",
                "btc_market_state",
                candidate.market_gate,
                "neighbor_impulse_ratio",
                "cluster_positive_return_ratio",
                "cluster_id",
            ]
            if col in one_min_signals.columns
        ]
        context = one_min_signals[context_cols].drop_duplicates(["exchange", "symbol", "feature_time"])
        rule, resolver = _rule(candidate.exit_rule, config)
        for cost in ONE_MIN_COST_BPS:
            trades = simulate_1m_execution(
                signal_rows,
                minute_bars,
                "__v07b1_1m_signal",
                candidate.candidate,
                candidate.entry_policy,
                candidate.exit_rule,
                rule,
                float(cost),
                resolver,
            )
            trades = _normalize_1m(trades, candidate, "signal_and_entry_gate")
            if trades.empty:
                continue
            trades = _entry_context_asof(trades, context, candidate.market_gate)
            trades = trades[trades["market_gate_at_entry"].fillna(False).astype(bool)].copy()
            trades = _attach_trade_context(trades, one_min_signals)
            if not trades.empty:
                trades["cost_single_side_bps"] = float(cost)
                trade_frames.append(trades)
    one_min_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    if one_min_trades.empty:
        return pd.DataFrame([{"status": "no_1m_fills"}]), pd.DataFrame()
    rows = []
    for (candidate, cost), group in one_min_trades.groupby(["candidate", "cost_single_side_bps"], sort=False):
        exits = group["exit_reason"].astype(str)
        net_1m = float(pd.to_numeric(group["net_return"], errors="coerce").mean())
        comp15 = trades_15m[
            trades_15m["candidate"].astype(str).eq(str(candidate))
            & trades_15m["baseline"].astype(str).eq("candidate_reclaim")
            & pd.to_numeric(trades_15m["cost_single_side_bps"], errors="coerce").eq(float(cost))
        ]
        net_15m = float(pd.to_numeric(comp15["net_return"], errors="coerce").mean()) if len(comp15) else np.nan
        rows.append(
            {
                "status": "ok",
                "candidate": candidate,
                "cost_single_side_bps": cost,
                "trades_15m": int(len(comp15)),
                "trades_1m": int(len(group)),
                "net_15m": net_15m,
                "net_1m": net_1m,
                "retention_vs_15m": float(net_1m / net_15m) if net_15m else np.nan,
                "tp_rate_1m": float(exits.str.startswith("tp").mean()),
                "sl_rate_1m": float(exits.str.startswith("sl").mean()),
                "timeout_rate_1m": float(exits.eq("max_hold").mean()),
                "same_bar_ambiguity_1m": float(group["unresolved_1m_same_bar"].fillna(False).mean()),
            }
        )
    return pd.DataFrame(rows), one_min_trades


def _one_min_reconciliation(trades_15m: pd.DataFrame, trades_1m: pd.DataFrame) -> pd.DataFrame:
    if trades_1m.empty:
        return pd.DataFrame()
    def prep(data: pd.DataFrame, suffix: str) -> pd.DataFrame:
        sample = data[
            data["baseline"].astype(str).eq("candidate_reclaim")
            & pd.to_numeric(data["cost_single_side_bps"], errors="coerce").eq(10.0)
        ].copy()
        sample["signal_time"] = pd.to_datetime(sample["signal_time"], utc=True, errors="coerce")
        sample["signal_id"] = (
            sample["exchange"].astype(str)
            + "|"
            + sample["symbol"].astype(str)
            + "|"
            + sample["signal_time"].astype(str)
            + "|"
            + sample["candidate"].astype(str)
        )
        keep = ["signal_id", "candidate", "symbol", "signal_time", "entry_time", "exit_time", "exit_reason", "net_return"]
        sample = sample[[col for col in keep if col in sample.columns]]
        return sample.rename(columns={col: f"{col}_{suffix}" for col in ["entry_time", "exit_time", "exit_reason", "net_return"]})

    left = prep(trades_15m, "15m")
    right = prep(trades_1m, "1m")
    merged = left.merge(right, on=["signal_id", "candidate", "symbol", "signal_time"], how="outer")
    merged["in_15m"] = merged["net_return_15m"].notna()
    merged["in_1m"] = merged["net_return_1m"].notna()
    merged["delta_net10"] = pd.to_numeric(merged["net_return_1m"], errors="coerce") - pd.to_numeric(
        merged["net_return_15m"],
        errors="coerce",
    )
    return merged


def _write_notes(report_root: Path, exact: pd.DataFrame, one_min: pd.DataFrame) -> None:
    lines = [
        "# v0.7B.1 NIR1 Frozen Validation",
        "",
        "Frozen validation for NIR1. No neighbor threshold, reclaim, or exit tuning.",
        "",
    ]
    if not exact.empty:
        sample = exact[exact["candidate"].astype(str).eq("B1_NIR1_neighbor_impulse_high")]
        if not sample.empty:
            row = sample.iloc[0]
            lines.append(
                f"- NIR1 15m: trades={int(row['trades'])}, net10={row['net10']:.4%}, net20={row['net20']:.4%}, "
                f"month_cap35_net20={row['month_cap35_net20']:.4%}."
            )
    if not one_min.empty and "status" in one_min.columns:
        lines.append(f"- 1m status: {', '.join(sorted(one_min['status'].dropna().astype(str).unique()))}.")
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v07b1_nir1_validation(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path = REPORT_ROOT,
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    rank30, rank90, symbols = _rank_inputs(feature_path, instruments, config)
    symbols = sorted(
        rank30[pd.to_numeric(rank30["dynamic_all_rank"], errors="coerce") <= TOP_N]["symbol"]
        .dropna()
        .astype(str)
        .unique()
    )
    regime = _build_regime_streaming(feature_path, rank30, rank90, symbols, config)
    trades, signals, edges, leakage, one_min_signals = _stream_15m(
        feature_path,
        rank30,
        rank90,
        regime,
        config,
        report_root,
    )
    long_summary = _summary_long(trades, signals)
    exact = _exact_replay(trades, long_summary)
    one_min_summary, one_min_trades = _one_min_validation(one_min_signals, trades, config)
    reconciliation = _one_min_reconciliation(trades, one_min_trades)
    outputs = {
        "leakage_audit": report_root / "leakage_audit.csv",
        "exact_replay": report_root / "exact_replay.csv",
        "one_min_execution": report_root / "one_min_execution.csv",
        "one_min_reconciliation": report_root / "one_min_reconciliation.csv",
        "monthly_attribution": report_root / "monthly_attribution.csv",
        "leave_one_month_out": report_root / "leave_one_month_out.csv",
        "month_cap_summary": report_root / "month_cap_summary.csv",
        "symbol_contribution": report_root / "symbol_contribution.csv",
        "cluster_contribution": report_root / "cluster_contribution.csv",
        "baseline_decomposition": report_root / "baseline_decomposition.csv",
        "portfolio_concurrency": report_root / "portfolio_concurrency.csv",
        "neighbor_graph_edges": report_root / "neighbor_graph_edges.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    leakage.to_csv(outputs["leakage_audit"], index=False)
    exact.to_csv(outputs["exact_replay"], index=False)
    one_min_summary.to_csv(outputs["one_min_execution"], index=False)
    reconciliation.to_csv(outputs["one_min_reconciliation"], index=False)
    _monthly_attribution(trades).to_csv(outputs["monthly_attribution"], index=False)
    _leave_one_month_out(trades).to_csv(outputs["leave_one_month_out"], index=False)
    _month_cap_summary(trades).to_csv(outputs["month_cap_summary"], index=False)
    _contribution(trades, "symbol", "symbol").to_csv(outputs["symbol_contribution"], index=False)
    _contribution(trades, "cluster_id", "cluster_id").to_csv(outputs["cluster_contribution"], index=False)
    exact.to_csv(outputs["baseline_decomposition"], index=False)
    _portfolio_concurrency(trades).to_csv(outputs["portfolio_concurrency"], index=False)
    edges.to_csv(outputs["neighbor_graph_edges"], index=False)
    _write_notes(report_root, exact, one_min_summary)
    return outputs
