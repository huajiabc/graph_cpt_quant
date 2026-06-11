from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.backtest.minute_execution import simulate_1m_execution
from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v06a1 import _read_symbol_features
from pressure_graph.reports.v06c import _build_regime_streaming, _rank_inputs
from pressure_graph.reports.v07a import RECLAIM_1, SIM_WINDOW_BARS, _add_motif_columns, _rule
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


REPORT_ROOT = Path("reports/v0_7b_neighbor_graph")
TOP_N = 50
GRAPH_LOOKBACK_DAYS = 30
GRAPH_TOP_K = 5
ONE_MIN_COST_BPS = [10, 20]


@dataclass(frozen=True)
class NeighborGate:
    candidate: str
    role: str
    gate_col: str
    description: str
    negative_control: bool = False


GATES = [
    NeighborGate(
        "B0_MIR1_raw",
        "raw_reference",
        "market_volume_impulse_density_high",
        "MIR1 raw market impulse density gate.",
    ),
    NeighborGate(
        "B1_neighbor_impulse_high",
        "neighbor_confirmation",
        "gate_neighbor_impulse_high",
        "MIR1 plus volume-impulse co-occurrence neighbors currently firing.",
    ),
    NeighborGate(
        "B2_cluster_breadth_high",
        "cluster_confirmation",
        "gate_cluster_positive_return_high",
        "MIR1 plus return-correlation neighbors broadly positive.",
    ),
    NeighborGate(
        "B3_leader_beta_lag",
        "leader_beta",
        "gate_leader_recent_lag",
        "MIR1 plus recent neighbor leader impulse while symbol is not extended.",
    ),
    NeighborGate(
        "B4_isolated_signal",
        "negative_control",
        "gate_isolated_signal",
        "MIR1 isolated from neighbor impulse and cluster breadth.",
        True,
    ),
    NeighborGate(
        "B5_low_neighbor_impulse",
        "negative_control",
        "gate_low_neighbor_impulse",
        "MIR1 with low neighbor impulse confirmation.",
        True,
    ),
]


def _candidate_for_gate(gate: NeighborGate) -> FrozenAtlasCandidate:
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


def _ensure_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def _safe_mean(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return frame[columns].mean(axis=1)


def _build_return_corr_edges(month_start: pd.Timestamp, hist: pd.DataFrame, symbols: list[str]) -> list[dict[str, object]]:
    if hist.empty:
        return []
    pivot = hist.pivot_table(index="feature_time", columns="symbol", values="ret_1h", aggfunc="last")
    pivot = pivot[[symbol for symbol in symbols if symbol in pivot.columns]]
    corr = pivot.corr(min_periods=96 * 7)
    rows: list[dict[str, object]] = []
    if corr.empty:
        return rows
    for source in corr.columns:
        scores = corr[source].drop(labels=[source], errors="ignore").dropna()
        scores = scores[scores > 0].sort_values(ascending=False).head(GRAPH_TOP_K)
        for rank, (target, weight) in enumerate(scores.items(), start=1):
            rows.append(
                {
                    "month_start": month_start,
                    "source_symbol": source,
                    "neighbor_symbol": target,
                    "edge_type": "return_corr_30d",
                    "edge_rank": rank,
                    "edge_weight": float(weight),
                    "lookback_days": GRAPH_LOOKBACK_DAYS,
                }
            )
    return rows


def _build_volume_cooccurrence_edges(
    month_start: pd.Timestamp,
    hist: pd.DataFrame,
    symbols: list[str],
) -> list[dict[str, object]]:
    if hist.empty:
        return []
    pivot = hist.pivot_table(
        index="feature_time",
        columns="symbol",
        values="bullish_volume_shock_state",
        aggfunc="max",
    )
    pivot = pivot[[symbol for symbol in symbols if symbol in pivot.columns]].fillna(False).astype("int8")
    if pivot.empty:
        return []
    matrix = pivot.T.dot(pivot)
    source_counts = pivot.sum(axis=0).replace(0, np.nan)
    rows: list[dict[str, object]] = []
    for source in matrix.columns:
        scores = (matrix[source] / source_counts.get(source, np.nan)).drop(labels=[source], errors="ignore").dropna()
        scores = scores[scores > 0].sort_values(ascending=False).head(GRAPH_TOP_K)
        for rank, (target, weight) in enumerate(scores.items(), start=1):
            rows.append(
                {
                    "month_start": month_start,
                    "source_symbol": source,
                    "neighbor_symbol": target,
                    "edge_type": "volume_impulse_cooccurrence_30d",
                    "edge_rank": rank,
                    "edge_weight": float(weight),
                    "lookback_days": GRAPH_LOOKBACK_DAYS,
                }
            )
    return rows


def build_neighbor_graph_edges(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    working = data.copy()
    working["bar_open_time"] = pd.to_datetime(working["bar_open_time"], utc=True, errors="coerce")
    working["feature_time"] = pd.to_datetime(working["feature_time"], utc=True, errors="coerce")
    rows: list[dict[str, object]] = []
    for month_start in sorted(working["month_start"].dropna().unique()):
        month_start = pd.Timestamp(month_start)
        month_rows = working[
            working["month_start"].eq(month_start)
            & (pd.to_numeric(working["dynamic_all_rank"], errors="coerce") <= TOP_N)
        ]
        symbols = sorted(month_rows["symbol"].dropna().astype(str).unique())
        if len(symbols) < 3:
            continue
        hist_start = month_start - pd.Timedelta(days=GRAPH_LOOKBACK_DAYS)
        hist = working[
            working["symbol"].astype(str).isin(symbols)
            & (working["bar_open_time"] >= hist_start)
            & (working["bar_open_time"] < month_start)
        ].copy()
        rows.extend(_build_return_corr_edges(month_start, hist, symbols))
        rows.extend(_build_volume_cooccurrence_edges(month_start, hist, symbols))
    return pd.DataFrame(rows)


def _build_neighbor_graph_edges_for_months(data: pd.DataFrame, months: list[pd.Timestamp]) -> pd.DataFrame:
    if data.empty or not months:
        return pd.DataFrame()
    working = data.copy()
    working["bar_open_time"] = pd.to_datetime(working["bar_open_time"], utc=True, errors="coerce")
    working["feature_time"] = pd.to_datetime(working["feature_time"], utc=True, errors="coerce")
    rows: list[dict[str, object]] = []
    for month_start in months:
        month_start = pd.Timestamp(month_start)
        month_rows = working[
            working["month_start"].eq(month_start)
            & (pd.to_numeric(working["dynamic_all_rank"], errors="coerce") <= TOP_N)
        ]
        symbols = sorted(month_rows["symbol"].dropna().astype(str).unique())
        if len(symbols) < 3:
            continue
        hist_start = month_start - pd.Timedelta(days=GRAPH_LOOKBACK_DAYS)
        hist = working[
            working["symbol"].astype(str).isin(symbols)
            & (working["bar_open_time"] >= hist_start)
            & (working["bar_open_time"] < month_start)
        ].copy()
        rows.extend(_build_return_corr_edges(month_start, hist, symbols))
        rows.extend(_build_volume_cooccurrence_edges(month_start, hist, symbols))
    return pd.DataFrame(rows)


def _neighbors(edges: pd.DataFrame, month_start: pd.Timestamp, symbol: str, edge_type: str) -> list[str]:
    sample = edges[
        edges["month_start"].eq(month_start)
        & edges["source_symbol"].astype(str).eq(symbol)
        & edges["edge_type"].astype(str).eq(edge_type)
    ].sort_values("edge_rank")
    return sample["neighbor_symbol"].dropna().astype(str).tolist()


def add_neighbor_graph_features(data: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    out = data.sort_values(["exchange", "symbol", "bar_open_time"]).copy()
    defaults = {
        "neighbor_impulse_ratio": 0.0,
        "neighbor_bullish_volume_shock_count": 0.0,
        "neighbor_reclaim_count": 0.0,
        "cluster_positive_return_ratio": np.nan,
        "leader_impulse_recent": False,
        "leader_return_1h": np.nan,
        "leader_return_4h": np.nan,
        "symbol_lag_vs_neighbors": np.nan,
        "isolated_signal_score": 1.0,
    }
    for col, value in defaults.items():
        out[col] = value
    if out.empty or edges.empty:
        return _add_neighbor_gates(out)
    for month_start, month_data in out.groupby("month_start", sort=True, observed=True):
        month_start = pd.Timestamp(month_start)
        sample = month_data[pd.to_numeric(month_data["dynamic_all_rank"], errors="coerce") <= TOP_N].copy()
        if sample.empty:
            continue
        state = sample.pivot_table(
            index="feature_time",
            columns="symbol",
            values="bullish_volume_shock_state",
            aggfunc="max",
            observed=True,
        ).fillna(False).infer_objects(copy=False).astype(bool)
        event = sample.pivot_table(
            index="feature_time",
            columns="symbol",
            values="bullish_volume_shock_event",
            aggfunc="max",
            observed=True,
        ).fillna(False).infer_objects(copy=False).astype(bool)
        ret1 = sample.pivot_table(
            index="feature_time",
            columns="symbol",
            values="ret_1h",
            aggfunc="last",
            observed=True,
        )
        ret4 = sample.pivot_table(
            index="feature_time",
            columns="symbol",
            values="ret_4h",
            aggfunc="last",
            observed=True,
        )
        posret = (ret4 > 0).fillna(False)
        recent_event = event.shift(1).rolling(4, min_periods=1).max().fillna(False).astype(bool)
        for symbol, group in sample.groupby("symbol", sort=False, observed=True):
            symbol = str(symbol)
            idx = group.index
            feature_time = pd.to_datetime(group["feature_time"], utc=True, errors="coerce")
            impulse_neighbors = _neighbors(
                edges,
                month_start,
                symbol,
                "volume_impulse_cooccurrence_30d",
            )
            corr_neighbors = _neighbors(edges, month_start, symbol, "return_corr_30d")
            impulse_neighbors = [item for item in impulse_neighbors if item in state.columns]
            corr_neighbors = [item for item in corr_neighbors if item in ret4.columns]
            if impulse_neighbors:
                count = state[impulse_neighbors].sum(axis=1).reindex(feature_time).to_numpy(dtype=float)
                ratio = count / float(len(impulse_neighbors))
                leader_recent = recent_event[impulse_neighbors].max(axis=1).reindex(feature_time).fillna(False)
                leader_ret1 = ret1[impulse_neighbors].max(axis=1).reindex(feature_time)
                leader_ret4 = ret4[impulse_neighbors].max(axis=1).reindex(feature_time)
                out.loc[idx, "neighbor_bullish_volume_shock_count"] = count
                out.loc[idx, "neighbor_impulse_ratio"] = ratio
                out.loc[idx, "leader_impulse_recent"] = leader_recent.to_numpy(dtype=bool)
                out.loc[idx, "leader_return_1h"] = pd.to_numeric(leader_ret1, errors="coerce").to_numpy()
                out.loc[idx, "leader_return_4h"] = pd.to_numeric(leader_ret4, errors="coerce").to_numpy()
            if corr_neighbors:
                cluster_ratio = posret[corr_neighbors].mean(axis=1).reindex(feature_time)
                neighbor_ret = _safe_mean(ret4, corr_neighbors).reindex(feature_time)
                own_ret = pd.to_numeric(group["ret_4h"], errors="coerce")
                out.loc[idx, "cluster_positive_return_ratio"] = pd.to_numeric(cluster_ratio, errors="coerce").to_numpy()
                out.loc[idx, "symbol_lag_vs_neighbors"] = (
                    pd.to_numeric(neighbor_ret, errors="coerce").to_numpy() - own_ret.to_numpy()
                )
    out["isolated_signal_score"] = 1.0 - pd.to_numeric(out["neighbor_impulse_ratio"], errors="coerce").fillna(0.0)
    return _add_neighbor_gates(out)


def _add_neighbor_gates(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    market = _ensure_bool(out.get("market_volume_impulse_density_high", pd.Series(False, index=out.index)))
    neighbor_ratio = pd.to_numeric(out.get("neighbor_impulse_ratio"), errors="coerce").fillna(0.0)
    cluster_ratio = pd.to_numeric(out.get("cluster_positive_return_ratio"), errors="coerce").fillna(0.0)
    lag = pd.to_numeric(out.get("symbol_lag_vs_neighbors"), errors="coerce")
    ret4_pct = pd.to_numeric(out.get("ret_4h_percentile"), errors="coerce")
    leader_recent = _ensure_bool(out.get("leader_impulse_recent", pd.Series(False, index=out.index)))
    out["gate_neighbor_impulse_high"] = market & (neighbor_ratio >= 0.20)
    out["gate_cluster_positive_return_high"] = market & (cluster_ratio >= 0.60)
    out["gate_leader_recent_lag"] = market & leader_recent & (lag > 0) & (ret4_pct < 90)
    out["gate_isolated_signal"] = market & (neighbor_ratio <= 0.0) & (cluster_ratio <= 0.40)
    out["gate_low_neighbor_impulse"] = market & (neighbor_ratio <= 0.10)
    return out


def _load_top100_dataset(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    symbols: list[str],
    regime: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    frames = []
    for idx, symbol in enumerate(symbols, start=1):
        data = _read_symbol_features(feature_path, rank30, rank90, symbol, config)
        if data.empty:
            continue
        data = data[pd.to_numeric(data["dynamic_all_rank"], errors="coerce") <= TOP_N].copy()
        if data.empty:
            continue
        data = _add_motif_columns(data, regime, config)
        frames.append(data)
        print(f"v0.7B load {idx}/{len(symbols)} {symbol}", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _month_symbols(rank30: pd.DataFrame, month_start: pd.Timestamp) -> list[str]:
    sample = rank30[
        pd.to_datetime(rank30["month_start"], utc=True, errors="coerce").eq(month_start)
        & (pd.to_numeric(rank30["dynamic_all_rank"], errors="coerce") <= TOP_N)
    ]
    return sorted(sample["symbol"].dropna().astype(str).unique())


def _load_month_window(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    symbols: list[str],
    regime: pd.DataFrame,
    config: ExperimentConfig,
    hist_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        data = _read_symbol_features(feature_path, rank30, rank90, symbol, config)
        if data.empty:
            continue
        data["bar_open_time"] = pd.to_datetime(data["bar_open_time"], utc=True, errors="coerce")
        data = data[
            (pd.to_numeric(data["dynamic_all_rank"], errors="coerce") <= TOP_N)
            & (data["bar_open_time"] >= hist_start)
            & (data["bar_open_time"] < window_end)
        ].copy()
        if data.empty:
            continue
        data = _add_motif_columns(data, regime, config)
        frames.append(data)
    return _prune_v07b_columns(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()


def _feature_signal_rows(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gate in GATES:
        mask = (
            pd.to_numeric(data["dynamic_all_rank"], errors="coerce").le(TOP_N)
            & _ensure_bool(data[gate.gate_col])
            & _ensure_bool(data["bullish_volume_shock_event"])
        )
        sample = data.loc[
            mask,
            [
                "neighbor_impulse_ratio",
                "cluster_positive_return_ratio",
                "leader_impulse_recent",
                "symbol_lag_vs_neighbors",
            ],
        ].copy()
        if sample.empty:
            continue
        sample.insert(0, "role", gate.role)
        sample.insert(0, "candidate", gate.candidate)
        rows.append(sample)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _feature_summary_from_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    out_rows = []
    for (candidate, role), sample in rows.groupby(["candidate", "role"], sort=False, dropna=False):
        out_rows.append(
            {
                "candidate": candidate,
                "role": role,
                "signals": int(len(sample)),
                "neighbor_impulse_ratio_median": float(
                    pd.to_numeric(sample["neighbor_impulse_ratio"], errors="coerce").median()
                ),
                "cluster_positive_return_ratio_median": float(
                    pd.to_numeric(sample["cluster_positive_return_ratio"], errors="coerce").median()
                ),
                "leader_impulse_recent_rate": float(_ensure_bool(sample["leader_impulse_recent"]).mean()),
                "symbol_lag_vs_neighbors_median": float(
                    pd.to_numeric(sample["symbol_lag_vs_neighbors"], errors="coerce").median()
                ),
            }
        )
    return pd.DataFrame(out_rows)


def _disable_out_of_month_events(data: pd.DataFrame, month_start: pd.Timestamp, next_month: pd.Timestamp) -> pd.DataFrame:
    out = data.copy()
    feature_time = pd.to_datetime(out["feature_time"], utc=True, errors="coerce")
    in_month = (feature_time >= month_start) & (feature_time < next_month)
    event_cols = sorted({baseline.event_col for baseline in BASELINES})
    for col in event_cols:
        if col in out.columns:
            out.loc[~in_month, col] = False
    return out


def _attach_neighbor_trade_context(trades: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    cols = [
        "exchange",
        "symbol",
        "feature_time",
        "neighbor_impulse_ratio",
        "neighbor_bullish_volume_shock_count",
        "neighbor_reclaim_count",
        "cluster_positive_return_ratio",
        "leader_impulse_recent",
        "leader_return_1h",
        "leader_return_4h",
        "symbol_lag_vs_neighbors",
        "isolated_signal_score",
    ]
    context_cols = [col for col in cols if col in data.columns]
    if len(context_cols) <= 3:
        return trades.copy()
    context = data[context_cols].drop_duplicates(["exchange", "symbol", "feature_time"])
    out = trades.drop(columns=[col for col in context_cols if col not in {"exchange", "symbol", "feature_time"}], errors="ignore")
    out = out.merge(
        context,
        left_on=["exchange", "symbol", "signal_time"],
        right_on=["exchange", "symbol", "feature_time"],
        how="left",
    ).drop(columns=["feature_time"], errors="ignore")
    return out


def _simulate_v07b_streaming(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    regime: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trade_path = report_root / "_v07b_trades_tmp.csv"
    if trade_path.exists():
        trade_path.unlink()
    wrote_header = False
    signal_rows: list[dict[str, object]] = []
    edge_frames: list[pd.DataFrame] = []
    feature_signal_frames: list[pd.DataFrame] = []
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
            pd.Timestamp(item) for item in pd.to_datetime(window["month_start"], utc=True, errors="coerce").dropna().unique()
        }
        months_for_edges = [month_start]
        if next_month in available_months:
            months_for_edges.append(next_month)
        edges = _build_neighbor_graph_edges_for_months(window, months_for_edges)
        if not edges.empty:
            edge_frames.append(edges[edges["month_start"].eq(month_start)].copy())
        sim_window = window[
            (pd.to_datetime(window["bar_open_time"], utc=True, errors="coerce") >= month_start)
            & (pd.to_datetime(window["bar_open_time"], utc=True, errors="coerce") < window_end)
        ].copy()
        sim_window = add_neighbor_graph_features(sim_window, edges)
        month_only = sim_window[
            (pd.to_datetime(sim_window["feature_time"], utc=True, errors="coerce") >= month_start)
            & (pd.to_datetime(sim_window["feature_time"], utc=True, errors="coerce") < next_month)
        ].copy()
        feature_rows = _feature_signal_rows(month_only)
        if not feature_rows.empty:
            feature_signal_frames.append(feature_rows)
        sim_window = _disable_out_of_month_events(sim_window, month_start, next_month)
        trades, signals = _simulate_gates(sim_window, config)
        if not trades.empty:
            trades = _attach_neighbor_trade_context(trades, sim_window)
            trades.to_csv(trade_path, mode="a", header=not wrote_header, index=False)
            wrote_header = True
            del trades
        if not signals.empty:
            signal_rows.extend(signals.to_dict("records"))
        print(f"v0.7B month {idx}/{len(months)} {month_start:%Y-%m} symbols={len(symbols)}", flush=True)
        del window, sim_window, month_only, edges
        gc.collect()
    trades = pd.read_csv(trade_path, low_memory=False) if wrote_header else pd.DataFrame()
    signals = pd.DataFrame(signal_rows)
    if not signals.empty:
        signals = signals.groupby(
            ["candidate", "gate_col", "gate_mode", "baseline"],
            as_index=False,
            sort=False,
            dropna=False,
        ).agg(signals=("signals", "sum"), pre_entry_gate_trades=("pre_entry_gate_trades", "sum"))
    edges = pd.concat(edge_frames, ignore_index=True).drop_duplicates() if edge_frames else pd.DataFrame()
    feature_rows = pd.concat(feature_signal_frames, ignore_index=True) if feature_signal_frames else pd.DataFrame()
    return trades, signals, edges, feature_rows


def _prune_v07b_columns(data: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "exchange",
        "symbol",
        "bar_open_time",
        "bar_close_time",
        "feature_time",
        "open",
        "high",
        "low",
        "close",
        "funding_time",
        "funding_rate_settled",
        "ret_1h",
        "ret_4h",
        "ret_4h_percentile",
        "volume_z_4h",
        "warmup_complete",
        "btc_market_state",
        "btc_ret_1h",
        "btc_ret_4h",
        "btc_volatility_4h",
        "dynamic_all_rank",
        "dynamic_all_trailing_turnover",
        "turnover_rank_90d",
        "month_start",
        "month",
        "liquidity_bucket",
        "symbol_volatility_percentile",
        "mfe_12h",
        "mae_12h",
        "mfe_24h",
        "mae_24h",
        "hit_10pct_12h",
        "hit_20pct_24h",
        "bullish_volume_shock_state",
        "bullish_volume_shock_event",
        "neutral_volume_event",
        "matched_random_event",
        "market_volume_impulse_density_high",
        "market_low_volume_impulse_density",
        "market_btc_up",
        "market_btc_chop",
        "volume_impulse_density",
    ]
    out = data[[col for col in keep if col in data.columns]].copy()
    for col in [
        "open",
        "high",
        "low",
        "close",
        "funding_rate_settled",
        "ret_1h",
        "ret_4h",
        "ret_4h_percentile",
        "volume_z_4h",
        "btc_ret_1h",
        "btc_ret_4h",
        "btc_volatility_4h",
        "dynamic_all_rank",
        "dynamic_all_trailing_turnover",
        "turnover_rank_90d",
        "symbol_volatility_percentile",
        "mfe_12h",
        "mae_12h",
        "mfe_24h",
        "mae_24h",
        "volume_impulse_density",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce", downcast="float")
    for col in ["exchange", "symbol", "btc_market_state", "month", "liquidity_bucket"]:
        if col in out.columns:
            out[col] = out[col].astype("category")
    for col in ["bar_open_time", "bar_close_time", "feature_time", "funding_time", "month_start"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    for col in [
        "warmup_complete",
        "bullish_volume_shock_state",
        "bullish_volume_shock_event",
        "neutral_volume_event",
        "matched_random_event",
        "market_volume_impulse_density_high",
        "market_low_volume_impulse_density",
        "market_btc_up",
        "market_btc_chop",
        "hit_10pct_12h",
        "hit_20pct_24h",
    ]:
        if col in out.columns:
            out[col] = out[col].fillna(False).astype(bool)
    return out


def _simulate_gates(data: pd.DataFrame, config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_rows: list[dict[str, object]] = []
    trade_frames = []
    for gate in GATES:
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


def _gate_summary(trades: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    long_summary = _summary_long(trades, signals)
    rows = []
    def median_col(frame: pd.DataFrame, col: str) -> float:
        if col not in frame.columns:
            return np.nan
        return float(pd.to_numeric(frame[col], errors="coerce").median())

    for gate in GATES:
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
            "gate_col": gate.gate_col,
            "negative_control": gate.negative_control,
            "signals": int(
                _lookup(long_summary, gate.candidate, "signal_and_entry_gate", "candidate_reclaim", 10, "signals")
            ),
            "pre_entry_gate_trades": int(
                _lookup(
                    long_summary,
                    gate.candidate,
                    "signal_and_entry_gate",
                    "candidate_reclaim",
                    10,
                    "pre_entry_gate_trades",
                )
            ),
            "trades": int(len(cand10)),
            "net10": net10,
            "net20": _metric(cand, 20),
            "net30": _metric(cand, 30),
            "net50": _metric(cand, 50),
            "month_cap35_net20": _month_cap_expectancy(cand, 20, 0.35),
            "ex_top_month_net10": _ex_top_month_net(cand, 10),
            "max_month_contribution": _max_contribution(cand, "month", 10),
            "max_symbol_contribution": _max_contribution(cand, "symbol", 10),
            "neighbor_impulse_ratio_median": median_col(cand10, "neighbor_impulse_ratio"),
            "cluster_positive_return_ratio_median": median_col(cand10, "cluster_positive_return_ratio"),
        }
        for baseline, col in [
            ("entry_only_reclaim", "entry_only_lift"),
            ("matched_random_reclaim", "matched_random_lift"),
            ("volume_shock_next_open", "volume_shock_next_open_lift"),
            ("volume_shock_pullback_only", "volume_shock_pullback_lift"),
        ]:
            row[col] = net10 - _lookup(
                long_summary,
                gate.candidate,
                "signal_and_entry_gate",
                baseline,
                10,
                "net_expectancy",
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gate in GATES:
        mask = (
            pd.to_numeric(data["dynamic_all_rank"], errors="coerce").le(TOP_N)
            & _ensure_bool(data[gate.gate_col])
            & _ensure_bool(data["bullish_volume_shock_event"])
        )
        sample = data[mask]
        rows.append(
            {
                "candidate": gate.candidate,
                "role": gate.role,
                "signals": int(len(sample)),
                "neighbor_impulse_ratio_median": float(
                    pd.to_numeric(sample.get("neighbor_impulse_ratio"), errors="coerce").median()
                ),
                "cluster_positive_return_ratio_median": float(
                    pd.to_numeric(sample.get("cluster_positive_return_ratio"), errors="coerce").median()
                ),
                "leader_impulse_recent_rate": float(
                    _ensure_bool(sample.get("leader_impulse_recent", pd.Series(dtype=bool))).mean()
                )
                if len(sample)
                else np.nan,
                "symbol_lag_vs_neighbors_median": float(
                    pd.to_numeric(sample.get("symbol_lag_vs_neighbors"), errors="coerce").median()
                ),
            }
        )
    return pd.DataFrame(rows)


def _month_cap_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for gate in GATES:
        sample = trades[
            trades["candidate"].astype(str).eq(gate.candidate)
            & trades["baseline"].astype(str).eq("candidate_reclaim")
        ]
        if sample.empty:
            continue
        rows.append(
            {
                "candidate": gate.candidate,
                "net10": _metric(sample, 10),
                "net20": _metric(sample, 20),
                "month_cap35_net10": _month_cap_expectancy(sample, 10, 0.35),
                "month_cap35_net20": _month_cap_expectancy(sample, 20, 0.35),
                "ex_top_month_net10": _ex_top_month_net(sample, 10),
                "max_month_contribution": _max_contribution(sample, "month", 10),
            }
        )
    return pd.DataFrame(rows)


def _one_min_summary(data: pd.DataFrame, config: ExperimentConfig, report_root: Path) -> pd.DataFrame:
    candidates = [_candidate_for_gate(gate) for gate in GATES]
    symbols = sorted(data["symbol"].dropna().astype(str).unique())
    minute_bars = _load_1m_cache(config, symbols)
    if minute_bars.empty:
        return pd.DataFrame([{"status": "pending_1m_data"}])
    rows = []
    for candidate in candidates:
        signal_rows = data[
            (pd.to_numeric(data["dynamic_all_rank"], errors="coerce") <= candidate.universe_top_n)
            & _ensure_bool(data[candidate.market_gate])
            & _ensure_bool(data[candidate.event_col])
        ].copy()
        if signal_rows.empty:
            continue
        signal_rows["__v07b_signal"] = True
        context = signal_rows[
            ["exchange", "symbol", "feature_time", "btc_market_state", candidate.market_gate]
        ].drop_duplicates(["exchange", "symbol", "feature_time"])
        rule, resolver = _rule(candidate.exit_rule, config)
        for cost in ONE_MIN_COST_BPS:
            trades = simulate_1m_execution(
                signal_rows,
                minute_bars,
                "__v07b_signal",
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
            if trades.empty:
                continue
            exits = trades["exit_reason"].astype(str)
            rows.append(
                {
                    "status": "ok",
                    "candidate": candidate.candidate,
                    "market_gate": candidate.market_gate,
                    "cost_single_side_bps": cost,
                    "trades": int(len(trades)),
                    "net_expectancy_1m": float(pd.to_numeric(trades["net_return"], errors="coerce").mean()),
                    "tp_rate_1m": float(exits.str.startswith("tp").mean()),
                    "sl_rate_1m": float(exits.str.startswith("sl").mean()),
                    "timeout_rate_1m": float(exits.eq("max_hold").mean()),
                    "same_bar_ambiguity_1m": float(trades["unresolved_1m_same_bar"].fillna(False).mean()),
                }
            )
    return pd.DataFrame(rows) if rows else pd.DataFrame([{"status": "no_1m_fills"}])


def _write_notes(report_root: Path, gate_summary: pd.DataFrame, one_min: pd.DataFrame) -> None:
    lines = [
        "# v0.7B Graph Neighbor Confirmation",
        "",
        "Frozen MIR1 neighbor graph validation. No MIR1 parameter tuning.",
        "",
        "Graphs:",
        "- return_corr_30d: monthly as-of top-5 positive 30d return-correlation neighbors.",
        "- volume_impulse_cooccurrence_30d: monthly as-of top-5 neighbors that historically co-fired volume impulses.",
        "",
    ]
    if gate_summary.empty:
        lines.append("- No gate summary rows were produced.")
    else:
        best = gate_summary.sort_values("net20", ascending=False).iloc[0]
        raw = gate_summary[gate_summary["candidate"].astype(str).eq("B0_MIR1_raw")]
        raw_net20 = float(raw["net20"].iloc[0]) if not raw.empty else np.nan
        lines.append(
            f"- Best net20 gate: {best['candidate']} with trades={int(best['trades'])}, "
            f"net10={best['net10']:.4%}, net20={best['net20']:.4%}."
        )
        if pd.notna(raw_net20):
            lines.append(f"- Raw MIR1 net20 reference: {raw_net20:.4%}.")
    if not one_min.empty and "status" in one_min.columns:
        lines.append(f"- 1m execution status: {', '.join(sorted(one_min['status'].dropna().astype(str).unique()))}.")
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v07b_neighbor_graph(
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
    trades, signals, edges, feature_rows = _simulate_v07b_streaming(
        feature_path,
        rank30,
        rank90,
        regime,
        config,
        report_root,
    )
    gate_summary = _gate_summary(trades, signals)
    feature_summary = _feature_summary_from_rows(feature_rows)
    month_cap = _month_cap_summary(trades)
    one_min = pd.DataFrame(
        [
            {
                "status": "pending_batched_1m_followup",
                "reason": "v0.7B runs month-batched on this server to avoid OOM; run 1m on frozen top gates after 15m pass.",
            }
        ]
    )
    outputs = {
        "neighbor_graph_edges": report_root / "neighbor_graph_edges.csv",
        "neighbor_feature_summary": report_root / "neighbor_feature_summary.csv",
        "mir1_neighbor_gate_summary": report_root / "mir1_neighbor_gate_summary.csv",
        "leader_beta_summary": report_root / "leader_beta_summary.csv",
        "isolated_signal_negative_control": report_root / "isolated_signal_negative_control.csv",
        "month_cap_summary": report_root / "month_cap_summary.csv",
        "one_min_execution_summary": report_root / "one_min_execution_summary.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    edges.to_csv(outputs["neighbor_graph_edges"], index=False)
    feature_summary.to_csv(outputs["neighbor_feature_summary"], index=False)
    gate_summary.to_csv(outputs["mir1_neighbor_gate_summary"], index=False)
    gate_summary[gate_summary["candidate"].astype(str).eq("B3_leader_beta_lag")].to_csv(
        outputs["leader_beta_summary"],
        index=False,
    )
    gate_summary[gate_summary["negative_control"].fillna(False)].to_csv(
        outputs["isolated_signal_negative_control"],
        index=False,
    )
    month_cap.to_csv(outputs["month_cap_summary"], index=False)
    one_min.to_csv(outputs["one_min_execution_summary"], index=False)
    _write_notes(report_root, gate_summary, one_min)
    return outputs
