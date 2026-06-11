from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v06c import _build_regime_streaming, _rank_inputs
from pressure_graph.reports.v07a import RECLAIM_1, SIM_WINDOW_BARS
from pressure_graph.reports.v07a1 import (
    BASELINES,
    BaselineSpec,
    FrozenAtlasCandidate,
    _ex_top_month_net,
    _max_contribution,
    _metric,
    _month_cap_expectancy,
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
from pressure_graph.reports.v07b1 import _edge_neighbors, _leakage_row, _stable_seed


REPORT_ROOT = Path("reports/v0_7b2_neighbor_edge_attribution")
RANDOM_PERMUTATIONS = 50
SHUFFLED_PERMUTATIONS = 20
EDGE_THRESHOLD = 0.20
CORE_BASELINE = BASELINES[0]
REAL_BASELINES = BASELINES


@dataclass(frozen=True)
class EdgeControl:
    candidate: str
    role: str
    gate_col: str
    description: str
    negative_control: bool = False
    baselines: tuple[BaselineSpec, ...] = (CORE_BASELINE,)


CORE_CONTROLS = [
    EdgeControl(
        "raw_mir1",
        "raw_reference",
        "market_volume_impulse_density_high",
        "MIR1 raw market impulse density gate.",
    ),
    EdgeControl(
        "real_neighbor",
        "primary_edge",
        "gate_neighbor_impulse_high",
        "Real volume-impulse co-occurrence neighbor confirmation.",
        False,
        tuple(REAL_BASELINES),
    ),
    EdgeControl(
        "anti_neighbor",
        "negative_control",
        "gate_anti_neighbor_impulse_high",
        "Low co-occurrence anti-neighbor pseudo graph.",
        True,
    ),
    EdgeControl(
        "isolated_signal",
        "negative_control",
        "gate_isolated_signal",
        "MIR1 signal isolated from neighbor impulse and cluster breadth.",
        True,
    ),
    EdgeControl(
        "low_neighbor_impulse",
        "negative_control",
        "gate_low_neighbor_impulse",
        "MIR1 with low real-neighbor impulse.",
        True,
    ),
]

FUTURE_CONTROLS = [
    EdgeControl(
        "future_neighbor_1h",
        "future_audit",
        "gate_future_neighbor_1h_high",
        "Neighbor impulse in the next 1h. Audit only; uses future information.",
        True,
    ),
    EdgeControl(
        "future_neighbor_4h",
        "future_audit",
        "gate_future_neighbor_4h_high",
        "Neighbor impulse in the next 4h. Audit only; uses future information.",
        True,
    ),
]

PAST_CONTROLS = [
    EdgeControl("past_neighbor_lag0", "lead_lag", "gate_neighbor_impulse_high", "Same-bar neighbor impulse."),
    EdgeControl("past_neighbor_15m", "lead_lag", "gate_past_neighbor_15m_high", "Neighbor impulse in prior 15m."),
    EdgeControl("past_neighbor_1h", "lead_lag", "gate_past_neighbor_1h_high", "Neighbor impulse in prior 1h."),
    EdgeControl("past_neighbor_4h", "lead_lag", "gate_past_neighbor_4h_high", "Neighbor impulse in prior 4h."),
]

EDGE_BUCKET_CONTROLS = [
    EdgeControl(f"edge_weight_q{bucket}", "edge_weight_bucket", f"gate_edge_weight_q{bucket}", f"Real neighbor active edge weight Q{bucket}.")
    for bucket in range(1, 5)
]


def _candidate(control: EdgeControl) -> FrozenAtlasCandidate:
    return FrozenAtlasCandidate(
        control.candidate,
        TOP_N,
        control.gate_col,
        "bullish_volume_shock_event",
        RECLAIM_1,
        "vol_regime_fast",
        control.role,
        control.negative_control,
    )


def _liquidity_bucket(rank: float) -> str:
    if pd.isna(rank):
        return "unknown"
    if rank <= 10:
        return "rank_1_10"
    if rank <= 30:
        return "rank_11_30"
    if rank <= 50:
        return "rank_31_50"
    return "rank_gt_50"


def _future_window_any(state: pd.DataFrame, bars: int) -> pd.DataFrame:
    return (
        state.shift(-1)
        .iloc[::-1]
        .rolling(bars, min_periods=1)
        .max()
        .iloc[::-1]
        .fillna(False)
        .infer_objects(copy=False)
        .astype(bool)
    )


def _past_window_any(state: pd.DataFrame, bars: int) -> pd.DataFrame:
    return state.shift(1).rolling(bars, min_periods=1).max().fillna(False).infer_objects(copy=False).astype(bool)


def _hist_cooccurrence_scores(hist: pd.DataFrame, symbols: list[str]) -> dict[str, dict[str, float]]:
    if hist.empty:
        return {symbol: {} for symbol in symbols}
    pivot = hist.pivot_table(
        index="feature_time",
        columns="symbol",
        values="bullish_volume_shock_state",
        aggfunc="max",
        observed=True,
    )
    pivot = pivot[[symbol for symbol in symbols if symbol in pivot.columns]].fillna(False).astype("int8")
    if pivot.empty:
        return {symbol: {} for symbol in symbols}
    matrix = pivot.T.dot(pivot)
    source_counts = pivot.sum(axis=0).replace(0, np.nan)
    out: dict[str, dict[str, float]] = {}
    for source in symbols:
        if source not in matrix.columns:
            out[source] = {}
            continue
        scores = (matrix[source] / source_counts.get(source, np.nan)).drop(labels=[source], errors="ignore").dropna()
        out[source] = {str(symbol): float(value) for symbol, value in scores.items()}
    return out


def _anti_neighbors(
    symbols: list[str],
    scores: dict[str, dict[str, float]],
    real_neighbors: dict[str, list[str]],
    k: int = 5,
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for symbol in symbols:
        real = set(real_neighbors.get(symbol, []))
        candidates = [item for item in symbols if item != symbol and item not in real]
        ranked = sorted(candidates, key=lambda item: (scores.get(symbol, {}).get(item, 0.0), item))
        out[symbol] = ranked[: min(k, len(ranked))]
    return out


def _rank_bucket_map(sample: pd.DataFrame) -> dict[str, str]:
    ranks = (
        sample.groupby("symbol", observed=True)["dynamic_all_rank"]
        .first()
        .apply(lambda value: _liquidity_bucket(float(value)))
    )
    return {str(symbol): str(bucket) for symbol, bucket in ranks.items()}


def _density_random_neighbors(
    month_start: pd.Timestamp,
    symbols: list[str],
    rank_buckets: dict[str, str],
    permutation: int,
    k: int = 5,
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for symbol in symbols:
        bucket = rank_buckets.get(symbol, "unknown")
        same_bucket = [item for item in symbols if item != symbol and rank_buckets.get(item, "unknown") == bucket]
        choices = same_bucket if len(same_bucket) >= 2 else [item for item in symbols if item != symbol]
        if not choices:
            out[symbol] = []
            continue
        rng = np.random.default_rng(_stable_seed("density_random", month_start.isoformat(), permutation, symbol, bucket))
        take = min(k, len(choices))
        out[symbol] = list(rng.choice(choices, size=take, replace=False))
    return out


def _shuffled_neighbors(
    month_start: pd.Timestamp,
    symbols: list[str],
    real_neighbors: dict[str, list[str]],
    permutation: int,
) -> dict[str, list[str]]:
    if not symbols:
        return {}
    rng = np.random.default_rng(_stable_seed("shuffle_v07b2", month_start.isoformat(), permutation))
    shuffled = symbols.copy()
    rng.shuffle(shuffled)
    mapping = dict(zip(symbols, shuffled, strict=False))
    out: dict[str, list[str]] = {}
    for symbol in symbols:
        mapped = [mapping.get(item, item) for item in real_neighbors.get(symbol, [])]
        out[symbol] = [item for item in mapped if item != symbol]
    return out


def _ratio_from_map(
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


def _weighted_active_edge(
    state: pd.DataFrame,
    feature_time: pd.Series,
    neighbors: list[str],
    weights: dict[str, float],
) -> np.ndarray:
    usable = [item for item in neighbors if item in state.columns]
    if not usable:
        return np.zeros(len(feature_time), dtype=float)
    active = state[usable].reindex(feature_time).fillna(False).astype(bool)
    weight_series = pd.Series({item: weights.get(item, 0.0) for item in usable}, dtype="float64")
    weighted = active.astype(float).mul(weight_series, axis=1).sum(axis=1)
    counts = active.sum(axis=1).replace(0, np.nan)
    return (weighted / counts).fillna(0.0).to_numpy(dtype=float)


def _edge_weights(edges: pd.DataFrame, month_start: pd.Timestamp) -> dict[str, dict[str, float]]:
    sample = edges[
        edges["month_start"].eq(month_start)
        & edges["edge_type"].astype(str).eq("volume_impulse_cooccurrence_30d")
    ]
    out: dict[str, dict[str, float]] = {}
    for source, group in sample.groupby("source_symbol", observed=True, sort=False):
        out[str(source)] = {
            str(row.neighbor_symbol): float(row.edge_weight)
            for row in group.itertuples(index=False)
        }
    return out


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


def _add_attribution_features(data: pd.DataFrame, window: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    defaults = {
        "anti_neighbor_impulse_ratio": 0.0,
        "future_neighbor_impulse_ratio_1h": 0.0,
        "future_neighbor_impulse_ratio_4h": 0.0,
        "past_neighbor_impulse_ratio_15m": 0.0,
        "past_neighbor_impulse_ratio_1h": 0.0,
        "past_neighbor_impulse_ratio_4h": 0.0,
        "active_neighbor_edge_weight_mean": 0.0,
        "cluster_id": "",
    }
    for col, value in defaults.items():
        out[col] = value
    if out.empty:
        return _add_attribution_gates(out)
    for month_start, month_data in out.groupby("month_start", observed=True, sort=True):
        month_start = pd.Timestamp(month_start)
        sample = month_data[pd.to_numeric(month_data["dynamic_all_rank"], errors="coerce") <= TOP_N].copy()
        if sample.empty:
            continue
        symbols = sorted(sample["symbol"].dropna().astype(str).unique())
        hist = window[
            (pd.to_datetime(window["bar_open_time"], utc=True, errors="coerce") < month_start)
            & (pd.to_datetime(window["bar_open_time"], utc=True, errors="coerce") >= month_start - pd.Timedelta(days=GRAPH_LOOKBACK_DAYS))
            & window["symbol"].astype(str).isin(symbols)
        ].copy()
        state = sample.pivot_table(
            index="feature_time",
            columns="symbol",
            values="bullish_volume_shock_state",
            aggfunc="max",
            observed=True,
        ).fillna(False).infer_objects(copy=False).astype(bool)
        real_neighbors = _edge_neighbors(edges, month_start, "volume_impulse_cooccurrence_30d")
        anti_neighbors = _anti_neighbors(symbols, _hist_cooccurrence_scores(hist, symbols), real_neighbors)
        edge_weights = _edge_weights(edges, month_start)
        future_1h = _future_window_any(state, 4)
        future_4h = _future_window_any(state, 16)
        past_15m = _past_window_any(state, 1)
        past_1h = _past_window_any(state, 4)
        past_4h = _past_window_any(state, 16)
        for symbol, group in sample.groupby("symbol", observed=True, sort=False):
            symbol = str(symbol)
            feature_time = pd.to_datetime(group["feature_time"], utc=True, errors="coerce")
            real = real_neighbors.get(symbol, [])
            out.loc[group.index, "anti_neighbor_impulse_ratio"] = _ratio_from_map(state, feature_time, anti_neighbors, symbol)
            out.loc[group.index, "future_neighbor_impulse_ratio_1h"] = _ratio_from_map(future_1h, feature_time, real_neighbors, symbol)
            out.loc[group.index, "future_neighbor_impulse_ratio_4h"] = _ratio_from_map(future_4h, feature_time, real_neighbors, symbol)
            out.loc[group.index, "past_neighbor_impulse_ratio_15m"] = _ratio_from_map(past_15m, feature_time, real_neighbors, symbol)
            out.loc[group.index, "past_neighbor_impulse_ratio_1h"] = _ratio_from_map(past_1h, feature_time, real_neighbors, symbol)
            out.loc[group.index, "past_neighbor_impulse_ratio_4h"] = _ratio_from_map(past_4h, feature_time, real_neighbors, symbol)
            out.loc[group.index, "active_neighbor_edge_weight_mean"] = _weighted_active_edge(
                state,
                feature_time,
                real,
                edge_weights.get(symbol, {}),
            )
        cluster_map = _top1_cluster_map(edges, month_start, symbols)
        out.loc[sample.index, "cluster_id"] = sample["symbol"].astype(str).map(cluster_map).fillna("")
    return _add_attribution_gates(out)


def _add_attribution_gates(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    market = _ensure_bool(out.get("market_volume_impulse_density_high", pd.Series(False, index=out.index)))
    out["gate_anti_neighbor_impulse_high"] = market & (
        pd.to_numeric(out["anti_neighbor_impulse_ratio"], errors="coerce").fillna(0.0) >= EDGE_THRESHOLD
    )
    out["gate_future_neighbor_1h_high"] = market & (
        pd.to_numeric(out["future_neighbor_impulse_ratio_1h"], errors="coerce").fillna(0.0) >= EDGE_THRESHOLD
    )
    out["gate_future_neighbor_4h_high"] = market & (
        pd.to_numeric(out["future_neighbor_impulse_ratio_4h"], errors="coerce").fillna(0.0) >= EDGE_THRESHOLD
    )
    out["gate_past_neighbor_15m_high"] = market & (
        pd.to_numeric(out["past_neighbor_impulse_ratio_15m"], errors="coerce").fillna(0.0) >= EDGE_THRESHOLD
    )
    out["gate_past_neighbor_1h_high"] = market & (
        pd.to_numeric(out["past_neighbor_impulse_ratio_1h"], errors="coerce").fillna(0.0) >= EDGE_THRESHOLD
    )
    out["gate_past_neighbor_4h_high"] = market & (
        pd.to_numeric(out["past_neighbor_impulse_ratio_4h"], errors="coerce").fillna(0.0) >= EDGE_THRESHOLD
    )
    out["edge_weight_bucket"] = pd.Series(pd.NA, index=out.index, dtype="object")
    active = _ensure_bool(out.get("gate_neighbor_impulse_high", pd.Series(False, index=out.index)))
    weights = pd.to_numeric(out["active_neighbor_edge_weight_mean"], errors="coerce")
    valid = active & weights.notna() & (weights > 0)
    if valid.sum() >= 4:
        try:
            buckets = pd.qcut(weights[valid], 4, labels=["q1", "q2", "q3", "q4"], duplicates="drop")
            out.loc[valid, "edge_weight_bucket"] = buckets.astype(str)
        except ValueError:
            out.loc[valid, "edge_weight_bucket"] = "q1"
    for bucket in range(1, 5):
        out[f"gate_edge_weight_q{bucket}"] = active & out["edge_weight_bucket"].astype(str).eq(f"q{bucket}")
    return out


def _add_neighbor_gate_from_map(
    data: pd.DataFrame,
    neighbor_map: dict[str, list[str]],
    gate_col: str,
) -> pd.DataFrame:
    out = data.copy()
    out[gate_col] = False
    sample = out[pd.to_numeric(out["dynamic_all_rank"], errors="coerce") <= TOP_N].copy()
    if sample.empty:
        return out
    state = sample.pivot_table(
        index="feature_time",
        columns="symbol",
        values="bullish_volume_shock_state",
        aggfunc="max",
        observed=True,
    ).fillna(False).infer_objects(copy=False).astype(bool)
    ratios = pd.Series(0.0, index=sample.index, dtype="float64")
    for symbol, group in sample.groupby("symbol", observed=True, sort=False):
        symbol = str(symbol)
        feature_time = pd.to_datetime(group["feature_time"], utc=True, errors="coerce")
        ratios.loc[group.index] = _ratio_from_map(state, feature_time, neighbor_map, symbol)
    market = _ensure_bool(sample.get("market_volume_impulse_density_high", pd.Series(False, index=sample.index)))
    out.loc[sample.index, gate_col] = (market & (ratios >= EDGE_THRESHOLD)).to_numpy(dtype=bool)
    return out


def _simulate_control(
    data: pd.DataFrame,
    control: EdgeControl,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_rows: list[dict[str, object]] = []
    frames = []
    candidate = _candidate(control)
    for baseline in control.baselines:
        trades, signal_n, pre_entry_gate_n = _simulate_variant(
            data,
            candidate,
            "signal_and_entry_gate",
            baseline,
            config,
        )
        signal_rows.append(
            {
                "candidate": control.candidate,
                "gate_col": control.gate_col,
                "gate_mode": "signal_and_entry_gate",
                "baseline": baseline.baseline,
                "signals": signal_n,
                "pre_entry_gate_trades": pre_entry_gate_n,
            }
        )
        if not trades.empty:
            trades["control_role"] = control.role
            trades["control_description"] = control.description
            frames.append(_attach_context(trades, data))
    trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    signals = pd.DataFrame(signal_rows)
    return trades, signals


def _context_cols(data: pd.DataFrame) -> list[str]:
    cols = [
        "exchange",
        "symbol",
        "feature_time",
        "volume_impulse_density",
        "dynamic_all_rank",
        "volume_z_1h",
        "close_location_value",
        "upper_wick_ratio",
        "neighbor_impulse_ratio",
        "anti_neighbor_impulse_ratio",
        "cluster_positive_return_ratio",
        "active_neighbor_edge_weight_mean",
        "edge_weight_bucket",
        "cluster_id",
    ]
    return [col for col in cols if col in data.columns]


def _attach_context(trades: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    cols = _context_cols(data)
    context = data[cols].drop_duplicates(["exchange", "symbol", "feature_time"])
    out = trades.drop(
        columns=[col for col in cols if col not in {"exchange", "symbol", "feature_time"}],
        errors="ignore",
    )
    out = out.merge(
        context,
        left_on=["exchange", "symbol", "signal_time"],
        right_on=["exchange", "symbol", "feature_time"],
        how="left",
    ).drop(columns=["feature_time"], errors="ignore")
    return out


def _summarize_candidates(trades: pd.DataFrame, signals: pd.DataFrame | None = None) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    long_summary = _summary_long(trades, signals if signals is not None else pd.DataFrame())
    rows = []
    for candidate, group in trades.groupby("candidate", sort=False, dropna=False):
        sample = group[group["baseline"].astype(str).eq("candidate_reclaim")]
        sample10 = sample[pd.to_numeric(sample["cost_single_side_bps"], errors="coerce").eq(10.0)]
        if sample10.empty:
            continue
        net10 = _metric(sample, 10)
        rows.append(
            {
                "candidate": candidate,
                "trades": int(len(sample10)),
                "net10": net10,
                "net20": _metric(sample, 20),
                "net30": _metric(sample, 30),
                "net50": _metric(sample, 50),
                "entry_only_lift": net10
                - _lookup_optional(long_summary, candidate, "entry_only_reclaim"),
                "matched_random_lift": net10
                - _lookup_optional(long_summary, candidate, "matched_random_reclaim"),
                "ex_top_month_net10": _ex_top_month_net(sample, 10),
                "month_cap35_net20": _month_cap_expectancy(sample, 20, 0.35),
                "max_month_contribution": _max_contribution(sample, "month", 10),
                "max_symbol_contribution": _max_contribution(sample, "symbol", 10),
                "max_cluster_contribution": _max_contribution(sample, "cluster_id", 10),
            }
        )
    return pd.DataFrame(rows)


def _lookup_optional(summary: pd.DataFrame, candidate: str, baseline: str) -> float:
    if summary.empty:
        return np.nan
    sample = summary[
        summary["candidate"].astype(str).eq(str(candidate))
        & summary["baseline"].astype(str).eq(str(baseline))
        & pd.to_numeric(summary["cost_single_side_bps"], errors="coerce").eq(10.0)
    ]
    if sample.empty:
        return np.nan
    return float(pd.to_numeric(sample.iloc[0]["net_expectancy"], errors="coerce"))


def _aggregate_permutations(summary: pd.DataFrame, family: str) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for metric in ["net10", "net20", "net50", "month_cap35_net20"]:
        values = pd.to_numeric(summary[metric], errors="coerce").dropna()
        rows.append(
            {
                "family": family,
                "metric": metric,
                "count": int(len(values)),
                "mean": float(values.mean()) if len(values) else np.nan,
                "median": float(values.median()) if len(values) else np.nan,
                "p75": float(values.quantile(0.75)) if len(values) else np.nan,
                "p90": float(values.quantile(0.90)) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _edge_control_summary(
    core: pd.DataFrame,
    random_summary: pd.DataFrame,
    shuffled_summary: pd.DataFrame,
) -> pd.DataFrame:
    out = core.copy()
    real = out[out["candidate"].astype(str).eq("real_neighbor")]
    if real.empty:
        return out
    real_net10 = float(real.iloc[0]["net10"])
    random_net = pd.to_numeric(random_summary.get("net10", pd.Series(dtype=float)), errors="coerce").dropna()
    shuffled_net = pd.to_numeric(shuffled_summary.get("net10", pd.Series(dtype=float)), errors="coerce").dropna()
    random_mean = float(random_net.mean()) if len(random_net) else np.nan
    random_median = float(random_net.median()) if len(random_net) else np.nan
    random_p75 = float(random_net.quantile(0.75)) if len(random_net) else np.nan
    random_p90 = float(random_net.quantile(0.90)) if len(random_net) else np.nan
    shuffled_mean = float(shuffled_net.mean()) if len(shuffled_net) else np.nan
    out["real_vs_random_mean_lift"] = real_net10 - random_mean
    out["real_vs_random_median_lift"] = real_net10 - random_median
    out["real_vs_random_p75_lift"] = real_net10 - random_p75
    out["real_vs_random_p90_lift"] = real_net10 - random_p90
    out["real_percentile_vs_random"] = float((random_net <= real_net10).mean()) if len(random_net) else np.nan
    out["real_vs_shuffled_mean_lift"] = real_net10 - shuffled_mean
    for control in ["anti_neighbor", "isolated_signal", "low_neighbor_impulse"]:
        sample = out[out["candidate"].astype(str).eq(control)]
        out[f"real_vs_{control}_lift"] = real_net10 - float(sample.iloc[0]["net10"]) if not sample.empty else np.nan
    return out


def _portfolio_diagnostic(trades: pd.DataFrame) -> pd.DataFrame:
    sample = trades[
        trades["candidate"].astype(str).eq("real_neighbor")
        & trades["baseline"].astype(str).eq("candidate_reclaim")
        & pd.to_numeric(trades["cost_single_side_bps"], errors="coerce").isin([10.0, 20.0])
    ].copy()
    if sample.empty:
        return pd.DataFrame()
    sample["entry_time"] = pd.to_datetime(sample["entry_time"], utc=True, errors="coerce")
    sample["exit_time"] = pd.to_datetime(sample["exit_time"], utc=True, errors="coerce")
    sample["rank_first_come_first_served"] = -sample["entry_time"].astype("int64")
    sample["rank_market_volume_impulse_density"] = _numeric_context(sample, "volume_impulse_density", -np.inf)
    sample["rank_local_volume_shock_strength"] = _numeric_context(sample, "volume_z_1h", -np.inf)
    close_loc = _numeric_context(sample, "close_location_value", 0.0)
    upper_wick = _numeric_context(sample, "upper_wick_ratio", 1.0)
    sample["rank_reclaim_quality"] = close_loc - upper_wick
    sample["rank_liquidity"] = -_numeric_context(sample, "dynamic_all_rank", 999.0)
    sample["rank_edge_weight"] = _numeric_context(sample, "active_neighbor_edge_weight_mean", -np.inf)
    hashed = pd.util.hash_pandas_object(sample[["symbol", "signal_time"]].astype(str), index=False)
    sample["rank_random"] = (hashed % 10_000_000).astype(float)
    rank_cols = [
        "rank_first_come_first_served",
        "rank_market_volume_impulse_density",
        "rank_local_volume_shock_strength",
        "rank_reclaim_quality",
        "rank_liquidity",
        "rank_edge_weight",
        "rank_random",
    ]
    rows = []
    for cost, cost_group in sample.groupby("cost_single_side_bps", sort=False, dropna=False):
        for rank_col in rank_cols:
            group = cost_group.sort_values(["entry_time", rank_col, "symbol"], ascending=[True, False, True]).reset_index(drop=True)
            for max_positions in [1, 3, 5, 10]:
                active: list[pd.Timestamp] = []
                selected = []
                skipped = 0
                max_concurrent = 0
                for row in group.itertuples(index=False):
                    entry_time = pd.Timestamp(row.entry_time)
                    active = [exit_time for exit_time in active if exit_time > entry_time]
                    if len(active) >= max_positions:
                        skipped += 1
                        continue
                    selected.append(row)
                    active.append(pd.Timestamp(row.exit_time))
                    max_concurrent = max(max_concurrent, len(active))
                net = pd.Series([float(getattr(item, "net_return", np.nan)) for item in selected], dtype="float64")
                equity = net.cumsum()
                dd = equity - equity.cummax()
                rows.append(
                    {
                        "ranking": rank_col.removeprefix("rank_"),
                        "cost_single_side_bps": cost,
                        "max_positions": max_positions,
                        "selected_trades": int(len(selected)),
                        "skipped_trades": int(skipped),
                        "net_expectancy": float(net.mean()) if len(net) else np.nan,
                        "total_net": float(net.sum()) if len(net) else 0.0,
                        "max_drawdown_proxy": float(dd.min()) if len(dd) else np.nan,
                        "max_concurrent_positions": int(max_concurrent),
                    }
                )
    return pd.DataFrame(rows)


def _numeric_context(data: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col not in data.columns:
        return pd.Series(default, index=data.index, dtype="float64")
    return pd.to_numeric(data[col], errors="coerce").fillna(default)


def _month_setup(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    regime: pd.DataFrame,
    config: ExperimentConfig,
    month_start: pd.Timestamp,
    next_month: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    symbols = _month_symbols(rank30, month_start)
    if not symbols:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []
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
        return pd.DataFrame(), pd.DataFrame(), window, symbols
    available_months = {
        pd.Timestamp(item)
        for item in pd.to_datetime(window["month_start"], utc=True, errors="coerce").dropna().unique()
    }
    months_for_edges = [month_start]
    if next_month in available_months:
        months_for_edges.append(next_month)
    edges = _build_neighbor_graph_edges_for_months(window, months_for_edges)
    sim_window = window[
        (pd.to_datetime(window["bar_open_time"], utc=True, errors="coerce") >= month_start)
        & (pd.to_datetime(window["bar_open_time"], utc=True, errors="coerce") < window_end)
    ].copy()
    sim_window = add_neighbor_graph_features(sim_window, edges)
    sim_window = _add_attribution_features(sim_window, window, edges)
    sim_window = _disable_out_of_month_events(sim_window, month_start, next_month)
    return sim_window, edges, window, symbols


def _stream_reports(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    regime: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trade_path = report_root / "_v07b2_trades_tmp.csv"
    signal_path = report_root / "_v07b2_signals_tmp.csv"
    if trade_path.exists():
        trade_path.unlink()
    if signal_path.exists():
        signal_path.unlink()
    wrote_trades = False
    wrote_signals = False
    edge_frames: list[pd.DataFrame] = []
    leakage_rows: list[dict[str, object]] = []
    months = sorted(
        pd.to_datetime(rank30["month_start"], utc=True, errors="coerce")
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    for idx, month_start in enumerate(months, start=1):
        month_start = pd.Timestamp(month_start)
        next_month = pd.Timestamp(months[idx]) if idx < len(months) else month_start + pd.DateOffset(months=1)
        sim_window, edges, window, symbols = _month_setup(
            feature_path,
            rank30,
            rank90,
            regime,
            config,
            month_start,
            next_month,
        )
        if sim_window.empty:
            continue
        leakage_rows.append(_leakage_row(month_start, symbols, window, edges[edges["month_start"].eq(month_start)]))
        if not edges.empty:
            edge_frames.append(edges[edges["month_start"].eq(month_start)].copy())
        trade_frames: list[pd.DataFrame] = []
        signal_frames: list[pd.DataFrame] = []
        for control in [*CORE_CONTROLS, *FUTURE_CONTROLS, *PAST_CONTROLS, *EDGE_BUCKET_CONTROLS]:
            trades, signals = _simulate_control(sim_window, control, config)
            if not trades.empty:
                trade_frames.append(trades)
            if not signals.empty:
                signal_frames.append(signals)
        sample = sim_window[pd.to_numeric(sim_window["dynamic_all_rank"], errors="coerce") <= TOP_N].copy()
        symbols_month = sorted(sample["symbol"].dropna().astype(str).unique())
        rank_buckets = _rank_bucket_map(sample)
        real_neighbors = _edge_neighbors(edges, month_start, "volume_impulse_cooccurrence_30d")
        for permutation in range(RANDOM_PERMUTATIONS):
            gate_col = "__density_random_gate"
            local = _add_neighbor_gate_from_map(
                sim_window,
                _density_random_neighbors(month_start, symbols_month, rank_buckets, permutation),
                gate_col,
            )
            control = EdgeControl(
                f"density_random_p{permutation:03d}",
                "density_matched_random",
                gate_col,
                "Density/liquidity matched random neighbor permutation.",
                True,
            )
            trades, signals = _simulate_control(local, control, config)
            if not trades.empty:
                trades["permutation"] = permutation
                trade_frames.append(trades)
            if not signals.empty:
                signals["permutation"] = permutation
                signal_frames.append(signals)
            del local
        for permutation in range(SHUFFLED_PERMUTATIONS):
            gate_col = "__shuffled_neighbor_gate"
            local = _add_neighbor_gate_from_map(
                sim_window,
                _shuffled_neighbors(month_start, symbols_month, real_neighbors, permutation),
                gate_col,
            )
            control = EdgeControl(
                f"shuffled_neighbor_p{permutation:03d}",
                "shuffled_neighbor",
                gate_col,
                "Within-month shuffled neighbor mapping.",
                True,
            )
            trades, signals = _simulate_control(local, control, config)
            if not trades.empty:
                trades["permutation"] = permutation
                trade_frames.append(trades)
            if not signals.empty:
                signals["permutation"] = permutation
                signal_frames.append(signals)
            del local
        month_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
        if not month_trades.empty:
            month_trades.to_csv(trade_path, mode="a", header=not wrote_trades, index=False)
            wrote_trades = True
        month_signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
        if not month_signals.empty:
            month_signals.to_csv(signal_path, mode="a", header=not wrote_signals, index=False)
            wrote_signals = True
        print(f"v0.7B.2 month {idx}/{len(months)} {month_start:%Y-%m} symbols={len(symbols)}", flush=True)
        del sim_window, edges, trade_frames, signal_frames, month_trades, month_signals
        gc.collect()
    trades = pd.read_csv(trade_path, low_memory=False) if wrote_trades else pd.DataFrame()
    signals = pd.read_csv(signal_path, low_memory=False) if wrote_signals else pd.DataFrame()
    for path in [trade_path, signal_path]:
        if path.exists():
            path.unlink()
    edges = pd.concat(edge_frames, ignore_index=True).drop_duplicates() if edge_frames else pd.DataFrame()
    leakage = pd.DataFrame(leakage_rows)
    return trades, signals, edges, leakage


def _candidate_family(candidate: str) -> str:
    if candidate.startswith("density_random_p"):
        return "density_matched_random"
    if candidate.startswith("shuffled_neighbor_p"):
        return "shuffled_neighbor"
    if candidate.startswith("future_neighbor"):
        return "future_neighbor"
    if candidate.startswith("past_neighbor"):
        return "past_neighbor"
    if candidate.startswith("edge_weight_q"):
        return "edge_weight_bucket"
    return str(candidate)


def _write_notes(
    report_root: Path,
    edge_summary: pd.DataFrame,
    random_summary: pd.DataFrame,
    shuffled_summary: pd.DataFrame,
    monotonicity: pd.DataFrame,
) -> None:
    lines = [
        "# v0.7B.2 Neighbor Edge Attribution",
        "",
        "Purpose: attribute whether NIR1 is true neighbor-edge alpha or a market-density proxy.",
        "",
    ]
    real = edge_summary[edge_summary["candidate"].astype(str).eq("real_neighbor")]
    if not real.empty:
        row = real.iloc[0]
        lines.append(f"- Real neighbor: trades={int(row['trades'])}, net10={row['net10']:.4%}, net20={row['net20']:.4%}.")
        lines.append(f"- Real percentile vs density-matched random: {row.get('real_percentile_vs_random', np.nan):.2%}.")
    if not random_summary.empty:
        lines.append(
            f"- Density-matched random net10 mean={random_summary['net10'].mean():.4%}, "
            f"p75={random_summary['net10'].quantile(0.75):.4%}, p90={random_summary['net10'].quantile(0.90):.4%}."
        )
    if not shuffled_summary.empty:
        lines.append(
            f"- Shuffled neighbor net10 mean={shuffled_summary['net10'].mean():.4%}, "
            f"p75={shuffled_summary['net10'].quantile(0.75):.4%}."
        )
    if not monotonicity.empty:
        ordered = monotonicity[monotonicity["candidate"].astype(str).str.startswith("edge_weight_q")]
        if not ordered.empty:
            lines.append("- Edge-weight buckets: " + ", ".join(f"{r.candidate}={r.net10:.4%}" for r in ordered.itertuples()))
    lines.extend(
        [
            "",
            "Interpretation rule:",
            "- Promote only if real neighbor beats density-matched random median/p75, shuffled controls, anti-neighbor, and isolated signal, with positive month-capped edge.",
            "- If random/shuffled remain close to real, keep MIR1 primary and treat NIR1 as market-density-enhanced rather than neighbor-edge alpha.",
        ]
    )
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v07b2_neighbor_edge_attribution(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path = REPORT_ROOT,
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    rank30, rank90, _ = _rank_inputs(feature_path, instruments, config)
    symbols = sorted(
        rank30[pd.to_numeric(rank30["dynamic_all_rank"], errors="coerce") <= TOP_N]["symbol"]
        .dropna()
        .astype(str)
        .unique()
    )
    regime = _build_regime_streaming(feature_path, rank30, rank90, symbols, config)
    trades, signals, edges, leakage = _stream_reports(feature_path, rank30, rank90, regime, config, report_root)
    signals_grouped = (
        signals.groupby(["candidate", "gate_mode", "baseline"], as_index=False, sort=False, dropna=False)
        .agg(signals=("signals", "sum"), pre_entry_gate_trades=("pre_entry_gate_trades", "sum"))
        if not signals.empty
        else pd.DataFrame()
    )
    summary = _summarize_candidates(trades, signals_grouped)
    summary["family"] = summary["candidate"].map(_candidate_family)
    random_summary = summary[summary["family"].eq("density_matched_random")].copy()
    shuffled_summary = summary[summary["family"].eq("shuffled_neighbor")].copy()
    core_summary = summary[
        summary["candidate"].isin([control.candidate for control in CORE_CONTROLS])
    ].copy()
    edge_summary = _edge_control_summary(core_summary, random_summary, shuffled_summary)
    future = summary[summary["family"].eq("future_neighbor")].copy()
    past = summary[summary["family"].eq("past_neighbor")].copy()
    monotonicity = summary[summary["family"].eq("edge_weight_bucket")].copy()
    anti = summary[summary["candidate"].eq("anti_neighbor")].copy()
    isolated = summary[summary["candidate"].eq("isolated_signal")].copy()
    portfolio = _portfolio_diagnostic(trades)
    outputs = {
        "edge_control_summary": report_root / "edge_control_summary.csv",
        "density_matched_random_permutation": report_root / "density_matched_random_permutation.csv",
        "density_matched_random_distribution": report_root / "density_matched_random_distribution.csv",
        "shuffled_neighbor_summary": report_root / "shuffled_neighbor_summary.csv",
        "anti_neighbor_negative_control": report_root / "anti_neighbor_negative_control.csv",
        "isolated_signal_control": report_root / "isolated_signal_control.csv",
        "future_neighbor_leakage_audit": report_root / "future_neighbor_leakage_audit.csv",
        "past_neighbor_lead_lag": report_root / "past_neighbor_lead_lag.csv",
        "edge_weight_monotonicity": report_root / "edge_weight_monotonicity.csv",
        "portfolio_concurrency_diagnostic": report_root / "portfolio_concurrency_diagnostic.csv",
        "neighbor_graph_edges": report_root / "neighbor_graph_edges.csv",
        "leakage_audit": report_root / "leakage_audit.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    edge_summary.to_csv(outputs["edge_control_summary"], index=False)
    random_summary.to_csv(outputs["density_matched_random_permutation"], index=False)
    _aggregate_permutations(random_summary, "density_matched_random").to_csv(
        outputs["density_matched_random_distribution"],
        index=False,
    )
    shuffled_summary.to_csv(outputs["shuffled_neighbor_summary"], index=False)
    anti.to_csv(outputs["anti_neighbor_negative_control"], index=False)
    isolated.to_csv(outputs["isolated_signal_control"], index=False)
    future.to_csv(outputs["future_neighbor_leakage_audit"], index=False)
    past.to_csv(outputs["past_neighbor_lead_lag"], index=False)
    monotonicity.to_csv(outputs["edge_weight_monotonicity"], index=False)
    portfolio.to_csv(outputs["portfolio_concurrency_diagnostic"], index=False)
    edges.to_csv(outputs["neighbor_graph_edges"], index=False)
    leakage.to_csv(outputs["leakage_audit"], index=False)
    _write_notes(report_root, edge_summary, random_summary, shuffled_summary, monotonicity)
    return outputs
