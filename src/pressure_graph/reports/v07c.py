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
    _disable_out_of_month_events,
    _ensure_bool,
    _load_month_window,
    _month_symbols,
)
from pressure_graph.reports.v07b1 import _stable_seed
from pressure_graph.reports.v07b2 import _aggregate_permutations, _numeric_context


REPORT_ROOT = Path("reports/v0_7c_leader_beta_rotation")
EDGE_TOP_K = 5
LEADER_RECENT_BARS = 16
FOLLOW_BARS = 16
EDGE_SHRINKAGE_K = 30
EDGE_MIN_SAMPLES = 3
RANDOM_PERMUTATIONS = 50
SHUFFLED_PERMUTATIONS = 20
CORE_BASELINE = BASELINES[0]


@dataclass(frozen=True)
class LeaderControl:
    candidate: str
    role: str
    gate_col: str
    description: str
    negative_control: bool = False
    baselines: tuple[BaselineSpec, ...] = (CORE_BASELINE,)


CORE_CONTROLS = [
    LeaderControl(
        "B0_MIR1_raw",
        "raw_reference",
        "market_volume_impulse_density_high",
        "MIR1 raw market impulse-density gate.",
        False,
        tuple(BASELINES),
    ),
    LeaderControl(
        "LBR1_directed_leader_beta",
        "directed_leader_beta",
        "gate_directed_leader_beta",
        "Real directed leader impulse in prior 4h, beta not extended, MIR1 local reclaim.",
        False,
        tuple(BASELINES),
    ),
    LeaderControl(
        "LBR2_leader_return_beta_laggard",
        "leader_return_laggard",
        "gate_leader_return_beta_laggard",
        "Leader already strong while beta is still lagging, then beta local reclaim.",
    ),
    LeaderControl(
        "LBR4_dispersion_shadow",
        "dispersion_shadow",
        "gate_dispersion_low_edge_beta",
        "Low-edge/anti-cluster leader impulse while beta is not extended; dispersion hypothesis.",
        True,
    ),
    LeaderControl(
        "low_edge_leader_control",
        "low_edge_control",
        "gate_low_edge_leader_beta",
        "Low directed edge-weight pseudo leader impulse control.",
        True,
    ),
    LeaderControl(
        "beta_already_extended_control",
        "negative_control",
        "gate_beta_already_extended",
        "Real leader impulse, but beta already extended.",
        True,
    ),
    LeaderControl(
        "no_leader_impulse_control",
        "negative_control",
        "gate_no_directed_leader_impulse",
        "MIR1 local event with no directed leader impulse.",
        True,
    ),
]

FUTURE_CONTROLS = [
    LeaderControl(
        "future_leader_1h",
        "future_audit",
        "gate_future_leader_1h",
        "Future leader impulse in next 1h. Audit only; uses future information.",
        True,
    ),
    LeaderControl(
        "future_leader_4h",
        "future_audit",
        "gate_future_leader_4h",
        "Future leader impulse in next 4h. Audit only; uses future information.",
        True,
    ),
]

EDGE_BUCKET_CONTROLS = [
    LeaderControl(
        f"edge_weight_q{bucket}",
        "edge_weight_bucket",
        f"gate_edge_weight_q{bucket}",
        f"Directed active leader edge-weight Q{bucket}.",
    )
    for bucket in range(1, 5)
]


def _candidate(control: LeaderControl) -> FrozenAtlasCandidate:
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


def _future_close_return(close: pd.DataFrame, bars: int) -> pd.DataFrame:
    return close.shift(-bars) / close - 1.0


def _future_any(state: pd.DataFrame, bars: int) -> pd.DataFrame:
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


def _past_any(state: pd.DataFrame, bars: int) -> pd.DataFrame:
    return state.shift(1).rolling(bars, min_periods=1).max().fillna(False).infer_objects(copy=False).astype(bool)


def _build_directed_edges(month_start: pd.Timestamp, hist: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if hist.empty or len(symbols) < 3:
        return pd.DataFrame()
    hist = hist.copy()
    hist["feature_time"] = pd.to_datetime(hist["feature_time"], utc=True, errors="coerce")
    hist = hist[hist["feature_time"] <= month_start - pd.Timedelta(minutes=15 * FOLLOW_BARS)].copy()
    if hist.empty:
        return pd.DataFrame()
    event = hist.pivot_table(
        index="feature_time",
        columns="symbol",
        values="bullish_volume_shock_event",
        aggfunc="max",
        observed=True,
    )
    close = hist.pivot_table(index="feature_time", columns="symbol", values="close", aggfunc="last", observed=True)
    event = event[[symbol for symbol in symbols if symbol in event.columns]].fillna(False).astype(bool)
    close = close[[symbol for symbol in symbols if symbol in close.columns]]
    if event.empty or close.empty:
        return pd.DataFrame()
    future_ret = _future_close_return(close, FOLLOW_BARS)
    future_impulse = _future_any(event, FOLLOW_BARS)
    rows: list[dict[str, object]] = []
    for leader in event.columns:
        leader_mask = event[leader].fillna(False)
        if int(leader_mask.sum()) < EDGE_MIN_SAMPLES:
            continue
        for beta in event.columns:
            if beta == leader:
                continue
            beta_ret = pd.to_numeric(future_ret.get(beta), errors="coerce")
            beta_impulse = future_impulse.get(beta, pd.Series(False, index=event.index)).fillna(False).astype(bool)
            sample_ret = beta_ret[leader_mask].dropna()
            if len(sample_ret) < EDGE_MIN_SAMPLES:
                continue
            sample_impulse = beta_impulse.reindex(sample_ret.index).fillna(False)
            success = (sample_ret > 0.01) | sample_impulse
            n = int(len(sample_ret))
            success_rate = float(success.mean())
            avg_follow_return = float(sample_ret.mean())
            raw_weight = success_rate * max(avg_follow_return, 0.0)
            adjusted = raw_weight * n / (n + EDGE_SHRINKAGE_K)
            if adjusted <= 0:
                continue
            rows.append(
                {
                    "month_start": month_start,
                    "leader_symbol": str(leader),
                    "beta_symbol": str(beta),
                    "edge_type": "directed_lead_lag_30d",
                    "sample_n": n,
                    "follow_success_rate": success_rate,
                    "avg_follow_return_4h": avg_follow_return,
                    "raw_edge_weight": raw_weight,
                    "edge_weight": float(adjusted),
                    "lookback_days": GRAPH_LOOKBACK_DAYS,
                    "follow_window_bars": FOLLOW_BARS,
                    "shrinkage_k": EDGE_SHRINKAGE_K,
                }
            )
    edges = pd.DataFrame(rows)
    if edges.empty:
        return edges
    edges = edges.sort_values(["beta_symbol", "edge_weight"], ascending=[True, False])
    edges["edge_rank"] = edges.groupby("beta_symbol", sort=False, observed=True).cumcount() + 1
    return edges


def _top_leader_map(edges: pd.DataFrame, k: int = EDGE_TOP_K) -> dict[str, list[str]]:
    if edges.empty:
        return {}
    sample = edges[edges["edge_rank"].le(k)].sort_values(["beta_symbol", "edge_rank"])
    out: dict[str, list[str]] = {}
    for beta, group in sample.groupby("beta_symbol", sort=False, dropna=False):
        out[str(beta)] = group["leader_symbol"].dropna().astype(str).tolist()
    return out


def _low_edge_leader_map(edges: pd.DataFrame, symbols: list[str], k: int = EDGE_TOP_K) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {symbol: [] for symbol in symbols}
    if edges.empty:
        return out
    for beta, group in edges.groupby("beta_symbol", sort=False, dropna=False):
        sample = group.sort_values("edge_weight", ascending=True)
        out[str(beta)] = sample["leader_symbol"].dropna().astype(str).head(k).tolist()
    return out


def _edge_weight_map(edges: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if edges.empty:
        return out
    for beta, group in edges.groupby("beta_symbol", sort=False, dropna=False):
        out[str(beta)] = {
            str(row.leader_symbol): float(row.edge_weight)
            for row in group.itertuples(index=False)
        }
    return out


def _rank_bucket_map(sample: pd.DataFrame) -> dict[str, str]:
    ranks = sample.groupby("symbol", observed=True)["dynamic_all_rank"].first()
    out: dict[str, str] = {}
    for symbol, rank in ranks.items():
        value = float(rank) if pd.notna(rank) else np.nan
        if pd.isna(value):
            bucket = "unknown"
        elif value <= 10:
            bucket = "rank_1_10"
        elif value <= 30:
            bucket = "rank_11_30"
        else:
            bucket = "rank_31_50"
        out[str(symbol)] = bucket
    return out


def _density_random_leaders(
    month_start: pd.Timestamp,
    symbols: list[str],
    rank_buckets: dict[str, str],
    permutation: int,
    k: int = EDGE_TOP_K,
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for beta in symbols:
        bucket = rank_buckets.get(beta, "unknown")
        choices = [item for item in symbols if item != beta and rank_buckets.get(item, "unknown") == bucket]
        if len(choices) < 2:
            choices = [item for item in symbols if item != beta]
        if not choices:
            out[beta] = []
            continue
        rng = np.random.default_rng(_stable_seed("v07c_density_random", month_start.isoformat(), permutation, beta, bucket))
        take = min(k, len(choices))
        out[beta] = list(rng.choice(choices, size=take, replace=False))
    return out


def _shuffled_leaders(
    month_start: pd.Timestamp,
    symbols: list[str],
    real_leaders: dict[str, list[str]],
    permutation: int,
) -> dict[str, list[str]]:
    if not symbols:
        return {}
    rng = np.random.default_rng(_stable_seed("v07c_shuffle", month_start.isoformat(), permutation))
    shuffled = symbols.copy()
    rng.shuffle(shuffled)
    mapping = dict(zip(symbols, shuffled, strict=False))
    out: dict[str, list[str]] = {}
    for beta in symbols:
        mapped = [mapping.get(item, item) for item in real_leaders.get(beta, [])]
        out[beta] = [item for item in mapped if item != beta]
    return out


def _ratio_from_map(state: pd.DataFrame, feature_time: pd.Series, leader_map: dict[str, list[str]], beta: str) -> np.ndarray:
    leaders = [item for item in leader_map.get(beta, []) if item in state.columns]
    if not leaders:
        return np.zeros(len(feature_time), dtype=float)
    count = state[leaders].sum(axis=1).reindex(feature_time).fillna(0.0).to_numpy(dtype=float)
    return count / float(len(leaders))


def _max_from_map(values: pd.DataFrame, feature_time: pd.Series, leader_map: dict[str, list[str]], beta: str) -> np.ndarray:
    leaders = [item for item in leader_map.get(beta, []) if item in values.columns]
    if not leaders:
        return np.full(len(feature_time), np.nan)
    return pd.to_numeric(values[leaders].max(axis=1).reindex(feature_time), errors="coerce").to_numpy(dtype=float)


def _active_edge_weight(
    state: pd.DataFrame,
    feature_time: pd.Series,
    leader_map: dict[str, list[str]],
    weights: dict[str, dict[str, float]],
    beta: str,
) -> np.ndarray:
    leaders = [item for item in leader_map.get(beta, []) if item in state.columns]
    if not leaders:
        return np.zeros(len(feature_time), dtype=float)
    active = state[leaders].reindex(feature_time).fillna(False).astype(bool)
    weight_series = pd.Series({leader: weights.get(beta, {}).get(leader, 0.0) for leader in leaders}, dtype="float64")
    weighted = active.astype(float).mul(weight_series, axis=1).sum(axis=1)
    counts = active.sum(axis=1).replace(0, np.nan)
    return (weighted / counts).fillna(0.0).to_numpy(dtype=float)


def _add_directed_features(data: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    defaults = {
        "directed_leader_recent_ratio": 0.0,
        "directed_leader_return_4h_max": np.nan,
        "directed_leader_edge_weight_active": 0.0,
        "low_edge_leader_recent_ratio": 0.0,
        "future_leader_1h_ratio": 0.0,
        "future_leader_4h_ratio": 0.0,
        "beta_not_extended": False,
        "beta_already_extended": False,
        "leader_beta_cluster_id": "",
    }
    for col, value in defaults.items():
        out[col] = value
    if out.empty:
        return _add_directed_gates(out)
    for month_start, month_data in out.groupby("month_start", observed=True, sort=True):
        month_start = pd.Timestamp(month_start)
        sample = month_data[pd.to_numeric(month_data["dynamic_all_rank"], errors="coerce") <= TOP_N].copy()
        if sample.empty:
            continue
        symbols = sorted(sample["symbol"].dropna().astype(str).unique())
        event = sample.pivot_table(
            index="feature_time",
            columns="symbol",
            values="bullish_volume_shock_event",
            aggfunc="max",
            observed=True,
        ).fillna(False).infer_objects(copy=False).astype(bool)
        ret4 = sample.pivot_table(index="feature_time", columns="symbol", values="ret_4h", aggfunc="last", observed=True)
        past_event = _past_any(event, LEADER_RECENT_BARS)
        future_1h = _future_any(event, 4)
        future_4h = _future_any(event, 16)
        month_edges = (
            edges[edges["month_start"].eq(month_start)]
            if "month_start" in edges.columns
            else pd.DataFrame()
        )
        real_leaders = _top_leader_map(month_edges)
        low_edge_leaders = _low_edge_leader_map(month_edges, symbols)
        weights = _edge_weight_map(month_edges)
        for beta, group in sample.groupby("symbol", observed=True, sort=False):
            beta = str(beta)
            idx = group.index
            feature_time = pd.to_datetime(group["feature_time"], utc=True, errors="coerce")
            out.loc[idx, "directed_leader_recent_ratio"] = _ratio_from_map(past_event, feature_time, real_leaders, beta)
            out.loc[idx, "low_edge_leader_recent_ratio"] = _ratio_from_map(past_event, feature_time, low_edge_leaders, beta)
            out.loc[idx, "future_leader_1h_ratio"] = _ratio_from_map(future_1h, feature_time, real_leaders, beta)
            out.loc[idx, "future_leader_4h_ratio"] = _ratio_from_map(future_4h, feature_time, real_leaders, beta)
            out.loc[idx, "directed_leader_return_4h_max"] = _max_from_map(ret4, feature_time, real_leaders, beta)
            out.loc[idx, "directed_leader_edge_weight_active"] = _active_edge_weight(
                past_event,
                feature_time,
                real_leaders,
                weights,
                beta,
            )
            leaders = real_leaders.get(beta, [])
            leader = leaders[0] if leaders else "none"
            out.loc[idx, "leader_beta_cluster_id"] = f"{month_start:%Y-%m}:lead:{leader}:beta:{beta}"
        ret_pct = pd.to_numeric(sample.get("ret_4h_percentile"), errors="coerce")
        out.loc[sample.index, "beta_not_extended"] = (ret_pct < 80).fillna(False).to_numpy(dtype=bool)
        out.loc[sample.index, "beta_already_extended"] = (ret_pct >= 90).fillna(False).to_numpy(dtype=bool)
    return _add_directed_gates(out)


def _add_directed_gates(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    market = _ensure_bool(out.get("market_volume_impulse_density_high", pd.Series(False, index=out.index)))
    leader_recent = pd.to_numeric(out.get("directed_leader_recent_ratio"), errors="coerce").fillna(0.0) >= 0.20
    low_edge_recent = pd.to_numeric(out.get("low_edge_leader_recent_ratio"), errors="coerce").fillna(0.0) >= 0.20
    future_1h = pd.to_numeric(out.get("future_leader_1h_ratio"), errors="coerce").fillna(0.0) >= 0.20
    future_4h = pd.to_numeric(out.get("future_leader_4h_ratio"), errors="coerce").fillna(0.0) >= 0.20
    leader_ret = pd.to_numeric(out.get("directed_leader_return_4h_max"), errors="coerce")
    beta_ret_pct = pd.to_numeric(out.get("ret_4h_percentile"), errors="coerce")
    beta_not_extended = _ensure_bool(out.get("beta_not_extended", pd.Series(False, index=out.index)))
    beta_already_extended = _ensure_bool(out.get("beta_already_extended", pd.Series(False, index=out.index)))
    out["gate_directed_leader_beta"] = market & leader_recent & beta_not_extended
    out["gate_leader_return_beta_laggard"] = market & (leader_ret > 0.03) & (beta_ret_pct < 70)
    out["gate_dispersion_low_edge_beta"] = market & low_edge_recent & beta_not_extended & ~leader_recent
    out["gate_low_edge_leader_beta"] = market & low_edge_recent & beta_not_extended
    out["gate_beta_already_extended"] = market & leader_recent & beta_already_extended
    out["gate_no_directed_leader_impulse"] = market & ~leader_recent & beta_not_extended
    out["gate_future_leader_1h"] = market & future_1h & beta_not_extended
    out["gate_future_leader_4h"] = market & future_4h & beta_not_extended
    out["directed_edge_weight_bucket"] = pd.Series(pd.NA, index=out.index, dtype="object")
    weight = pd.to_numeric(out.get("directed_leader_edge_weight_active"), errors="coerce")
    valid = market & leader_recent & beta_not_extended & weight.notna() & (weight > 0)
    if valid.sum() >= 4:
        try:
            buckets = pd.qcut(weight[valid], 4, labels=["q1", "q2", "q3", "q4"], duplicates="drop")
            out.loc[valid, "directed_edge_weight_bucket"] = buckets.astype(str)
        except ValueError:
            out.loc[valid, "directed_edge_weight_bucket"] = "q1"
    for bucket in range(1, 5):
        out[f"gate_edge_weight_q{bucket}"] = valid & out["directed_edge_weight_bucket"].astype(str).eq(f"q{bucket}")
    return out


def _add_map_gate(data: pd.DataFrame, leader_map: dict[str, list[str]], gate_col: str) -> pd.DataFrame:
    out = data.copy()
    out[gate_col] = False
    sample = out[pd.to_numeric(out["dynamic_all_rank"], errors="coerce") <= TOP_N].copy()
    if sample.empty:
        return out
    event = sample.pivot_table(
        index="feature_time",
        columns="symbol",
        values="bullish_volume_shock_event",
        aggfunc="max",
        observed=True,
    ).fillna(False).infer_objects(copy=False).astype(bool)
    past_event = _past_any(event, LEADER_RECENT_BARS)
    ratios = pd.Series(0.0, index=sample.index, dtype="float64")
    for beta, group in sample.groupby("symbol", observed=True, sort=False):
        feature_time = pd.to_datetime(group["feature_time"], utc=True, errors="coerce")
        ratios.loc[group.index] = _ratio_from_map(past_event, feature_time, leader_map, str(beta))
    market = _ensure_bool(sample.get("market_volume_impulse_density_high", pd.Series(False, index=sample.index)))
    beta_not_extended = _ensure_bool(sample.get("beta_not_extended", pd.Series(False, index=sample.index)))
    out.loc[sample.index, gate_col] = (market & beta_not_extended & (ratios >= 0.20)).to_numpy(dtype=bool)
    return out


def _simulate_control(
    data: pd.DataFrame,
    control: LeaderControl,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = _candidate(control)
    signal_rows: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []
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
        "directed_leader_recent_ratio",
        "directed_leader_return_4h_max",
        "directed_leader_edge_weight_active",
        "low_edge_leader_recent_ratio",
        "future_leader_1h_ratio",
        "future_leader_4h_ratio",
        "beta_not_extended",
        "beta_already_extended",
        "directed_edge_weight_bucket",
        "leader_beta_cluster_id",
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
                "entry_only_lift": net10 - _lookup_optional(long_summary, candidate, "entry_only_reclaim"),
                "matched_random_lift": net10 - _lookup_optional(long_summary, candidate, "matched_random_reclaim"),
                "ex_top_month_net10": _ex_top_month_net(sample, 10),
                "month_cap35_net20": _month_cap_expectancy(sample, 20, 0.35),
                "max_month_contribution": _max_contribution(sample, "month", 10),
                "max_symbol_contribution": _max_contribution(sample, "symbol", 10),
                "max_cluster_contribution": _max_contribution(sample, "leader_beta_cluster_id", 10),
            }
        )
    return pd.DataFrame(rows)


def _leader_control_summary(
    core: pd.DataFrame,
    random_summary: pd.DataFrame,
    shuffled_summary: pd.DataFrame,
) -> pd.DataFrame:
    out = core.copy()
    real = out[out["candidate"].astype(str).eq("LBR1_directed_leader_beta")]
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
    for control in [
        "low_edge_leader_control",
        "beta_already_extended_control",
        "no_leader_impulse_control",
        "LBR4_dispersion_shadow",
    ]:
        sample = out[out["candidate"].astype(str).eq(control)]
        out[f"real_vs_{control}_lift"] = real_net10 - float(sample.iloc[0]["net10"]) if not sample.empty else np.nan
    return out


def _portfolio_diagnostic(trades: pd.DataFrame) -> pd.DataFrame:
    sample = trades[
        trades["candidate"].astype(str).eq("LBR1_directed_leader_beta")
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
    sample["rank_leader_edge_weight"] = _numeric_context(sample, "directed_leader_edge_weight_active", -np.inf)
    sample["rank_leader_return"] = _numeric_context(sample, "directed_leader_return_4h_max", -np.inf)
    sample["rank_liquidity"] = -_numeric_context(sample, "dynamic_all_rank", 999.0)
    hashed = pd.util.hash_pandas_object(sample[["symbol", "signal_time"]].astype(str), index=False)
    sample["rank_random"] = (hashed % 10_000_000).astype(float)
    rank_cols = [
        "rank_first_come_first_served",
        "rank_market_volume_impulse_density",
        "rank_local_volume_shock_strength",
        "rank_leader_edge_weight",
        "rank_leader_return",
        "rank_liquidity",
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


def _month_setup(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    regime: pd.DataFrame,
    config: ExperimentConfig,
    month_start: pd.Timestamp,
    next_month: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    symbols = _month_symbols(rank30, month_start)
    if not symbols:
        return pd.DataFrame(), pd.DataFrame(), []
    hist_start = month_start - pd.Timedelta(days=GRAPH_LOOKBACK_DAYS)
    window_end = next_month + pd.Timedelta(minutes=15 * SIM_WINDOW_BARS)
    window = _load_month_window(feature_path, rank30, rank90, symbols, regime, config, hist_start, window_end)
    if window.empty:
        return pd.DataFrame(), pd.DataFrame(), symbols
    hist = window[
        (pd.to_datetime(window["bar_open_time"], utc=True, errors="coerce") >= hist_start)
        & (pd.to_datetime(window["bar_open_time"], utc=True, errors="coerce") < month_start)
    ].copy()
    edges = _build_directed_edges(month_start, hist, symbols)
    sim_window = window[
        (pd.to_datetime(window["bar_open_time"], utc=True, errors="coerce") >= month_start)
        & (pd.to_datetime(window["bar_open_time"], utc=True, errors="coerce") < window_end)
    ].copy()
    sim_window = _add_directed_features(sim_window, edges)
    sim_window = _disable_out_of_month_events(sim_window, month_start, next_month)
    return sim_window, edges, symbols


def _stream_reports(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    regime: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trade_path = report_root / "_v07c_trades_tmp.csv"
    signal_path = report_root / "_v07c_signals_tmp.csv"
    for path in [trade_path, signal_path]:
        if path.exists():
            path.unlink()
    wrote_trades = False
    wrote_signals = False
    edge_frames: list[pd.DataFrame] = []
    months = sorted(
        pd.to_datetime(rank30["month_start"], utc=True, errors="coerce")
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    for idx, month_start in enumerate(months, start=1):
        month_start = pd.Timestamp(month_start)
        next_month = pd.Timestamp(months[idx]) if idx < len(months) else month_start + pd.DateOffset(months=1)
        sim_window, edges, symbols = _month_setup(feature_path, rank30, rank90, regime, config, month_start, next_month)
        if sim_window.empty:
            continue
        if not edges.empty:
            edge_frames.append(edges.copy())
        trade_frames: list[pd.DataFrame] = []
        signal_frames: list[pd.DataFrame] = []
        for control in [*CORE_CONTROLS, *FUTURE_CONTROLS, *EDGE_BUCKET_CONTROLS]:
            trades, signals = _simulate_control(sim_window, control, config)
            if not trades.empty:
                trade_frames.append(trades)
            if not signals.empty:
                signal_frames.append(signals)
        sample = sim_window[pd.to_numeric(sim_window["dynamic_all_rank"], errors="coerce") <= TOP_N].copy()
        symbols_month = sorted(sample["symbol"].dropna().astype(str).unique())
        rank_buckets = _rank_bucket_map(sample)
        real_leaders = _top_leader_map(edges)
        for permutation in range(RANDOM_PERMUTATIONS):
            gate_col = "__density_random_leader_gate"
            local = _add_map_gate(
                sim_window,
                _density_random_leaders(month_start, symbols_month, rank_buckets, permutation),
                gate_col,
            )
            control = LeaderControl(
                f"density_random_leader_p{permutation:03d}",
                "density_matched_random_leader",
                gate_col,
                "Density/liquidity matched random directed leader permutation.",
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
            gate_col = "__shuffled_leader_gate"
            local = _add_map_gate(
                sim_window,
                _shuffled_leaders(month_start, symbols_month, real_leaders, permutation),
                gate_col,
            )
            control = LeaderControl(
                f"shuffled_leader_p{permutation:03d}",
                "shuffled_leader",
                gate_col,
                "Within-month shuffled directed leader mapping.",
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
        print(f"v0.7C month {idx}/{len(months)} {month_start:%Y-%m} symbols={len(symbols)}", flush=True)
        del sim_window, edges, trade_frames, signal_frames, month_trades, month_signals
        gc.collect()
    trades = pd.read_csv(trade_path, low_memory=False) if wrote_trades else pd.DataFrame()
    signals = pd.read_csv(signal_path, low_memory=False) if wrote_signals else pd.DataFrame()
    for path in [trade_path, signal_path]:
        if path.exists():
            path.unlink()
    edges = pd.concat(edge_frames, ignore_index=True).drop_duplicates() if edge_frames else pd.DataFrame()
    return trades, signals, edges


def _candidate_family(candidate: str) -> str:
    if candidate.startswith("density_random_leader_p"):
        return "density_matched_random_leader"
    if candidate.startswith("shuffled_leader_p"):
        return "shuffled_leader"
    if candidate.startswith("future_leader"):
        return "future_leader"
    if candidate.startswith("edge_weight_q"):
        return "edge_weight_bucket"
    return str(candidate)


def _write_notes(
    report_root: Path,
    control_summary: pd.DataFrame,
    random_summary: pd.DataFrame,
    shuffled_summary: pd.DataFrame,
    monotonicity: pd.DataFrame,
) -> None:
    lines = [
        "# v0.7C Leader-Beta Rotation Graph",
        "",
        "Purpose: test whether directed prior-window leader -> beta edges add alpha beyond MIR1 market-density context.",
        "",
    ]
    real = control_summary[control_summary["candidate"].astype(str).eq("LBR1_directed_leader_beta")]
    if not real.empty:
        row = real.iloc[0]
        lines.append(f"- LBR1 real directed edge: trades={int(row['trades'])}, net10={row['net10']:.4%}, net20={row['net20']:.4%}.")
        lines.append(f"- Real percentile vs density-matched random leader: {row.get('real_percentile_vs_random', np.nan):.2%}.")
        lines.append(f"- Month-cap35 net20: {row.get('month_cap35_net20', np.nan):.4%}.")
    if not random_summary.empty:
        lines.append(
            f"- Random leader net10 mean={random_summary['net10'].mean():.4%}, "
            f"p75={random_summary['net10'].quantile(0.75):.4%}, p90={random_summary['net10'].quantile(0.90):.4%}."
        )
    if not shuffled_summary.empty:
        lines.append(
            f"- Shuffled leader net10 mean={shuffled_summary['net10'].mean():.4%}, "
            f"p75={shuffled_summary['net10'].quantile(0.75):.4%}."
        )
    if not monotonicity.empty:
        ordered = monotonicity[monotonicity["candidate"].astype(str).str.startswith("edge_weight_q")]
        if not ordered.empty:
            lines.append("- Edge-weight buckets: " + ", ".join(f"{r.candidate}={r.net10:.4%}" for r in ordered.itertuples()))
    lines.extend(
        [
            "",
            "Decision rule:",
            "- Directed edge is promoted only if real LBR1 beats density-matched random/shuffled/low-edge controls, beta-not-extended beats beta-extended, future leader control is not the real source, and month-capped net20 stays positive.",
            "- If random/shuffled remain close or stronger, keep MIR1 primary and treat leader-beta as unproven density proxy.",
        ]
    )
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v07c_leader_beta_rotation(
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
    trades, signals, edges = _stream_reports(feature_path, rank30, rank90, regime, config, report_root)
    signals_grouped = (
        signals.groupby(["candidate", "gate_mode", "baseline"], as_index=False, sort=False, dropna=False)
        .agg(signals=("signals", "sum"), pre_entry_gate_trades=("pre_entry_gate_trades", "sum"))
        if not signals.empty
        else pd.DataFrame()
    )
    summary = _summarize_candidates(trades, signals_grouped)
    summary["family"] = summary["candidate"].map(_candidate_family) if not summary.empty else pd.Series(dtype=object)
    random_summary = summary[summary["family"].eq("density_matched_random_leader")].copy()
    shuffled_summary = summary[summary["family"].eq("shuffled_leader")].copy()
    core_names = [control.candidate for control in CORE_CONTROLS]
    core_summary = summary[summary["candidate"].isin(core_names)].copy()
    control_summary = _leader_control_summary(core_summary, random_summary, shuffled_summary)
    future = summary[summary["family"].eq("future_leader")].copy()
    monotonicity = summary[summary["family"].eq("edge_weight_bucket")].copy()
    portfolio = _portfolio_diagnostic(trades)
    outputs = {
        "leader_beta_edge_summary": report_root / "leader_beta_edge_summary.csv",
        "leader_beta_control_summary": report_root / "leader_beta_control_summary.csv",
        "density_matched_random_leader": report_root / "density_matched_random_leader.csv",
        "density_matched_random_distribution": report_root / "density_matched_random_distribution.csv",
        "shuffled_leader_summary": report_root / "shuffled_leader_summary.csv",
        "low_edge_leader_control": report_root / "low_edge_leader_control.csv",
        "future_leader_leakage_audit": report_root / "future_leader_leakage_audit.csv",
        "beta_extension_control": report_root / "beta_extension_control.csv",
        "edge_weight_monotonicity": report_root / "edge_weight_monotonicity.csv",
        "portfolio_concurrency_diagnostic": report_root / "portfolio_concurrency_diagnostic.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    edges.to_csv(outputs["leader_beta_edge_summary"], index=False)
    control_summary.to_csv(outputs["leader_beta_control_summary"], index=False)
    random_summary.to_csv(outputs["density_matched_random_leader"], index=False)
    _aggregate_permutations(random_summary, "density_matched_random_leader").to_csv(
        outputs["density_matched_random_distribution"],
        index=False,
    )
    shuffled_summary.to_csv(outputs["shuffled_leader_summary"], index=False)
    control_summary[control_summary["candidate"].astype(str).eq("low_edge_leader_control")].to_csv(
        outputs["low_edge_leader_control"],
        index=False,
    )
    future.to_csv(outputs["future_leader_leakage_audit"], index=False)
    control_summary[control_summary["candidate"].astype(str).eq("beta_already_extended_control")].to_csv(
        outputs["beta_extension_control"],
        index=False,
    )
    monotonicity.to_csv(outputs["edge_weight_monotonicity"], index=False)
    portfolio.to_csv(outputs["portfolio_concurrency_diagnostic"], index=False)
    _write_notes(report_root, control_summary, random_summary, shuffled_summary, monotonicity)
    return outputs
