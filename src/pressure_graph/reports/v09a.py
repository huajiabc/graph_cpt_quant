from __future__ import annotations

import gc
import ctypes
import multiprocessing as mp
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
    _lookup,
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
    _load_month_window,
    _month_symbols,
)
from pressure_graph.reports.v07d1 import _ex_month_net


REPORT_ROOT = Path("reports/v0_9a_cluster_impulse_graph")
FOCAL_MONTH = "2025-08"
GRAPH_TOP_K = 5
HYBRID_WEIGHT_RETURN = 0.5
HYBRID_WEIGHT_VOLUME = 0.5
CLUSTER_EDGE_THRESHOLD = 0.08
CLUSTER_IMPULSE_HIGH = 0.20
CLUSTER_IMPULSE_LOW = 0.05
RANDOM_PERMUTATIONS = 5
SHUFFLED_PERMUTATIONS = 3


@dataclass(frozen=True)
class ClusterCandidate:
    candidate: str
    role: str
    gate_col: str
    description: str
    negative_control: bool = False
    baselines: tuple[BaselineSpec, ...] = tuple(BASELINES)


CORE_BASELINE = next(item for item in BASELINES if item.baseline == "candidate_reclaim")


CANDIDATES = [
    ClusterCandidate(
        "CG0_CIC1_reference",
        "cic_primary_reference",
        "c9a_cg0_cic1_reference",
        "Current CIC1-filtered MIR1 reference: market impulse high + beta extreme continuation.",
    ),
    ClusterCandidate(
        "CG1_beta_extreme_cluster_high_any_market",
        "cluster_context",
        "c9a_cg1_beta_extreme_cluster_high_any_market",
        "Beta extreme continuation with cluster impulse high, independent of market-wide gate.",
    ),
    ClusterCandidate(
        "CG2_cic1_cluster_high_market_high",
        "market_plus_cluster",
        "c9a_cg2_cic1_cluster_high_market_high",
        "CIC1 with cluster impulse high and market impulse high.",
    ),
    ClusterCandidate(
        "CG3_cluster_high_market_not_high",
        "local_rotation_probe",
        "c9a_cg3_cluster_high_market_not_high",
        "Cluster impulse high while market-wide impulse density is not high.",
    ),
    ClusterCandidate(
        "CG4_market_high_cluster_low",
        "negative_control",
        "c9a_cg4_market_high_cluster_low",
        "CIC1 with market impulse high but cluster impulse low.",
        True,
    ),
    ClusterCandidate(
        "CG5_cic2_cluster_high",
        "broad_continuation_cluster",
        "c9a_cg5_cic2_cluster_high",
        "Broader CIC2 continuation with cluster impulse high.",
    ),
    ClusterCandidate(
        "CG6_isolated_beta_extreme",
        "isolated_negative_control",
        "c9a_cg6_isolated_beta_extreme",
        "Beta extreme continuation with isolated cluster impulse.",
        True,
    ),
]


def _candidate_for_gate(candidate: ClusterCandidate) -> FrozenAtlasCandidate:
    return FrozenAtlasCandidate(
        candidate.candidate,
        TOP_N,
        candidate.gate_col,
        "bullish_volume_shock_event",
        RECLAIM_1,
        "vol_regime_fast",
        candidate.role,
        candidate.negative_control,
    )


def _ensure_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def _month_key(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m")


def _release_memory() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _month_slug(month_start: pd.Timestamp) -> str:
    return pd.Timestamp(month_start).strftime("%Y%m")


def _month_artifact_paths(report_root: Path, month_start: pd.Timestamp) -> dict[str, Path]:
    slug = _month_slug(month_start)
    return {
        "trades": report_root / f"_v09a_month_{slug}_trades.csv",
        "signals": report_root / f"_v09a_month_{slug}_signals.csv",
        "membership": report_root / f"_v09a_month_{slug}_membership.csv",
        "edges": report_root / f"_v09a_month_{slug}_edges.csv",
        "features": report_root / f"_v09a_month_{slug}_features.csv",
    }


def _write_csv_frame(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        path.write_text("", encoding="utf-8")
        return
    frame.to_csv(path, index=False)


def _read_csv_frames(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path, low_memory=False) for path in paths if path.exists() and path.stat().st_size > 0]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _run_child_process(target, args: tuple[object, ...], label: str) -> None:
    ctx = mp.get_context("spawn")
    process = ctx.Process(target=target, args=args, name=label)
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"{label} failed with exit code {process.exitcode}")


def _positive_corr_matrix(hist: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    pivot = hist.pivot_table(
        index="feature_time",
        columns="symbol",
        values="ret_1h",
        aggfunc="last",
        observed=True,
    )
    cols = [symbol for symbol in symbols if symbol in pivot.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    corr = pivot[cols].corr(min_periods=96).clip(lower=0.0).fillna(0.0)
    np.fill_diagonal(corr.values, 0.0)
    return corr


def _volume_cooccurrence_matrix(hist: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    pivot = hist.pivot_table(
        index="feature_time",
        columns="symbol",
        values="bullish_volume_shock_state",
        aggfunc="max",
        observed=True,
    )
    cols = [symbol for symbol in symbols if symbol in pivot.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    event = pivot[cols].infer_objects(copy=False).fillna(False).astype("int8")
    cooc = event.T.dot(event).astype("float64")
    counts = event.sum(axis=0).astype("float64")
    denom = counts.values[:, None] + counts.values[None, :] - cooc.values
    score = np.divide(cooc.values, denom, out=np.zeros_like(cooc.values), where=denom > 0)
    out = pd.DataFrame(score, index=cooc.index, columns=cooc.columns)
    np.fill_diagonal(out.values, 0.0)
    return out


def _hybrid_related_matrix(hist: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    corr = _positive_corr_matrix(hist, symbols)
    cooc = _volume_cooccurrence_matrix(hist, symbols)
    cols = sorted(set(corr.columns).union(cooc.columns))
    if len(cols) < 2:
        return pd.DataFrame()
    corr = corr.reindex(index=cols, columns=cols).fillna(0.0)
    cooc = cooc.reindex(index=cols, columns=cols).fillna(0.0)
    hybrid = HYBRID_WEIGHT_RETURN * corr + HYBRID_WEIGHT_VOLUME * cooc
    np.fill_diagonal(hybrid.values, 0.0)
    return hybrid


def _topk_adjacency(scores: pd.DataFrame) -> dict[str, set[str]]:
    top: dict[str, set[str]] = {}
    if scores.empty:
        return top
    for symbol in scores.columns:
        ranked = scores[symbol].drop(labels=[symbol], errors="ignore")
        ranked = ranked[ranked >= CLUSTER_EDGE_THRESHOLD].sort_values(ascending=False).head(GRAPH_TOP_K)
        top[str(symbol)] = set(ranked.index.astype(str))
    adjacency: dict[str, set[str]] = {str(symbol): set() for symbol in scores.columns}
    for source, targets in top.items():
        for target in targets:
            if source in top.get(target, set()):
                adjacency[source].add(target)
                adjacency.setdefault(target, set()).add(source)
    return adjacency


def _connected_components(adjacency: dict[str, set[str]]) -> list[list[str]]:
    remaining = set(adjacency)
    components: list[list[str]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        comp = {start}
        while stack:
            node = stack.pop()
            for neighbor in adjacency.get(node, set()):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    comp.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(comp))
    return sorted(components, key=lambda items: (-len(items), items[0]))


def build_cluster_membership(
    month_start: pd.Timestamp,
    hist: pd.DataFrame,
    symbols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = _hybrid_related_matrix(hist, symbols)
    adjacency = _topk_adjacency(scores) if not scores.empty else {symbol: set() for symbol in symbols}
    for symbol in symbols:
        adjacency.setdefault(symbol, set())
    components = _connected_components(adjacency)
    rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    for cluster_idx, members in enumerate(components, start=1):
        cluster_id = f"{month_start:%Y-%m}:C{cluster_idx:02d}"
        for symbol in members:
            rows.append(
                {
                    "month_start": month_start,
                    "symbol": symbol,
                    "cluster_id": cluster_id,
                    "cluster_size": len(members),
                }
            )
        for source in members:
            for target in sorted(adjacency.get(source, set())):
                if source < target:
                    edge_rows.append(
                        {
                            "month_start": month_start,
                            "cluster_id": cluster_id,
                            "source_symbol": source,
                            "target_symbol": target,
                            "edge_weight": float(scores.loc[source, target]) if not scores.empty else np.nan,
                            "edge_type": "hybrid_mutual_topk",
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(edge_rows)


def _membership_map(membership: pd.DataFrame) -> dict[str, list[str]]:
    if membership.empty:
        return {}
    out: dict[str, list[str]] = {}
    for cluster_id, group in membership.groupby("cluster_id", sort=False):
        out[str(cluster_id)] = sorted(group["symbol"].dropna().astype(str).unique())
    return out


def _randomized_membership(
    membership: pd.DataFrame,
    *,
    seed: int,
    control_type: str,
) -> pd.DataFrame:
    if membership.empty:
        return membership.copy()
    rng = np.random.default_rng(seed)
    symbols = membership["symbol"].dropna().astype(str).tolist()
    sizes = membership.groupby("cluster_id", sort=False).size().astype(int).tolist()
    shuffled = symbols.copy()
    rng.shuffle(shuffled)
    rows = []
    pos = 0
    month_start = pd.Timestamp(membership["month_start"].iloc[0])
    for idx, size in enumerate(sizes, start=1):
        cluster_id = f"{month_start:%Y-%m}:{control_type}:{idx:02d}"
        for symbol in shuffled[pos : pos + size]:
            rows.append(
                {
                    "month_start": month_start,
                    "symbol": symbol,
                    "cluster_id": cluster_id,
                    "cluster_size": size,
                    "control_type": control_type,
                }
            )
        pos += size
    return pd.DataFrame(rows)


def add_cluster_graph_features(
    data: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    prefix: str = "cluster",
) -> pd.DataFrame:
    out = data.copy()
    cols = {
        f"{prefix}_id": "",
        f"{prefix}_size": 1,
        f"{prefix}_peer_count": 0,
        f"{prefix}_impulse_density": 0.0,
        f"{prefix}_bullish_volume_shock_count": 0.0,
        f"{prefix}_positive_return_ratio": np.nan,
        f"{prefix}_beta_extreme_count": 0.0,
        f"{prefix}_beta_extreme_density": 0.0,
        f"{prefix}_symbol_rank_within_cluster": np.nan,
        f"{prefix}_lag_vs_cluster": np.nan,
        f"{prefix}_isolated_impulse_flag": True,
    }
    for col, value in cols.items():
        out[col] = value
    if out.empty or membership.empty:
        return _add_cluster_gates(out, prefix)
    membership = membership.copy()
    membership["symbol"] = membership["symbol"].astype(str)
    symbol_to_cluster = membership.set_index("symbol")["cluster_id"].astype(str).to_dict()
    symbol_to_size = membership.set_index("symbol")["cluster_size"].astype(int).to_dict()
    clusters = _membership_map(membership)
    sample = out[pd.to_numeric(out["dynamic_all_rank"], errors="coerce") <= TOP_N].copy()
    if sample.empty:
        return _add_cluster_gates(out, prefix)
    state = sample.pivot_table(
        index="feature_time",
        columns="symbol",
        values="bullish_volume_shock_state",
        aggfunc="max",
        observed=True,
    )
    state = state.where(state.notna(), False).infer_objects(copy=False).astype(bool)
    ret4 = sample.pivot_table(index="feature_time", columns="symbol", values="ret_4h", aggfunc="last", observed=True)
    ret_pct = sample.pivot_table(
        index="feature_time",
        columns="symbol",
        values="ret_4h_percentile",
        aggfunc="last",
        observed=True,
    )
    positive = (ret4 > 0).fillna(False)
    beta_extreme = (ret_pct >= 95).fillna(False)
    for symbol, group in sample.groupby("symbol", observed=True, sort=False):
        symbol = str(symbol)
        cluster_id = symbol_to_cluster.get(symbol, "")
        members = [item for item in clusters.get(cluster_id, []) if item in state.columns and item != symbol]
        idx = group.index
        feature_time = pd.to_datetime(group["feature_time"], utc=True, errors="coerce")
        out.loc[idx, f"{prefix}_id"] = cluster_id
        out.loc[idx, f"{prefix}_size"] = int(symbol_to_size.get(symbol, 1))
        out.loc[idx, f"{prefix}_peer_count"] = len(members)
        if not members:
            continue
        impulse_count = state[members].sum(axis=1).reindex(feature_time).to_numpy(dtype=float)
        positive_ratio = positive[members].mean(axis=1).reindex(feature_time).to_numpy(dtype=float)
        beta_members = [item for item in members if item in beta_extreme.columns]
        beta_count = (
            beta_extreme[beta_members].sum(axis=1).reindex(feature_time).to_numpy(dtype=float)
            if beta_members
            else np.zeros(len(group), dtype=float)
        )
        cluster_ret = ret4[members].mean(axis=1).reindex(feature_time).to_numpy(dtype=float)
        own_ret = pd.to_numeric(group["ret_4h"], errors="coerce").to_numpy(dtype=float)
        out.loc[idx, f"{prefix}_bullish_volume_shock_count"] = impulse_count
        out.loc[idx, f"{prefix}_impulse_density"] = impulse_count / float(len(members))
        out.loc[idx, f"{prefix}_positive_return_ratio"] = positive_ratio
        out.loc[idx, f"{prefix}_beta_extreme_count"] = beta_count
        out.loc[idx, f"{prefix}_beta_extreme_density"] = beta_count / float(len(members))
        out.loc[idx, f"{prefix}_lag_vs_cluster"] = cluster_ret - own_ret
        rank_symbols = [*members, symbol]
        rank_symbols = [item for item in rank_symbols if item in ret4.columns]
        if symbol in rank_symbols:
            rank = ret4[rank_symbols].rank(axis=1, pct=True)[symbol].reindex(feature_time)
            out.loc[idx, f"{prefix}_symbol_rank_within_cluster"] = pd.to_numeric(rank, errors="coerce").to_numpy()
    return _add_cluster_gates(out, prefix)


def _add_cluster_gates(data: pd.DataFrame, prefix: str = "cluster") -> pd.DataFrame:
    out = data.copy()
    density = pd.to_numeric(out.get(f"{prefix}_impulse_density"), errors="coerce").fillna(0.0)
    out[f"{prefix}_impulse_high"] = density >= CLUSTER_IMPULSE_HIGH
    out[f"{prefix}_impulse_low"] = density <= CLUSTER_IMPULSE_LOW
    out[f"{prefix}_isolated_impulse_flag"] = out[f"{prefix}_impulse_low"]
    if prefix != "cluster":
        return out
    market = _ensure_bool(out.get("market_volume_impulse_density_high", pd.Series(False, index=out.index)))
    beta_pct = pd.to_numeric(out.get("ret_4h_percentile"), errors="coerce")
    beta_extreme = beta_pct >= 95
    beta_continuation = beta_pct >= 80
    out["c9a_beta_extreme_continuation"] = beta_extreme
    out["c9a_beta_broad_continuation"] = beta_continuation
    out["c9a_cg0_cic1_reference"] = market & beta_extreme
    out["c9a_cg1_beta_extreme_cluster_high_any_market"] = beta_extreme & out["cluster_impulse_high"]
    out["c9a_cg2_cic1_cluster_high_market_high"] = market & beta_extreme & out["cluster_impulse_high"]
    out["c9a_cg3_cluster_high_market_not_high"] = (~market) & beta_extreme & out["cluster_impulse_high"]
    out["c9a_cg4_market_high_cluster_low"] = market & beta_extreme & out["cluster_impulse_low"]
    out["c9a_cg5_cic2_cluster_high"] = market & beta_continuation & out["cluster_impulse_high"]
    out["c9a_cg6_isolated_beta_extreme"] = beta_extreme & out["cluster_isolated_impulse_flag"]
    return out


def _control_candidate(
    control_type: str,
    idx: int,
    gate_col: str,
) -> ClusterCandidate:
    return ClusterCandidate(
        f"{control_type}_cluster_p{idx:03d}",
        f"{control_type}_cluster_control",
        gate_col,
        f"{control_type.title()} cluster membership preserving real cluster size distribution.",
        True,
        (CORE_BASELINE,),
    )


def _control_cluster_frame(
    data: pd.DataFrame,
    real_membership: pd.DataFrame,
    *,
    month_start: pd.Timestamp,
    control_type: str,
    idx: int,
) -> tuple[pd.DataFrame, ClusterCandidate]:
    prefix = f"{control_type}_cluster_p{idx:03d}"
    seed_base = 1000 if control_type == "random" else 2000
    seed = int(month_start.strftime("%Y%m")) * seed_base + idx + (777 if control_type == "shuffled" else 0)
    membership = _randomized_membership(
        real_membership,
        seed=seed,
        control_type=f"{control_type}{idx:03d}",
    )
    out = add_cluster_graph_features(data.copy(), membership, prefix=prefix)
    gate = f"c9a_{control_type}_cluster_high_p{idx:03d}"
    out[gate] = (
        _ensure_bool(out["market_volume_impulse_density_high"])
        & pd.to_numeric(out["ret_4h_percentile"], errors="coerce").ge(95)
        & _ensure_bool(out[f"{prefix}_impulse_high"])
    )
    return out, _control_candidate(control_type, idx, gate)


def _attach_cluster_trade_context(trades: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    cols = [
        "exchange",
        "symbol",
        "feature_time",
        "cluster_id",
        "cluster_size",
        "cluster_peer_count",
        "cluster_impulse_density",
        "cluster_bullish_volume_shock_count",
        "cluster_positive_return_ratio",
        "cluster_beta_extreme_count",
        "cluster_beta_extreme_density",
        "cluster_symbol_rank_within_cluster",
        "cluster_lag_vs_cluster",
        "cluster_isolated_impulse_flag",
        "market_volume_impulse_density_high",
        "volume_impulse_density",
        "ret_4h_percentile",
        "volume_z_4h",
    ]
    context_cols = [col for col in cols if col in data.columns]
    context = data[context_cols].drop_duplicates(["exchange", "symbol", "feature_time"])
    out = trades.drop(
        columns=[col for col in context_cols if col not in {"exchange", "symbol", "feature_time"}],
        errors="ignore",
    )
    return out.merge(
        context,
        left_on=["exchange", "symbol", "signal_time"],
        right_on=["exchange", "symbol", "feature_time"],
        how="left",
    ).drop(columns=["feature_time"], errors="ignore")


def _simulate_candidates(
    data: pd.DataFrame,
    candidates: list[ClusterCandidate],
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_rows: list[dict[str, object]] = []
    trade_frames: list[pd.DataFrame] = []
    for item in candidates:
        candidate = _candidate_for_gate(item)
        for baseline in item.baselines:
            trades, signal_n, pre_entry_gate_n = _simulate_variant(
                data,
                candidate,
                "signal_and_entry_gate",
                baseline,
                config,
            )
            signal_rows.append(
                {
                    "candidate": item.candidate,
                    "role": item.role,
                    "gate_col": item.gate_col,
                    "gate_mode": "signal_and_entry_gate",
                    "baseline": baseline.baseline,
                    "signals": signal_n,
                    "pre_entry_gate_trades": pre_entry_gate_n,
                }
            )
            if not trades.empty:
                trades["cluster_candidate_role"] = item.role
                trades["gate_description"] = item.description
                trade_frames.append(_attach_cluster_trade_context(trades, data))
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    signals = pd.DataFrame(signal_rows)
    return trades, signals


def _stream_v09a(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    regime: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trade_path = report_root / "_v09a_trades_tmp.csv"
    signal_path = report_root / "_v09a_signals_tmp.csv"
    for path in [trade_path, signal_path]:
        if path.exists():
            path.unlink()
    wrote_trades = False
    wrote_signals = False
    membership_frames: list[pd.DataFrame] = []
    edge_frames: list[pd.DataFrame] = []
    feature_frames: list[pd.DataFrame] = []
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
        window = _load_month_window(feature_path, rank30, rank90, symbols, regime, config, hist_start, window_end)
        if window.empty:
            continue
        bar_time = pd.to_datetime(window["bar_open_time"], utc=True, errors="coerce")
        hist = window[(bar_time >= hist_start) & (bar_time < month_start)].copy()
        membership, edges = build_cluster_membership(month_start, hist, symbols)
        sim_window = window[(bar_time >= month_start) & (bar_time < window_end)].copy()
        sim_window = add_cluster_graph_features(sim_window, membership)
        sim_window = _disable_out_of_month_events(sim_window, month_start, next_month)
        trades, signals = _simulate_candidates(sim_window, CANDIDATES, config)
        if not trades.empty:
            trades.to_csv(trade_path, mode="a", header=not wrote_trades, index=False)
            wrote_trades = True
        if not signals.empty:
            signals.to_csv(signal_path, mode="a", header=not wrote_signals, index=False)
            wrote_signals = True
        for control_type, permutations in [
            ("random", RANDOM_PERMUTATIONS),
            ("shuffled", SHUFFLED_PERMUTATIONS),
        ]:
            for perm_idx in range(permutations):
                control_window, control_candidate = _control_cluster_frame(
                    sim_window,
                    membership,
                    month_start=month_start,
                    control_type=control_type,
                    idx=perm_idx,
                )
                control_trades, control_signals = _simulate_candidates(
                    control_window,
                    [control_candidate],
                    config,
                )
                if not control_trades.empty:
                    control_trades.to_csv(trade_path, mode="a", header=not wrote_trades, index=False)
                    wrote_trades = True
                if not control_signals.empty:
                    control_signals.to_csv(signal_path, mode="a", header=not wrote_signals, index=False)
                    wrote_signals = True
                del control_window, control_trades, control_signals
                _release_memory()
        if not membership.empty:
            membership_frames.append(membership)
        if not edges.empty:
            edge_frames.append(edges)
        feature_frames.append(_cluster_feature_snapshot(sim_window, month_start))
        print(f"v0.9A month {idx}/{len(months)} {month_start:%Y-%m} symbols={len(symbols)}", flush=True)
        del window, hist, sim_window, trades, signals
        _release_memory()
    trades = pd.read_csv(trade_path, low_memory=False) if wrote_trades else pd.DataFrame()
    signals = pd.read_csv(signal_path, low_memory=False) if wrote_signals else pd.DataFrame()
    for path in [trade_path, signal_path]:
        if path.exists():
            path.unlink()
    if not signals.empty:
        signals = signals.groupby(
            ["candidate", "role", "gate_col", "gate_mode", "baseline"],
            as_index=False,
            sort=False,
            dropna=False,
        ).agg(signals=("signals", "sum"), pre_entry_gate_trades=("pre_entry_gate_trades", "sum"))
    membership = pd.concat(membership_frames, ignore_index=True) if membership_frames else pd.DataFrame()
    edges = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    features = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
    return trades, signals, membership, edges, features


def _write_regime_cache_worker(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    symbols: list[str],
    config: ExperimentConfig,
    regime_path: Path,
) -> None:
    regime = _build_regime_streaming(feature_path, rank30, rank90, symbols, config)
    regime.to_parquet(regime_path, index=False)


def _write_month_worker(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    regime_path: Path,
    config: ExperimentConfig,
    report_root: Path,
    month_start: pd.Timestamp,
    next_month: pd.Timestamp,
    month_idx: int,
    month_total: int,
) -> None:
    report_root = ensure_dir(report_root)
    month_start = pd.Timestamp(month_start)
    next_month = pd.Timestamp(next_month)
    paths = _month_artifact_paths(report_root, month_start)
    for path in paths.values():
        if path.exists():
            path.unlink()
    symbols = _month_symbols(rank30, month_start)
    if not symbols:
        for path in paths.values():
            path.write_text("", encoding="utf-8")
        print(f"v0.9A month {month_idx}/{month_total} {month_start:%Y-%m} symbols=0", flush=True)
        return
    hist_start = month_start - pd.Timedelta(days=GRAPH_LOOKBACK_DAYS)
    window_end = next_month + pd.Timedelta(minutes=15 * SIM_WINDOW_BARS)
    regime = pd.read_parquet(regime_path)
    if "feature_time" in regime.columns:
        regime_time = pd.to_datetime(regime["feature_time"], utc=True, errors="coerce")
        regime = regime[(regime_time >= hist_start) & (regime_time < window_end)].copy()
    window = _load_month_window(feature_path, rank30, rank90, symbols, regime, config, hist_start, window_end)
    if window.empty:
        for path in paths.values():
            path.write_text("", encoding="utf-8")
        print(f"v0.9A month {month_idx}/{month_total} {month_start:%Y-%m} symbols={len(symbols)} empty", flush=True)
        return
    bar_time = pd.to_datetime(window["bar_open_time"], utc=True, errors="coerce")
    hist = window[(bar_time >= hist_start) & (bar_time < month_start)].copy()
    membership, edges = build_cluster_membership(month_start, hist, symbols)
    sim_window = window[(bar_time >= month_start) & (bar_time < window_end)].copy()
    sim_window = add_cluster_graph_features(sim_window, membership)
    sim_window = _disable_out_of_month_events(sim_window, month_start, next_month)
    trade_frames: list[pd.DataFrame] = []
    signal_frames: list[pd.DataFrame] = []
    trades, signals = _simulate_candidates(sim_window, CANDIDATES, config)
    if not trades.empty:
        trade_frames.append(trades)
    if not signals.empty:
        signal_frames.append(signals)
    for control_type, permutations in [
        ("random", RANDOM_PERMUTATIONS),
        ("shuffled", SHUFFLED_PERMUTATIONS),
    ]:
        for perm_idx in range(permutations):
            control_window, control_candidate = _control_cluster_frame(
                sim_window,
                membership,
                month_start=month_start,
                control_type=control_type,
                idx=perm_idx,
            )
            control_trades, control_signals = _simulate_candidates(control_window, [control_candidate], config)
            if not control_trades.empty:
                trade_frames.append(control_trades)
            if not control_signals.empty:
                signal_frames.append(control_signals)
            del control_window, control_trades, control_signals
            _release_memory()
    feature_rows = _cluster_feature_snapshot(sim_window, month_start)
    _write_csv_frame(pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame(), paths["trades"])
    _write_csv_frame(pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame(), paths["signals"])
    _write_csv_frame(membership, paths["membership"])
    _write_csv_frame(edges, paths["edges"])
    _write_csv_frame(feature_rows, paths["features"])
    print(f"v0.9A month {month_idx}/{month_total} {month_start:%Y-%m} symbols={len(symbols)}", flush=True)


def _stream_v09a_batched(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    symbols: list[str],
    config: ExperimentConfig,
    report_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    report_root = ensure_dir(report_root)
    for path in report_root.glob("_v09a_month_*_*.csv"):
        path.unlink()
    for path in [
        report_root / "_v09a_regime_cache.parquet",
        report_root / "_v09a_trades_tmp.csv",
        report_root / "_v09a_signals_tmp.csv",
    ]:
        if path.exists():
            path.unlink()
    regime_path = report_root / "_v09a_regime_cache.parquet"
    _run_child_process(
        _write_regime_cache_worker,
        (feature_path, rank30, rank90, symbols, config, regime_path),
        "v09a-regime-cache",
    )
    months = sorted(
        pd.to_datetime(rank30["month_start"], utc=True, errors="coerce")
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    for idx, month_start in enumerate(months, start=1):
        month_start = pd.Timestamp(month_start)
        next_month = pd.Timestamp(months[idx]) if idx < len(months) else month_start + pd.DateOffset(months=1)
        _run_child_process(
            _write_month_worker,
            (feature_path, rank30, rank90, regime_path, config, report_root, month_start, next_month, idx, len(months)),
            f"v09a-month-{month_start:%Y-%m}",
        )
    month_paths = [_month_artifact_paths(report_root, pd.Timestamp(month)) for month in months]
    trades = _read_csv_frames([paths["trades"] for paths in month_paths])
    signals = _read_csv_frames([paths["signals"] for paths in month_paths])
    membership = _read_csv_frames([paths["membership"] for paths in month_paths])
    edges = _read_csv_frames([paths["edges"] for paths in month_paths])
    features = _read_csv_frames([paths["features"] for paths in month_paths])
    if not signals.empty:
        signals = signals.groupby(
            ["candidate", "role", "gate_col", "gate_mode", "baseline"],
            as_index=False,
            sort=False,
            dropna=False,
        ).agg(signals=("signals", "sum"), pre_entry_gate_trades=("pre_entry_gate_trades", "sum"))
    if regime_path.exists():
        regime_path.unlink()
    for paths in month_paths:
        for path in paths.values():
            if path.exists():
                path.unlink()
    return trades, signals, membership, edges, features


def _cluster_feature_snapshot(data: pd.DataFrame, month_start: pd.Timestamp) -> pd.DataFrame:
    cols = [
        "symbol",
        "cluster_id",
        "cluster_size",
        "cluster_impulse_density",
        "cluster_positive_return_ratio",
        "cluster_beta_extreme_density",
        "cluster_symbol_rank_within_cluster",
        "cluster_lag_vs_cluster",
        "cluster_isolated_impulse_flag",
        "market_volume_impulse_density_high",
        "ret_4h_percentile",
    ]
    sample = data[
        (pd.to_datetime(data["feature_time"], utc=True, errors="coerce") >= month_start)
        & (pd.to_datetime(data["feature_time"], utc=True, errors="coerce") < month_start + pd.DateOffset(months=1))
        & pd.to_numeric(data["dynamic_all_rank"], errors="coerce").le(TOP_N)
        & _ensure_bool(data["bullish_volume_shock_event"])
    ].copy()
    if sample.empty:
        return pd.DataFrame()
    out = sample[[col for col in cols if col in sample.columns]].copy()
    out.insert(0, "month_start", month_start)
    return out


def _gate_summary(trades: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    long_summary = _summary_long(trades, signals.rename(columns={"role": "candidate_role"}))
    rows = []
    for item in CANDIDATES:
        sample = trades[
            trades["candidate"].astype(str).eq(item.candidate)
            & trades["baseline"].astype(str).eq("candidate_reclaim")
        ]
        if sample.empty:
            continue
        net10 = _metric(sample, 10)
        row = {
            "candidate": item.candidate,
            "role": item.role,
            "negative_control": item.negative_control,
            "gate_col": item.gate_col,
            "signals": int(_lookup(long_summary, item.candidate, "signal_and_entry_gate", "candidate_reclaim", 10, "signals")),
            "pre_entry_gate_trades": int(
                _lookup(long_summary, item.candidate, "signal_and_entry_gate", "candidate_reclaim", 10, "pre_entry_gate_trades")
            ),
            "trades": int(
                len(sample[pd.to_numeric(sample["cost_single_side_bps"], errors="coerce").eq(10.0)])
            ),
            "net10": net10,
            "net20": _metric(sample, 20),
            "net30": _metric(sample, 30),
            "net50": _metric(sample, 50),
            f"ex_{FOCAL_MONTH}_net10": _ex_month_net(sample, FOCAL_MONTH, 10),
            "ex_top_month_net10": _ex_top_month_net(sample, 10),
            "month_cap35_net20": _month_cap_expectancy(sample, 20, 0.35),
            "max_month_contribution": _max_contribution(sample, "month", 10),
            "max_symbol_contribution": _max_contribution(sample, "symbol", 10),
            "cluster_impulse_density_median": _median(sample, "cluster_impulse_density", 10),
            "cluster_size_median": _median(sample, "cluster_size", 10),
        }
        for baseline, col in [
            ("entry_only_reclaim", "entry_only_lift"),
            ("matched_random_reclaim", "matched_random_lift"),
            ("volume_shock_next_open", "volume_shock_next_open_lift"),
            ("volume_shock_pullback_only", "volume_shock_pullback_lift"),
        ]:
            row[col] = net10 - _lookup(long_summary, item.candidate, "signal_and_entry_gate", baseline, 10, "net_expectancy")
        rows.append(row)
    return pd.DataFrame(rows)


def _median(sample: pd.DataFrame, col: str, cost: float) -> float:
    data = sample[pd.to_numeric(sample["cost_single_side_bps"], errors="coerce").eq(float(cost))]
    if data.empty or col not in data.columns:
        return np.nan
    return float(pd.to_numeric(data[col], errors="coerce").median())


def _ex_top_month_net(sample: pd.DataFrame, cost: float) -> float:
    data = sample[pd.to_numeric(sample["cost_single_side_bps"], errors="coerce").eq(float(cost))]
    if data.empty:
        return np.nan
    top_month = data.groupby("month", dropna=False, sort=False)["net_return"].sum().abs().idxmax()
    return _ex_month_net(sample, str(top_month), cost)


def _cluster_control_summary(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    candidate = trades[trades["baseline"].astype(str).eq("candidate_reclaim")].copy()
    cost10 = candidate[pd.to_numeric(candidate["cost_single_side_bps"], errors="coerce").eq(10.0)]
    real = _metric(candidate[candidate["candidate"].astype(str).eq("CG2_cic1_cluster_high_market_high")], 10)
    random_rows = []
    shuffled_rows = []
    for label, frame, rows in [
        ("random_cluster_control", cost10[cost10["cluster_candidate_role"].astype(str).eq("random_cluster_control")], random_rows),
        ("shuffled_cluster_control", cost10[cost10["cluster_candidate_role"].astype(str).eq("shuffled_cluster_control")], shuffled_rows),
    ]:
        for name, group in frame.groupby("candidate", sort=False):
            rows.append(
                {
                    "control_type": label,
                    "candidate": name,
                    "trades": int(len(group)),
                    "net10": float(pd.to_numeric(group["net_return"], errors="coerce").mean()),
                    "net20": _metric(candidate[candidate["candidate"].astype(str).eq(str(name))], 20),
                    "real_cg2_net10": real,
                    "real_minus_control_net10": real - float(pd.to_numeric(group["net_return"], errors="coerce").mean()),
                }
            )
    random_df = pd.DataFrame(random_rows)
    shuffled_df = pd.DataFrame(shuffled_rows)
    summary_rows = []
    for label, df in [("random", random_df), ("shuffled", shuffled_df)]:
        if df.empty:
            continue
        vals = pd.to_numeric(df["net10"], errors="coerce")
        summary_rows.append(
            {
                "control_type": label,
                "real_cg2_net10": real,
                "control_mean_net10": float(vals.mean()),
                "control_median_net10": float(vals.median()),
                "control_p75_net10": float(vals.quantile(0.75)),
                "control_p90_net10": float(vals.quantile(0.90)),
                "real_percentile_vs_control": float((vals <= real).mean()),
            }
        )
    return pd.DataFrame(summary_rows), random_df, shuffled_df


def _cic_cluster_overlap(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    mapping = [
        ("CIC1_reference", "CG0_CIC1_reference"),
        ("CIC1_cluster_high", "CG2_cic1_cluster_high_market_high"),
        ("CIC1_cluster_low", "CG4_market_high_cluster_low"),
        ("CIC1_isolated", "CG6_isolated_beta_extreme"),
        ("CIC2_cluster_high", "CG5_cic2_cluster_high"),
        ("cluster_high_market_not_high", "CG3_cluster_high_market_not_high"),
    ]
    for bucket, candidate in mapping:
        sample = trades[
            trades["candidate"].astype(str).eq(candidate)
            & trades["baseline"].astype(str).eq("candidate_reclaim")
        ]
        rows.append(
            {
                "bucket": bucket,
                "candidate": candidate,
                "trades": int(len(sample[pd.to_numeric(sample.get("cost_single_side_bps"), errors="coerce").eq(10.0)])),
                "net10": _metric(sample, 10),
                "net20": _metric(sample, 20),
                "month_cap35_net20": _month_cap_expectancy(sample, 20, 0.35),
                "max_symbol_contribution": _max_contribution(sample, "symbol", 10),
                "cluster_impulse_density_median": _median(sample, "cluster_impulse_density", 10),
            }
        )
    return pd.DataFrame(rows)


def _cluster_feature_summary(feature_rows: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not membership.empty:
        for month, group in membership.groupby("month_start", sort=False):
            rows.append(
                {
                    "section": "membership",
                    "month_start": month,
                    "symbols": int(group["symbol"].nunique()),
                    "clusters": int(group["cluster_id"].nunique()),
                    "median_cluster_size": float(pd.to_numeric(group["cluster_size"], errors="coerce").median()),
                    "max_cluster_size": int(pd.to_numeric(group["cluster_size"], errors="coerce").max()),
                }
            )
    if not feature_rows.empty:
        rows.append(
            {
                "section": "signal_features",
                "month_start": "all",
                "symbols": int(feature_rows["symbol"].nunique()),
                "clusters": int(feature_rows["cluster_id"].nunique()),
                "median_cluster_size": float(pd.to_numeric(feature_rows["cluster_size"], errors="coerce").median()),
                "max_cluster_size": int(pd.to_numeric(feature_rows["cluster_size"], errors="coerce").max()),
                "cluster_impulse_density_median": float(
                    pd.to_numeric(feature_rows["cluster_impulse_density"], errors="coerce").median()
                ),
                "cluster_impulse_density_p75": float(
                    pd.to_numeric(feature_rows["cluster_impulse_density"], errors="coerce").quantile(0.75)
                ),
            }
        )
    return pd.DataFrame(rows)


def _portfolio_cluster_width(trades: pd.DataFrame) -> pd.DataFrame:
    base = trades[
        trades["baseline"].astype(str).eq("candidate_reclaim")
        & pd.to_numeric(trades["cost_single_side_bps"], errors="coerce").isin([10.0, 20.0])
        & trades["candidate"].astype(str).isin(
            [
                "CG0_CIC1_reference",
                "CG2_cic1_cluster_high_market_high",
                "CG5_cic2_cluster_high",
            ]
        )
    ].copy()
    if base.empty:
        return pd.DataFrame()
    base["entry_time"] = pd.to_datetime(base["entry_time"], utc=True, errors="coerce")
    base["exit_time"] = pd.to_datetime(base["exit_time"], utc=True, errors="coerce")
    rankings = {
        "first_come_first_served": lambda df: -df["entry_time"].astype("int64"),
        "cluster_impulse_density_high": lambda df: pd.to_numeric(df["cluster_impulse_density"], errors="coerce").fillna(-np.inf),
        "cluster_size_high": lambda df: pd.to_numeric(df["cluster_size"], errors="coerce").fillna(-np.inf),
        "beta_extreme_strength_high": lambda df: pd.to_numeric(df["ret_4h_percentile"], errors="coerce").fillna(-np.inf),
        "local_volume_shock_strength_high": lambda df: pd.to_numeric(df["volume_z_4h"], errors="coerce").fillna(-np.inf),
        "market_impulse_density_high": lambda df: pd.to_numeric(df["volume_impulse_density"], errors="coerce").fillna(-np.inf),
        "liquidity_high": lambda df: -pd.to_numeric(df["dynamic_all_rank"], errors="coerce").fillna(999.0),
    }
    rows = []
    for (candidate, cost), group in base.groupby(["candidate", "cost_single_side_bps"], sort=False):
        for ranking, scorer in rankings.items():
            local = group.copy()
            local["score"] = scorer(local)
            ranked = local.sort_values(["entry_time", "score", "symbol"], ascending=[True, False, True])
            for max_positions in [1, 3, 5, 10, 10_000]:
                active: list[pd.Timestamp] = []
                selected = []
                skipped = 0
                for row in ranked.itertuples(index=False):
                    entry = pd.Timestamp(row.entry_time)
                    active = [exit_time for exit_time in active if exit_time > entry]
                    if len(active) >= max_positions:
                        skipped += 1
                        continue
                    selected.append(row)
                    active.append(pd.Timestamp(row.exit_time))
                net = pd.Series([float(getattr(row, "net_return", np.nan)) for row in selected], dtype="float64")
                equity = net.cumsum()
                dd = equity - equity.cummax()
                rows.append(
                    {
                        "candidate": candidate,
                        "cost_single_side_bps": cost,
                        "ranking": ranking,
                        "max_positions": "unlimited" if max_positions >= 10_000 else max_positions,
                        "selected_trades": int(len(selected)),
                        "skipped_trades": int(skipped),
                        "net_expectancy": float(net.mean()) if len(net) else np.nan,
                        "total_net": float(net.sum()) if len(net) else 0.0,
                        "max_drawdown_proxy": float(dd.min()) if len(dd) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _write_notes(report_root: Path, gate_summary: pd.DataFrame, control_summary: pd.DataFrame) -> None:
    lines = [
        "# v0.9A Cluster Impulse Graph",
        "",
        "Purpose: test whether CIC-filtered MIR1 is improved by a prior-window cluster impulse graph node.",
        "",
        "No CIC parameter tuning was performed. The only new gates are cluster-context gates and structural controls.",
        "",
    ]
    for candidate in ["CG0_CIC1_reference", "CG2_cic1_cluster_high_market_high", "CG3_cluster_high_market_not_high", "CG4_market_high_cluster_low"]:
        row = gate_summary[gate_summary["candidate"].astype(str).eq(candidate)].head(1)
        if not row.empty:
            item = row.iloc[0]
            lines.append(
                f"- {candidate}: trades={int(item.get('trades', 0))}, "
                f"net10={item.get('net10', np.nan):.4%}, net20={item.get('net20', np.nan):.4%}, "
                f"month_cap35_net20={item.get('month_cap35_net20', np.nan):.4%}."
            )
    if not control_summary.empty:
        for row in control_summary.itertuples(index=False):
            lines.append(
                f"- real CG2 vs {row.control_type}: real={row.real_cg2_net10:.4%}, "
                f"median={row.control_median_net10:.4%}, p90={row.control_p90_net10:.4%}, "
                f"percentile={row.real_percentile_vs_control:.2%}."
            )
    lines.extend(
        [
            "",
            "Interpretation rule:",
            "- If CG2 beats random/shuffled controls and CG4 is weak, cluster impulse is a real graph node.",
            "- If random/shuffled are similar or stronger, the effect is still mostly market-density proxy.",
        ]
    )
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v09a_cluster_impulse_graph(
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
    trades, signals, membership, edges, feature_rows = _stream_v09a_batched(
        feature_path,
        rank30,
        rank90,
        symbols,
        config,
        report_root,
    )
    gate_summary = _gate_summary(trades, signals)
    control_summary, random_perm, shuffled = _cluster_control_summary(trades)
    overlap = _cic_cluster_overlap(trades)
    feature_summary = _cluster_feature_summary(feature_rows, membership)
    portfolio = _portfolio_cluster_width(trades)
    outputs = {
        "cluster_membership": report_root / "cluster_membership.csv",
        "cluster_edges": report_root / "cluster_edges.csv",
        "cluster_feature_summary": report_root / "cluster_feature_summary.csv",
        "cluster_gate_summary": report_root / "cluster_gate_summary.csv",
        "cluster_control_summary": report_root / "cluster_control_summary.csv",
        "cluster_random_permutation": report_root / "cluster_random_permutation.csv",
        "cluster_shuffled_summary": report_root / "cluster_shuffled_summary.csv",
        "cic_cluster_overlap": report_root / "cic_cluster_overlap.csv",
        "portfolio_cluster_width": report_root / "portfolio_cluster_width.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    membership.to_csv(outputs["cluster_membership"], index=False)
    edges.to_csv(outputs["cluster_edges"], index=False)
    feature_summary.to_csv(outputs["cluster_feature_summary"], index=False)
    gate_summary.to_csv(outputs["cluster_gate_summary"], index=False)
    control_summary.to_csv(outputs["cluster_control_summary"], index=False)
    random_perm.to_csv(outputs["cluster_random_permutation"], index=False)
    shuffled.to_csv(outputs["cluster_shuffled_summary"], index=False)
    overlap.to_csv(outputs["cic_cluster_overlap"], index=False)
    portfolio.to_csv(outputs["portfolio_cluster_width"], index=False)
    _write_notes(report_root, gate_summary, control_summary)
    return outputs


__all__ = [
    "REPORT_ROOT",
    "add_cluster_graph_features",
    "build_cluster_membership",
    "write_v09a_cluster_impulse_graph",
]
