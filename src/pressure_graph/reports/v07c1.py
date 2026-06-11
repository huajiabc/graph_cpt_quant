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
    BaselineSpec,
    FrozenAtlasCandidate,
    _simulate_variant,
)
from pressure_graph.reports.v07b import (
    GRAPH_LOOKBACK_DAYS,
    TOP_N,
    _disable_out_of_month_events,
    _ensure_bool,
    _load_month_window,
    _month_symbols,
)
from pressure_graph.reports.v07b2 import _aggregate_permutations, _numeric_context
from pressure_graph.reports.v07c import (
    CORE_BASELINE,
    EDGE_TOP_K,
    LEADER_RECENT_BARS,
    RANDOM_PERMUTATIONS,
    SHUFFLED_PERMUTATIONS,
    _active_edge_weight,
    _build_directed_edges,
    _density_random_leaders,
    _edge_weight_map,
    _future_any,
    _low_edge_leader_map,
    _max_from_map,
    _past_any,
    _rank_bucket_map,
    _ratio_from_map,
    _shuffled_leaders,
    _summarize_candidates,
    _top_leader_map,
)


REPORT_ROOT = Path("reports/v0_7c1_gate_width_audit")


@dataclass(frozen=True)
class AuditGate:
    candidate: str
    role: str
    gate_col: str
    description: str
    negative_control: bool = False
    baseline: BaselineSpec = CORE_BASELINE


FUNNEL_GATES = [
    AuditGate("S0_MIR1_raw", "funnel", "audit_market_density", "MIR1 raw market impulse-density gate."),
    AuditGate("S1_directed_edge_exists", "funnel", "audit_directed_edge_exists", "MIR1 plus a prior-window directed leader edge exists."),
    AuditGate("S2_leader_prior_4h", "funnel", "audit_leader_prior_4h", "MIR1 plus directed leader impulse in prior 4h."),
    AuditGate("S3_beta_not_extended", "funnel", "audit_beta_not_extended", "Prior leader impulse plus beta ret_4h percentile < 80."),
]

BETA_BUCKET_GATES = [
    AuditGate("B0_beta_laggard", "beta_extension_bucket", "audit_beta_bucket_laggard", "Beta lags leader by at least 2pct or ret_4h percentile < 40."),
    AuditGate("B1_beta_neutral", "beta_extension_bucket", "audit_beta_bucket_neutral", "Beta ret_4h percentile 40-60."),
    AuditGate("B2_beta_mildly_extended", "beta_extension_bucket", "audit_beta_bucket_mildly_extended", "Beta ret_4h percentile 60-80."),
    AuditGate("B3_beta_strongly_extended", "beta_extension_bucket", "audit_beta_bucket_strongly_extended", "Beta ret_4h percentile 80-95."),
    AuditGate("B4_beta_overextended", "beta_extension_bucket", "audit_beta_bucket_overextended", "Beta ret_4h percentile >= 95."),
]

LEADER_WINDOW_GATES = [
    AuditGate("leader_impulse_15m", "leader_window", "audit_leader_prior_15m", "Directed leader impulse in prior 15m."),
    AuditGate("leader_impulse_1h", "leader_window", "audit_leader_prior_1h", "Directed leader impulse in prior 1h."),
    AuditGate("leader_impulse_4h", "leader_window", "audit_leader_prior_4h", "Directed leader impulse in prior 4h."),
    AuditGate("leader_impulse_12h", "leader_window", "audit_leader_prior_12h", "Directed leader impulse in prior 12h."),
]

CONTROL_GATES = [
    AuditGate("low_edge_leader", "control", "audit_low_edge_leader_prior_4h", "Low edge-weight leader impulse control.", True),
    AuditGate("no_leader_impulse", "control", "audit_no_leader_prior_4h", "No directed leader impulse in prior 4h.", True),
    AuditGate("future_leader_1h", "future_audit", "audit_future_leader_1h", "Future leader impulse in next 1h. Audit only.", True),
    AuditGate("future_leader_4h", "future_audit", "audit_future_leader_4h", "Future leader impulse in next 4h. Audit only.", True),
]

ALL_FIXED_GATES = [*FUNNEL_GATES, *BETA_BUCKET_GATES, *LEADER_WINDOW_GATES, *CONTROL_GATES]


def _candidate(gate: AuditGate) -> FrozenAtlasCandidate:
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


def _context_cols(data: pd.DataFrame) -> list[str]:
    cols = [
        "exchange",
        "symbol",
        "feature_time",
        "volume_impulse_density",
        "dynamic_all_rank",
        "volume_z_1h",
        "ret_4h",
        "ret_4h_percentile",
        "close_location_value",
        "upper_wick_ratio",
        "audit_directed_edge_exists",
        "audit_leader_prior_15m_ratio",
        "audit_leader_prior_1h_ratio",
        "audit_leader_prior_4h_ratio",
        "audit_leader_prior_12h_ratio",
        "audit_low_edge_leader_prior_4h_ratio",
        "audit_future_leader_1h_ratio",
        "audit_future_leader_4h_ratio",
        "audit_directed_leader_return_4h_max",
        "audit_directed_edge_weight_active",
        "audit_beta_extension_bucket",
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


def _bucket_beta_extension(beta_ret: pd.Series, beta_pct: pd.Series, leader_ret: pd.Series) -> pd.Series:
    gap = leader_ret - beta_ret
    out = pd.Series("neutral", index=beta_ret.index, dtype="object")
    out[(gap >= 0.02) | (beta_pct < 40)] = "laggard"
    out[(beta_pct >= 40) & (beta_pct < 60) & (gap < 0.02)] = "neutral"
    out[(beta_pct >= 60) & (beta_pct < 80)] = "mildly_extended"
    out[(beta_pct >= 80) & (beta_pct < 95)] = "strongly_extended"
    out[beta_pct >= 95] = "overextended"
    return out


def _add_audit_features(data: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    defaults: dict[str, object] = {
        "audit_directed_edge_exists": False,
        "audit_leader_prior_15m_ratio": 0.0,
        "audit_leader_prior_1h_ratio": 0.0,
        "audit_leader_prior_4h_ratio": 0.0,
        "audit_leader_prior_12h_ratio": 0.0,
        "audit_low_edge_leader_prior_4h_ratio": 0.0,
        "audit_future_leader_1h_ratio": 0.0,
        "audit_future_leader_4h_ratio": 0.0,
        "audit_directed_leader_return_4h_max": np.nan,
        "audit_directed_edge_weight_active": 0.0,
        "audit_beta_extension_bucket": "unknown",
        "leader_beta_cluster_id": "",
    }
    for col, value in defaults.items():
        out[col] = value
    if out.empty:
        return _add_audit_gates(out)
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
        month_edges = (
            edges[edges["month_start"].eq(month_start)]
            if "month_start" in edges.columns
            else pd.DataFrame()
        )
        real_leaders = _top_leader_map(month_edges)
        low_edge_leaders = _low_edge_leader_map(month_edges, symbols)
        weights = _edge_weight_map(month_edges)
        past_windows = {
            "15m": _past_any(event, 1),
            "1h": _past_any(event, 4),
            "4h": _past_any(event, LEADER_RECENT_BARS),
            "12h": _past_any(event, 48),
        }
        low_edge_4h = _past_any(event, LEADER_RECENT_BARS)
        future_1h = _future_any(event, 4)
        future_4h = _future_any(event, 16)
        for beta, group in sample.groupby("symbol", observed=True, sort=False):
            beta = str(beta)
            idx = group.index
            feature_time = pd.to_datetime(group["feature_time"], utc=True, errors="coerce")
            leaders = real_leaders.get(beta, [])
            out.loc[idx, "audit_directed_edge_exists"] = bool(leaders)
            for label, state in past_windows.items():
                out.loc[idx, f"audit_leader_prior_{label}_ratio"] = _ratio_from_map(state, feature_time, real_leaders, beta)
            out.loc[idx, "audit_low_edge_leader_prior_4h_ratio"] = _ratio_from_map(
                low_edge_4h,
                feature_time,
                low_edge_leaders,
                beta,
            )
            out.loc[idx, "audit_future_leader_1h_ratio"] = _ratio_from_map(future_1h, feature_time, real_leaders, beta)
            out.loc[idx, "audit_future_leader_4h_ratio"] = _ratio_from_map(future_4h, feature_time, real_leaders, beta)
            out.loc[idx, "audit_directed_leader_return_4h_max"] = _max_from_map(ret4, feature_time, real_leaders, beta)
            out.loc[idx, "audit_directed_edge_weight_active"] = _active_edge_weight(
                past_windows["4h"],
                feature_time,
                real_leaders,
                weights,
                beta,
            )
            top_leader = leaders[0] if leaders else "none"
            out.loc[idx, "leader_beta_cluster_id"] = f"{month_start:%Y-%m}:lead:{top_leader}:beta:{beta}"
        beta_ret = pd.to_numeric(sample.get("ret_4h"), errors="coerce")
        beta_pct = pd.to_numeric(sample.get("ret_4h_percentile"), errors="coerce")
        leader_ret = pd.to_numeric(out.loc[sample.index, "audit_directed_leader_return_4h_max"], errors="coerce")
        out.loc[sample.index, "audit_beta_extension_bucket"] = _bucket_beta_extension(beta_ret, beta_pct, leader_ret).to_numpy()
    return _add_audit_gates(out)


def _add_audit_gates(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    market = _ensure_bool(out.get("market_volume_impulse_density_high", pd.Series(False, index=out.index)))
    edge_exists = _ensure_bool(out.get("audit_directed_edge_exists", pd.Series(False, index=out.index)))
    leader_15m = pd.to_numeric(out.get("audit_leader_prior_15m_ratio"), errors="coerce").fillna(0.0) >= 0.20
    leader_1h = pd.to_numeric(out.get("audit_leader_prior_1h_ratio"), errors="coerce").fillna(0.0) >= 0.20
    leader_4h = pd.to_numeric(out.get("audit_leader_prior_4h_ratio"), errors="coerce").fillna(0.0) >= 0.20
    leader_12h = pd.to_numeric(out.get("audit_leader_prior_12h_ratio"), errors="coerce").fillna(0.0) >= 0.20
    low_edge_4h = pd.to_numeric(out.get("audit_low_edge_leader_prior_4h_ratio"), errors="coerce").fillna(0.0) >= 0.20
    future_1h = pd.to_numeric(out.get("audit_future_leader_1h_ratio"), errors="coerce").fillna(0.0) >= 0.20
    future_4h = pd.to_numeric(out.get("audit_future_leader_4h_ratio"), errors="coerce").fillna(0.0) >= 0.20
    bucket = out.get("audit_beta_extension_bucket", pd.Series("unknown", index=out.index)).astype(str)
    out["audit_market_density"] = market
    out["audit_directed_edge_exists"] = market & edge_exists
    out["audit_leader_prior_15m"] = market & edge_exists & leader_15m
    out["audit_leader_prior_1h"] = market & edge_exists & leader_1h
    out["audit_leader_prior_4h"] = market & edge_exists & leader_4h
    out["audit_leader_prior_12h"] = market & edge_exists & leader_12h
    out["audit_beta_not_extended"] = out["audit_leader_prior_4h"] & bucket.isin(["laggard", "neutral", "mildly_extended"])
    out["audit_low_edge_leader_prior_4h"] = market & low_edge_4h
    out["audit_no_leader_prior_4h"] = market & edge_exists & ~leader_4h
    out["audit_future_leader_1h"] = market & edge_exists & future_1h
    out["audit_future_leader_4h"] = market & edge_exists & future_4h
    for item in ["laggard", "neutral", "mildly_extended", "strongly_extended", "overextended"]:
        out[f"audit_beta_bucket_{item}"] = out["audit_leader_prior_4h"] & bucket.eq(item)
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
    out.loc[sample.index, gate_col] = (market & (ratios >= 0.20)).to_numpy(dtype=bool)
    return out


def _simulate_gate(data: pd.DataFrame, gate: AuditGate, config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = _candidate(gate)
    trades, signal_n, pre_entry_gate_n = _simulate_variant(
        data,
        candidate,
        "signal_and_entry_gate",
        gate.baseline,
        config,
    )
    signals = pd.DataFrame(
        [
            {
                "candidate": gate.candidate,
                "gate_mode": "signal_and_entry_gate",
                "baseline": gate.baseline.baseline,
                "signals": signal_n,
                "pre_entry_gate_trades": pre_entry_gate_n,
            }
        ]
    )
    if trades.empty:
        return trades, signals
    trades["audit_role"] = gate.role
    trades["audit_description"] = gate.description
    return _attach_context(trades, data), signals


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
    window = _load_month_window(feature_path, rank30, rank90, symbols, regime, config, hist_start, window_end)
    if window.empty:
        return pd.DataFrame(), pd.DataFrame(), window, symbols
    hist = window[
        (pd.to_datetime(window["bar_open_time"], utc=True, errors="coerce") >= hist_start)
        & (pd.to_datetime(window["bar_open_time"], utc=True, errors="coerce") < month_start)
    ].copy()
    edges = _build_directed_edges(month_start, hist, symbols)
    sim_window = window[
        (pd.to_datetime(window["bar_open_time"], utc=True, errors="coerce") >= month_start)
        & (pd.to_datetime(window["bar_open_time"], utc=True, errors="coerce") < window_end)
    ].copy()
    sim_window = _add_audit_features(sim_window, edges)
    sim_window = _disable_out_of_month_events(sim_window, month_start, next_month)
    return sim_window, edges, window, symbols


def _raw_gate_counts(data: pd.DataFrame, gates: list[AuditGate]) -> pd.DataFrame:
    rows = []
    rank = pd.to_numeric(data["dynamic_all_rank"], errors="coerce").le(TOP_N)
    local = _ensure_bool(data.get("bullish_volume_shock_event", pd.Series(False, index=data.index)))
    for gate in gates:
        mask = rank & _ensure_bool(data.get(gate.gate_col, pd.Series(False, index=data.index)))
        rows.append(
            {
                "candidate": gate.candidate,
                "role": gate.role,
                "gate_col": gate.gate_col,
                "raw_rows": int(mask.sum()),
                "local_signal_events": int((mask & local).sum()),
            }
        )
    return pd.DataFrame(rows)


def _graph_coverage(sim_window: pd.DataFrame, edges: pd.DataFrame, month_start: pd.Timestamp) -> pd.DataFrame:
    sample = sim_window[pd.to_numeric(sim_window["dynamic_all_rank"], errors="coerce") <= TOP_N].copy()
    if sample.empty:
        return pd.DataFrame()
    rows = []
    for symbol, group in sample.groupby("symbol", observed=True, sort=False):
        symbol = str(symbol)
        local_edges = edges[edges["beta_symbol"].astype(str).eq(symbol)] if "beta_symbol" in edges.columns else pd.DataFrame()
        leaders = local_edges[local_edges["edge_rank"].le(EDGE_TOP_K)] if "edge_rank" in local_edges.columns else local_edges
        local_events = _ensure_bool(group.get("bullish_volume_shock_event", pd.Series(False, index=group.index)))
        leader_gate = _ensure_bool(group.get("audit_leader_prior_4h", pd.Series(False, index=group.index)))
        rows.append(
            {
                "month_start": month_start,
                "symbol": symbol,
                "num_leaders": int(len(leaders)),
                "avg_edge_weight": float(pd.to_numeric(leaders.get("edge_weight"), errors="coerce").mean()) if not leaders.empty else np.nan,
                "max_edge_weight": float(pd.to_numeric(leaders.get("edge_weight"), errors="coerce").max()) if not leaders.empty else np.nan,
                "leader_signal_count": int(leader_gate.sum()),
                "beta_signal_count": int(local_events.sum()),
                "leader_beta_overlap_count": int((leader_gate & local_events).sum()),
            }
        )
    return pd.DataFrame(rows)


def _stream_reports(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    regime: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trade_path = report_root / "_v07c1_trades_tmp.csv"
    signal_path = report_root / "_v07c1_signals_tmp.csv"
    raw_path = report_root / "_v07c1_raw_tmp.csv"
    for path in [trade_path, signal_path, raw_path]:
        if path.exists():
            path.unlink()
    wrote_trades = False
    wrote_signals = False
    wrote_raw = False
    edge_frames: list[pd.DataFrame] = []
    coverage_frames: list[pd.DataFrame] = []
    months = sorted(
        pd.to_datetime(rank30["month_start"], utc=True, errors="coerce")
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    for idx, month_start in enumerate(months, start=1):
        month_start = pd.Timestamp(month_start)
        next_month = pd.Timestamp(months[idx]) if idx < len(months) else month_start + pd.DateOffset(months=1)
        sim_window, edges, _, symbols = _month_setup(feature_path, rank30, rank90, regime, config, month_start, next_month)
        if sim_window.empty:
            continue
        if not edges.empty:
            edge_frames.append(edges.copy())
        coverage = _graph_coverage(sim_window, edges, month_start)
        if not coverage.empty:
            coverage_frames.append(coverage)
        fixed_gates = ALL_FIXED_GATES
        raw_counts = _raw_gate_counts(sim_window, fixed_gates)
        if not raw_counts.empty:
            raw_counts.to_csv(raw_path, mode="a", header=not wrote_raw, index=False)
            wrote_raw = True
        trade_frames: list[pd.DataFrame] = []
        signal_frames: list[pd.DataFrame] = []
        for gate in fixed_gates:
            trades, signals = _simulate_gate(sim_window, gate, config)
            if not trades.empty:
                trade_frames.append(trades)
            if not signals.empty:
                signal_frames.append(signals)
        sample = sim_window[pd.to_numeric(sim_window["dynamic_all_rank"], errors="coerce") <= TOP_N].copy()
        symbols_month = sorted(sample["symbol"].dropna().astype(str).unique())
        rank_buckets = _rank_bucket_map(sample)
        real_leaders = _top_leader_map(edges)
        for permutation in range(RANDOM_PERMUTATIONS):
            gate_col = "__audit_density_random_leader_gate"
            local = _add_map_gate(
                sim_window,
                _density_random_leaders(month_start, symbols_month, rank_buckets, permutation),
                gate_col,
            )
            gate = AuditGate(
                f"density_random_leader_p{permutation:03d}",
                "density_matched_random_leader",
                gate_col,
                "Density/liquidity matched random leader audit.",
                True,
            )
            trades, signals = _simulate_gate(local, gate, config)
            if not trades.empty:
                trades["permutation"] = permutation
                trade_frames.append(trades)
            if not signals.empty:
                signals["permutation"] = permutation
                signal_frames.append(signals)
            del local
        for permutation in range(SHUFFLED_PERMUTATIONS):
            gate_col = "__audit_shuffled_leader_gate"
            local = _add_map_gate(sim_window, _shuffled_leaders(month_start, symbols_month, real_leaders, permutation), gate_col)
            gate = AuditGate(
                f"shuffled_leader_p{permutation:03d}",
                "shuffled_leader",
                gate_col,
                "Within-month shuffled leader mapping audit.",
                True,
            )
            trades, signals = _simulate_gate(local, gate, config)
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
        print(f"v0.7C.1 month {idx}/{len(months)} {month_start:%Y-%m} symbols={len(symbols)}", flush=True)
        del sim_window, edges, trade_frames, signal_frames, month_trades, month_signals
        gc.collect()
    trades = pd.read_csv(trade_path, low_memory=False) if wrote_trades else pd.DataFrame()
    signals = pd.read_csv(signal_path, low_memory=False) if wrote_signals else pd.DataFrame()
    raw_counts = pd.read_csv(raw_path, low_memory=False) if wrote_raw else pd.DataFrame()
    for path in [trade_path, signal_path, raw_path]:
        if path.exists():
            path.unlink()
    edges = pd.concat(edge_frames, ignore_index=True).drop_duplicates() if edge_frames else pd.DataFrame()
    coverage = pd.concat(coverage_frames, ignore_index=True) if coverage_frames else pd.DataFrame()
    return trades, signals, raw_counts, edges, coverage


def _summary(trades: pd.DataFrame, signals: pd.DataFrame, raw_counts: pd.DataFrame) -> pd.DataFrame:
    grouped_signals = (
        signals.groupby(["candidate", "gate_mode", "baseline"], as_index=False, sort=False, dropna=False)
        .agg(signals=("signals", "sum"), pre_entry_gate_trades=("pre_entry_gate_trades", "sum"))
        if not signals.empty
        else pd.DataFrame()
    )
    summary = _summarize_candidates(trades, grouped_signals)
    if raw_counts.empty:
        return summary
    raw = raw_counts.groupby(["candidate", "role", "gate_col"], as_index=False, sort=False, dropna=False).agg(
        raw_rows=("raw_rows", "sum"),
        local_signal_events=("local_signal_events", "sum"),
    )
    return raw.merge(summary, on="candidate", how="outer")


def _funnel_summary(summary: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    order = [gate.candidate for gate in FUNNEL_GATES]
    rows = []
    previous = np.nan
    mir1 = np.nan
    for idx, candidate in enumerate(order):
        sample = summary[summary["candidate"].astype(str).eq(candidate)]
        if sample.empty:
            continue
        row = sample.iloc[0].to_dict()
        local_events = float(row.get("local_signal_events", np.nan))
        if idx == 0:
            mir1 = local_events
        row["stage"] = f"S{idx}"
        row["description"] = next(g.description for g in FUNNEL_GATES if g.candidate == candidate)
        row["candidate_events"] = int(local_events) if pd.notna(local_events) else 0
        row["pass_rate_vs_previous"] = local_events / previous if previous and pd.notna(previous) else np.nan
        row["pass_rate_vs_mir1"] = local_events / mir1 if mir1 and pd.notna(mir1) else np.nan
        rows.append(row)
        previous = local_events
    strict = summary[summary["candidate"].astype(str).eq("S3_beta_not_extended")]
    if not strict.empty:
        row = strict.iloc[0].to_dict()
        row["stage"] = "S4"
        row["candidate"] = "S4_local_bullish_volume_shock"
        row["description"] = "Same strict LBR1 gate; candidate_events are beta local bullish volume shock events."
        row["candidate_events"] = int(row.get("local_signal_events", 0))
        rows.append(row)
        row = strict.iloc[0].to_dict()
        row["stage"] = "S5"
        row["candidate"] = "S5_reclaim_pre_entry_fill"
        row["description"] = "Same strict LBR1 gate after reclaim_1pct produces a pre-entry fill."
        row["candidate_events"] = int(row.get("pre_entry_gate_trades", 0)) if "pre_entry_gate_trades" in row else int(row.get("trades", 0))
        rows.append(row)
        row = strict.iloc[0].to_dict()
        row["stage"] = "S6"
        row["candidate"] = "S6_strict_signal_and_entry_gate"
        row["description"] = "Same strict LBR1 gate after signal and entry both pass as-of gate."
        row["candidate_events"] = int(row.get("trades", 0)) if pd.notna(row.get("trades", np.nan)) else 0
        rows.append(row)
    out = pd.DataFrame(rows)
    preferred = [
        "stage",
        "candidate",
        "description",
        "raw_rows",
        "local_signal_events",
        "candidate_events",
        "signals",
        "pre_entry_gate_trades",
        "trades",
        "pass_rate_vs_previous",
        "pass_rate_vs_mir1",
        "net10",
        "net20",
        "ex_top_month_net10",
        "month_cap35_net20",
        "max_symbol_contribution",
        "max_month_contribution",
    ]
    return out[[col for col in preferred if col in out.columns]]


def _candidate_family(candidate: str) -> str:
    if str(candidate).startswith("density_random_leader_p"):
        return "density_matched_random_leader"
    if str(candidate).startswith("shuffled_leader_p"):
        return "shuffled_leader"
    return str(candidate)


def _random_edge_summary(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = summary.copy()
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    data["family"] = data["candidate"].map(_candidate_family)
    random = data[data["family"].eq("density_matched_random_leader")].copy()
    shuffled = data[data["family"].eq("shuffled_leader")].copy()
    real = data[data["candidate"].astype(str).eq("S2_leader_prior_4h")]
    real_net10 = float(real.iloc[0]["net10"]) if not real.empty and pd.notna(real.iloc[0].get("net10")) else np.nan
    for frame in [random, shuffled]:
        if not frame.empty and pd.notna(real_net10):
            frame["real_net10"] = real_net10
            frame["real_vs_control_lift"] = real_net10 - pd.to_numeric(frame["net10"], errors="coerce")
            frame["real_percentile_vs_family"] = float((pd.to_numeric(frame["net10"], errors="coerce") <= real_net10).mean())
    return random, shuffled


def _portfolio_width_audit(trades: pd.DataFrame) -> pd.DataFrame:
    sample = trades[
        trades["candidate"].astype(str).isin(["S2_leader_prior_4h", "B2_beta_mildly_extended", "B3_beta_strongly_extended"])
        & trades["baseline"].astype(str).eq("candidate_reclaim")
        & pd.to_numeric(trades["cost_single_side_bps"], errors="coerce").isin([10.0, 20.0])
    ].copy()
    if sample.empty:
        return pd.DataFrame()
    sample["entry_time"] = pd.to_datetime(sample["entry_time"], utc=True, errors="coerce")
    sample["exit_time"] = pd.to_datetime(sample["exit_time"], utc=True, errors="coerce")
    sample["rank_first_come_first_served"] = -sample["entry_time"].astype("int64")
    sample["rank_leader_edge_weight"] = _numeric_context(sample, "audit_directed_edge_weight_active", -np.inf)
    sample["rank_leader_impulse_strength"] = _numeric_context(sample, "audit_leader_prior_4h_ratio", -np.inf)
    sample["rank_lower_beta_extension"] = -_numeric_context(sample, "ret_4h_percentile", 999.0)
    sample["rank_local_volume_shock_strength"] = _numeric_context(sample, "volume_z_1h", -np.inf)
    sample["rank_market_volume_impulse_density"] = _numeric_context(sample, "volume_impulse_density", -np.inf)
    rank_cols = [
        "rank_first_come_first_served",
        "rank_leader_edge_weight",
        "rank_leader_impulse_strength",
        "rank_lower_beta_extension",
        "rank_local_volume_shock_strength",
        "rank_market_volume_impulse_density",
    ]
    rows = []
    for (candidate, cost), group in sample.groupby(["candidate", "cost_single_side_bps"], sort=False, dropna=False):
        for rank_col in rank_cols:
            ranked = group.sort_values(["entry_time", rank_col, "symbol"], ascending=[True, False, True]).reset_index(drop=True)
            for max_positions in [1, 3, 5, 10, 10_000]:
                active: list[pd.Timestamp] = []
                selected = []
                skipped = 0
                max_concurrent = 0
                for row in ranked.itertuples(index=False):
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
                        "candidate": candidate,
                        "ranking": rank_col.removeprefix("rank_"),
                        "cost_single_side_bps": cost,
                        "max_positions": "unlimited" if max_positions >= 10_000 else max_positions,
                        "selected_trades": int(len(selected)),
                        "skipped_trades": int(skipped),
                        "net_expectancy": float(net.mean()) if len(net) else np.nan,
                        "total_net": float(net.sum()) if len(net) else 0.0,
                        "max_drawdown_proxy": float(dd.min()) if len(dd) else np.nan,
                        "max_concurrent_positions": int(max_concurrent),
                    }
                )
    return pd.DataFrame(rows)


def _write_notes(
    report_root: Path,
    funnel: pd.DataFrame,
    beta: pd.DataFrame,
    random: pd.DataFrame,
    shuffled: pd.DataFrame,
) -> None:
    lines = [
        "# v0.7C.1 Gate Width Audit",
        "",
        "Purpose: attribute why strict LBR1 collapsed from MIR1's broad event pool to a tiny directed leader-beta sample.",
        "",
    ]
    if not funnel.empty:
        lines.append("## Funnel")
        for row in funnel.itertuples(index=False):
            lines.append(
                f"- {row.stage} {row.candidate}: local_events={getattr(row, 'local_signal_events', np.nan)}, "
                f"trades={getattr(row, 'trades', np.nan)}, net10={getattr(row, 'net10', np.nan):.4%}."
            )
    if not beta.empty:
        best = beta.sort_values("net10", ascending=False).head(1)
        if not best.empty:
            row = best.iloc[0]
            lines.append("")
            lines.append(
                f"- Best beta extension bucket by net10: {row['candidate']} "
                f"trades={int(row.get('trades', 0))}, net10={row.get('net10', np.nan):.4%}."
            )
    if not random.empty:
        real = random["real_net10"].dropna().iloc[0] if "real_net10" in random.columns and random["real_net10"].notna().any() else np.nan
        lines.append("")
        lines.append(
            f"- Directed leader prior-4h real net10={real:.4%}; "
            f"density-random mean={random['net10'].mean():.4%}, p75={random['net10'].quantile(0.75):.4%}."
        )
    if not shuffled.empty:
        lines.append(f"- Shuffled leader mean net10={shuffled['net10'].mean():.4%}, p75={shuffled['net10'].quantile(0.75):.4%}.")
    lines.extend(
        [
            "",
            "Interpretation:",
            "- This audit should not promote a candidate by itself.",
            "- If beta continuation buckets beat laggard buckets and real leader beats density/shuffled controls with enough trades, next step is v0.7C.2 Leader-Beta Continuation Validation.",
        ]
    )
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v07c1_gate_width_audit(
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
    trades, signals, raw_counts, edges, coverage = _stream_reports(feature_path, rank30, rank90, regime, config, report_root)
    summary = _summary(trades, signals, raw_counts)
    funnel = _funnel_summary(summary, signals)
    beta = summary[summary["candidate"].astype(str).isin([gate.candidate for gate in BETA_BUCKET_GATES])].copy()
    leader_window = summary[summary["candidate"].astype(str).isin([gate.candidate for gate in LEADER_WINDOW_GATES])].copy()
    controls = summary[summary["candidate"].astype(str).isin([gate.candidate for gate in CONTROL_GATES])].copy()
    random, shuffled = _random_edge_summary(summary)
    portfolio = _portfolio_width_audit(trades)
    outputs = {
        "gate_funnel": report_root / "gate_funnel.csv",
        "beta_extension_bucket_summary": report_root / "beta_extension_bucket_summary.csv",
        "leader_impulse_window_summary": report_root / "leader_impulse_window_summary.csv",
        "leader_graph_coverage": report_root / "leader_graph_coverage.csv",
        "density_matched_random_leader": report_root / "density_matched_random_leader.csv",
        "density_matched_random_distribution": report_root / "density_matched_random_distribution.csv",
        "shuffled_within_month_leader": report_root / "shuffled_within_month_leader.csv",
        "control_summary": report_root / "control_summary.csv",
        "portfolio_width_audit": report_root / "portfolio_width_audit.csv",
        "leader_beta_edges": report_root / "leader_beta_edges.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    funnel.to_csv(outputs["gate_funnel"], index=False)
    beta.to_csv(outputs["beta_extension_bucket_summary"], index=False)
    leader_window.to_csv(outputs["leader_impulse_window_summary"], index=False)
    coverage.to_csv(outputs["leader_graph_coverage"], index=False)
    random.to_csv(outputs["density_matched_random_leader"], index=False)
    _aggregate_permutations(random, "density_matched_random_leader").to_csv(
        outputs["density_matched_random_distribution"],
        index=False,
    )
    shuffled.to_csv(outputs["shuffled_within_month_leader"], index=False)
    controls.to_csv(outputs["control_summary"], index=False)
    portfolio.to_csv(outputs["portfolio_width_audit"], index=False)
    edges.to_csv(outputs["leader_beta_edges"], index=False)
    _write_notes(report_root, funnel, beta, random, shuffled)
    return outputs
