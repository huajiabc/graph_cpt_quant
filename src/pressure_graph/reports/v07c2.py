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
    RANDOM_PERMUTATIONS,
    SHUFFLED_PERMUTATIONS,
    _active_edge_weight,
    _build_directed_edges,
    _density_random_leaders,
    _edge_weight_map,
    _future_any,
    _max_from_map,
    _past_any,
    _rank_bucket_map,
    _ratio_from_map,
    _shuffled_leaders,
    _summarize_candidates,
    _top_leader_map,
)


REPORT_ROOT = Path("reports/v0_7c2_leader_beta_continuation")
LEADER_GATE_THRESHOLD = 0.20


@dataclass(frozen=True)
class ContinuationControl:
    candidate: str
    role: str
    gate_col: str
    description: str
    negative_control: bool = False
    baselines: tuple[BaselineSpec, ...] = (CORE_BASELINE,)


FIXED_CONTROLS = [
    ContinuationControl(
        "C0_MIR1_raw",
        "raw_reference",
        "c2_mir1_raw",
        "MIR1 raw: market volume impulse-density high.",
        False,
        tuple(BASELINES),
    ),
    ContinuationControl(
        "C1_beta_continuation",
        "beta_state",
        "c2_beta_continuation",
        "MIR1 plus beta extension bucket in extended/extreme-overextended.",
        False,
        tuple(BASELINES),
    ),
    ContinuationControl(
        "C2_leader_prior_1h",
        "leader_window",
        "c2_leader_prior_1h",
        "MIR1 plus real directed leader impulse in prior 1h.",
    ),
    ContinuationControl(
        "C3_directed_edge_leader_prior_1h",
        "directed_edge",
        "c2_directed_edge_leader_prior_1h",
        "MIR1 plus directed edge exists and real leader impulse in prior 1h.",
    ),
    ContinuationControl(
        "C4_beta_continuation_leader_prior_1h",
        "beta_plus_leader",
        "c2_beta_continuation_leader_prior_1h",
        "MIR1 plus beta continuation and real leader impulse in prior 1h.",
    ),
    ContinuationControl(
        "C5_LBC1_real_directed_leader_beta_continuation",
        "lbc1_primary_attribution",
        "c2_lbc1_real_directed_leader_beta_continuation",
        "Frozen LBC1: market impulse density, directed leader prior 1h, beta continuation, local shock and reclaim.",
        False,
        tuple(BASELINES),
    ),
    ContinuationControl(
        "C8_future_leader_1h_control",
        "future_control",
        "c2_beta_continuation_future_leader_1h",
        "Future leader impulse in next 1h with beta continuation. Audit only; uses future information.",
        True,
    ),
    ContinuationControl(
        "C9_beta_continuation_no_leader",
        "no_leader_control",
        "c2_beta_continuation_no_leader",
        "MIR1 beta continuation with no real directed leader impulse in prior 1h.",
        True,
    ),
]

BETA_BUCKET_CONTROLS = [
    ContinuationControl(
        f"bucket_{bucket}",
        "beta_extension_bucket",
        f"c2_bucket_{bucket}",
        f"MIR1 plus beta extension bucket {bucket}.",
    )
    for bucket in [
        "beta_laggard",
        "beta_neutral",
        "beta_confirmed",
        "beta_extended",
        "beta_extreme_overextended",
    ]
]

TIME_DIRECTION_CONTROLS = [
    ContinuationControl("leader_same_time", "time_direction", "c2_beta_continuation_leader_same_time", "Same-bar leader impulse control."),
    ContinuationControl("leader_prior_15m", "time_direction", "c2_beta_continuation_leader_prior_15m", "Leader impulse in prior 15m."),
    ContinuationControl("leader_prior_30m", "time_direction", "c2_beta_continuation_leader_prior_30m", "Leader impulse in prior 30m."),
    ContinuationControl("leader_prior_1h", "time_direction", "c2_beta_continuation_leader_prior_1h", "Leader impulse in prior 1h."),
    ContinuationControl("leader_prior_2h", "time_direction", "c2_beta_continuation_leader_prior_2h", "Leader impulse in prior 2h."),
    ContinuationControl("leader_prior_4h", "time_direction", "c2_beta_continuation_leader_prior_4h", "Leader impulse in prior 4h."),
    ContinuationControl(
        "leader_future_1h",
        "future_time_control",
        "c2_beta_continuation_future_leader_1h",
        "Future leader impulse in next 1h. Audit only.",
        True,
    ),
]

ALL_FIXED_CONTROLS = [*FIXED_CONTROLS, *BETA_BUCKET_CONTROLS, *TIME_DIRECTION_CONTROLS]


def _candidate(control: ContinuationControl) -> FrozenAtlasCandidate:
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
        "c2_beta_extension_bucket",
        "c2_beta_extension_score",
        "c2_leader_same_time_ratio",
        "c2_leader_prior_15m_ratio",
        "c2_leader_prior_30m_ratio",
        "c2_leader_prior_1h_ratio",
        "c2_leader_prior_2h_ratio",
        "c2_leader_prior_4h_ratio",
        "c2_future_leader_1h_ratio",
        "c2_directed_leader_return_4h_max",
        "c2_directed_edge_weight_active_1h",
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
    out = pd.Series("beta_neutral", index=beta_ret.index, dtype="object")
    out[(gap >= 0.02) | (beta_pct < 40)] = "beta_laggard"
    out[(beta_pct >= 40) & (beta_pct < 60) & (gap < 0.02)] = "beta_neutral"
    out[(beta_pct >= 60) & (beta_pct < 80)] = "beta_confirmed"
    out[(beta_pct >= 80) & (beta_pct < 95)] = "beta_extended"
    out[beta_pct >= 95] = "beta_extreme_overextended"
    return out


def _add_continuation_features(data: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    defaults: dict[str, object] = {
        "c2_leader_same_time_ratio": 0.0,
        "c2_leader_prior_15m_ratio": 0.0,
        "c2_leader_prior_30m_ratio": 0.0,
        "c2_leader_prior_1h_ratio": 0.0,
        "c2_leader_prior_2h_ratio": 0.0,
        "c2_leader_prior_4h_ratio": 0.0,
        "c2_future_leader_1h_ratio": 0.0,
        "c2_directed_leader_return_4h_max": np.nan,
        "c2_directed_edge_weight_active_1h": 0.0,
        "c2_beta_extension_bucket": "unknown",
        "c2_beta_extension_score": np.nan,
        "leader_beta_cluster_id": "",
    }
    for col, value in defaults.items():
        out[col] = value
    if out.empty:
        return _add_continuation_gates(out)
    for month_start, month_data in out.groupby("month_start", observed=True, sort=True):
        month_start = pd.Timestamp(month_start)
        sample = month_data[pd.to_numeric(month_data["dynamic_all_rank"], errors="coerce") <= TOP_N].copy()
        if sample.empty:
            continue
        event = sample.pivot_table(
            index="feature_time",
            columns="symbol",
            values="bullish_volume_shock_event",
            aggfunc="max",
            observed=True,
        ).fillna(False).infer_objects(copy=False).astype(bool)
        ret4 = sample.pivot_table(index="feature_time", columns="symbol", values="ret_4h", aggfunc="last", observed=True)
        month_edges = edges[edges["month_start"].eq(month_start)] if "month_start" in edges.columns else pd.DataFrame()
        leaders = _top_leader_map(month_edges)
        weights = _edge_weight_map(month_edges)
        windows = {
            "same_time": event,
            "prior_15m": _past_any(event, 1),
            "prior_30m": _past_any(event, 2),
            "prior_1h": _past_any(event, 4),
            "prior_2h": _past_any(event, 8),
            "prior_4h": _past_any(event, 16),
            "future_1h": _future_any(event, 4),
        }
        for beta, group in sample.groupby("symbol", observed=True, sort=False):
            beta = str(beta)
            idx = group.index
            feature_time = pd.to_datetime(group["feature_time"], utc=True, errors="coerce")
            for label, state in windows.items():
                out.loc[idx, f"c2_leader_{label}_ratio" if label != "future_1h" else "c2_future_leader_1h_ratio"] = _ratio_from_map(
                    state,
                    feature_time,
                    leaders,
                    beta,
                )
            out.loc[idx, "c2_directed_leader_return_4h_max"] = _max_from_map(ret4, feature_time, leaders, beta)
            out.loc[idx, "c2_directed_edge_weight_active_1h"] = _active_edge_weight(
                windows["prior_1h"],
                feature_time,
                leaders,
                weights,
                beta,
            )
            top_leader = leaders.get(beta, ["none"])[0] if leaders.get(beta) else "none"
            out.loc[idx, "leader_beta_cluster_id"] = f"{month_start:%Y-%m}:lead:{top_leader}:beta:{beta}"
        beta_ret = pd.to_numeric(sample.get("ret_4h"), errors="coerce")
        beta_pct = pd.to_numeric(sample.get("ret_4h_percentile"), errors="coerce")
        leader_ret = pd.to_numeric(out.loc[sample.index, "c2_directed_leader_return_4h_max"], errors="coerce")
        out.loc[sample.index, "c2_beta_extension_bucket"] = _bucket_beta_extension(beta_ret, beta_pct, leader_ret).to_numpy()
        out.loc[sample.index, "c2_beta_extension_score"] = beta_pct.to_numpy(dtype=float)
    return _add_continuation_gates(out)


def _add_continuation_gates(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    market = _ensure_bool(out.get("market_volume_impulse_density_high", pd.Series(False, index=out.index)))
    same_time = pd.to_numeric(out.get("c2_leader_same_time_ratio"), errors="coerce").fillna(0.0) >= LEADER_GATE_THRESHOLD
    prior_15m = pd.to_numeric(out.get("c2_leader_prior_15m_ratio"), errors="coerce").fillna(0.0) >= LEADER_GATE_THRESHOLD
    prior_30m = pd.to_numeric(out.get("c2_leader_prior_30m_ratio"), errors="coerce").fillna(0.0) >= LEADER_GATE_THRESHOLD
    prior_1h = pd.to_numeric(out.get("c2_leader_prior_1h_ratio"), errors="coerce").fillna(0.0) >= LEADER_GATE_THRESHOLD
    prior_2h = pd.to_numeric(out.get("c2_leader_prior_2h_ratio"), errors="coerce").fillna(0.0) >= LEADER_GATE_THRESHOLD
    prior_4h = pd.to_numeric(out.get("c2_leader_prior_4h_ratio"), errors="coerce").fillna(0.0) >= LEADER_GATE_THRESHOLD
    future_1h = pd.to_numeric(out.get("c2_future_leader_1h_ratio"), errors="coerce").fillna(0.0) >= LEADER_GATE_THRESHOLD
    bucket = out.get("c2_beta_extension_bucket", pd.Series("unknown", index=out.index)).astype(str)
    continuation = bucket.isin(["beta_extended", "beta_extreme_overextended"])
    edge_exists = out.get("leader_beta_cluster_id", pd.Series("", index=out.index)).astype(str).str.contains("lead:none", regex=False).eq(False)
    out["c2_mir1_raw"] = market
    out["c2_beta_continuation"] = market & continuation
    out["c2_leader_prior_1h"] = market & prior_1h
    out["c2_directed_edge_leader_prior_1h"] = market & edge_exists & prior_1h
    out["c2_beta_continuation_leader_same_time"] = market & continuation & same_time
    out["c2_beta_continuation_leader_prior_15m"] = market & continuation & prior_15m
    out["c2_beta_continuation_leader_prior_30m"] = market & continuation & prior_30m
    out["c2_beta_continuation_leader_prior_1h"] = market & continuation & prior_1h
    out["c2_beta_continuation_leader_prior_2h"] = market & continuation & prior_2h
    out["c2_beta_continuation_leader_prior_4h"] = market & continuation & prior_4h
    out["c2_lbc1_real_directed_leader_beta_continuation"] = market & edge_exists & continuation & prior_1h
    out["c2_beta_continuation_future_leader_1h"] = market & continuation & future_1h
    out["c2_beta_continuation_no_leader"] = market & continuation & ~prior_1h
    for item in [
        "beta_laggard",
        "beta_neutral",
        "beta_confirmed",
        "beta_extended",
        "beta_extreme_overextended",
    ]:
        out[f"c2_bucket_{item}"] = market & bucket.eq(item)
    return out


def _add_map_gate(
    data: pd.DataFrame,
    leader_map: dict[str, list[str]],
    gate_col: str,
    *,
    require_beta_continuation: bool = True,
) -> pd.DataFrame:
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
    past_event = _past_any(event, 4)
    ratios = pd.Series(0.0, index=sample.index, dtype="float64")
    for beta, group in sample.groupby("symbol", observed=True, sort=False):
        feature_time = pd.to_datetime(group["feature_time"], utc=True, errors="coerce")
        ratios.loc[group.index] = _ratio_from_map(past_event, feature_time, leader_map, str(beta))
    market = _ensure_bool(sample.get("market_volume_impulse_density_high", pd.Series(False, index=sample.index)))
    mask = market & (ratios >= LEADER_GATE_THRESHOLD)
    if require_beta_continuation:
        mask = mask & _ensure_bool(sample.get("c2_beta_continuation", pd.Series(False, index=sample.index)))
    out.loc[sample.index, gate_col] = mask.to_numpy(dtype=bool)
    return out


def _simulate_control(
    data: pd.DataFrame,
    control: ContinuationControl,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = _candidate(control)
    rows: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []
    for baseline in control.baselines:
        trades, signal_n, pre_entry_gate_n = _simulate_variant(
            data,
            candidate,
            "signal_and_entry_gate",
            baseline,
            config,
        )
        rows.append(
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
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), pd.DataFrame(rows))


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
    sim_window = _add_continuation_features(sim_window, edges)
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
    trade_path = report_root / "_v07c2_trades_tmp.csv"
    signal_path = report_root / "_v07c2_signals_tmp.csv"
    for path in [trade_path, signal_path]:
        if path.exists():
            path.unlink()
    wrote_trades = False
    wrote_signals = False
    edge_frames: list[pd.DataFrame] = []
    months = sorted(pd.to_datetime(rank30["month_start"], utc=True, errors="coerce").dropna().drop_duplicates().tolist())
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
        for control in ALL_FIXED_CONTROLS:
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
            gate_col = "__c2_density_random_leader_prior_1h"
            local = _add_map_gate(
                sim_window,
                _density_random_leaders(month_start, symbols_month, rank_buckets, permutation),
                gate_col,
            )
            control = ContinuationControl(
                f"C6_density_random_leader_p{permutation:03d}",
                "density_matched_random_leader",
                gate_col,
                "Beta continuation plus density/liquidity matched random leader prior 1h.",
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
            gate_col = "__c2_shuffled_leader_prior_1h"
            local = _add_map_gate(sim_window, _shuffled_leaders(month_start, symbols_month, real_leaders, permutation), gate_col)
            control = ContinuationControl(
                f"C7_shuffled_leader_p{permutation:03d}",
                "shuffled_leader",
                gate_col,
                "Beta continuation plus within-month shuffled leader prior 1h.",
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
        print(f"v0.7C.2 month {idx}/{len(months)} {month_start:%Y-%m} symbols={len(symbols)}", flush=True)
        del sim_window, edges, trade_frames, signal_frames, month_trades, month_signals
        gc.collect()
    trades = pd.read_csv(trade_path, low_memory=False) if wrote_trades else pd.DataFrame()
    signals = pd.read_csv(signal_path, low_memory=False) if wrote_signals else pd.DataFrame()
    for path in [trade_path, signal_path]:
        if path.exists():
            path.unlink()
    edges = pd.concat(edge_frames, ignore_index=True).drop_duplicates() if edge_frames else pd.DataFrame()
    return trades, signals, edges


def _signals_grouped(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    return signals.groupby(["candidate", "gate_mode", "baseline"], as_index=False, sort=False, dropna=False).agg(
        signals=("signals", "sum"),
        pre_entry_gate_trades=("pre_entry_gate_trades", "sum"),
    )


def _candidate_family(candidate: str) -> str:
    text = str(candidate)
    if text.startswith("C6_density_random_leader_p"):
        return "density_matched_random_leader"
    if text.startswith("C7_shuffled_leader_p"):
        return "shuffled_leader"
    if text.startswith("bucket_"):
        return "beta_extension_bucket"
    if text.startswith("leader_"):
        return "leader_time_direction"
    return text


def _control_lifts(controls: pd.DataFrame, full_summary: pd.DataFrame) -> pd.DataFrame:
    out = controls.copy()
    if out.empty:
        return out
    real = out[out["candidate"].astype(str).eq("C5_LBC1_real_directed_leader_beta_continuation")]
    if real.empty:
        return out
    real_net10 = float(real.iloc[0]["net10"])
    random_net = pd.to_numeric(
        full_summary.loc[full_summary["family"].eq("density_matched_random_leader"), "net10"],
        errors="coerce",
    ).dropna()
    shuffled_net = pd.to_numeric(
        full_summary.loc[full_summary["family"].eq("shuffled_leader"), "net10"],
        errors="coerce",
    ).dropna()
    for label, values in [("random", random_net), ("shuffled", shuffled_net)]:
        out[f"real_vs_{label}_mean_lift"] = real_net10 - (float(values.mean()) if len(values) else np.nan)
        out[f"real_vs_{label}_median_lift"] = real_net10 - (float(values.median()) if len(values) else np.nan)
        out[f"real_vs_{label}_p75_lift"] = real_net10 - (float(values.quantile(0.75)) if len(values) else np.nan)
        out[f"real_vs_{label}_p90_lift"] = real_net10 - (float(values.quantile(0.90)) if len(values) else np.nan)
        out[f"real_percentile_vs_{label}"] = float((values <= real_net10).mean()) if len(values) else np.nan
    for control in [
        "C1_beta_continuation",
        "C8_future_leader_1h_control",
        "C9_beta_continuation_no_leader",
    ]:
        sample = out[out["candidate"].astype(str).eq(control)]
        out[f"real_vs_{control}_lift"] = real_net10 - float(sample.iloc[0]["net10"]) if not sample.empty else np.nan
    return out


def _attribution_controls(summary: pd.DataFrame) -> pd.DataFrame:
    names = [item.candidate for item in FIXED_CONTROLS]
    return _control_lifts(summary[summary["candidate"].astype(str).isin(names)].copy(), summary)


def _random_tables(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    random = summary[summary["family"].eq("density_matched_random_leader")].copy()
    shuffled = summary[summary["family"].eq("shuffled_leader")].copy()
    real = summary[summary["candidate"].astype(str).eq("C5_LBC1_real_directed_leader_beta_continuation")]
    real_net10 = float(real.iloc[0]["net10"]) if not real.empty and pd.notna(real.iloc[0].get("net10")) else np.nan
    for frame in [random, shuffled]:
        if frame.empty or pd.isna(real_net10):
            continue
        frame["real_net10"] = real_net10
        frame["real_vs_control_lift"] = real_net10 - pd.to_numeric(frame["net10"], errors="coerce")
        frame["real_percentile_vs_family"] = float((pd.to_numeric(frame["net10"], errors="coerce") <= real_net10).mean())
    return random, shuffled


def _portfolio_ranking(trades: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        "C1_beta_continuation",
        "C5_LBC1_real_directed_leader_beta_continuation",
    ]
    sample = trades[
        trades["candidate"].astype(str).isin(candidates)
        & trades["baseline"].astype(str).eq("candidate_reclaim")
        & pd.to_numeric(trades["cost_single_side_bps"], errors="coerce").isin([10.0, 20.0])
    ].copy()
    if sample.empty:
        return pd.DataFrame()
    sample["entry_time"] = pd.to_datetime(sample["entry_time"], utc=True, errors="coerce")
    sample["exit_time"] = pd.to_datetime(sample["exit_time"], utc=True, errors="coerce")
    sample["rank_first_come_first_served"] = -sample["entry_time"].astype("int64")
    sample["rank_leader_edge_weight"] = _numeric_context(sample, "c2_directed_edge_weight_active_1h", -np.inf)
    sample["rank_leader_impulse_strength"] = _numeric_context(sample, "c2_leader_prior_1h_ratio", -np.inf)
    sample["rank_beta_extension_medium"] = -np.abs(_numeric_context(sample, "c2_beta_extension_score", np.nan) - 90.0)
    sample["rank_local_volume_shock_strength"] = _numeric_context(sample, "volume_z_1h", -np.inf)
    sample["rank_market_volume_impulse_density"] = _numeric_context(sample, "volume_impulse_density", -np.inf)
    hashed = pd.util.hash_pandas_object(sample[["symbol", "signal_time"]].astype(str), index=False)
    sample["rank_random"] = (hashed % 10_000_000).astype(float)
    rank_cols = [
        "rank_first_come_first_served",
        "rank_leader_edge_weight",
        "rank_leader_impulse_strength",
        "rank_beta_extension_medium",
        "rank_local_volume_shock_strength",
        "rank_market_volume_impulse_density",
        "rank_random",
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
                    active = [exit for exit in active if exit > entry_time]
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


def _single_summary(trades: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    grouped = _signals_grouped(signals)
    summary = _summarize_candidates(trades, grouped)
    if summary.empty:
        return summary
    summary["family"] = summary["candidate"].map(_candidate_family)
    return summary


def _write_notes(
    report_root: Path,
    controls: pd.DataFrame,
    random: pd.DataFrame,
    shuffled: pd.DataFrame,
    beta: pd.DataFrame,
) -> None:
    lines = [
        "# v0.7C.2 Leader-Beta Continuation Validation",
        "",
        "Purpose: test whether beta continuation after market impulse is a real directed leader-beta edge, or just co-impulse / market-density continuation.",
        "",
    ]
    real = controls[controls["candidate"].astype(str).eq("C5_LBC1_real_directed_leader_beta_continuation")]
    if not real.empty:
        row = real.iloc[0]
        lines.append(f"- LBC1 real directed continuation: trades={int(row.get('trades', 0))}, net10={row.get('net10', np.nan):.4%}, net20={row.get('net20', np.nan):.4%}.")
        lines.append(f"- Month-cap35 net20={row.get('month_cap35_net20', np.nan):.4%}; ex-top-month net10={row.get('ex_top_month_net10', np.nan):.4%}.")
        lines.append(f"- Real percentile vs random={row.get('real_percentile_vs_random', np.nan):.2%}; vs shuffled={row.get('real_percentile_vs_shuffled', np.nan):.2%}.")
    if not random.empty:
        lines.append(f"- Density-matched random net10 median={random['net10'].median():.4%}, p75={random['net10'].quantile(0.75):.4%}, p90={random['net10'].quantile(0.90):.4%}.")
    if not shuffled.empty:
        lines.append(f"- Shuffled leader net10 median={shuffled['net10'].median():.4%}, p75={shuffled['net10'].quantile(0.75):.4%}.")
    if not beta.empty:
        best = beta.sort_values("net10", ascending=False).head(1)
        if not best.empty:
            row = best.iloc[0]
            lines.append(f"- Best beta bucket: {row['candidate']} trades={int(row.get('trades', 0))}, net10={row.get('net10', np.nan):.4%}.")
    lines.extend(
        [
            "",
            "Decision rule:",
            "- If real directed leader does not beat density-matched random, shuffled, and future controls, call it Co-Impulse Continuation rather than Leader-Beta alpha.",
            "- This report does not change MIR1 paper-live primary status by itself.",
        ]
    )
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v07c2_leader_beta_continuation(
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
    summary = _single_summary(trades, signals)
    controls = _attribution_controls(summary)
    beta = summary[summary["family"].eq("beta_extension_bucket")].copy()
    time_direction = summary[summary["family"].eq("leader_time_direction")].copy()
    random, shuffled = _random_tables(summary)
    portfolio = _portfolio_ranking(trades)
    outputs = {
        "candidate_summary": report_root / "candidate_summary.csv",
        "attribution_controls": report_root / "attribution_controls.csv",
        "beta_extension_bucket_summary": report_root / "beta_extension_bucket_summary.csv",
        "leader_time_direction": report_root / "leader_time_direction.csv",
        "density_matched_random_leader": report_root / "density_matched_random_leader.csv",
        "density_matched_random_distribution": report_root / "density_matched_random_distribution.csv",
        "shuffled_leader": report_root / "shuffled_leader.csv",
        "portfolio_ranking": report_root / "portfolio_ranking.csv",
        "leader_beta_edges": report_root / "leader_beta_edges.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    summary.to_csv(outputs["candidate_summary"], index=False)
    controls.to_csv(outputs["attribution_controls"], index=False)
    beta.to_csv(outputs["beta_extension_bucket_summary"], index=False)
    time_direction.to_csv(outputs["leader_time_direction"], index=False)
    random.to_csv(outputs["density_matched_random_leader"], index=False)
    _aggregate_permutations(random, "density_matched_random_leader").to_csv(
        outputs["density_matched_random_distribution"],
        index=False,
    )
    shuffled.to_csv(outputs["shuffled_leader"], index=False)
    portfolio.to_csv(outputs["portfolio_ranking"], index=False)
    edges.to_csv(outputs["leader_beta_edges"], index=False)
    _write_notes(report_root, controls, random, shuffled, beta)
    return outputs
