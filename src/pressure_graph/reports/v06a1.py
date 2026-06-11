from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from pressure_graph.backtest import EntryPolicy, simulate_entry_policy_trades
from pressure_graph.config import ExperimentConfig
from pressure_graph.config.models import ExecutionRule
from pressure_graph.io import ensure_dir, write_parquet
from pressure_graph.reports.v03 import (
    V03_REQUIRED_COLUMNS,
    V03_TURNOVER_COLUMNS,
    _add_v03_report_columns,
    _concat_or_empty,
    _downcast_frame,
    _event_flags,
    _month_start,
    _read_existing_columns,
    build_dynamic_rank_table,
)
from pressure_graph.reports.v04 import _rank_table_with_lookback


REPORT_ROOT = Path("reports/v0_6a1_impulse_reclaim_validation")
EVENT_OUTPUT = Path("data/processed/v0_6a1/frozen_candidate_events.parquet")
MAX_BOUNDARY_TOP_N = 100
R1_RECLAIM = EntryPolicy(
    "R1_pullback_1.0pct_reclaim_signal_close_valid_8_bars",
    "pullback_reclaim",
    valid_bars=8,
    pullback_pct=0.010,
)
R2_RECLAIM = EntryPolicy(
    "R2_pullback_0.5pct_reclaim_signal_close_valid_8_bars",
    "pullback_reclaim",
    valid_bars=8,
    pullback_pct=0.005,
)
NEXT_OPEN = EntryPolicy("B0_bullish_volume_shock_next_open", "next_open")


@dataclass(frozen=True)
class FrozenImpulseReclaimCandidate:
    candidate: str
    universe_top_n: int
    gate: str
    anchor: str
    entry_policy: str
    execution_rule: str
    rationale: str


@dataclass(frozen=True)
class V06A1Config:
    candidates: list[FrozenImpulseReclaimCandidate]
    cost_bps: list[float]
    boundary_top_ns: list[int]


def load_v06a1_config(path: str | Path = "configs/v0_6a1_frozen_candidates.yaml") -> V06A1Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    candidates = [FrozenImpulseReclaimCandidate(**item) for item in payload.get("candidates", [])]
    return V06A1Config(
        candidates=candidates,
        cost_bps=[float(item) for item in payload.get("cost_bps", [5, 10, 20, 30, 50])],
        boundary_top_ns=[int(item) for item in payload.get("boundary_top_ns", [30, 50, 100])],
    )


def _policy(name: str) -> EntryPolicy:
    if name == R1_RECLAIM.name:
        return R1_RECLAIM
    if name == R2_RECLAIM.name:
        return R2_RECLAIM
    if name == NEXT_OPEN.name:
        return NEXT_OPEN
    if name.startswith("B1_pullback_1.0pct"):
        return EntryPolicy(name, "pullback", valid_bars=8, pullback_pct=0.010)
    if name.startswith("B1_pullback_0.5pct"):
        return EntryPolicy(name, "pullback", valid_bars=8, pullback_pct=0.005)
    raise KeyError(name)


def _vol_regime_rule(row: pd.Series) -> ExecutionRule:
    vol_pct = pd.to_numeric(row.get("symbol_volatility_percentile"), errors="coerce")
    if pd.isna(vol_pct) or vol_pct < 40:
        return ExecutionRule(tp=0.03, sl=0.02, max_hold_bars=16)
    if vol_pct < 80:
        return ExecutionRule(tp=0.04, sl=0.025, max_hold_bars=16)
    return ExecutionRule(tp=0.05, sl=0.03, max_hold_bars=16)


def _rule(name: str, base_config: ExperimentConfig) -> tuple[ExecutionRule, Callable[[pd.Series], ExecutionRule] | None]:
    if name == "swing":
        return ExecutionRule(tp=0.05, sl=0.03, max_hold_bars=48), None
    if name == "vol_regime_fast":
        return base_config.execution.rules["fast"], _vol_regime_rule
    raise KeyError(name)


def _v06a1_required_columns() -> list[str]:
    return list(dict.fromkeys([*V03_REQUIRED_COLUMNS, "ret_15m", "btc_ret_1h", "btc_volatility_4h"]))


def _top_symbols(ranks: pd.DataFrame, top_n: int = MAX_BOUNDARY_TOP_N) -> list[str]:
    return sorted(
        ranks[pd.to_numeric(ranks["dynamic_all_rank"], errors="coerce") <= top_n]["symbol"]
        .dropna()
        .astype(str)
        .unique()
    )


def _read_symbol_features(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    symbol: str,
    config: ExperimentConfig,
) -> pd.DataFrame:
    data = _read_existing_columns(
        feature_path,
        _v06a1_required_columns(),
        filters=[("symbol", "==", symbol)],
    )
    if data.empty:
        return data
    data = _downcast_frame(data)
    data["month_start"] = _month_start(data["bar_open_time"])
    data = data.merge(rank30, on=["month_start", "symbol"], how="left")
    data = data.merge(rank90, on=["month_start", "symbol"], how="left")
    data = data[pd.to_numeric(data["dynamic_all_rank"], errors="coerce") <= MAX_BOUNDARY_TOP_N].copy()
    if data.empty:
        return data
    data["dynamic_all_rank"] = pd.to_numeric(data["dynamic_all_rank"], errors="coerce")
    data["turnover_rank_90d"] = pd.to_numeric(data["turnover_rank_90d"], errors="coerce")
    data = _add_v03_report_columns(data, config)
    return _add_impulse_reclaim_columns(data, config)


def _add_impulse_reclaim_columns(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    out = df.sort_values(["exchange", "symbol", "bar_open_time"]).copy()
    warmup = out.get("warmup_complete", True)
    if not isinstance(warmup, pd.Series):
        warmup = pd.Series(bool(warmup), index=out.index)

    volume_z_4h = pd.to_numeric(out.get("volume_z_4h"), errors="coerce")
    ret_4h = pd.to_numeric(out.get("ret_4h"), errors="coerce")
    out["bullish_volume_shock_state"] = warmup & (volume_z_4h > 2.0) & (ret_4h > 0)
    out["bearish_volume_shock_state"] = warmup & (volume_z_4h > 2.0) & (ret_4h < 0)
    out["neutral_volume_state"] = warmup & (volume_z_4h.abs() <= 0.5)
    for state_col in [
        "bullish_volume_shock_state",
        "bearish_volume_shock_state",
        "neutral_volume_state",
    ]:
        event_col = state_col.replace("_state", "_event")
        out[event_col] = (
            out.groupby(["exchange", "symbol"], group_keys=False, sort=False, observed=True)[state_col]
            .apply(lambda series: _event_flags(series, config.events.cooldown_bars_4h))
            .reindex(out.index)
        )
    state = out["btc_market_state"].astype("string")
    out["gate_BTC_up"] = state.eq("BTC_up")
    out["gate_BTC_chop"] = state.eq("BTC_chop")
    out["gate_BTC_down"] = state.eq("BTC_down")
    out["core_liquidity"] = (out["dynamic_all_rank"] <= 30) & (out["turnover_rank_90d"] <= 50)
    out["transient_hot"] = (out["dynamic_all_rank"] <= 30) & (out["turnover_rank_90d"] > 50)
    out["liquidity_bucket"] = pd.cut(
        pd.to_numeric(out["dynamic_all_rank"], errors="coerce"),
        bins=[0, 10, 30, 50, 100, np.inf],
        labels=["rank_1_10", "rank_11_30", "rank_31_50", "rank_51_100", "rank_101_plus"],
    ).astype("string")
    out["vol_bucket"] = pd.cut(
        pd.to_numeric(out["symbol_volatility_percentile"], errors="coerce"),
        bins=[-1, 20, 40, 60, 80, 101],
        labels=["v0_20", "v20_40", "v40_60", "v60_80", "v80_100"],
    ).astype("string")
    out["volume_bucket"] = pd.cut(
        pd.to_numeric(out["volume_z_4h"], errors="coerce"),
        bins=[-np.inf, 0, 1, 2, 3, np.inf],
        labels=["vol_z_lt0", "vol_z_0_1", "vol_z_1_2", "vol_z_2_3", "vol_z_gt3"],
    ).astype("string")
    return out


def _anchor_col(name: str) -> str:
    return f"{name}_event"


def _signal_mask(data: pd.DataFrame, candidate: FrozenImpulseReclaimCandidate, anchor: str | None = None) -> pd.Series:
    anchor_name = anchor or candidate.anchor
    return (
        (pd.to_numeric(data["dynamic_all_rank"], errors="coerce") <= candidate.universe_top_n)
        & data[f"gate_{candidate.gate}"].fillna(False)
        & data[_anchor_col(anchor_name)].fillna(False)
    )


def _simulate_rows(
    data: pd.DataFrame,
    signal_mask: pd.Series,
    candidate_name: str,
    anchor_name: str,
    policy: EntryPolicy,
    rule_name: str,
    base_config: ExperimentConfig,
) -> pd.DataFrame:
    sim = data.copy()
    sim["__v06a1_signal_event"] = signal_mask.fillna(False).astype(bool)
    if not sim["__v06a1_signal_event"].any():
        return pd.DataFrame()
    rule, resolver = _rule(rule_name, base_config)
    trades = simulate_entry_policy_trades(
        sim,
        "__v06a1_signal_event",
        anchor_name,
        policy,
        rule,
        0,
        "sl_first",
        True,
        "__v06a1_signal_event",
        resolver,
    )
    if trades.empty:
        return trades
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
        "vol_bucket",
        "volume_bucket",
        "core_liquidity",
        "transient_hot",
        "turnover_rank_90d",
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
    out["candidate"] = candidate_name
    out["anchor_name"] = anchor_name
    out["entry_policy"] = policy.name
    out["execution_rule"] = rule_name
    return out


def _expand_costs(trades: pd.DataFrame, costs: list[float]) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    frames = []
    for cost in costs:
        data = trades.copy()
        data["cost_single_side_bps"] = float(cost)
        data["net_return"] = pd.to_numeric(data["gross_return"], errors="coerce") - 2.0 * float(cost) / 10_000.0
        if "funding_cost" in data.columns:
            data["net_return_funding"] = data["net_return"] - pd.to_numeric(data["funding_cost"], errors="coerce").fillna(0)
        frames.append(data)
    return pd.concat(frames, ignore_index=True)


def _summary_row(trades: pd.DataFrame, signal_n: int, **keys: object) -> dict[str, object]:
    net = pd.to_numeric(trades.get("net_return"), errors="coerce")
    gross = pd.to_numeric(trades.get("gross_return"), errors="coerce")
    exit_reason = trades.get("exit_reason", pd.Series(dtype=str)).astype(str)
    return {
        **keys,
        "signals": int(signal_n),
        "trades": int(len(trades)),
        "fill_rate": float(len(trades) / signal_n) if signal_n else np.nan,
        "gross_expectancy": float(gross.mean()) if len(trades) else np.nan,
        "net_expectancy": float(net.mean()) if len(trades) else np.nan,
        "net_sum": float(net.sum()) if len(trades) else 0.0,
        "win_rate": float((net > 0).mean()) if len(trades) else np.nan,
        "tp_rate": float(exit_reason.str.startswith("tp").mean()) if len(trades) else np.nan,
        "sl_rate": float(exit_reason.str.startswith("sl").mean()) if len(trades) else np.nan,
        "timeout_rate": float(exit_reason.eq("max_hold").mean()) if len(trades) else np.nan,
        "p25_net": float(net.quantile(0.25)) if len(trades) else np.nan,
        "p75_net": float(net.quantile(0.75)) if len(trades) else np.nan,
        "max_loss": float(net.min()) if len(trades) else np.nan,
    }


def _matched_random_rows(data: pd.DataFrame, signal_mask: pd.Series, seed: int = 610) -> pd.Series:
    pool = data.copy()
    pool["__signal"] = signal_mask.fillna(False).astype(bool)
    hashed = pd.util.hash_pandas_object(pool[["symbol", "bar_open_time"]], index=False)
    pool["__rand"] = ((hashed + seed) % 1_000_000).astype(float)
    out = pd.Series(False, index=data.index)
    keys = ["symbol", "month", "vol_bucket", "volume_bucket", "liquidity_bucket", "btc_market_state"]
    for _, group in pool.groupby(keys, sort=False, dropna=False, observed=True):
        target = int(group["__signal"].sum())
        if target <= 0:
            continue
        anchors = group[~group["__signal"]].sort_values("__rand").head(target)
        out.loc[anchors.index] = True
    return out


def _candidate_boundary_mask(data: pd.DataFrame, candidate: FrozenImpulseReclaimCandidate, top_n: int, bucket: str | None) -> pd.Series:
    mask = data[f"gate_{candidate.gate}"].fillna(False) & data[_anchor_col(candidate.anchor)].fillna(False)
    rank = pd.to_numeric(data["dynamic_all_rank"], errors="coerce")
    if bucket is not None:
        mask &= data["liquidity_bucket"].astype(str).eq(bucket)
    else:
        mask &= rank <= top_n
    return mask


def _candidate_events(data: pd.DataFrame, candidate: FrozenImpulseReclaimCandidate) -> pd.DataFrame:
    events = data[_signal_mask(data, candidate)].copy()
    if events.empty:
        return events
    cols = [
        "exchange",
        "symbol",
        "bar_open_time",
        "bar_close_time",
        "feature_time",
        "month",
        "dynamic_all_rank",
        "liquidity_bucket",
        "btc_market_state",
        "volume_z_4h",
        "ret_4h",
        "close",
    ]
    out = events[[col for col in cols if col in events.columns]].copy()
    out["candidate"] = candidate.candidate
    out["entry_policy"] = candidate.entry_policy
    out["execution_rule"] = candidate.execution_rule
    return out


def _aggregate_exact(trades: pd.DataFrame, signals: dict[tuple[str, str], int], costs: list[float]) -> pd.DataFrame:
    rows = []
    if trades.empty:
        return pd.DataFrame()
    for (candidate, baseline, cost), group in trades.groupby(["candidate", "baseline", "cost_single_side_bps"], sort=False):
        rows.append(_summary_row(group, signals.get((candidate, baseline), 0), candidate=candidate, baseline=baseline, cost_single_side_bps=cost))
    return pd.DataFrame(rows)


def _monthly(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if trades.empty:
        return pd.DataFrame()
    for (candidate, baseline, cost), cost_group in trades.groupby(["candidate", "baseline", "cost_single_side_bps"], sort=False):
        total = pd.to_numeric(cost_group["net_return"], errors="coerce").sum()
        for month, group in cost_group.groupby("month", dropna=False, sort=False):
            net_sum = pd.to_numeric(group["net_return"], errors="coerce").sum()
            rows.append(
                {
                    **_summary_row(group, len(group), candidate=candidate, baseline=baseline, cost_single_side_bps=cost, month=month),
                    "net_contribution": net_sum / total if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _leave_one_month(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if trades.empty:
        return pd.DataFrame()
    months = sorted(trades["month"].dropna().astype(str).unique())
    for (candidate, baseline, cost), cost_group in trades.groupby(["candidate", "baseline", "cost_single_side_bps"], sort=False):
        for month in months:
            sample = cost_group[~cost_group["month"].astype(str).eq(month)]
            rows.append(_summary_row(sample, len(sample), candidate=candidate, baseline=baseline, cost_single_side_bps=cost, excluded_month=month))
    return pd.DataFrame(rows)


def _symbol_attr(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if trades.empty:
        return pd.DataFrame()
    for (candidate, baseline, cost), cost_group in trades.groupby(["candidate", "baseline", "cost_single_side_bps"], sort=False):
        total = pd.to_numeric(cost_group["net_return"], errors="coerce").sum()
        for symbol, group in cost_group.groupby("symbol", dropna=False, sort=False):
            net_sum = pd.to_numeric(group["net_return"], errors="coerce").sum()
            row = _summary_row(group, len(group), candidate=candidate, baseline=baseline, cost_single_side_bps=cost, symbol=symbol)
            row["net_contribution"] = net_sum / total if total else np.nan
            row["avg_turnover_rank"] = pd.to_numeric(group.get("dynamic_all_rank"), errors="coerce").mean()
            row["avg_volatility_rank"] = pd.to_numeric(group.get("symbol_volatility_percentile"), errors="coerce").mean()
            rows.append(row)
    return pd.DataFrame(rows)


def _boundary(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if trades.empty:
        return pd.DataFrame()
    for (candidate, boundary, cost), group in trades.groupby(["candidate", "boundary", "cost_single_side_bps"], sort=False):
        rows.append(_summary_row(group, len(group), candidate=candidate, boundary=boundary, cost_single_side_bps=cost))
    return pd.DataFrame(rows)


def _reclassification(exact: pd.DataFrame, monthly: pd.DataFrame, symbol: pd.DataFrame, report_root: Path) -> None:
    lines = ["# v0.6A.1 Candidate Reclassification", ""]
    primary = exact[(exact["baseline"].eq("candidate")) & (exact["cost_single_side_bps"].eq(10))]
    for row in primary.sort_values("net_expectancy", ascending=False).itertuples(index=False):
        lines.append(
            f"- {row.candidate}: trades={row.trades}, net10={row.net_expectancy:.4%}, "
            f"fill={row.fill_rate:.2%}, sl={row.sl_rate:.2%}"
        )
        m = monthly[
            monthly["candidate"].eq(row.candidate)
            & monthly["baseline"].eq("candidate")
            & monthly["cost_single_side_bps"].eq(10)
        ].copy()
        s = symbol[
            symbol["candidate"].eq(row.candidate)
            & symbol["baseline"].eq("candidate")
            & symbol["cost_single_side_bps"].eq(10)
        ].copy()
        if not m.empty:
            top_month = m.assign(abs_contrib=m["net_contribution"].abs()).sort_values("abs_contrib", ascending=False).iloc[0]
            lines.append(f"  - max_month_contribution={top_month.net_contribution:.2%} ({top_month.month})")
        if not s.empty:
            top_symbol = s.assign(abs_contrib=s["net_contribution"].abs()).sort_values("abs_contrib", ascending=False).iloc[0]
            lines.append(f"  - max_symbol_contribution={top_symbol.net_contribution:.2%} ({top_symbol.symbol})")
    lines.append("")
    lines.append("Execution reality check is pending. Do not promote to paper-live until 1m/tick validation is complete.")
    (report_root / "10_candidate_reclassification.md").write_text("\n".join(lines), encoding="utf-8")


def write_v06a1_impulse_reclaim_validation(
    feature_path: Path,
    instruments: pd.DataFrame,
    base_config: ExperimentConfig,
    v06a1_config: V06A1Config,
    report_root: Path = REPORT_ROOT,
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    turnover = _read_existing_columns(feature_path, V03_TURNOVER_COLUMNS)
    rank30 = build_dynamic_rank_table(turnover, instruments, base_config)
    rank90 = _rank_table_with_lookback(turnover, instruments, base_config, 90, "turnover_rank_90d", "trailing_90d_turnover")
    symbols = _top_symbols(rank30, MAX_BOUNDARY_TOP_N)
    del turnover

    trade_frames: list[pd.DataFrame] = []
    boundary_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    signal_counts: dict[tuple[str, str], int] = {}

    for idx, symbol in enumerate(symbols, start=1):
        data = _read_symbol_features(feature_path, rank30, rank90, symbol, base_config)
        if data.empty:
            continue
        print(f"v0.6A.1 processing {idx}/{len(symbols)} {symbol}", flush=True)
        for candidate in v06a1_config.candidates:
            policy = _policy(candidate.entry_policy)
            signal = _signal_mask(data, candidate)
            signal_counts[(candidate.candidate, "candidate")] = signal_counts.get((candidate.candidate, "candidate"), 0) + int(signal.sum())
            event_frame = _candidate_events(data, candidate)
            if not event_frame.empty:
                event_frames.append(event_frame)
            candidate_trades = _simulate_rows(
                data,
                signal,
                candidate.candidate,
                candidate.anchor,
                policy,
                candidate.execution_rule,
                base_config,
            )
            if not candidate_trades.empty:
                candidate_trades["baseline"] = "candidate"
                trade_frames.append(_expand_costs(candidate_trades, v06a1_config.cost_bps))

            for baseline, anchor_name, baseline_policy in [
                ("entry_only_reclaim", "neutral_volume", policy),
                ("volume_shock_next_open", candidate.anchor, NEXT_OPEN),
                (
                    "volume_shock_pullback_only",
                    candidate.anchor,
                    _policy("B1_pullback_1.0pct_valid_8_bars")
                    if "1.0pct" in policy.name
                    else _policy("B1_pullback_0.5pct_valid_8_bars"),
                ),
                ("bearish_volume_shock_control", "bearish_volume_shock", policy),
                ("matched_random", candidate.anchor, policy),
            ]:
                if baseline == "matched_random":
                    base_mask = _matched_random_rows(data, signal)
                else:
                    base_mask = (
                        (pd.to_numeric(data["dynamic_all_rank"], errors="coerce") <= candidate.universe_top_n)
                        & data[f"gate_{candidate.gate}"].fillna(False)
                        & data[_anchor_col(anchor_name)].fillna(False)
                    )
                signal_counts[(candidate.candidate, baseline)] = signal_counts.get((candidate.candidate, baseline), 0) + int(base_mask.sum())
                trades = _simulate_rows(
                    data,
                    base_mask,
                    candidate.candidate,
                    anchor_name,
                    baseline_policy,
                    candidate.execution_rule,
                    base_config,
                )
                if not trades.empty:
                    trades["baseline"] = baseline
                    trade_frames.append(_expand_costs(trades, v06a1_config.cost_bps))

            for top_n in v06a1_config.boundary_top_ns:
                boundary_mask = _candidate_boundary_mask(data, candidate, top_n, None)
                trades = _simulate_rows(
                    data,
                    boundary_mask,
                    candidate.candidate,
                    candidate.anchor,
                    policy,
                    candidate.execution_rule,
                    base_config,
                )
                if not trades.empty:
                    trades["boundary"] = f"dynamic_all_top{top_n}"
                    boundary_frames.append(_expand_costs(trades, v06a1_config.cost_bps))
            for bucket in ["rank_1_10", "rank_11_30", "rank_31_50", "rank_51_100"]:
                boundary_mask = _candidate_boundary_mask(data, candidate, candidate.universe_top_n, bucket)
                trades = _simulate_rows(
                    data,
                    boundary_mask,
                    candidate.candidate,
                    candidate.anchor,
                    policy,
                    candidate.execution_rule,
                    base_config,
                )
                if not trades.empty:
                    trades["boundary"] = bucket
                    boundary_frames.append(_expand_costs(trades, v06a1_config.cost_bps))

    trades = _concat_or_empty(trade_frames)
    boundary_trades = _concat_or_empty(boundary_frames)
    events = _concat_or_empty(event_frames)
    write_parquet(events, EVENT_OUTPUT)

    frozen = pd.DataFrame([candidate.__dict__ for candidate in v06a1_config.candidates])
    exact = _aggregate_exact(trades, signal_counts, v06a1_config.cost_bps)
    monthly = _monthly(trades)
    leave_one = _leave_one_month(trades)
    symbol = _symbol_attr(trades)
    cost = exact.copy()
    baseline = exact.copy()
    boundary = _boundary(boundary_trades)
    controls = exact[exact["baseline"].isin(["bearish_volume_shock_control", "matched_random"])].copy()
    execution = pd.DataFrame(
        [
            {
                "candidate": candidate.candidate,
                "status": "pending_1m_tick_collection",
                "event_file": str(EVENT_OUTPUT),
                "events": int((events["candidate"].eq(candidate.candidate)).sum()) if not events.empty else 0,
            }
            for candidate in v06a1_config.candidates
        ]
    )

    outputs = {
        "frozen_candidates": report_root / "00_frozen_candidates.csv",
        "exact_gate_replay": report_root / "01_exact_gate_replay.csv",
        "monthly_attribution": report_root / "02_monthly_attribution.csv",
        "leave_one_month_out": report_root / "03_leave_one_month_out.csv",
        "symbol_attribution": report_root / "04_symbol_attribution.csv",
        "cost_stress": report_root / "05_cost_stress.csv",
        "execution_1m_tick": report_root / "06_1m_tick_execution.csv",
        "baseline_decomposition": report_root / "07_baseline_decomposition.csv",
        "universe_boundary": report_root / "08_universe_boundary.csv",
        "regime_negative_controls": report_root / "09_regime_negative_controls.csv",
        "candidate_reclassification": report_root / "10_candidate_reclassification.md",
        "frozen_candidate_events": EVENT_OUTPUT,
    }
    frozen.to_csv(outputs["frozen_candidates"], index=False)
    exact.to_csv(outputs["exact_gate_replay"], index=False)
    monthly.to_csv(outputs["monthly_attribution"], index=False)
    leave_one.to_csv(outputs["leave_one_month_out"], index=False)
    symbol.to_csv(outputs["symbol_attribution"], index=False)
    cost.to_csv(outputs["cost_stress"], index=False)
    execution.to_csv(outputs["execution_1m_tick"], index=False)
    baseline.to_csv(outputs["baseline_decomposition"], index=False)
    boundary.to_csv(outputs["universe_boundary"], index=False)
    controls.to_csv(outputs["regime_negative_controls"], index=False)
    _reclassification(exact, monthly, symbol, report_root)
    return outputs
