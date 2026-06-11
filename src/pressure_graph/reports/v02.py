from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from pressure_graph.backtest.trade_sequence import SequenceTrade, _entry_trade_sequence, _exit_trade_sequence
from pressure_graph.backtest.entry_policies import ENTRY_POLICIES, EntryPolicy
from pressure_graph.backtest.trade_sequence import (
    simulate_trade_sequence_candidate,
    summarize_sequence_trades,
    write_trade_sequence_outputs,
)
from pressure_graph.clients.bybit_public import load_public_trade_file
from pressure_graph.config import ExperimentConfig
from pressure_graph.config.models import ExecutionRule
from pressure_graph.config.v02 import FrozenCandidate, V02Config
from pressure_graph.io import ensure_dir


def policy_by_name(name: str) -> EntryPolicy:
    for policy in ENTRY_POLICIES:
        if policy.name == name:
            return policy
    raise KeyError(f"unknown entry policy: {name}")


def execution_rule_by_name(name: str, row: pd.Series | None, base_config: ExperimentConfig) -> ExecutionRule:
    if name == "fast":
        return base_config.execution.rules["fast"]
    if name == "swing":
        return base_config.execution.rules["swing"]
    if name == "vol_regime_fast":
        vol_pct = pd.to_numeric(row.get("symbol_volatility_percentile") if row is not None else None, errors="coerce")
        if pd.isna(vol_pct) or vol_pct < 40:
            return ExecutionRule(tp=0.03, sl=0.02, max_hold_bars=16)
        if vol_pct < 80:
            return ExecutionRule(tp=0.04, sl=0.025, max_hold_bars=16)
        return ExecutionRule(tp=0.05, sl=0.03, max_hold_bars=16)
    raise KeyError(f"unknown execution rule: {name}")


def load_public_trades_for_available_days(cache_root: Path) -> pd.DataFrame:
    from pressure_graph.clients.bybit_public import load_public_trade_file

    frames = []
    for file in sorted(cache_root.glob("*/*.csv.gz")):
        try:
            frames.append(load_public_trade_file(file))
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True).sort_values(["exchange", "symbol", "timestamp"]) if frames else pd.DataFrame()


def available_public_trade_days(cache_root: Path) -> dict[str, set[str]]:
    days: dict[str, set[str]] = {}
    for file in sorted(cache_root.glob("*/*.csv.gz")):
        symbol = file.parent.name
        suffix = file.name.removeprefix(symbol).removesuffix(".csv.gz")
        if len(suffix) == 10:
            days.setdefault(symbol, set()).add(suffix)
    return days


def _day_keys(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    day = pd.Timestamp(start).floor("D")
    last = pd.Timestamp(end).floor("D")
    keys = []
    while day <= last:
        keys.append(day.strftime("%Y-%m-%d"))
        day += pd.Timedelta(days=1)
    return keys


def restrict_signals_to_public_trade_days(
    signal_rows: pd.DataFrame,
    day_map: dict[str, set[str]],
    max_lookahead_hours: int = 16,
) -> pd.DataFrame:
    if signal_rows.empty:
        return signal_rows.copy()
    mask = []
    for row in signal_rows.itertuples(index=False):
        start = pd.Timestamp(row.feature_time)
        end = start + pd.Timedelta(hours=max_lookahead_hours)
        available = day_map.get(str(row.symbol), set())
        mask.append(all(day in available for day in _day_keys(start, end)))
    return signal_rows.loc[mask].copy()


def _load_symbol_day_trades(cache_root: Path, symbol: str, day_key: str) -> pd.DataFrame:
    path = cache_root / symbol / f"{symbol}{day_key}.csv.gz"
    if not path.exists() or path.stat().st_size <= 0:
        return pd.DataFrame()
    parquet = cache_root.with_name("public_trading_parquet") / symbol / f"{symbol}{day_key}.parquet"
    if parquet.exists() and parquet.stat().st_size > 0:
        return pd.read_parquet(parquet)
    trades = load_public_trade_file(path)
    ensure_dir(parquet.parent)
    tmp = parquet.with_name(parquet.name + ".tmp.parquet")
    trades.to_parquet(tmp, index=False)
    tmp.replace(parquet)
    return trades


def _load_symbol_execution_window(cache_root: Path, symbol: str, day_key: str) -> pd.DataFrame:
    next_day = (pd.Timestamp(day_key) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    frames = [
        frame
        for frame in [
            _load_symbol_day_trades(cache_root, symbol, day_key),
            _load_symbol_day_trades(cache_root, symbol, next_day),
        ]
        if not frame.empty
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("timestamp")


def run_tick_execution(
    signal_rows: pd.DataFrame,
    public_trades: pd.DataFrame,
    base_config: ExperimentConfig,
    v02_config: V02Config,
) -> pd.DataFrame:
    frames = []
    for candidate in v02_config.candidates:
        for fill_policy in v02_config.fill_policies:
            for cost in v02_config.cost_bps:
                rule = execution_rule_by_name(candidate.execution_rule, None, base_config)
                resolver = (
                    (lambda row, name=candidate.execution_rule: execution_rule_by_name(name, row, base_config))
                    if candidate.execution_rule == "vol_regime_fast"
                    else None
                )
                frames.append(
                    simulate_trade_sequence_candidate(
                        signal_rows,
                        public_trades,
                        candidate.candidate,
                        candidate.path_name,
                        candidate.signal_col,
                        candidate.entry_policy,
                        candidate.execution_rule,
                        rule,
                        fill_policy,
                        cost,
                        resolver,
                    )
                )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_tick_execution_from_cache(
    signal_rows: pd.DataFrame,
    cache_root: Path,
    base_config: ExperimentConfig,
    v02_config: V02Config,
) -> pd.DataFrame:
    rows = []
    for (exchange, symbol), group in signal_rows.sort_values(
        ["exchange", "symbol", "bar_open_time"]
    ).groupby(["exchange", "symbol"], sort=False):
        group = group.reset_index(drop=True)
        candidate_days: dict[str, dict[str, list[int]]] = {}
        all_days: set[str] = set()
        for candidate in v02_config.candidates:
            if candidate.signal_col not in group.columns:
                continue
            signal_mask = group[candidate.signal_col].astype("boolean").fillna(False).astype(bool)
            for idx in signal_mask[signal_mask].index:
                day_key = pd.Timestamp(group.iloc[int(idx)]["feature_time"]).floor("D").strftime("%Y-%m-%d")
                candidate_days.setdefault(candidate.candidate, {}).setdefault(day_key, []).append(int(idx))
                all_days.add(day_key)
        if not all_days:
            continue

        active_until = {
            (candidate.candidate, fill_policy.name): pd.Timestamp.min.tz_localize("UTC")
            for candidate in v02_config.candidates
            for fill_policy in v02_config.fill_policies
        }
        for day_key in sorted(all_days):
            trades = _load_symbol_execution_window(cache_root, str(symbol), day_key)
            if trades.empty:
                continue
            for candidate in v02_config.candidates:
                day_indices = candidate_days.get(candidate.candidate, {}).get(day_key, [])
                if not day_indices:
                    continue
                state_signal_col = candidate.signal_col.removesuffix("_event")
                if state_signal_col not in group.columns:
                    group[state_signal_col] = group[candidate.signal_col].fillna(False)
                for fill_policy in v02_config.fill_policies:
                    active_key = (candidate.candidate, fill_policy.name)
                    for idx in day_indices:
                        signal = group.iloc[idx]
                        signal_time = pd.Timestamp(signal["feature_time"])
                        if signal_time <= active_until[active_key]:
                            continue
                        entry = _entry_trade_sequence(
                            group,
                            idx,
                            trades,
                            candidate.entry_policy,
                            fill_policy,
                            state_signal_col,
                        )
                        if entry is None:
                            for cost in v02_config.cost_bps:
                                rows.append(
                                    asdict(
                                        SequenceTrade(
                                            candidate.candidate,
                                            str(exchange),
                                            str(symbol),
                                            candidate.path_name,
                                            candidate.entry_policy,
                                            candidate.execution_rule,
                                            fill_policy.name,
                                            float(cost),
                                            signal_time,
                                            None,
                                            None,
                                            None,
                                            None,
                                            None,
                                            None,
                                            "missed_entry",
                                            False,
                                            False,
                                            None,
                                            None,
                                        )
                                    )
                                )
                            continue
                        entry_time, entry_price = entry
                        trade_rule = execution_rule_by_name(candidate.execution_rule, signal, base_config)
                        exit_result = _exit_trade_sequence(trades, entry_time, entry_price, trade_rule)
                        if exit_result is None:
                            continue
                        exit_time, exit_price, exit_reason = exit_result
                        gross = exit_price / entry_price - 1.0
                        active_until[active_key] = exit_time
                        for cost in v02_config.cost_bps:
                            net = gross - 2.0 * float(cost) / 10_000.0
                            rows.append(
                                asdict(
                                    SequenceTrade(
                                        candidate.candidate,
                                        str(exchange),
                                        str(symbol),
                                        candidate.path_name,
                                        candidate.entry_policy,
                                        candidate.execution_rule,
                                        fill_policy.name,
                                        float(cost),
                                        signal_time,
                                        entry_time,
                                        exit_time,
                                        entry_price,
                                        exit_price,
                                        gross,
                                        net,
                                        exit_reason,
                                        True,
                                        False,
                                        float((exit_time - entry_time) / pd.Timedelta(minutes=1)),
                                        float((entry_time - signal_time) / pd.Timedelta(minutes=1)),
                                    )
                                )
                            )
    return pd.DataFrame(rows)


def restrict_signals_to_trade_coverage(
    signal_rows: pd.DataFrame,
    public_trades: pd.DataFrame,
    max_lookahead_hours: int = 16,
) -> pd.DataFrame:
    if signal_rows.empty or public_trades.empty:
        return signal_rows.iloc[0:0].copy()
    windows = (
        public_trades.groupby(["exchange", "symbol"], as_index=False)["timestamp"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "trade_start", "max": "trade_end"})
    )
    out = signal_rows.merge(windows, on=["exchange", "symbol"], how="inner")
    feature_time = pd.to_datetime(out["feature_time"], utc=True)
    safe_end = pd.to_datetime(out["trade_end"], utc=True) - pd.Timedelta(hours=max_lookahead_hours)
    out = out[(feature_time >= pd.to_datetime(out["trade_start"], utc=True)) & (feature_time <= safe_end)]
    return out.drop(columns=["trade_start", "trade_end"])


def write_cost_stress(execution_1m: pd.DataFrame, tick_execution: pd.DataFrame, report_root: Path) -> Path:
    rows = []
    for granularity, summary in [("1m", execution_1m), ("trade_sequence", tick_execution)]:
        if summary.empty:
            continue
        net_col = "net_expectancy_1m" if granularity == "1m" else "net_expectancy"
        for row in summary.itertuples(index=False):
            rows.append(
                {
                    "candidate": getattr(row, "candidate", getattr(row, "path_name", "")),
                    "execution_granularity": granularity,
                    "fill_policy": getattr(row, "fill_policy", ""),
                    "cost_single_side_bps": row.cost_single_side_bps,
                    "net_expectancy": getattr(row, net_col),
                    "trades": getattr(row, "trade_n", getattr(row, "trades", 0)),
                }
            )
    path = report_root / "cost_stress.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_fill_assumption_stress(tick_summary: pd.DataFrame, report_root: Path) -> Path:
    if tick_summary.empty:
        out = tick_summary
    else:
        out = tick_summary[
            [
                "candidate",
                "fill_policy",
                "cost_single_side_bps",
                "signals",
                "trades",
                "fill_rate",
                "net_expectancy",
                "tp_first_rate",
                "sl_first_rate",
                "timeout_rate",
            ]
        ].copy()
    path = report_root / "fill_assumption_stress.csv"
    out.to_csv(path, index=False)
    return path


def write_symbol_month_contribution(trades: pd.DataFrame, report_root: Path) -> Path:
    rows = []
    filled = trades[trades.get("filled", False).fillna(False)] if not trades.empty else trades
    if not filled.empty:
        filled = filled.copy()
        filled["month"] = pd.to_datetime(filled["signal_time"], utc=True).dt.strftime("%Y-%m")
        for candidate, group in filled.groupby("candidate"):
            for kind, col in [("symbol", "symbol"), ("month", "month")]:
                counts = group[col].value_counts(normalize=True)
                if not counts.empty:
                    rows.append(
                        {
                            "candidate": candidate,
                            "dimension": kind,
                            "top_value": counts.index[0],
                            "top_contribution": float(counts.iloc[0]),
                        }
                    )
    path = report_root / "symbol_month_contribution.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _add_match_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = pd.to_datetime(out["bar_open_time"], utc=True).dt.strftime("%Y-%m")
    out["vol_bucket"] = pd.cut(
        pd.to_numeric(out["symbol_volatility_percentile"], errors="coerce"),
        bins=[-1, 20, 40, 60, 80, 101],
        labels=["v0_20", "v20_40", "v40_60", "v60_80", "v80_100"],
    ).astype(str)
    return out


def _matched_random_rows(signal_rows: pd.DataFrame, signal_col: str, seed: int = 42) -> pd.DataFrame:
    data = _add_match_keys(signal_rows)
    rng = pd.Series(index=data.index, data=pd.util.hash_pandas_object(data[["symbol", "bar_open_time"]], index=False))
    data["_rand"] = ((rng + seed) % 1_000_000).astype(float)
    selected = []
    keys = ["symbol", "month", "vol_bucket", "btc_market_state"]
    for _, group in data.groupby(keys, sort=False):
        signal_count = int(group[signal_col].fillna(False).sum())
        if signal_count <= 0:
            continue
        non_signal = group[~group[signal_col].fillna(False)].sort_values("_rand")
        if non_signal.empty:
            continue
        selected.append(non_signal.head(signal_count))
    if not selected:
        return signal_rows.iloc[0:0].copy()
    out = pd.concat(selected, ignore_index=True).drop(columns=["_rand"])
    out["baseline_event"] = True
    return out


def _entry_only_anchor_rows(
    signal_rows: pd.DataFrame,
    signal_col: str | None = None,
    cooldown_bars: int = 16,
) -> pd.DataFrame:
    if signal_col:
        data = _add_match_keys(signal_rows)
        rows = []
        keys = ["symbol", "month", "vol_bucket", "btc_market_state"]
        for _, group in data.sort_values("bar_open_time").groupby(keys, sort=False):
            target = int(group[signal_col].fillna(False).sum())
            if target <= 0:
                continue
            anchors = group[~group[signal_col].fillna(False)].iloc[::cooldown_bars].head(target)
            if not anchors.empty:
                rows.append(anchors)
        if not rows:
            return signal_rows.iloc[0:0].copy()
        out = pd.concat(rows, ignore_index=True)
        out["baseline_event"] = True
        return out

    rows = []
    for _, group in signal_rows.sort_values(["exchange", "symbol", "bar_open_time"]).groupby(
        ["exchange", "symbol"], sort=False
    ):
        rows.append(group.iloc[::cooldown_bars].copy())
    if not rows:
        return signal_rows.iloc[0:0].copy()
    out = pd.concat(rows, ignore_index=True)
    out["baseline_event"] = True
    return out


def run_tick_baseline(
    signal_rows: pd.DataFrame,
    public_trades: pd.DataFrame,
    base_config: ExperimentConfig,
    v02_config: V02Config,
    baseline_kind: str,
) -> pd.DataFrame:
    frames = []
    for candidate in v02_config.candidates:
        if baseline_kind == "matched_random":
            baseline_rows = _matched_random_rows(signal_rows, candidate.signal_col)
        elif baseline_kind == "entry_only":
            baseline_rows = _entry_only_anchor_rows(signal_rows, candidate.signal_col)
        else:
            raise ValueError(f"unknown baseline kind: {baseline_kind}")
        if baseline_rows.empty:
            continue
        for fill_policy in v02_config.fill_policies:
            for cost in v02_config.cost_bps:
                rule = execution_rule_by_name(candidate.execution_rule, None, base_config)
                resolver = (
                    (lambda row, name=candidate.execution_rule: execution_rule_by_name(name, row, base_config))
                    if candidate.execution_rule == "vol_regime_fast"
                    else None
                )
                result = simulate_trade_sequence_candidate(
                    baseline_rows,
                    public_trades,
                    candidate.candidate,
                    candidate.path_name,
                    "baseline_event",
                    candidate.entry_policy,
                    candidate.execution_rule,
                    rule,
                    fill_policy,
                    cost,
                    resolver,
                )
                if not result.empty:
                    result["baseline_kind"] = baseline_kind
                    frames.append(result)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _single_candidate_config(
    v02_config: V02Config,
    candidate: FrozenCandidate,
    signal_col: str,
) -> V02Config:
    baseline_candidate = FrozenCandidate(
        candidate=candidate.candidate,
        path_name=candidate.path_name,
        signal_col=signal_col,
        entry_policy=candidate.entry_policy,
        execution_rule=candidate.execution_rule,
        priority=candidate.priority,
        rationale=candidate.rationale,
    )
    return V02Config(
        name=v02_config.name,
        source_config=v02_config.source_config,
        universe_col=v02_config.universe_col,
        final_holdout_only=v02_config.final_holdout_only,
        candidates=[baseline_candidate],
        fill_policies=v02_config.fill_policies,
        cost_bps=v02_config.cost_bps,
        max_positions=v02_config.max_positions,
        rankings=v02_config.rankings,
    )


def run_tick_baseline_from_cache(
    signal_rows: pd.DataFrame,
    cache_root: Path,
    base_config: ExperimentConfig,
    v02_config: V02Config,
    baseline_kind: str,
) -> pd.DataFrame:
    frames = []
    baseline_candidates = []
    for candidate in v02_config.candidates:
        if baseline_kind == "matched_random":
            baseline_rows = _matched_random_rows(signal_rows, candidate.signal_col)
        elif baseline_kind == "entry_only":
            baseline_rows = _entry_only_anchor_rows(signal_rows, candidate.signal_col)
        else:
            raise ValueError(f"unknown baseline kind: {baseline_kind}")
        if baseline_rows.empty:
            continue
        signal_col = f"baseline_event_{candidate.candidate}"
        baseline_rows = baseline_rows.copy()
        baseline_rows[signal_col] = True
        frames.append(baseline_rows)
        baseline_candidates.append(
            FrozenCandidate(
                candidate=candidate.candidate,
                path_name=candidate.path_name,
                signal_col=signal_col,
                entry_policy=candidate.entry_policy,
                execution_rule=candidate.execution_rule,
                priority=candidate.priority,
                rationale=candidate.rationale,
            )
        )
    if not frames:
        return pd.DataFrame()
    config = V02Config(
        name=v02_config.name,
        source_config=v02_config.source_config,
        universe_col=v02_config.universe_col,
        final_holdout_only=v02_config.final_holdout_only,
        candidates=baseline_candidates,
        fill_policies=v02_config.fill_policies,
        cost_bps=v02_config.cost_bps,
        max_positions=v02_config.max_positions,
        rankings=v02_config.rankings,
    )
    result = run_tick_execution_from_cache(pd.concat(frames, ignore_index=True), cache_root, base_config, config)
    if not result.empty:
        result["baseline_kind"] = baseline_kind
    return result


def write_baseline_summary(baseline_trades: pd.DataFrame, report_root: Path, filename: str) -> Path:
    summary = summarize_sequence_trades(baseline_trades)
    path = report_root / filename
    summary.to_csv(path, index=False)
    return path


def write_capital_overlap(trades: pd.DataFrame, v02_config: V02Config, report_root: Path) -> Path:
    rows = []
    filled = trades[trades.get("filled", False).fillna(False)] if not trades.empty else trades
    if not filled.empty:
        filled = filled.copy()
        filled["entry_time"] = pd.to_datetime(filled["entry_time"], utc=True)
        filled["exit_time"] = pd.to_datetime(filled["exit_time"], utc=True)
        group_cols = ["candidate", "fill_policy", "cost_single_side_bps"]
        for key, group in filled.groupby(group_cols):
            candidate, fill_policy, cost = key
            points = sorted(set(group["entry_time"]).union(set(group["exit_time"])))
            active_counts = []
            for ts in points:
                active = group[(group["entry_time"] <= ts) & (group["exit_time"] > ts)]
                active_counts.append(len(active))
            for max_pos in v02_config.max_positions:
                rows.append(
                    {
                        "candidate": candidate,
                        "fill_policy": fill_policy,
                        "cost_single_side_bps": cost,
                        "max_positions": max_pos,
                        "max_concurrent_positions": max(active_counts) if active_counts else 0,
                        "avg_concurrent_positions": (
                            sum(active_counts) / len(active_counts) if active_counts else 0
                        ),
                        "capital_utilization_proxy": (
                            min(max(active_counts), max_pos) / max_pos if active_counts else 0
                        ),
                    }
                )
    path = report_root / "capital_overlap.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_v02_candidate_summary(
    v02_config: V02Config,
    tick_summary: pd.DataFrame,
    report_root: Path,
) -> Path:
    rows = []
    for candidate in v02_config.candidates:
        rows.append(
            {
                "candidate": candidate.candidate,
                "path_name": candidate.path_name,
                "entry_policy": candidate.entry_policy,
                "execution_rule": candidate.execution_rule,
                "priority": candidate.priority,
                "rationale": candidate.rationale,
            }
        )
    path = report_root / "candidate_frozen_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    lines = ["# Crypto Pressure Graph v0.2 Candidate List", ""]
    lines.append("Frozen candidates only; no new optimization in v0.2.")
    if not tick_summary.empty:
        lines.append("")
        lines.append("## Available Tick/Trade-Sequence Summary")
        best = tick_summary[tick_summary["cost_single_side_bps"].eq(5)].sort_values(
            "net_expectancy", ascending=False
        )
        for row in best.head(12).itertuples(index=False):
            lines.append(
                f"- {row.candidate} / {row.fill_policy}: trades={row.trades}, "
                f"fill={row.fill_rate:.2%}, net5={row.net_expectancy:.4%}, "
                f"tp={row.tp_first_rate:.2%}, sl={row.sl_first_rate:.2%}"
            )
    (report_root / "candidate_list.md").write_text("\n".join(lines), encoding="utf-8")
    return path


def write_v02_reports(
    signal_rows: pd.DataFrame,
    one_min_summary: pd.DataFrame,
    public_trade_cache_root: Path | None,
    base_config: ExperimentConfig,
    v02_config: V02Config,
    include_baselines: bool = True,
) -> dict[str, Path]:
    report_root = ensure_dir(Path("reports/v0_2"))
    outputs: dict[str, Path] = {}

    one_min_path = report_root / "execution_1m_comparison.csv"
    one_min_summary.to_csv(one_min_path, index=False)
    outputs["execution_1m_comparison"] = one_min_path

    day_map = available_public_trade_days(public_trade_cache_root) if public_trade_cache_root else {}
    covered_signal_rows = restrict_signals_to_public_trade_days(signal_rows, day_map)
    tick_trades = (
        run_tick_execution_from_cache(covered_signal_rows, public_trade_cache_root, base_config, v02_config)
        if public_trade_cache_root
        else pd.DataFrame()
    )
    outputs.update(write_trade_sequence_outputs(tick_trades, report_root))
    tick_summary = summarize_sequence_trades(tick_trades)
    if include_baselines and public_trade_cache_root:
        matched_random = run_tick_baseline_from_cache(
            covered_signal_rows, public_trade_cache_root, base_config, v02_config, "matched_random"
        )
        entry_only = run_tick_baseline_from_cache(
            covered_signal_rows, public_trade_cache_root, base_config, v02_config, "entry_only"
        )
    else:
        matched_random = pd.DataFrame()
        entry_only = pd.DataFrame()
    outputs["matched_random_baseline"] = write_baseline_summary(
        matched_random, report_root, "matched_random_baseline.csv"
    )
    outputs["entry_only_baseline"] = write_baseline_summary(
        entry_only, report_root, "entry_only_baseline.csv"
    )
    outputs["fill_assumption_stress"] = write_fill_assumption_stress(tick_summary, report_root)
    outputs["cost_stress"] = write_cost_stress(one_min_summary, tick_summary, report_root)
    outputs["symbol_month_contribution"] = write_symbol_month_contribution(tick_trades, report_root)
    outputs["capital_overlap"] = write_capital_overlap(tick_trades, v02_config, report_root)
    outputs["candidate_frozen_summary"] = write_v02_candidate_summary(
        v02_config, tick_summary, report_root
    )
    outputs["candidate_list"] = report_root / "candidate_list.md"
    return outputs
