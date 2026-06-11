from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pressure_graph.backtest.entry_policies import EntryPolicy
from pressure_graph.backtest.minute_execution import simulate_1m_execution
from pressure_graph.backtest.trade_sequence import summarize_sequence_trades
from pressure_graph.clients import BybitClient
from pressure_graph.clients.bybit_public import download_public_trading_day, public_trades_to_1m_ohlcv
from pressure_graph.config import ExperimentConfig
from pressure_graph.config.v02 import FillPolicy, FrozenCandidate, V02Config
from pressure_graph.io import ensure_dir, read_parquet, write_parquet
from pressure_graph.reports.v02 import (
    available_public_trade_days,
    restrict_signals_to_public_trade_days,
    run_tick_execution_from_cache,
)
from pressure_graph.reports.v03 import (
    V03_TURNOVER_COLUMNS,
    _concat_or_empty,
    _read_existing_columns,
    build_dynamic_rank_table,
)
from pressure_graph.reports.v04 import _rank_table_with_lookback
from pressure_graph.reports.v06a1 import (
    MAX_BOUNDARY_TOP_N,
    FrozenImpulseReclaimCandidate,
    _read_symbol_features,
    _rule,
    _signal_mask,
)


REPORT_ROOT = Path("reports/v0_6a2_execution_concentration")
SIGNAL_ROWS_OUTPUT = Path("data/processed/v0_6a2/frozen_candidate_signal_rows.parquet")
ONE_MIN_CACHE_DATASET = "klines_1m_v06a2"
PUBLIC_TRADE_CACHE_DATASET = "public_trading"
ENTRY_SEQUENCE_MAP = {
    "R1_pullback_1.0pct_reclaim_signal_close_valid_8_bars": (
        "E7_pullback_1.0pct_then_reclaim_signal_close_valid_8_bars"
    ),
    "R2_pullback_0.5pct_reclaim_signal_close_valid_8_bars": (
        "E6_pullback_0.5pct_then_reclaim_signal_close_valid_8_bars"
    ),
}
ENTRY_POLICY_MAP = {
    "R1_pullback_1.0pct_reclaim_signal_close_valid_8_bars": EntryPolicy(
        "R1_pullback_1.0pct_reclaim_signal_close_valid_8_bars",
        "pullback_reclaim",
        valid_bars=8,
        pullback_pct=0.010,
    ),
    "R2_pullback_0.5pct_reclaim_signal_close_valid_8_bars": EntryPolicy(
        "R2_pullback_0.5pct_reclaim_signal_close_valid_8_bars",
        "pullback_reclaim",
        valid_bars=8,
        pullback_pct=0.005,
    ),
}


@dataclass(frozen=True)
class V06A2ExecutionValidation:
    max_lookahead_hours: int = 16
    one_min_cache_dataset: str = ONE_MIN_CACHE_DATASET
    public_trade_cache_dataset: str = PUBLIC_TRADE_CACHE_DATASET
    month_cap_levels: list[float] | None = None
    focal_month: str = "2025-08"
    tick_required_for_paper_live: bool = True


@dataclass(frozen=True)
class V06A2Config:
    name: str
    source_config: Path
    source_candidates: Path
    frozen: bool
    mode: str
    candidates: list[FrozenImpulseReclaimCandidate]
    cost_bps: list[float]
    fill_policies: list[FillPolicy]
    execution_validation: V06A2ExecutionValidation


def load_v06a2_config(
    path: str | Path = "configs/v0_6a2_execution_concentration.yaml",
) -> V06A2Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    experiment = payload.get("experiment", {})
    validation = payload.get("execution_validation", {})
    fill_policies = [
        FillPolicy(name=name, **settings)
        for name, settings in (payload.get("fill_policies") or {}).items()
    ]
    if not fill_policies:
        fill_policies = [
            FillPolicy("normal_2bp", 2.0, "limit"),
            FillPolicy("conservative_5bp", 5.0, "limit"),
        ]
    return V06A2Config(
        name=str(experiment.get("name", "v0_6a2_execution_concentration")),
        source_config=Path(experiment.get("source_config", "configs/v0_3.yaml")),
        source_candidates=Path(
            experiment.get("source_candidates", "configs/v0_6a1_frozen_candidates.yaml")
        ),
        frozen=bool(experiment.get("frozen", True)),
        mode=str(experiment.get("mode", "validation_only")),
        candidates=[FrozenImpulseReclaimCandidate(**item) for item in payload.get("candidates", [])],
        cost_bps=[float(item) for item in payload.get("cost_bps", [5, 10, 20, 30, 50])],
        fill_policies=fill_policies,
        execution_validation=V06A2ExecutionValidation(
            max_lookahead_hours=int(validation.get("max_lookahead_hours", 16)),
            one_min_cache_dataset=str(
                validation.get("one_min_cache_dataset", ONE_MIN_CACHE_DATASET)
            ),
            public_trade_cache_dataset=str(
                validation.get("public_trade_cache_dataset", PUBLIC_TRADE_CACHE_DATASET)
            ),
            month_cap_levels=[
                float(item) for item in validation.get("month_cap_levels", [0.35, 0.40, 0.50])
            ],
            focal_month=str(validation.get("focal_month", "2025-08")),
            tick_required_for_paper_live=bool(
                validation.get("tick_required_for_paper_live", True)
            ),
        ),
    )


def _signal_col(candidate: FrozenImpulseReclaimCandidate) -> str:
    return f"{candidate.candidate}_signal_event"


def _entry_sequence_name(candidate: FrozenImpulseReclaimCandidate) -> str:
    try:
        return ENTRY_SEQUENCE_MAP[candidate.entry_policy]
    except KeyError as exc:
        raise KeyError(f"unsupported v0.6A.2 entry policy: {candidate.entry_policy}") from exc


def _entry_policy(candidate: FrozenImpulseReclaimCandidate) -> EntryPolicy:
    try:
        return ENTRY_POLICY_MAP[candidate.entry_policy]
    except KeyError as exc:
        raise KeyError(f"unsupported v0.6A.2 entry policy: {candidate.entry_policy}") from exc


def _v02_adapter(v06a2_config: V06A2Config) -> V02Config:
    candidates = [
        FrozenCandidate(
            candidate=candidate.candidate,
            path_name=candidate.anchor,
            signal_col=_signal_col(candidate),
            entry_policy=_entry_sequence_name(candidate),
            execution_rule=candidate.execution_rule,
            priority="frozen_validation",
            rationale=candidate.rationale,
        )
        for candidate in v06a2_config.candidates
    ]
    return V02Config(
        name=v06a2_config.name,
        source_config=v06a2_config.source_config,
        universe_col="dynamic_all_frozen_candidate_events",
        final_holdout_only=False,
        candidates=candidates,
        fill_policies=v06a2_config.fill_policies,
        cost_bps=v06a2_config.cost_bps,
        max_positions=[1],
        rankings=["frozen_order"],
    )


def _minimal_signal_columns(candidates: list[FrozenImpulseReclaimCandidate]) -> list[str]:
    return list(
        dict.fromkeys(
            [
                "exchange",
                "symbol",
                "bar_open_time",
                "bar_close_time",
                "feature_time",
                "open",
                "high",
                "low",
                "close",
                "month",
                "dynamic_all_rank",
                "dynamic_all_trailing_turnover",
                "turnover_rank_90d",
                "trailing_90d_turnover",
                "liquidity_bucket",
                "btc_market_state",
                "btc_ret_1h",
                "btc_ret_4h",
                "btc_volatility_4h",
                "symbol_volatility_percentile",
                "vol_bucket",
                "volume_bucket",
                "volume_z_4h",
                "ret_4h",
                "core_liquidity",
                "transient_hot",
                *[_signal_col(candidate) for candidate in candidates],
            ]
        )
    )


def build_v06a2_signal_rows(
    feature_path: Path,
    instruments: pd.DataFrame,
    base_config: ExperimentConfig,
    v06a2_config: V06A2Config,
    output_path: Path = SIGNAL_ROWS_OUTPUT,
) -> pd.DataFrame:
    if output_path.exists() and output_path.stat().st_size > 0:
        return read_parquet(output_path)

    turnover = _read_existing_columns(feature_path, V03_TURNOVER_COLUMNS)
    rank30 = build_dynamic_rank_table(turnover, instruments, base_config)
    rank90 = _rank_table_with_lookback(
        turnover,
        instruments,
        base_config,
        90,
        "turnover_rank_90d",
        "trailing_90d_turnover",
    )
    symbols = sorted(
        rank30[pd.to_numeric(rank30["dynamic_all_rank"], errors="coerce") <= MAX_BOUNDARY_TOP_N][
            "symbol"
        ]
        .dropna()
        .astype(str)
        .unique()
    )
    del turnover

    frames: list[pd.DataFrame] = []
    for idx, symbol in enumerate(symbols, start=1):
        data = _read_symbol_features(feature_path, rank30, rank90, symbol, base_config)
        if data.empty:
            continue
        print(f"v0.6A.2 signal rows {idx}/{len(symbols)} {symbol}", flush=True)
        combined = pd.Series(False, index=data.index)
        for candidate in v06a2_config.candidates:
            col = _signal_col(candidate)
            data[col] = _signal_mask(data, candidate).fillna(False).astype(bool)
            combined |= data[col]
        if not combined.any():
            continue
        cols = [col for col in _minimal_signal_columns(v06a2_config.candidates) if col in data.columns]
        frames.append(data.loc[combined, cols].copy())

    signal_rows = _concat_or_empty(frames)
    if not signal_rows.empty:
        signal_rows = signal_rows.sort_values(
            ["exchange", "symbol", "bar_open_time"]
        ).reset_index(drop=True)
    write_parquet(signal_rows, output_path)
    return signal_rows


def _load_1m_bars(base_config: ExperimentConfig, v06a2_config: V06A2Config) -> pd.DataFrame:
    path = (
        base_config.paths.data_root
        / "raw"
        / "bybit"
        / v06a2_config.execution_validation.one_min_cache_dataset
    )
    if not path.exists():
        return pd.DataFrame()
    frames = [pd.read_parquet(file) for file in sorted(path.glob("*.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _merge_intervals(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + pd.Timedelta(minutes=1):
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def collect_v06a2_1m_execution_data(
    feature_path: Path,
    instruments: pd.DataFrame,
    base_config: ExperimentConfig,
    v06a2_config: V06A2Config,
    source: str = "api",
    max_symbol_days: int | None = None,
    symbol_day_offset: int = 0,
    public_trade_workers: int = 4,
) -> Path:
    signal_rows = build_v06a2_signal_rows(feature_path, instruments, base_config, v06a2_config)
    if source == "public-trades":
        return _collect_v06a2_public_trades(
            signal_rows,
            base_config,
            v06a2_config,
            max_symbol_days,
            symbol_day_offset,
            public_trade_workers,
        )

    out_dir = ensure_dir(
        base_config.paths.data_root
        / "raw"
        / "bybit"
        / v06a2_config.execution_validation.one_min_cache_dataset
    )
    if signal_rows.empty:
        return out_dir
    client = BybitClient(str(base_config.exchanges.bybit.base_url), base_config.exchanges.bybit.category)
    try:
        for idx, (symbol, group) in enumerate(signal_rows.groupby("symbol", sort=True), start=1):
            print(f"v0.6A.2 collect 1m {idx} {symbol}", flush=True)
            intervals = [
                (
                    pd.Timestamp(row.feature_time),
                    pd.Timestamp(row.feature_time)
                    + pd.Timedelta(hours=v06a2_config.execution_validation.max_lookahead_hours),
                )
                for row in group.itertuples(index=False)
            ]
            frames = []
            for start, end in _merge_intervals(intervals):
                frames.append(client.klines(str(symbol), start, end, "1m"))
            if not frames:
                continue
            out_path = out_dir / f"{symbol}.parquet"
            if out_path.exists():
                frames.insert(0, read_parquet(out_path))
            data = (
                pd.concat(frames, ignore_index=True)
                .drop_duplicates(["exchange", "symbol", "bar_open_time"])
                .sort_values("bar_open_time")
            )
            write_parquet(data, out_path)
        return out_dir
    finally:
        client.close()


def _collect_v06a2_public_trades(
    signal_rows: pd.DataFrame,
    base_config: ExperimentConfig,
    v06a2_config: V06A2Config,
    max_symbol_days: int | None,
    symbol_day_offset: int,
    public_trade_workers: int,
) -> Path:
    cache_root = ensure_dir(
        base_config.paths.data_root
        / "raw"
        / "bybit"
        / v06a2_config.execution_validation.public_trade_cache_dataset
    )
    minute_out = ensure_dir(
        base_config.paths.data_root
        / "raw"
        / "bybit"
        / f"{v06a2_config.execution_validation.one_min_cache_dataset}_public"
    )
    pairs: list[tuple[str, pd.Timestamp]] = []
    for row in signal_rows.itertuples(index=False):
        day = pd.Timestamp(row.feature_time).floor("D")
        last = (
            pd.Timestamp(row.feature_time)
            + pd.Timedelta(hours=v06a2_config.execution_validation.max_lookahead_hours)
        ).floor("D")
        while day <= last:
            pairs.append((str(row.symbol), day))
            day += pd.Timedelta(days=1)
    pairs = sorted(set(pairs), key=lambda item: (item[0], item[1]))
    if symbol_day_offset:
        pairs = pairs[symbol_day_offset:]
    if max_symbol_days is not None:
        pairs = pairs[:max_symbol_days]

    def fetch(pair: tuple[str, pd.Timestamp]) -> tuple[str, Path | None]:
        symbol, day = pair
        try:
            path = download_public_trading_day(symbol, day, cache_root)
        except Exception as exc:
            print(f"public trade download failed {symbol} {day.date()}: {exc}", flush=True)
            return symbol, None
        return symbol, path

    workers = max(1, int(public_trade_workers))
    if workers == 1:
        results = [fetch(pair) for pair in pairs]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(fetch, pair) for pair in pairs]
            for future in as_completed(futures):
                results.append(future.result())

    by_symbol: dict[str, list[pd.DataFrame]] = {}
    for symbol, path in results:
        if path is None:
            continue
        minute = public_trades_to_1m_ohlcv(path)
        if minute.empty:
            continue
        by_symbol.setdefault(symbol, []).append(minute)
    for symbol, frames in by_symbol.items():
        out_path = minute_out / f"{symbol}.parquet"
        if out_path.exists():
            frames.insert(0, read_parquet(out_path))
        data = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(["exchange", "symbol", "bar_open_time"])
            .sort_values("bar_open_time")
        )
        write_parquet(data, out_path)
    return cache_root


def _one_min_execution(
    signal_rows: pd.DataFrame,
    minute_bars: pd.DataFrame,
    base_config: ExperimentConfig,
    v06a2_config: V06A2Config,
) -> pd.DataFrame:
    if signal_rows.empty or minute_bars.empty:
        return pd.DataFrame()
    frames = []
    for candidate in v06a2_config.candidates:
        rule, resolver = _rule(candidate.execution_rule, base_config)
        for cost in v06a2_config.cost_bps:
            trades = simulate_1m_execution(
                signal_rows,
                minute_bars,
                _signal_col(candidate),
                candidate.candidate,
                _entry_policy(candidate),
                candidate.execution_rule,
                rule,
                float(cost),
                resolver,
            )
            if trades.empty:
                continue
            trades["candidate"] = candidate.candidate
            trades["fill_policy"] = "1m_ohlc_reclaim"
            frames.append(trades)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _summarize_1m(trades: pd.DataFrame, signal_rows: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    signal_counts = {
        candidate: int(signal_rows[f"{candidate}_signal_event"].fillna(False).sum())
        for candidate in sorted(trades["candidate"].dropna().astype(str).unique())
    }
    group_cols = ["candidate", "fill_policy", "entry_policy", "execution_rule", "cost_single_side_bps"]
    for key, group in trades.groupby(group_cols, sort=False, dropna=False):
        candidate, fill_policy, entry_policy, execution_rule, cost = key
        net = pd.to_numeric(group["net_expectancy"], errors="coerce")
        gross = pd.to_numeric(group["gross_return"], errors="coerce")
        exit_reason = group["exit_reason"].astype(str)
        rows.append(
            {
                "candidate": candidate,
                "execution_granularity": "1m_bar",
                "fill_policy": fill_policy,
                "entry_policy": entry_policy,
                "execution_rule": execution_rule,
                "cost_single_side_bps": cost,
                "signals": signal_counts.get(str(candidate), np.nan),
                "trades": int(len(group)),
                "fill_rate": (
                    float(len(group) / signal_counts[str(candidate)])
                    if signal_counts.get(str(candidate), 0)
                    else np.nan
                ),
                "gross_expectancy": float(gross.mean()),
                "net_expectancy": float(net.mean()),
                "tp_first_rate": float(exit_reason.str.startswith("tp").mean()),
                "sl_first_rate": float(exit_reason.str.startswith("sl").mean()),
                "timeout_rate": float(exit_reason.eq("max_hold").mean()),
                "same_bar_ambiguous_rate": float(group["unresolved_1m_same_bar"].mean()),
                "median_holding_minutes": float(
                    pd.to_numeric(group["holding_minutes"], errors="coerce").median()
                ),
                "p25_return": float(net.quantile(0.25)),
                "p75_return": float(net.quantile(0.75)),
                "max_loss": float(net.min()),
            }
        )
    return pd.DataFrame(rows)


def _fifteen_min_summary(report_root: Path) -> pd.DataFrame:
    path = report_root.parent / "v0_6a1_impulse_reclaim_validation" / "01_exact_gate_replay.csv"
    if not path.exists():
        path = Path("reports/v0_6a1_impulse_reclaim_validation/01_exact_gate_replay.csv")
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path)
    data = data[data["baseline"].eq("candidate")].copy()
    if data.empty:
        return data
    return pd.DataFrame(
        {
            "candidate": data["candidate"],
            "execution_granularity": "15m_bar",
            "fill_policy": "15m_ohlc_sl_first",
            "entry_policy": "",
            "execution_rule": "",
            "cost_single_side_bps": data["cost_single_side_bps"],
            "signals": data["signals"],
            "trades": data["trades"],
            "fill_rate": data["fill_rate"],
            "gross_expectancy": data["gross_expectancy"],
            "net_expectancy": data["net_expectancy"],
            "tp_first_rate": data["tp_rate"],
            "sl_first_rate": data["sl_rate"],
            "timeout_rate": data["timeout_rate"],
            "same_bar_ambiguous_rate": 0.0,
            "median_holding_minutes": np.nan,
            "p25_return": data["p25_net"],
            "p75_return": data["p75_net"],
            "max_loss": data["max_loss"],
        }
    )


def _tick_execution(
    signal_rows: pd.DataFrame,
    base_config: ExperimentConfig,
    v06a2_config: V06A2Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_root = (
        base_config.paths.data_root
        / "raw"
        / "bybit"
        / v06a2_config.execution_validation.public_trade_cache_dataset
    )
    if not cache_root.exists():
        return pd.DataFrame(), _coverage_summary(
            signal_rows, signal_rows.iloc[0:0], v06a2_config
        )
    day_map = available_public_trade_days(cache_root)
    covered = restrict_signals_to_public_trade_days(
        signal_rows,
        day_map,
        max_lookahead_hours=v06a2_config.execution_validation.max_lookahead_hours,
    )
    coverage = _coverage_summary(signal_rows, covered, v06a2_config)
    if covered.empty:
        return pd.DataFrame(), coverage
    trades = run_tick_execution_from_cache(covered, cache_root, base_config, _v02_adapter(v06a2_config))
    if not trades.empty:
        trades["execution_granularity"] = "trade_sequence"
    return trades, coverage


def _coverage_summary(
    signal_rows: pd.DataFrame,
    covered_signal_rows: pd.DataFrame,
    v06a2_config: V06A2Config,
) -> pd.DataFrame:
    rows = []
    for candidate in v06a2_config.candidates:
        col = _signal_col(candidate)
        total = int(signal_rows[col].fillna(False).sum()) if col in signal_rows.columns else 0
        covered = (
            int(covered_signal_rows[col].fillna(False).sum())
            if col in covered_signal_rows.columns
            else 0
        )
        rows.append(
            {
                "candidate": candidate.candidate,
                "signals": total,
                "tick_covered_signals": covered,
                "tick_coverage_rate": float(covered / total) if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _execution_comparison(
    report_root: Path,
    one_min_summary: pd.DataFrame,
    tick_summary: pd.DataFrame,
) -> pd.DataFrame:
    fifteen = _fifteen_min_summary(report_root)
    frames = [frame for frame in [fifteen, one_min_summary, tick_summary] if not frame.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    baseline = (
        out[out["execution_granularity"].eq("15m_bar")][
            ["candidate", "cost_single_side_bps", "net_expectancy"]
        ]
        .rename(columns={"net_expectancy": "net_expectancy_15m"})
        .drop_duplicates(["candidate", "cost_single_side_bps"])
    )
    out = out.merge(baseline, on=["candidate", "cost_single_side_bps"], how="left")
    out["expectancy_retention_vs_15m"] = out["net_expectancy"] / out["net_expectancy_15m"]
    return out


def _fill_stress(tick_summary: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    if tick_summary.empty:
        out = coverage.copy()
        out["status"] = "pending_public_trade_cache"
        return out
    cols = [
        "candidate",
        "fill_policy",
        "cost_single_side_bps",
        "signals",
        "trades",
        "fill_rate",
        "gross_expectancy",
        "net_expectancy",
        "tp_first_rate",
        "sl_first_rate",
        "timeout_rate",
    ]
    return tick_summary[[col for col in cols if col in tick_summary.columns]].copy()


def _reclaim_order_validity(
    one_min_summary: pd.DataFrame,
    tick_summary: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for granularity, summary in [("1m_bar", one_min_summary), ("trade_sequence", tick_summary)]:
        if summary.empty:
            continue
        data = summary.copy()
        data["valid_reclaim_rate"] = data["fill_rate"]
        data["missed_or_invalid_order_rate"] = 1.0 - data["fill_rate"]
        data["invalid_order_rate"] = np.nan
        data["execution_granularity"] = granularity
        frames.append(data)
    if frames:
        return pd.concat(frames, ignore_index=True, sort=False)
    out = coverage.copy()
    out["status"] = "pending_execution_cache"
    return out


def _load_v06a1_report(filename: str) -> pd.DataFrame:
    path = Path("reports/v0_6a1_impulse_reclaim_validation") / filename
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _month_capped_expectancy(v06a2_config: V06A2Config) -> pd.DataFrame:
    monthly = _load_v06a1_report("02_monthly_attribution.csv")
    if monthly.empty:
        return monthly
    sample = monthly[
        monthly["baseline"].eq("candidate")
        & monthly["cost_single_side_bps"].isin(v06a2_config.cost_bps)
    ].copy()
    rows = []
    for (candidate, cost), group in sample.groupby(["candidate", "cost_single_side_bps"], sort=False):
        total_net = pd.to_numeric(group["net_sum"], errors="coerce").sum()
        total_trades = pd.to_numeric(group["trades"], errors="coerce").sum()
        for cap in v06a2_config.execution_validation.month_cap_levels or [0.35, 0.40, 0.50]:
            cap_value = total_net * float(cap) if total_net > 0 else 0.0
            capped = []
            for value in pd.to_numeric(group["net_sum"], errors="coerce").fillna(0):
                if value > 0 and cap_value > 0:
                    capped.append(min(value, cap_value))
                else:
                    capped.append(value)
            capped_sum = float(np.sum(capped))
            rows.append(
                {
                    "candidate": candidate,
                    "cost_single_side_bps": cost,
                    "month_cap": cap,
                    "original_net_sum": float(total_net),
                    "capped_net_sum": capped_sum,
                    "original_expectancy": float(total_net / total_trades)
                    if total_trades
                    else np.nan,
                    "capped_expectancy": float(capped_sum / total_trades)
                    if total_trades
                    else np.nan,
                    "trades": int(total_trades),
                }
            )
    return pd.DataFrame(rows)


def _ex_top_month_summary() -> pd.DataFrame:
    exact = _load_v06a1_report("01_exact_gate_replay.csv")
    monthly = _load_v06a1_report("02_monthly_attribution.csv")
    leave_one = _load_v06a1_report("03_leave_one_month_out.csv")
    if exact.empty or monthly.empty or leave_one.empty:
        return pd.DataFrame()
    rows = []
    for candidate in exact["candidate"].dropna().unique():
        m10 = monthly[
            monthly["candidate"].eq(candidate)
            & monthly["baseline"].eq("candidate")
            & monthly["cost_single_side_bps"].eq(10)
        ].copy()
        if m10.empty:
            continue
        top_month = (
            m10.assign(abs_contribution=m10["net_contribution"].abs())
            .sort_values("abs_contribution", ascending=False)
            .iloc[0]
        )
        full10 = exact[
            exact["candidate"].eq(candidate)
            & exact["baseline"].eq("candidate")
            & exact["cost_single_side_bps"].eq(10)
        ]
        ex10 = leave_one[
            leave_one["candidate"].eq(candidate)
            & leave_one["baseline"].eq("candidate")
            & leave_one["cost_single_side_bps"].eq(10)
            & leave_one["excluded_month"].astype(str).eq(str(top_month["month"]))
        ]
        ex20 = leave_one[
            leave_one["candidate"].eq(candidate)
            & leave_one["baseline"].eq("candidate")
            & leave_one["cost_single_side_bps"].eq(20)
            & leave_one["excluded_month"].astype(str).eq(str(top_month["month"]))
        ]
        rows.append(
            {
                "candidate": candidate,
                "top_month": top_month["month"],
                "top_month_contribution": top_month["net_contribution"],
                "full_net10": full10["net_expectancy"].iloc[0] if not full10.empty else np.nan,
                "ex_top_month_net10": ex10["net_expectancy"].iloc[0] if not ex10.empty else np.nan,
                "ex_top_month_net20": ex20["net_expectancy"].iloc[0] if not ex20.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _regime_attribution(signal_rows: pd.DataFrame, v06a2_config: V06A2Config) -> pd.DataFrame:
    if signal_rows.empty:
        return pd.DataFrame()
    monthly = _load_v06a1_report("02_monthly_attribution.csv")
    rows = []
    focal = v06a2_config.execution_validation.focal_month
    numeric_cols = [
        "btc_ret_1h",
        "btc_ret_4h",
        "btc_volatility_4h",
        "volume_z_4h",
        "ret_4h",
        "symbol_volatility_percentile",
        "dynamic_all_rank",
        "turnover_rank_90d",
    ]
    for candidate in v06a2_config.candidates:
        data = signal_rows[signal_rows[_signal_col(candidate)].fillna(False)].copy()
        if data.empty:
            continue
        positive_months: set[str] = set()
        negative_months: set[str] = set()
        if not monthly.empty:
            cm = monthly[
                monthly["candidate"].eq(candidate.candidate)
                & monthly["baseline"].eq("candidate")
                & monthly["cost_single_side_bps"].eq(10)
            ]
            positive_months = set(cm[pd.to_numeric(cm["net_expectancy"], errors="coerce") > 0]["month"].astype(str))
            negative_months = set(cm[pd.to_numeric(cm["net_expectancy"], errors="coerce") <= 0]["month"].astype(str))
        groups = {
            f"only_{focal}": data[data["month"].astype(str).eq(focal)],
            f"positive_ex_{focal}": data[
                data["month"].astype(str).isin(positive_months)
                & ~data["month"].astype(str).eq(focal)
            ],
            "negative_months": data[data["month"].astype(str).isin(negative_months)],
            "all": data,
        }
        for bucket, group in groups.items():
            row: dict[str, object] = {
                "candidate": candidate.candidate,
                "month_group": bucket,
                "signals": int(len(group)),
                "unique_symbols": int(group["symbol"].nunique()) if not group.empty else 0,
            }
            for col in numeric_cols:
                row[f"avg_{col}"] = (
                    float(pd.to_numeric(group[col], errors="coerce").mean())
                    if col in group.columns and not group.empty
                    else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _universe_boundary_cost() -> pd.DataFrame:
    boundary = _load_v06a1_report("08_universe_boundary.csv")
    if boundary.empty:
        return boundary
    return boundary[boundary["baseline"].fillna("candidate").eq("candidate")] if "baseline" in boundary.columns else boundary


def _write_reclassification(
    report_root: Path,
    execution: pd.DataFrame,
    ex_top: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    lines = ["# v0.6A.2 Execution & Concentration Reclassification", ""]
    lines.append("Frozen candidates only: IR1, IR2, IR3. No parameter tuning in this report.")
    lines.append("")
    if execution.empty:
        lines.append("- Execution comparison is empty; signal rows or cached execution data are missing.")
    else:
        primary = execution[
            execution["cost_single_side_bps"].eq(10)
            & execution["execution_granularity"].isin(["15m_bar", "1m_bar", "trade_sequence"])
        ].copy()
        for row in primary.sort_values(
            ["candidate", "execution_granularity", "fill_policy"], ascending=True
        ).itertuples(index=False):
            net = getattr(row, "net_expectancy", np.nan)
            retention = getattr(row, "expectancy_retention_vs_15m", np.nan)
            lines.append(
                f"- {row.candidate} {row.execution_granularity}/{row.fill_policy}: "
                f"trades={int(row.trades)}, net10={net:.4%}, retention={retention:.2f}"
            )
    if not ex_top.empty:
        lines.append("")
        lines.append("## Top-Month Risk")
        for row in ex_top.itertuples(index=False):
            lines.append(
                f"- {row.candidate}: top_month={row.top_month}, "
                f"contribution={row.top_month_contribution:.2%}, "
                f"ex_top_net10={row.ex_top_month_net10:.4%}"
            )
    if not coverage.empty:
        lines.append("")
        lines.append("## Tick Coverage")
        for row in coverage.itertuples(index=False):
            lines.append(
                f"- {row.candidate}: covered={row.tick_covered_signals}/{row.signals} "
                f"({row.tick_coverage_rate:.2%})"
            )
    lines.append("")
    lines.append(
        "Promotion rule: require tick net10 >= +0.30%, tick net20 >= +0.10%, "
        "retention >= 70%, ex-top-month net10 > 0, and concentration checks passing."
    )
    (report_root / "10_candidate_reclassification.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_v06a2_execution_concentration(
    feature_path: Path,
    instruments: pd.DataFrame,
    base_config: ExperimentConfig,
    v06a2_config: V06A2Config,
    report_root: Path = REPORT_ROOT,
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    signal_rows = build_v06a2_signal_rows(feature_path, instruments, base_config, v06a2_config)

    minute_bars = _load_1m_bars(base_config, v06a2_config)
    one_min_trades = _one_min_execution(signal_rows, minute_bars, base_config, v06a2_config)
    one_min_summary = _summarize_1m(one_min_trades, signal_rows)

    tick_trades, coverage = _tick_execution(signal_rows, base_config, v06a2_config)
    tick_summary = summarize_sequence_trades(tick_trades)
    execution = _execution_comparison(report_root, one_min_summary, tick_summary)
    fill_stress = _fill_stress(tick_summary, coverage)
    validity = _reclaim_order_validity(one_min_summary, tick_summary, coverage)
    month_cap = _month_capped_expectancy(v06a2_config)
    ex_top = _ex_top_month_summary()
    regime = _regime_attribution(signal_rows, v06a2_config)
    boundary = _universe_boundary_cost()

    outputs = {
        "tick_execution_comparison": report_root / "01_tick_execution_comparison.csv",
        "fill_stress": report_root / "02_fill_stress.csv",
        "reclaim_order_validity": report_root / "03_reclaim_order_validity.csv",
        "monthly_attribution": report_root / "04_monthly_attribution.csv",
        "leave_one_month_out": report_root / "05_leave_one_month_out.csv",
        "ex_top_month_summary": report_root / "06_ex_top_month_summary.csv",
        "month_capped_expectancy": report_root / "07_month_capped_expectancy.csv",
        "regime_attribution": report_root / "08_2025_08_regime_attribution.csv",
        "universe_boundary_cost": report_root / "09_universe_boundary_cost.csv",
        "candidate_reclassification": report_root / "10_candidate_reclassification.md",
        "signal_rows": SIGNAL_ROWS_OUTPUT,
        "one_min_trades": report_root / "debug_1m_trades.csv",
        "tick_trades": report_root / "debug_tick_trades.csv",
        "tick_coverage": report_root / "debug_tick_coverage.csv",
    }

    execution.to_csv(outputs["tick_execution_comparison"], index=False)
    fill_stress.to_csv(outputs["fill_stress"], index=False)
    validity.to_csv(outputs["reclaim_order_validity"], index=False)
    _load_v06a1_report("02_monthly_attribution.csv").to_csv(
        outputs["monthly_attribution"], index=False
    )
    _load_v06a1_report("03_leave_one_month_out.csv").to_csv(
        outputs["leave_one_month_out"], index=False
    )
    ex_top.to_csv(outputs["ex_top_month_summary"], index=False)
    month_cap.to_csv(outputs["month_capped_expectancy"], index=False)
    regime.to_csv(outputs["regime_attribution"], index=False)
    boundary.to_csv(outputs["universe_boundary_cost"], index=False)
    one_min_trades.to_csv(outputs["one_min_trades"], index=False)
    tick_trades.to_csv(outputs["tick_trades"], index=False)
    coverage.to_csv(outputs["tick_coverage"], index=False)
    _write_reclassification(report_root, execution, ex_top, coverage)
    return outputs
