from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.backtest.simulator import _funding_cost
from pressure_graph.config import ExperimentConfig
from pressure_graph.config.v05 import V05Config
from pressure_graph.io import ensure_dir, write_parquet
from pressure_graph.reports.v03 import (
    C2_SIGNAL_COL,
    V03_TURNOVER_COLUMNS,
    _concat_or_empty,
    _read_existing_columns,
)
from pressure_graph.reports.v04 import (
    RANK30_COL,
    RANK90_COL,
    _rank_table_with_lookback,
    _read_symbol_v04,
)


REPORT_ROOT = Path("reports/v0_5_paper_live")
PAPER_DATA_ROOT = Path("data/paper/v0_5")
FILL_MODEL_BUFFER_BPS = {
    "optimistic_0bp": 0.0,
    "normal_2bp": 2.0,
    "conservative_5bp": 5.0,
}
SIGNAL_LOG_COLUMNS = [
    "signal_id",
    "created_at_utc",
    "exchange",
    "symbol",
    "bar_open_time",
    "bar_close_time",
    "feature_time",
    "universe_mode",
    "turnover_rank_30d",
    "turnover_rank_90d",
    "core_liquidity",
    "transient_hot",
    "btc_state",
    "btc_ret_1h",
    "btc_ret_4h",
    "btc_volatility_4h",
    "c2_signal",
    "oi_delta_1h_percentile",
    "oi_delta_4h_percentile",
    "funding_percentile",
    "funding_z",
    "ret_1h",
    "ret_4h",
    "volume_z_1h",
    "volume_z_4h",
    "signal_close",
    "entry_limit",
    "entry_valid_until",
    "status",
    "skip_reason",
]
TRADE_LOG_COLUMNS = [
    "trade_id",
    "signal_id",
    "exchange",
    "symbol",
    "armed_time",
    "fill_time_normal_2bp",
    "fill_price_normal_2bp",
    "fill_time_conservative_5bp",
    "fill_price_conservative_5bp",
    "exit_time",
    "exit_reason",
    "tp_price",
    "sl_price",
    "max_hold_time",
    "gross_return",
    "net_return_5bp",
    "net_return_10bp",
    "net_return_20bp",
    "holding_minutes",
    "mae",
    "mfe",
    "tp_first",
    "sl_first",
    "timeout",
    "same_bar_ambiguity",
    "btc_state_at_entry",
    "btc_state_at_exit",
    "concurrent_positions_at_entry",
]


def _to_timestamp(value: object) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC") if pd.Timestamp(value).tzinfo else pd.Timestamp(value, tz="UTC")


def _safe_float(value: object, default: float = np.nan) -> float:
    out = pd.to_numeric(value, errors="coerce")
    return float(out) if pd.notna(out) else default


def _bool(value: object) -> bool:
    return bool(value) if pd.notna(value) else False


def _signal_id(row: pd.Series) -> str:
    bar_time = pd.Timestamp(row["bar_open_time"]).strftime("%Y%m%dT%H%M%SZ")
    return f"{row['exchange']}:{row['symbol']}:{bar_time}"


def _fill_model_buffer_bps(name: str) -> float:
    if name in FILL_MODEL_BUFFER_BPS:
        return FILL_MODEL_BUFFER_BPS[name]
    if name.endswith("bp"):
        token = name.rsplit("_", maxsplit=1)[-1].removesuffix("bp")
        return float(token)
    raise KeyError(f"unknown v0.5 fill model: {name}")


def _find_pullback_fill(
    group: pd.DataFrame,
    signal_idx: int,
    entry_limit: float,
    valid_bars: int,
    touch_buffer_bps: float,
) -> tuple[int, pd.Timestamp, float] | None:
    trigger = entry_limit * (1.0 - touch_buffer_bps / 10_000.0)
    end_idx = min(signal_idx + valid_bars, len(group) - 1)
    for idx in range(signal_idx + 1, end_idx + 1):
        if _safe_float(group.iloc[idx]["low"]) <= trigger:
            return idx, pd.Timestamp(group.iloc[idx]["bar_open_time"]), entry_limit
    return None


def _exit_from_fill(
    group: pd.DataFrame,
    fill_idx: int,
    entry_price: float,
    tp_pct: float,
    sl_pct: float,
    max_hold_bars: int,
) -> dict[str, object]:
    tp_price = entry_price * (1.0 + tp_pct)
    sl_price = entry_price * (1.0 - sl_pct)
    target_exit_idx = fill_idx + max_hold_bars - 1
    max_exit_idx = min(target_exit_idx, len(group) - 1)
    exit_idx = max_exit_idx
    exit_price = _safe_float(group.iloc[max_exit_idx]["close"])
    exit_reason = "open" if target_exit_idx >= len(group) else "max_hold"
    same_bar_ambiguity = False

    for idx in range(fill_idx, max_exit_idx + 1):
        row = group.iloc[idx]
        high_hit = _safe_float(row["high"]) >= tp_price
        low_hit = _safe_float(row["low"]) <= sl_price
        if high_hit and low_hit:
            exit_idx = idx
            exit_price = sl_price
            exit_reason = "sl_ambiguous"
            same_bar_ambiguity = True
            break
        if low_hit:
            exit_idx = idx
            exit_price = sl_price
            exit_reason = "sl"
            break
        if high_hit:
            exit_idx = idx
            exit_price = tp_price
            exit_reason = "tp"
            break

    holding = group.iloc[fill_idx : exit_idx + 1]
    mae = _safe_float(holding["low"].min()) / entry_price - 1.0
    mfe = _safe_float(holding["high"].max()) / entry_price - 1.0
    entry_time = pd.Timestamp(group.iloc[fill_idx]["bar_open_time"])
    exit_time = pd.Timestamp(group.iloc[exit_idx]["bar_close_time"])
    gross = exit_price / entry_price - 1.0
    return {
        "exit_idx": exit_idx,
        "exit_time": exit_time,
        "exit_reason": exit_reason,
        "exit_price": exit_price,
        "tp_price": tp_price,
        "sl_price": sl_price,
        "max_hold_time": pd.Timestamp(group.iloc[max_exit_idx]["bar_close_time"]),
        "gross_return": gross,
        "holding_minutes": (exit_time - entry_time).total_seconds() / 60.0,
        "mae": mae,
        "mfe": mfe,
        "tp_first": exit_reason.startswith("tp"),
        "sl_first": exit_reason.startswith("sl"),
        "timeout": exit_reason in {"max_hold", "open"},
        "same_bar_ambiguity": same_bar_ambiguity,
        "funding_cost": _funding_cost(group, entry_time, exit_time),
    }


def _signal_skip_reason(row: pd.Series, config: V05Config) -> str | None:
    btc_state = str(row.get("btc_market_state", "unknown"))
    if config.regime.btc_gate.enabled and btc_state != config.regime.btc_gate.required_state:
        return f"btc_not_up:{btc_state}"
    rank30 = _safe_float(row.get(RANK30_COL))
    rank90 = _safe_float(row.get(RANK90_COL))
    if pd.isna(rank30):
        return "rank30_missing"
    if rank30 > config.universe.turnover_rank_30d_max:
        return "outside_dynamic_top30"
    if pd.isna(rank90):
        return "rank90_missing"
    if (
        config.universe.transient_hot.enabled
        and rank30 <= config.universe.transient_hot.turnover_rank_30d_max
        and rank90 > config.universe.transient_hot.turnover_rank_90d_min
    ):
        return "transient_hot"
    if not _bool(row.get("core_liquidity")):
        return "non_core_liquidity"
    return None


def _base_signal_row(row: pd.Series, config: V05Config, created_at: pd.Timestamp) -> dict[str, object]:
    signal_close = _safe_float(row.get("close"))
    entry_limit = signal_close * (1.0 - config.entry.pullback_pct)
    feature_time = pd.Timestamp(row.get("feature_time"))
    return {
        "signal_id": _signal_id(row),
        "created_at_utc": created_at,
        "exchange": str(row.get("exchange")),
        "symbol": str(row.get("symbol")),
        "bar_open_time": pd.Timestamp(row.get("bar_open_time")),
        "bar_close_time": pd.Timestamp(row.get("bar_close_time")),
        "feature_time": feature_time,
        "universe_mode": config.universe.mode,
        "turnover_rank_30d": _safe_float(row.get(RANK30_COL)),
        "turnover_rank_90d": _safe_float(row.get(RANK90_COL)),
        "core_liquidity": _bool(row.get("core_liquidity")),
        "transient_hot": _bool(row.get("transient_hot")),
        "btc_state": str(row.get("btc_market_state", "unknown")),
        "btc_ret_1h": _safe_float(row.get("btc_ret_1h")),
        "btc_ret_4h": _safe_float(row.get("btc_ret_4h")),
        "btc_volatility_4h": _safe_float(row.get("btc_volatility_4h")),
        "c2_signal": True,
        "oi_delta_1h_percentile": _safe_float(row.get("oi_value_delta_1h_percentile")),
        "oi_delta_4h_percentile": _safe_float(row.get("oi_value_delta_4h_percentile")),
        "funding_percentile": _safe_float(row.get("funding_percentile")),
        "funding_z": _safe_float(row.get("funding_z")),
        "ret_1h": _safe_float(row.get("ret_1h")),
        "ret_4h": _safe_float(row.get("ret_4h")),
        "volume_z_1h": _safe_float(row.get("volume_z_1h")),
        "volume_z_4h": _safe_float(row.get("volume_z_4h")),
        "signal_close": signal_close,
        "entry_limit": entry_limit,
        "entry_valid_until": feature_time + pd.Timedelta(minutes=15 * config.entry.valid_bars),
        "status": "detected",
        "skip_reason": "",
    }


def _simulate_signal_trade(
    group: pd.DataFrame,
    signal_idx: int,
    signal: dict[str, object],
    config: V05Config,
) -> dict[str, object] | None:
    entry_limit = float(signal["entry_limit"])
    fills: dict[str, tuple[int, pd.Timestamp, float] | None] = {}
    for model_name in config.entry.fill_models:
        fills[model_name] = _find_pullback_fill(
            group,
            signal_idx,
            entry_limit,
            config.entry.valid_bars,
            _fill_model_buffer_bps(model_name),
        )
    normal = fills.get("normal_2bp")
    if normal is None:
        return None

    fill_idx, fill_time, fill_price = normal
    exit_data = _exit_from_fill(
        group,
        fill_idx,
        fill_price,
        config.exit.tp,
        config.exit.sl,
        config.exit.max_hold_bars,
    )
    conservative = fills.get("conservative_5bp")
    conservative_exit = None
    if conservative is not None:
        conservative_exit = _exit_from_fill(
            group,
            conservative[0],
            conservative[2],
            config.exit.tp,
            config.exit.sl,
            config.exit.max_hold_bars,
        )

    gross = float(exit_data["gross_return"])
    trade = {
        "trade_id": f"{signal['signal_id']}:normal_2bp",
        "signal_id": signal["signal_id"],
        "exchange": signal["exchange"],
        "symbol": signal["symbol"],
        "armed_time": signal["feature_time"],
        "fill_time_normal_2bp": fill_time,
        "fill_price_normal_2bp": fill_price,
        "fill_time_conservative_5bp": conservative[1] if conservative else pd.NaT,
        "fill_price_conservative_5bp": conservative[2] if conservative else np.nan,
        "exit_time": exit_data["exit_time"],
        "exit_reason": exit_data["exit_reason"],
        "tp_price": exit_data["tp_price"],
        "sl_price": exit_data["sl_price"],
        "max_hold_time": exit_data["max_hold_time"],
        "gross_return": gross,
        "holding_minutes": exit_data["holding_minutes"],
        "mae": exit_data["mae"],
        "mfe": exit_data["mfe"],
        "tp_first": exit_data["tp_first"],
        "sl_first": exit_data["sl_first"],
        "timeout": exit_data["timeout"],
        "same_bar_ambiguity": exit_data["same_bar_ambiguity"],
        "funding_cost": exit_data["funding_cost"],
        "btc_state_at_entry": signal["btc_state"],
        "btc_state_at_exit": str(group.iloc[int(exit_data["exit_idx"])].get("btc_market_state", "unknown")),
        "concurrent_positions_at_entry": 0,
        "turnover_30d": _safe_float(group.iloc[signal_idx].get("trailing_30d_turnover")),
        "transient_hot": signal["transient_hot"],
    }
    for cost in config.costs.single_side_bps:
        suffix = f"{int(cost)}bp" if float(cost).is_integer() else f"{cost}bp"
        trade[f"net_return_{suffix}"] = gross - 2.0 * float(cost) / 10_000.0
        trade[f"net_return_{suffix}_funding"] = (
            trade[f"net_return_{suffix}"] - float(exit_data["funding_cost"])
        )
        if conservative_exit is not None:
            c_gross = float(conservative_exit["gross_return"])
            trade[f"conservative_net_return_{suffix}"] = c_gross - 2.0 * float(cost) / 10_000.0
    if conservative_exit is not None:
        trade["conservative_gross_return"] = conservative_exit["gross_return"]
        trade["conservative_exit_reason"] = conservative_exit["exit_reason"]
    return trade


def _apply_primary_portfolio(
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    config: V05Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if signals.empty or trades.empty:
        return signals, trades
    signals = signals.copy().set_index("signal_id", drop=False)
    data = trades.copy()
    data["_sort_time"] = pd.to_datetime(data["fill_time_normal_2bp"], utc=True, errors="coerce")
    data["_sort_turnover"] = pd.to_numeric(data.get("turnover_30d"), errors="coerce").fillna(-np.inf)
    data["_sort_transient"] = data.get("transient_hot", False).fillna(False).astype(bool)
    data["_sort_armed"] = pd.to_datetime(data["armed_time"], utc=True, errors="coerce")
    data = data.sort_values(
        ["_sort_time", "_sort_turnover", "_sort_transient", "_sort_armed", "symbol"],
        ascending=[True, False, True, True, True],
    )

    accepted = []
    active: list[tuple[pd.Timestamp, str]] = []
    for row in data.itertuples(index=False):
        fill_time = pd.Timestamp(row.fill_time_normal_2bp)
        active = [(exit_time, symbol) for exit_time, symbol in active if exit_time > fill_time]
        active_symbols = {symbol for _, symbol in active}
        signal_id = str(row.signal_id)
        if config.portfolio.one_position_per_symbol and row.symbol in active_symbols:
            signals.loc[signal_id, "status"] = "skipped"
            signals.loc[signal_id, "skip_reason"] = "symbol_position_open"
            continue
        if len(active) >= config.portfolio.primary_max_positions:
            signals.loc[signal_id, "status"] = "skipped"
            signals.loc[signal_id, "skip_reason"] = "max_positions"
            continue
        record = row._asdict()
        record["concurrent_positions_at_entry"] = len(active)
        record.pop("_sort_time", None)
        record.pop("_sort_turnover", None)
        record.pop("_sort_transient", None)
        record.pop("_sort_armed", None)
        accepted.append(record)
        exit_time = pd.Timestamp(row.exit_time)
        active.append((exit_time, str(row.symbol)))
        signals.loc[signal_id, "status"] = "open" if str(row.exit_reason) == "open" else "exited"
        signals.loc[signal_id, "skip_reason"] = ""
    out_trades = pd.DataFrame(accepted)
    return signals.reset_index(drop=True), out_trades


def _portfolio_shadow_summary(trades: pd.DataFrame, config: V05Config) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    max_positions = [*config.portfolio.shadow_max_positions, config.portfolio.primary_max_positions]
    for limit in sorted(set(max_positions)):
        shadow_config = V05Config(
            experiment=config.experiment,
            exchange=config.exchange,
            universe=config.universe,
            regime=config.regime,
            signal=config.signal,
            entry=config.entry,
            exit=config.exit,
            costs=config.costs,
            portfolio=type(config.portfolio)(
                primary_max_positions=limit,
                shadow_max_positions=config.portfolio.shadow_max_positions,
                one_position_per_symbol=config.portfolio.one_position_per_symbol,
                pyramiding=config.portfolio.pyramiding,
                sizing=config.portfolio.sizing,
                ranking=config.portfolio.ranking,
            ),
            logging=config.logging,
            stops=config.stops,
        )
        _, accepted = _apply_primary_portfolio(
            pd.DataFrame({"signal_id": trades["signal_id"], "status": "filled", "skip_reason": ""}),
            trades,
            shadow_config,
        )
        completed = accepted[accepted["exit_reason"].ne("open")] if not accepted.empty else accepted
        rows.append(
            {
                "max_positions": limit,
                "trades": len(completed),
                "net_5bp_avg": _safe_float(completed.get("net_return_5bp", pd.Series(dtype=float)).mean()),
                "net_10bp_avg": _safe_float(completed.get("net_return_10bp", pd.Series(dtype=float)).mean()),
                "net_20bp_avg": _safe_float(completed.get("net_return_20bp", pd.Series(dtype=float)).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_v05_paper_ledger(
    prepared: pd.DataFrame,
    config: V05Config,
    signal_start_time: pd.Timestamp | None = None,
    created_at: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if prepared.empty:
        empty_signals = pd.DataFrame(columns=SIGNAL_LOG_COLUMNS)
        empty_trades = pd.DataFrame(columns=TRADE_LOG_COLUMNS)
        return empty_signals, empty_trades, pd.DataFrame()
    created_at = created_at or pd.Timestamp.now(tz="UTC")
    data = prepared.sort_values(["exchange", "symbol", "bar_open_time"]).copy()
    for col in ["bar_open_time", "bar_close_time", "feature_time"]:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], utc=True, errors="coerce")
    if signal_start_time is not None:
        signal_start_time = pd.Timestamp(signal_start_time)
        if signal_start_time.tzinfo is None:
            signal_start_time = signal_start_time.tz_localize("UTC")
        else:
            signal_start_time = signal_start_time.tz_convert("UTC")

    signal_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    for _, group in data.groupby(["exchange", "symbol"], sort=False, observed=True):
        group = group.reset_index(drop=True)
        events = group[C2_SIGNAL_COL].fillna(False).astype(bool) if C2_SIGNAL_COL in group else []
        for signal_idx, is_signal in enumerate(events):
            if not is_signal:
                continue
            row = group.iloc[signal_idx]
            feature_time = pd.Timestamp(row["feature_time"])
            if signal_start_time is not None and feature_time < signal_start_time:
                continue
            signal = _base_signal_row(row, config, created_at)
            skip_reason = _signal_skip_reason(row, config)
            if skip_reason:
                signal["status"] = "skipped"
                signal["skip_reason"] = skip_reason
                signal_rows.append(signal)
                continue
            signal["status"] = "armed"
            trade = _simulate_signal_trade(group, signal_idx, signal, config)
            if trade is None:
                signal["status"] = "expired_unfilled"
                signal["skip_reason"] = ""
            else:
                signal["status"] = "filled"
                trade_rows.append(trade)
            signal_rows.append(signal)

    signals = pd.DataFrame(signal_rows)
    trades = pd.DataFrame(trade_rows)
    if signals.empty:
        signals = pd.DataFrame(columns=SIGNAL_LOG_COLUMNS)
    if trades.empty:
        trades = pd.DataFrame(columns=TRADE_LOG_COLUMNS)
    signals, accepted_trades = _apply_primary_portfolio(signals, trades, config)
    shadow = _portfolio_shadow_summary(trades, config)
    return signals, accepted_trades, shadow


def _daily_summary(signals: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    signal_dates = (
        pd.to_datetime(signals.get("feature_time"), utc=True, errors="coerce").dt.date
        if not signals.empty
        else pd.Series(dtype=object)
    )
    trade_dates = (
        pd.to_datetime(trades.get("fill_time_normal_2bp"), utc=True, errors="coerce").dt.date
        if not trades.empty
        else pd.Series(dtype=object)
    )
    dates = sorted(set(signal_dates.dropna()).union(set(trade_dates.dropna())))
    rows = []
    for date in dates:
        day_signals = signals[signal_dates.eq(date)] if not signals.empty else signals
        day_trades = trades[trade_dates.eq(date)] if not trades.empty else trades
        completed = day_trades[day_trades["exit_reason"].ne("open")] if not day_trades.empty else day_trades
        rows.append(
            {
                "date": date,
                "signals": len(day_signals),
                "armed": int(day_signals["status"].isin(["armed", "filled", "exited", "open"]).sum())
                if not day_signals.empty
                else 0,
                "filled": int(day_signals["status"].isin(["filled", "exited", "open"]).sum())
                if not day_signals.empty
                else 0,
                "expired": int(day_signals["status"].eq("expired_unfilled").sum())
                if not day_signals.empty
                else 0,
                "skipped": int(day_signals["status"].eq("skipped").sum()) if not day_signals.empty else 0,
                "trades": len(completed),
                "gross_avg": _safe_float(completed.get("gross_return", pd.Series(dtype=float)).mean()),
                "net_5bp_avg": _safe_float(completed.get("net_return_5bp", pd.Series(dtype=float)).mean()),
                "net_10bp_avg": _safe_float(completed.get("net_return_10bp", pd.Series(dtype=float)).mean()),
                "net_20bp_avg": _safe_float(completed.get("net_return_20bp", pd.Series(dtype=float)).mean()),
                "tp_rate": _safe_float(completed.get("tp_first", pd.Series(dtype=bool)).mean()),
                "sl_rate": _safe_float(completed.get("sl_first", pd.Series(dtype=bool)).mean()),
                "timeout_rate": _safe_float(completed.get("timeout", pd.Series(dtype=bool)).mean()),
                "avg_holding_minutes": _safe_float(
                    completed.get("holding_minutes", pd.Series(dtype=float)).mean()
                ),
                "max_concurrent_positions": _safe_float(
                    day_trades.get("concurrent_positions_at_entry", pd.Series(dtype=float)).max()
                ),
            }
        )
    return pd.DataFrame(rows)


def _latest_btc_state(prepared: pd.DataFrame) -> str:
    if prepared.empty or "BTCUSDT" not in set(prepared["symbol"].astype(str)):
        return "unknown"
    btc = prepared[prepared["symbol"].astype(str).eq("BTCUSDT")].sort_values("feature_time")
    return str(btc.iloc[-1].get("btc_market_state", "unknown")) if not btc.empty else "unknown"


def _write_status_reports(
    report_root: Path,
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    shadow: pd.DataFrame,
    prepared: pd.DataFrame,
    config: V05Config,
) -> None:
    latest_time = (
        pd.to_datetime(prepared["feature_time"], utc=True, errors="coerce").max()
        if not prepared.empty and "feature_time" in prepared
        else pd.NaT
    )
    btc_state = _latest_btc_state(prepared)
    completed = trades[trades["exit_reason"].ne("open")] if not trades.empty else trades
    net10 = _safe_float(completed.get("net_return_10bp", pd.Series(dtype=float)).mean())
    net5 = _safe_float(completed.get("net_return_5bp", pd.Series(dtype=float)).mean())
    net20 = _safe_float(completed.get("net_return_20bp", pd.Series(dtype=float)).mean())
    stale_cutoff = pd.Timedelta(minutes=15 * config.stops.stale_data_bars)
    data_stale = (
        pd.notna(latest_time)
        and pd.Timestamp.now(tz="UTC") - pd.Timestamp(latest_time) > stale_cutoff
    )
    current_lines = [
        "# v0.5 Paper-Live Current Status",
        "",
        f"- candidate: {config.experiment.candidate}",
        f"- latest_feature_time: {latest_time}",
        f"- btc_state: {btc_state}",
        f"- data_stale: {bool(data_stale)}",
        f"- primary_max_positions: {config.portfolio.primary_max_positions}",
        f"- total_signals_logged: {len(signals)}",
        f"- completed_trades: {len(completed)}",
        f"- open_trades: {int(trades['exit_reason'].eq('open').sum()) if not trades.empty else 0}",
        f"- net_5bp_avg: {net5:.4%}" if pd.notna(net5) else "- net_5bp_avg: n/a",
        f"- net_10bp_avg: {net10:.4%}" if pd.notna(net10) else "- net_10bp_avg: n/a",
        f"- net_20bp_avg: {net20:.4%}" if pd.notna(net20) else "- net_20bp_avg: n/a",
    ]
    (report_root / "current_status.md").write_text("\n".join(current_lines), encoding="utf-8")

    candidate_lines = [
        "# C2_G3_BTC_UP_SHORT_SQUEEZE_PULLBACK",
        "",
        "Status: B- conditional paper-live overlay candidate.",
        "",
        "Frozen definition:",
        "- Base path: C2 Short Squeeze E4 pullback 0.5% / swing.",
        "- Graph gate: BTC_up only.",
        "- Liquidity veto: transient_hot excluded; core monthly dynamic Top30/90d quality required.",
        "- Exit: TP 5%, SL 3%, max hold 12h.",
        "- No C2 parameter tuning is allowed in v0.5.",
        "",
        "Primary readout:",
        f"- signals={len(signals)}, completed_trades={len(completed)}",
        f"- net5={net5:.4%}" if pd.notna(net5) else "- net5=n/a",
        f"- net10={net10:.4%}" if pd.notna(net10) else "- net10=n/a",
        f"- net20={net20:.4%}" if pd.notna(net20) else "- net20=n/a",
    ]
    if not shadow.empty:
        candidate_lines.extend(["", "Portfolio shadows:"])
        for row in shadow.itertuples(index=False):
            candidate_lines.append(
                f"- max_positions={row.max_positions}: trades={row.trades}, "
                f"net10={row.net_10bp_avg:.4%}"
            )
    (report_root / "candidate_status.md").write_text("\n".join(candidate_lines), encoding="utf-8")


def _load_v05_prepared_features(
    feature_path: Path,
    instruments: pd.DataFrame,
    base_config: ExperimentConfig,
    config: V05Config,
    days: int | None = None,
) -> pd.DataFrame:
    turnover = _read_existing_columns(feature_path, V03_TURNOVER_COLUMNS)
    rank30 = _rank_table_with_lookback(
        turnover,
        instruments,
        base_config,
        30,
        RANK30_COL,
        "trailing_30d_turnover",
    )
    rank90 = _rank_table_with_lookback(
        turnover,
        instruments,
        base_config,
        90,
        RANK90_COL,
        "trailing_90d_turnover",
    )
    max_time = pd.to_datetime(turnover["bar_open_time"], utc=True, errors="coerce").max()
    cutoff = max_time - pd.Timedelta(days=days) if days is not None and pd.notna(max_time) else None
    del turnover
    top_symbols = sorted(
        rank30[
            pd.to_numeric(rank30[RANK30_COL], errors="coerce") <= config.universe.turnover_rank_30d_max
        ]["symbol"]
        .dropna()
        .astype(str)
        .unique()
    )
    frames = []
    for symbol in top_symbols:
        data = _read_symbol_v04(feature_path, rank30, rank90, symbol, base_config)
        if data.empty:
            continue
        if cutoff is not None:
            data = data[pd.to_datetime(data["bar_open_time"], utc=True, errors="coerce") >= cutoff].copy()
        frames.append(data)
    return _concat_or_empty(frames)


def write_v05_paper_live(
    feature_path: Path,
    instruments: pd.DataFrame,
    base_config: ExperimentConfig,
    config: V05Config,
    report_root: Path = REPORT_ROOT,
    paper_data_root: Path = PAPER_DATA_ROOT,
    days: int | None = None,
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    paper_data_root = ensure_dir(paper_data_root)
    prepared = _load_v05_prepared_features(feature_path, instruments, base_config, config, days)
    signal_start_time = None
    if days is not None and not prepared.empty:
        latest = pd.to_datetime(prepared["feature_time"], utc=True, errors="coerce").max()
        signal_start_time = latest - pd.Timedelta(days=days)

    signals, trades, shadow = build_v05_paper_ledger(prepared, config, signal_start_time)
    daily = _daily_summary(signals, trades)
    skipped = signals[signals["status"].eq("skipped")].copy() if not signals.empty else signals.copy()

    outputs = {
        "paper_signals": report_root / "paper_signals.parquet",
        "paper_trades": report_root / "paper_trades.parquet",
        "daily_summary": report_root / "daily_summary.csv",
        "skipped_signals": report_root / "skipped_signals.csv",
        "portfolio_shadow_summary": report_root / "portfolio_shadow_summary.csv",
        "current_status": report_root / "current_status.md",
        "candidate_status": report_root / "candidate_status.md",
        "paper_signals_data": paper_data_root / "paper_signals.parquet",
        "paper_trades_data": paper_data_root / "paper_trades.parquet",
    }
    write_parquet(signals, outputs["paper_signals"])
    write_parquet(trades, outputs["paper_trades"])
    write_parquet(signals, outputs["paper_signals_data"])
    write_parquet(trades, outputs["paper_trades_data"])
    daily.to_csv(outputs["daily_summary"], index=False)
    skipped.to_csv(outputs["skipped_signals"], index=False)
    shadow.to_csv(outputs["portfolio_shadow_summary"], index=False)
    _write_status_reports(report_root, signals, trades, shadow, prepared, config)
    return outputs
