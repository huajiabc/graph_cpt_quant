from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.clients import BybitClient
from pressure_graph.config import load_config
from pressure_graph.config.v05 import load_v05_config
from pressure_graph.features import build_feature_table
from pressure_graph.io import ensure_dir, raw_path, read_parquet, write_parquet
from pressure_graph.paper_live.v05 import (
    PAPER_DATA_ROOT,
    REPORT_ROOT,
    _daily_summary,
    _write_status_reports,
    build_v05_paper_ledger,
)
from pressure_graph.reports.v03 import (
    C2_SIGNAL_COL,
    V03_TURNOVER_COLUMNS,
    _add_v03_report_columns,
    _read_existing_columns,
)
from pressure_graph.reports.v04 import (
    RANK30_COL,
    RANK90_COL,
    _rank_table_with_lookback,
)


def _floor_15m(ts: pd.Timestamp | None = None) -> pd.Timestamp:
    ts = ts or pd.Timestamp.now(tz="UTC")
    # Use only fully closed 15m bars. Bybit may return the currently forming
    # candle when queried at the current floor timestamp.
    return ts.floor("15min") - pd.Timedelta(minutes=15)


def _read_optional(path: Path) -> pd.DataFrame:
    return read_parquet(path) if path.exists() else pd.DataFrame()


def _concat_symbol_dir(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frames = [pd.read_parquet(file) for file in sorted(path.glob("*.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _month_start(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return pd.to_datetime({"year": ts.dt.year, "month": ts.dt.month, "day": 1}, utc=True)


def _append_pruned(
    existing: pd.DataFrame,
    new: pd.DataFrame,
    key_cols: list[str],
    time_col: str,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    frames = [frame for frame in [existing, new] if not frame.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out[time_col] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    out = out[out[time_col] >= cutoff].copy()
    return out.sort_values(key_cols).drop_duplicates(key_cols, keep="last")


def _rank_context(
    feature_path: Path,
    instruments: pd.DataFrame,
    base_config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return rank30, rank90


def _live_symbols(rank30: pd.DataFrame, now: pd.Timestamp, max_symbols: int | None = None) -> list[str]:
    month_start = pd.Timestamp(year=now.year, month=now.month, day=1, tz="UTC")
    ranked = rank30[rank30["month_start"].eq(month_start)].copy()
    if ranked.empty:
        ranked = rank30[rank30["month_start"].eq(rank30["month_start"].max())].copy()
    ranked = ranked[pd.to_numeric(ranked[RANK30_COL], errors="coerce") <= 30].sort_values(RANK30_COL)
    symbols = ranked["symbol"].dropna().astype(str).tolist()
    if "BTCUSDT" not in symbols:
        symbols = ["BTCUSDT", *symbols]
    symbols = list(dict.fromkeys(symbols))
    return symbols[:max_symbols] if max_symbols else symbols


def _start_for_symbol(path: Path, time_col: str, end: pd.Timestamp, history_days: int) -> pd.Timestamp:
    if not path.exists():
        return end - pd.Timedelta(days=history_days)
    data = pd.read_parquet(path, columns=[time_col])
    if data.empty:
        return end - pd.Timedelta(days=history_days)
    last = pd.to_datetime(data[time_col], utc=True, errors="coerce").max()
    if pd.isna(last):
        return end - pd.Timedelta(days=history_days)
    return max(last - pd.Timedelta(hours=1), end - pd.Timedelta(days=history_days))


def _refresh_live_raw(
    symbols: list[str],
    base_config,
    live_root: Path,
    history_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    end = _floor_15m()
    cutoff = end - pd.Timedelta(days=history_days)
    raw_root = live_root / "raw" / "bybit"
    client = BybitClient(str(base_config.exchanges.bybit.base_url), base_config.exchanges.bybit.category)
    try:
        instruments = client.instruments(base_config.exchanges.bybit.settle_coin)
        tickers = client.tickers(base_config.exchanges.bybit.settle_coin)
        write_parquet(instruments, raw_root / "instruments.parquet")
        write_parquet(tickers, raw_root / "tickers.parquet")

        for idx, symbol in enumerate(symbols, start=1):
            print(f"refresh {idx}/{len(symbols)} {symbol}", flush=True)
            kline_path = raw_root / "klines" / f"{symbol}.parquet"
            funding_path = raw_root / "funding" / f"{symbol}.parquet"
            oi_path = raw_root / "open_interest" / f"{symbol}.parquet"

            k_start = _start_for_symbol(kline_path, "bar_open_time", end, history_days)
            f_start = _start_for_symbol(funding_path, "funding_time", end, history_days)
            oi_start = _start_for_symbol(oi_path, "oi_time", end, history_days)

            klines = client.klines(symbol, k_start, end, base_config.experiment.base_interval)
            funding = client.funding_history(symbol, f_start, end)
            oi = client.open_interest(symbol, oi_start, end, base_config.experiment.base_interval)

            safe_klines = _append_pruned(
                _read_optional(kline_path),
                klines,
                ["exchange", "symbol", "bar_open_time"],
                "bar_open_time",
                cutoff,
            )
            if not safe_klines.empty:
                safe_klines = safe_klines[
                    pd.to_datetime(safe_klines["bar_open_time"], utc=True, errors="coerce") <= end
                ].copy()
            write_parquet(safe_klines, kline_path)
            write_parquet(
                _append_pruned(
                    _read_optional(funding_path),
                    funding,
                    ["exchange", "symbol", "funding_time"],
                    "funding_time",
                    cutoff - pd.Timedelta(days=1),
                ),
                funding_path,
            )
            write_parquet(
                _append_pruned(
                    _read_optional(oi_path),
                    oi,
                    ["exchange", "symbol", "oi_time"],
                    "oi_time",
                    cutoff,
                ),
                oi_path,
            )
    finally:
        client.close()
    return instruments, tickers


def _build_live_prepared(
    live_root: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    base_config,
) -> pd.DataFrame:
    raw_root = live_root / "raw" / "bybit"
    instruments = _read_optional(raw_root / "instruments.parquet")
    klines = _concat_symbol_dir(raw_root / "klines")
    funding = _concat_symbol_dir(raw_root / "funding")
    oi = _concat_symbol_dir(raw_root / "open_interest")
    if klines.empty:
        raise FileNotFoundError("No live klines found after refresh.")

    features = build_feature_table(klines, funding, oi, instruments, base_config)
    features["month_start"] = _month_start(features["bar_open_time"])
    context30 = rank30.rename(
        columns={
            RANK30_COL: "turnover_rank_30d",
            "trailing_30d_turnover": "trailing_30d_turnover",
        }
    )[["month_start", "symbol", "turnover_rank_30d", "trailing_30d_turnover"]]
    context90 = rank90.rename(
        columns={
            RANK90_COL: "turnover_rank_90d",
            "trailing_90d_turnover": "trailing_90d_turnover",
        }
    )[["month_start", "symbol", "turnover_rank_90d", "trailing_90d_turnover"]]
    features = features.merge(context30, on=["month_start", "symbol"], how="left")
    features = features.merge(context90, on=["month_start", "symbol"], how="left")
    features[RANK30_COL] = pd.to_numeric(features["turnover_rank_30d"], errors="coerce")
    features[RANK90_COL] = pd.to_numeric(features["turnover_rank_90d"], errors="coerce")
    features["dynamic_all_rank"] = features[RANK30_COL]
    features["dynamic_all_trailing_turnover"] = pd.to_numeric(
        features["trailing_30d_turnover"], errors="coerce"
    )
    features["core_liquidity"] = (features[RANK30_COL] <= 30) & (features[RANK90_COL] <= 50)
    features["transient_hot"] = (features[RANK30_COL] <= 30) & (features[RANK90_COL] > 50)
    features["liquidity_quality"] = np.select(
        [
            features["core_liquidity"],
            features["transient_hot"],
            features[RANK90_COL].isna(),
        ],
        ["core_liquidity", "transient_hot", "rank90_missing"],
        default="non_core_liquidity",
    )
    prepared = _add_v03_report_columns(features, base_config)
    prepared = prepared[pd.to_numeric(prepared[RANK30_COL], errors="coerce").le(30)].copy()
    return prepared


def _write_outputs(
    prepared: pd.DataFrame,
    paper_config,
    signal_days: int,
    report_root: Path,
    paper_root: Path,
) -> None:
    report_root = ensure_dir(report_root)
    paper_root = ensure_dir(paper_root)
    latest = pd.to_datetime(prepared["feature_time"], utc=True, errors="coerce").max()
    signal_start = latest - pd.Timedelta(days=signal_days)
    signals, trades, shadow = build_v05_paper_ledger(prepared, paper_config, signal_start)
    daily = _daily_summary(signals, trades)
    skipped = signals[signals["status"].eq("skipped")].copy() if not signals.empty else signals.copy()

    for root in [report_root, paper_root]:
        write_parquet(signals, root / "paper_signals.parquet")
        write_parquet(trades, root / "paper_trades.parquet")
    daily.to_csv(report_root / "daily_summary.csv", index=False)
    skipped.to_csv(report_root / "skipped_signals.csv", index=False)
    shadow.to_csv(report_root / "portfolio_shadow_summary.csv", index=False)
    _write_status_reports(report_root, signals, trades, shadow, prepared, paper_config)
    print(f"latest_feature_time={latest}")
    print(f"signals={len(signals)} trades={len(trades)}")
    if C2_SIGNAL_COL in prepared.columns:
        recent = prepared[pd.to_datetime(prepared["feature_time"], utc=True) >= signal_start]
        print(f"recent_c2_events={int(recent[C2_SIGNAL_COL].fillna(False).sum())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one isolated v0.5 paper-live refresh.")
    parser.add_argument("--base-config", default="configs/v0_3.yaml")
    parser.add_argument("--paper-config", default="configs/v0_5_paper_live.yaml")
    parser.add_argument("--live-root", default="data/live_v05")
    parser.add_argument("--history-days", type=int, default=45)
    parser.add_argument("--signal-days", type=int, default=7)
    parser.add_argument("--max-symbols", type=int, default=0)
    args = parser.parse_args()

    base_config = load_config(args.base_config)
    paper_config = load_v05_config(args.paper_config)
    feature_path = base_config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments_path = raw_path(base_config.paths.data_root, "bybit", "instruments")
    instruments = _read_optional(instruments_path)
    if instruments.empty:
        raise FileNotFoundError(f"Missing instruments: {instruments_path}")

    rank30, rank90 = _rank_context(feature_path, instruments, base_config)
    symbols = _live_symbols(rank30, _floor_15m(), args.max_symbols or None)
    live_root = Path(args.live_root)
    print(f"live_symbols={len(symbols)} {','.join(symbols)}")
    _refresh_live_raw(symbols, base_config, live_root, args.history_days)
    prepared = _build_live_prepared(live_root, rank30, rank90, base_config)
    processed_path = live_root / "processed" / "v0_5_live_features.parquet"
    write_parquet(prepared, processed_path)
    _write_outputs(prepared, paper_config, args.signal_days, REPORT_ROOT, PAPER_DATA_ROOT)


if __name__ == "__main__":
    main()
