from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from pressure_graph.clients.bybit import BybitClient
from pressure_graph.clients.deribit import DeribitClient
from pressure_graph.io import ensure_dir


DEFAULT_START = pd.Timestamp("2025-06-03 06:00:00", tz="UTC")
DEFAULT_END = pd.Timestamp("2026-06-03 06:00:00", tz="UTC")
MEMBERSHIP_PATH = Path("reports/v11_0_balanced_topology_break/monthly_balanced_membership.csv")
OUTPUT_ROOT = Path("data/external/orthogonal_volatility")


def _atomic_parquet(frame: pd.DataFrame, output: Path) -> Path:
    ensure_dir(output.parent)
    temporary = output.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(output)
    return output


def _monthly_last_wednesdays(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    months = pd.date_range(start=start.floor("D").replace(day=1), end=end, freq="MS", tz="UTC")
    expiries = []
    for month in months:
        month_end = month + pd.offsets.MonthEnd(0)
        days_back = (month_end.weekday() - 2) % 7
        expiries.append(pd.Timestamp(month_end - pd.Timedelta(days=days_back)))
    return expiries


def _dvol_instrument(expiry: pd.Timestamp) -> str:
    return f"BTCDVOL_USDC-{expiry:%d%b%y}".upper()


def _download_account_ratio(
    symbol: str, start: pd.Timestamp, end: pd.Timestamp, output_root: Path
) -> tuple[str, int]:
    client = BybitClient()
    try:
        frame = client.account_ratio(symbol, start, end, interval="1h")
    finally:
        client.close()
    if not frame.empty:
        _atomic_parquet(frame, output_root / "bybit_account_ratio_1h" / f"{symbol}.parquet")
    return symbol, len(frame)


def _download_chunked_chart(
    client: DeribitClient,
    instrument: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_days: int = 30,
) -> pd.DataFrame:
    frames = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + pd.Timedelta(days=chunk_days), end)
        frame = client.chart_data(instrument, cursor, chunk_end, resolution_minutes=60)
        if not frame.empty:
            frames.append(frame)
        cursor = chunk_end + pd.Timedelta(milliseconds=1)
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("bar_open_time", keep="last")
        .sort_values("bar_open_time")
        .reset_index(drop=True)
    )


def backfill(
    start: pd.Timestamp,
    end: pd.Timestamp,
    output_root: Path = OUTPUT_ROOT,
    workers: int = 8,
    skip_account_ratio: bool = False,
) -> None:
    if not skip_account_ratio:
        membership = pd.read_csv(MEMBERSHIP_PATH)
        symbols = sorted(set(membership["symbol"].astype(str)) | {"BTCUSDT"})
        print(f"account-ratio symbols={len(symbols)} start={start} end={end}")
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(
                    _download_account_ratio, symbol, start, end, output_root
                ): symbol
                for symbol in symbols
            }
            for future in as_completed(futures):
                symbol, rows = future.result()
                print(f"account-ratio {symbol}: {rows} rows")

    client = DeribitClient()
    try:
        deribit_start = min(start, pd.Timestamp("2021-03-24", tz="UTC"))
        for currency in ("BTC", "ETH"):
            frames = []
            cursor = deribit_start
            while cursor < end:
                chunk_end = min(cursor + pd.Timedelta(days=30), end)
                frame = client.volatility_index_data(currency, cursor, chunk_end)
                if not frame.empty:
                    frames.append(frame)
                cursor = chunk_end + pd.Timedelta(milliseconds=1)
            if frames:
                dvol = (
                    pd.concat(frames, ignore_index=True)
                    .drop_duplicates("dvol_time", keep="last")
                    .sort_values("dvol_time")
                )
                _atomic_parquet(dvol, output_root / "deribit_dvol_1h" / f"{currency}.parquet")
                print(f"dvol {currency}: {len(dvol)} rows")

        perpetual_start = min(deribit_start, pd.Timestamp("2021-03-01", tz="UTC"))
        perpetual = _download_chunked_chart(client, "BTC-PERPETUAL", perpetual_start, end)
        if not perpetual.empty:
            _atomic_parquet(
                perpetual, output_root / "deribit_perpetual_1h" / "BTC-PERPETUAL.parquet"
            )
            print(f"BTC-PERPETUAL: {len(perpetual)} rows")

        futures_root = output_root / "deribit_dvol_futures_1h"
        contract_start = max(deribit_start, pd.Timestamp("2023-05-01", tz="UTC"))
        for expiry in _monthly_last_wednesdays(contract_start, end + pd.Timedelta(days=40)):
            instrument = _dvol_instrument(expiry)
            try:
                frame = client.chart_data(
                    instrument,
                    expiry - pd.Timedelta(days=42),
                    expiry + pd.Timedelta(hours=9),
                    resolution_minutes=60,
                )
            except RuntimeError:
                continue
            if frame.empty:
                continue
            frame = frame[
                frame["bar_open_time"].ge(expiry - pd.Timedelta(days=42))
                & frame["bar_open_time"].le(expiry + pd.Timedelta(hours=9))
            ].copy()
            frame["expiration_time"] = expiry + pd.Timedelta(hours=8)
            _atomic_parquet(frame, futures_root / f"{instrument}.parquet")
            print(f"dvol-future {instrument}: {len(frame)} rows")
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=str(DEFAULT_START))
    parser.add_argument("--end", default=str(DEFAULT_END))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-account-ratio", action="store_true")
    args = parser.parse_args()
    backfill(
        pd.Timestamp(args.start).tz_convert("UTC"),
        pd.Timestamp(args.end).tz_convert("UTC"),
        args.output_root,
        args.workers,
        args.skip_account_ratio,
    )


if __name__ == "__main__":
    main()
